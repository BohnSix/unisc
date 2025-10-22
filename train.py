import argparse

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import yaml
import importlib
import torchvision.transforms as T
import clip

from src.dataset import DinoClipDataset, COCOCaptions
from src.metrics import get_image_and_text_tensor, i2t, t2i
from src.model import ProjectionLayer
from src.train_util import do_train, set_seed
from tqdm import tqdm

device = "cuda"


def train_and_eval(
    config_file,
    train_dataset,
    val_dataset,
    texts=None,
    images=None,
    model_type="cls",
    test_set=None,
    optimizer="adam",
    weight_decay=0.05,
    scheduler="linear",
    warmup=0,
    name_pedix="",
    save_head_activations=None,
    wandb=False,
    model_k=None,
    out_dir="weights4",
):
    set_seed(123)
    model_name = os.path.basename(config_file).split(".")[0]
    if name_pedix != "":
        model_name += f"_{name_pedix}"
    if model_type == "":
        out_path = os.path.join(out_dir, f"{model_name}")
    else:
        out_path = os.path.join(out_dir, f"{model_name}_{model_type}")

    config = {}
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if model_k is not None:
        config["model"]["k"] = model_k
        out_path = out_path + f"_k{config['model']['k']}"

    model_class_name = config["model"].get("model_class", "ProjectionLayer")
    ModelClass = getattr(importlib.import_module("src.model"), model_class_name)

    model = ModelClass.from_config(config["model"])
    model.to(device)
    # print(model)

    model, train_losses, val_losses = do_train(
        model,
        train_dataset,
        val_dataset,
        config["train"],
        optimizer_name=optimizer,
        weight_decay=weight_decay,
        scheduler_name=scheduler,
        warmup=warmup,
        save_head_attivations=save_head_activations,
        wandb=wandb,
        out_path=out_path,
        verbose=True,
    )

    if model_k is not None:
        plot_losses(
            train_losses, val_losses, f"plot_loss/train_{model_name}_k{model_k}.png"
        )
    else:
        plot_losses(train_losses, val_losses, f"plot_loss/train_{model_name}.png")

    torch.save(model.state_dict(), f"{out_path}.pth")
    print(f"Saved model at {out_path}.pth\n")

    if model_type == "patch_tokens":
        # if we are working with weighted attention head, images test tensors must be calculated after the model is trained
        images, texts = get_image_and_text_tensor(
            args.test_dataset, args.feature_name, model=model
        )

    if texts is not None:
        texts_proj = model.project_clip_txt(texts.to(device).float()).detach().cpu()
        print("Retrieval results (t2i, i2t):")
        t2i_rk = t2i(images.numpy(), texts_proj.numpy())
        i2t_rk = i2t(images.numpy(), texts_proj.numpy())

        data = [["t2i"] + list(t2i_rk), ["i2t"] + list(i2t_rk)]

        columns = ["type", "r1", "r5", "r10", "median_rank", "mean_rank"]

        df = pd.DataFrame(data, columns=columns)
        print(df)


def plot_losses(train_losses, val_losses, figname="losses.png"):
    plt.figure(figsize=(10, 6))

    plt.plot(train_losses, label="Training Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="o")

    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(figname)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crop_dim",
        type=int,
        default=518,
        help="Crop dimension, irrelevant in case of pre-extracted features",
    )
    parser.add_argument(
        "--data_dir", type=str, default="../coco/", help="Directory of the images"
    )
    parser.add_argument(
        "--feature_name",
        type=str,
        default="disentangled_self_attn",
        help="Name of the field of the features",
    )
    parser.add_argument(
        "--text_features",
        type=str,
        default="ann_feats",
        help="Name of the field of the text features",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="dinov2_vitl14_reg",
        help="Model configuration",
    )
    parser.add_argument(
        "--resize_dim",
        type=int,
        default=518,
        help="Resize dimension, irrelevant in case of pre-extracted features",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="../coco2014_b14_448_dense/test.pth",
        help="Directory of the test file",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="../coco2014_b14_448_dense/val/shard{00000..00009}.tar",
        help="Directory of the train file",
    )
    parser.add_argument(
        "--val_dataset",
        type=str,
        default="../coco2014_b14_448_dense/train/shard{00000..00009}.tar",
        help="Directory of the validation file",
    )
    parser.add_argument(
        "--use_wandb",
        default=False,
        action="store_true",
        help="If setted wandb will be used",
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="Optimizer to be used"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="Weight decay to be used"
    )
    parser.add_argument(
        "--scheduler", type=str, default="linear", help="Scheduler to be used"
    )
    parser.add_argument(
        "--name_pedix",
        type=str,
        default="",
        help="Model name to append to name of the configuration for weights name",
    )
    parser.add_argument(
        "--save_head_activations",
        type=str,
        default=None,
        help="If setted the occurences of the head activation of the last epoch will be saved at that path",
    )
    parser.add_argument("--warmup", type=int, default=0, help="Number of warmup epochs")
    parser.add_argument(
        "--channelk", type=int, default=None, help="Number of channels to be compress"
    )
    args = parser.parse_args()

    if args.use_wandb:
        import wandb

        os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"
        wandb.init(project="dino-clip")

    # if the model config name contains 'dino', it means that we do not work with pre-extracted features
    if not ("dino" in args.model_config):
        print("Loading val dataset...")
        val_dataset = DinoClipDataset(
            args.val_dataset,
            features_name=(
                "avg_self_attn_out"
                if args.feature_name == "disentangled_self_attn"
                else args.feature_name
            ),
            # features_name=args.feature_name,
            text_features=args.text_features,
            # load_attn_maps=args.feature_name == 'patch_tokens',
            load_attn_maps=False,
            is_wds=".tar" in args.val_dataset,
        )
        print("Loading train dataset...")
        train_dataset = DinoClipDataset(
            args.train_dataset,
            features_name=args.feature_name,
            text_features=args.text_features,
            # load_attn_maps=args.feature_name == 'patch_tokens',
            load_attn_maps=False,
            is_wds=".tar" in args.train_dataset,
        )
        print("Datasets loaded!")

    else:
        image_transforms = T.Compose(
            [
                T.Resize(args.resize_dim, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(args.crop_dim),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        train_dataset = COCOCaptions(
            args.train_dataset,
            "coco/train2014",
            "train",
            image_transforms,
            clip.tokenize,
        )
        val_dataset = COCOCaptions(
            args.val_dataset, "coco/val2014", "val", image_transforms, clip.tokenize
        )

    if args.feature_name == "patch_tokens" and False:
        if args.text_features == "clip_second_last_out":
            images, texts, text_argmax = get_image_and_text_tensor(
                args.test_dataset, args.feature_name, args.text_features
            )
        else:
            images, texts = get_image_and_text_tensor(
                args.test_dataset, args.feature_name, args.text_features
            )
    else:
        images = None
        texts = None

    train_and_eval(
        args.model_config,
        train_dataset,
        val_dataset,
        texts,
        images,
        test_set=args.test_dataset,
        model_type="",
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        warmup=args.warmup,
        name_pedix=args.name_pedix,
        save_head_activations=args.save_head_activations,
        wandb=args.use_wandb,
        model_k=args.channelk,
    )

    from wx_push import tell_me

    tell_me(
        title=f"Ablation Study: Training First finished",
        content="",
    )


"""
CUDA_VISIBLE_DEVICES=3 nohup python -u train.py --model configs/second_stage/vitb_jscc_infonce.yaml --feature_name patch_tokens --train_dataset /data1/bohnsix/talking2dino_cvpr/coco2014_dinov2reg_b_p14/train.pth --val_dataset /data1/bohnsix/talking2dino_cvpr/coco2014_dinov2reg_b_p14/val.pth --channelk 12 > train_cbr01.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u train.py --model configs/second_stage/vitb_jscc_infonce.yaml --feature_name patch_tokens --train_dataset /data1/bohnsix/talking2dino_cvpr/coco2014_dinov2reg_b_p14/train.pth --val_dataset /data1/bohnsix/talking2dino_cvpr/coco2014_dinov2reg_b_p14/val.pth --channelk 24 > train_cbr02.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --model configs/second_stage/vitb_jscc_infonce.yaml --feature_name patch_tokens --train_dataset /data1/bohnsix/talking2dino_cvpr/coco2014_dinov2reg_b_p14/train.pth --val_dataset /data1/bohnsix/talking2dino_cvpr/coco2014_dinov2reg_b_p14/val.pth --channelk 36 > train_cbr03.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --model configs/second_stage/vitb_jscc_infonce.yaml --feature_name patch_tokens --train_dataset /data1/bohnsix/talking2dino_cvpr/coco2014_dinov2reg_b_p14/train.pth --val_dataset /data1/bohnsix/talking2dino_cvpr/coco2014_dinov2reg_b_p14/val.pth --channelk 48 > train_cbr04.log 2>&1 &

### Ablation Study
CUDA_VISIBLE_DEVICES=3 nohup python -u train.py --model configs/second_stage/vitb_jscc_infonce.yaml --feature_name patch_tokens --train_dataset /data1/bohnsix/talking2dino_cvpr/coco2014_dinov2reg_b_p14/train.pth --val_dataset /data1/bohnsix/talking2dino_cvpr/coco2014_dinov2reg_b_p14/val.pth --channelk 48  > train_ablation_cbr01.log 2>&1 &

1708286

"""
