from transformers import Blip2Processor, Blip2ForConditionalGeneration, AddedToken
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from src.webdatasets_util import (
    cc2coco_format,
    create_webdataset_tar,
    read_coco_format_wds,
)
from src.hooks import (
    get_self_attention,
    process_self_attention,
    get_second_last_out,
    get_vit_out,
    get_dinov1_patches,
    feats,
)
from io import BytesIO
import numpy as np
import torchvision.transforms as T
import torch
import timm
import tarfile
import webdataset as wds
import requests
import math
import json
import clip
import argparse
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
"""
CUDA_VISIBLE_DEVICES=3 python dino_extraction_v2.py --ann_path ../coco/captions_val2014.json --out_path ../coco2014_b14/tmp_val.pth --model dinov2_vitb14_reg --resize_dim 448 --crop_dim 448 --extract_avg_self_attn --extract_disentangled_self_attn --extract_patch_tokens --extract_cls
CUDA_VISIBLE_DEVICES=2 python dino_extraction_v2.py --ann_path ../coco/captions_train2014.json --out_path ../coco2014_b14/tmp_train.pth --model dinov2_vitb14_reg --resize_dim 448 --crop_dim 448 --extract_avg_self_attn --extract_disentangled_self_attn --extract_patch_tokens --extract_cls

CUDA_VISIBLE_DEVICES=3 python dino_extraction_v2.py --ann_path ../coco/captions_val2014.json --out_path ../coco2014_b14/tmp_val.pth --model dinov2_vitb14_reg --resize_dim 448 --crop_dim 448 --extract_cls
CUDA_VISIBLE_DEVICES=2 python dino_extraction_v2.py --ann_path ../coco/captions_train2014.json --out_path ../coco2014_b14/tmp_train.pth --model dinov2_vitb14_reg --resize_dim 448 --crop_dim 448 --extract_cls

CUDA_VISIBLE_DEVICES=2 python text_features_extraction.py --ann_path ../coco2014_b14/tmp_train.pth
CUDA_VISIBLE_DEVICES=3 python text_features_extraction.py --ann_path ../coco2014_b14/tmp_val.pth

python dino_extraction_v2.py --ann_path ../coco/captions_train2014.json --out_path /data1/bohnsix/talking2dino/coco2014_dinov2reg_l_p14/train.pth --model dinov2_vitl14_reg --resize_dim 448 --crop_dim 448 --extract_avg_self_attn --extract_disentangled_self_attn --extract_patch_tokens
python dino_extraction_v2.py --ann_path ../coco/captions_val2014.json --out_path /data1/bohnsix/talking2dino/coco2014_dinov2reg_l_p14/val.pth --model dinov2_vitl14_reg --resize_dim 448 --crop_dim 448 --extract_avg_self_attn --extract_disentangled_self_attn --extract_patch_tokens

python text_features_extraction.py --ann_path /data1/bohnsix/talking2dino/coco2014_dinov2reg_l_p14/train.pth
python text_features_extraction.py --ann_path /data1/bohnsix/talking2dino/coco2014_dinov2reg_l_p14/val.pth

mkdir -p /data1/bohnsix/talking2dino/coco2014_dinov2reg_l_p14  
mkdir avg_self_attn_out  
mkdir cls_token  
mkdir disentangled_self_attn  
mkdir patch_tokens
"""

# Initialize global variables
# feats = {}
# num_global_tokens = 1
# num_patch_tokens = 518 // 14 * 518 // 14
# num_tokens = num_global_tokens + num_patch_tokens
# embed_dim = 1024
# num_attn_heads = 16
# scale = 0.125
# batch_size_ = 1


def generate_caption(model, processor, images, prompt="a photography of"):
    image_token = AddedToken("<image>", normalized=False, special=True)
    processor.tokenizer.add_tokens([image_token], special_tokens=True)

    # pad for efficient computation
    model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64)
    model.config.image_token_index = len(processor.tokenizer) - 1
    inputs = processor(
        images=images, text=[prompt] * len(images), return_tensors="pt"
    ).to(next(model.parameters()).device)
    inputs["pixel_values"] = inputs["pixel_values"].float()

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [x.strip() for x in generated_text]


def run_dinov2_extraction(
    model_name,
    data_dir,
    ann_path,
    batch_size,
    resize_dim=518,
    crop_dim=518,
    out_path=None,
    write_as_wds=False,
    num_shards=25,
    n_in_splits=4,
    in_batch_offset=0,
    out_offset=0,
    extract_cls=False,
    extract_avg_self_attn=False,
    extract_second_last_out=False,
    extract_patch_tokens=False,
    extract_self_attn_maps=False,
    extract_disentangled_self_attn=False,
    blip_model_name=None,
):
    device = "cuda" if torch.cuda.is_available else "cpu"

    # global num_global_tokens, num_patch_tokens, num_tokens, embed_dim, num_attn_heads, scale, batch_size_

    num_global_tokens = 1 if "reg" not in model_name else 5
    num_patch_tokens = crop_dim // 14 * crop_dim // 14
    num_tokens = num_global_tokens + num_patch_tokens
    if "vitl" in model_name or "vit_large" in model_name or "ViT-L" in model_name:
        embed_dim = 1024
    elif "vitb" in model_name or "vit_base" in model_name or "ViT-B" in model_name:
        embed_dim = 768
    elif "vits" in model_name or "vit_small" in model_name:
        embed_dim = 384
    else:
        raise Exception("Unknown ViT model")

    num_attn_heads = 16 if not "vits" in model_name else 6
    scale = 0.125
    batch_size_ = batch_size

    # loading the model
    if "dinov2" in model_name:
        model_family = "facebookresearch/dinov2"
        model = torch.hub.load(
            "/home/bohnsix/.cache/torch/hub/facebookresearch_dinov2_main",
            model_name,
            source="local",
        )
        image_transforms = T.Compose(
            [
                T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(crop_dim),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    elif (
        "mae" in model_name
        or "sam" in model_name
        or "clip" in model_name
        or "dino" in model_name
    ):
        model = timm.create_model(
            model_name,
            num_classes=0,  # remove classifier nn.Linear
            # pretrained=True,
            # img_size=crop_dim
        )
        local_ckpt = f"/data/bohnsix/timm_offline/{model_name}.pth"
        state = torch.load(local_ckpt, map_location="cpu")
        model.load_state_dict(state)
        # the resize dimension will be the one native of the model
        data_config = timm.data.resolve_model_data_config(model)
        data_config["input_size"] = (3, crop_dim, crop_dim)
        image_transforms = timm.data.create_transform(**data_config, is_training=False)

        # adjusting the dimensions
        if "mae" in model_name:
            num_patch_tokens = crop_dim // 16 * crop_dim // 16
            num_tokens = 1 + num_patch_tokens
            model.patch_embed.img_size = (448, 448)
            model.patch_embed.grid_size = (448 // 16, 448 // 16)  # 56×56
            old_pos_embed = model.pos_embed  # shape: [1, 197, 1024] (cls+14*14)
            cls_pos, patch_pos = (
                old_pos_embed[:, :1],
                old_pos_embed[:, 1:],
            )  # 拆分 cls 与 patch
            B, L, C = patch_pos.shape
            H_old = W_old = int(L**0.5)  # 14
            H_new = W_new = 448 // 16  # 56
            patch_pos = patch_pos.reshape(B, C, H_old, W_old)
            patch_pos = F.interpolate(
                patch_pos, size=(H_new, W_new), mode="bicubic", align_corners=False
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(B, H_new * W_new, C)
            new_pos_embed = torch.cat([cls_pos, patch_pos], dim=1)  # [1, 3137, 1024]
            model.pos_embed = torch.nn.Parameter(new_pos_embed)
        if "dino" in model_name:
            num_patch_tokens = crop_dim // 16 * crop_dim // 16
            num_tokens = 1 + num_patch_tokens
        elif "sam" in model_name:
            num_patch_tokens = crop_dim // 16 * crop_dim // 16
            num_tokens = num_patch_tokens
            num_global_tokens = 0
            model.blocks[-1].register_forward_hook(get_vit_out)
        elif "clip" in model_name:
            crop_dim = resize_dim = 224
            num_patch_tokens = (
                crop_dim // 16 * crop_dim // 16
                if "vit_base" in model_name
                else crop_dim // 14 * crop_dim // 14
            )
            num_tokens = 1 + num_patch_tokens
    elif "ViT" in model_name:
        # CLIP extraction using clip library
        # use it only for CLS token
        model, image_transforms = clip.load(model_name, device)
    else:
        raise Exception("Unknown ViT model")

    model.eval()
    model.to(device)

    if blip_model_name is not None:
        blip_processor = Blip2Processor.from_pretrained(blip_model_name)
        blip_model = Blip2ForConditionalGeneration.from_pretrained(
            blip_model_name,
            torch_dtype=torch.float16,  # device_map ='auto'
        ).to(device)
        blip_processor.num_query_tokens = blip_model.config.num_query_tokens

    if os.path.isdir(ann_path):
        # if we have a dir as path we assume that the path refere to gcc3m webdataset
        data = cc2coco_format(ann_path, n_in_splits, in_batch_offset)
    elif ".tar" in ann_path:
        # if we have a webdataset template, we read the input dataset as webdatset assuming that it is in COCO format
        data = read_coco_format_wds(ann_path)
    else:
        # otherwise we treat the dataset as a COCO dataset
        if ann_path.endswith(".json"):
            print("Loading the annotations JSON")
            with open(ann_path, "r") as f:
                data = json.load(f)
        else:
            print("Loading the annotations PTH")
            data = torch.load(ann_path)

    if extract_second_last_out:
        model.blocks[-2].register_forward_hook(get_second_last_out)
    if (
        extract_avg_self_attn
        or extract_self_attn_maps
        or extract_disentangled_self_attn
    ):
        model.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)

    print("Starting the features extraction...")
    n_imgs = len(data["images"])
    n_batch = math.ceil(n_imgs / batch_size)
    n_errors = 0
    for i in tqdm(range(n_batch)):
        start = i * batch_size
        end = start + batch_size if i < n_batch - 1 else n_imgs
        batch_size_ = end - start
        raw_imgs = []
        failed_ids = []
        for j in range(start, end):
            if "jpg" in data["images"][j]:
                # CC3M case
                pil_img = data["images"][j]["jpg"]
                # saving space by eliminating the jpg
                del data["images"][j]["jpg"]
            else:
                # COCO or Recap case

                # Recap
                if "http" in data["images"][j]["file_name"]:
                    try:
                        pil_img = Image.open(
                            BytesIO(
                                requests.get(data["images"][j]["file_name"]).content
                            )
                        )
                    except Exception as e:
                        # genererate dummy image
                        pil_img = Image.new("RGB", (224, 224))
                        failed_ids.append(j)
                        n_errors += 1
                # COCO 2014
                elif "train" in data["images"][j]["file_name"]:
                    pil_img = Image.open(
                        os.path.join(
                            data_dir, f"train2014/{data['images'][j]['file_name']}"
                        )
                    )
                elif "val" in data["images"][j]["file_name"]:
                    pil_img = Image.open(
                        os.path.join(
                            data_dir, f"val2014/{data['images'][j]['file_name']}"
                        )
                    )
                elif "test" in data["images"][j]["file_name"]:
                    pil_img = Image.open(
                        os.path.join(
                            data_dir, f"test2014/{data['images'][j]['file_name']}"
                        )
                    )
                # COCO 2017
                elif "train" in data["images"][j]["coco_url"]:
                    pil_img = Image.open(
                        os.path.join(
                            data_dir, f"train2017/{data['images'][j]['file_name']}"
                        )
                    )
                elif "val" in data["images"][j]["coco_url"]:
                    pil_img = Image.open(
                        os.path.join(
                            data_dir, f"val2017/{data['images'][j]['file_name']}"
                        )
                    )
                else:
                    # genererate dummy image
                    pil_img = Image.new("RGB", (224, 224))
                    failed_ids.append(j)
                    n_errors += 1

            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            raw_imgs.append(pil_img)

        batch_imgs = torch.stack([image_transforms(img) for img in raw_imgs]).to(device)

        with torch.no_grad():
            if "dinov2" in model_name:
                outs = model(batch_imgs, is_training=True)
            elif "mae" in model_name or "clip" in model_name or "dino" in model_name:
                output = model.forward_features(batch_imgs)
                # reporting output in DINOv2 format
                outs = {
                    "x_norm_clstoken": output[:, 0, :],
                    "x_norm_patchtokens": output[:, 1:, :],
                }
            elif "sam" in model_name:
                sam_output = model.forward_features(batch_imgs)
                if extract_cls:
                    cls = model.forward_head(sam_output, pre_logits=True)
                else:
                    cls = None
                outs = {
                    "x_norm_clstoken": cls,
                    "x_norm_patchtokens": feats["vit_out"].reshape(
                        batch_size_, num_patch_tokens, embed_dim
                    ),
                }
            elif "ViT" in model_name:
                outs = {"x_norm_clstoken": model.encode_image(batch_imgs)}
            cls_token = outs["x_norm_clstoken"]
            if (
                extract_avg_self_attn
                or extract_self_attn_maps
                or extract_disentangled_self_attn
            ):
                self_attn, self_attn_maps = process_self_attention(
                    feats["self_attn"],
                    (end - start),
                    num_tokens,
                    num_attn_heads,
                    embed_dim,
                    scale,
                    num_global_tokens,
                    ret_self_attn_maps=True,
                )
                # print(self_attn.shape, self_attn_maps.shape)  # [64, 1024] [64, 16, 1024]

            if extract_avg_self_attn:
                avg_self_attn_token = (
                    self_attn.unsqueeze(-1) * outs["x_norm_patchtokens"]
                ).mean(
                    dim=1
                )  # [64, 768]
            if extract_disentangled_self_attn:
                self_attn_maps = self_attn_maps.softmax(dim=-1)  # [64, 16, 1024]
                # [64, 1, 1024, 768] * [64, 16, 1024, 1] -> [64, 16, 1024, 768] -> [64, 16, 768]
                disentangled_self_attn = (
                    outs["x_norm_patchtokens"].unsqueeze(1)
                    * self_attn_maps.unsqueeze(-1)
                ).mean(dim=2)
            if extract_second_last_out:
                # keeping only the CLS token
                second_last_cls = feats["second_last_out"][:, 0, :]
            if blip_model_name is not None:
                new_capts = generate_caption(blip_model, blip_processor, raw_imgs)

        # writing the outputs in the original data
        for j in range(start, end):
            if j in failed_ids:
                continue

            if extract_cls or (
                not extract_avg_self_attn and not extract_second_last_out
            ):
                # data['images'][j]['dino_features'] = cls_token[j-start].to('cpu')
                cls_token_path = os.path.join(os.path.dirname(out_path), "cls_token")
                np.save(
                    os.path.join(cls_token_path, f"{data['images'][j]['id']}.npy"),
                    cls_token[j - start].to("cpu"),
                )
            if extract_avg_self_attn:
                data["images"][j]["avg_self_attn_out"] = avg_self_attn_token[
                    j - start
                ].to("cpu")
            if extract_second_last_out:
                data["images"][j]["second_last_out"] = second_last_cls[j - start].to(
                    "cpu"
                )
            if extract_patch_tokens:
                # data['images'][j]['patch_tokens'] = outs['x_norm_patchtokens'][j - start].to('cpu')
                patch_tokes_path = os.path.join(
                    os.path.dirname(out_path), "patch_tokens"
                )
                np.save(
                    os.path.join(patch_tokes_path, f"{data['images'][j]['id']}.npy"),
                    outs["x_norm_patchtokens"][j - start].to("cpu"),
                )
            if extract_self_attn_maps:
                data["images"][j]["self_attn_maps"] = self_attn_maps[j - start].to(
                    "cpu"
                )
            if extract_disentangled_self_attn:
                data["images"][j]["disentangled_self_attn"] = disentangled_self_attn[
                    j - start
                ].to("cpu")
            if blip_model_name is not None:
                data["annotations"][j]["caption"] = new_capts[j - start]

    print("Feature extraction done!")
    print(f"Failed to extract {n_errors} of {len(data['images'])}")

    if write_as_wds:
        os.makedirs(out_path, exist_ok=True)
        create_webdataset_tar(data, out_path, num_shards, out_offset)
    else:
        if out_path is None:
            # we use as output path the ann_path but with the extension pth
            out_path = os.path.splitext(ann_path)[0] + ".pth"
        torch.save(data, out_path)
        print(f"Features saved at {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann_path",
        type=str,
        default="coco/test1k.json",
        help="Directory of the annotation file",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--data_dir", type=str, default="../coco/", help="Directory of the images"
    )
    parser.add_argument(
        "--blip_model",
        type=str,
        default=None,
        help="BLIP model to recaption with. If None we will use standard captions. For CC3M we use Salesforce/blip2-opt-6.7b-coco",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dinov2_vitl14_reg",
        help="Model configuration to extract features from",
    )
    parser.add_argument("--resize_dim", type=int, default=518, help="Resize dimension")
    parser.add_argument("--crop_dim", type=int, default=518, help="Crop dimension")
    parser.add_argument(
        "--extract_cls",
        default=False,
        action="store_true",
        help="If setted, adds the CLS token to the output",
    )
    parser.add_argument(
        "--extract_avg_self_attn",
        default=False,
        action="store_true",
        help="If setted, adds the token obtained by weighting using the self attention to the output",
    )
    parser.add_argument(
        "--extract_second_last_out",
        default=False,
        action="store_true",
        help="If setted, adds the second last CLS to the output",
    )
    parser.add_argument(
        "--extract_patch_tokens",
        default=False,
        action="store_true",
        help="If setted, we extract all the patch tokens",
    )
    parser.add_argument(
        "--extract_self_attn_maps",
        default=False,
        action="store_true",
        help="If setted, we extract all the self-attention maps",
    )
    parser.add_argument(
        "--extract_disentangled_self_attn",
        default=False,
        action="store_true",
        help="If setted, adds the token obtained by weighting using the self attention to the output, without averaging the attention heads",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Pth of the output file, if setted to None. out_pat = ann_path",
    )
    parser.add_argument(
        "--write_as_wds",
        action="store_true",
        default=False,
        help="If setted, the output will be written as a webdataset",
    )
    parser.add_argument(
        "--n_shards",
        type=int,
        default=25,
        help="Number of shards in which the webdataset is splitted. Only relevant if --write_as_wds is setted.",
    )
    parser.add_argument(
        "--n_in_split",
        type=int,
        default=1,
        help="Number of splits in which we want to divide the tar files. For example, with 4 n_split we elaborate 332 // 4 = 83 tar files.",
    )
    parser.add_argument(
        "--in_batch_offset",
        type=int,
        default=0,
        help="Of the n_splits in which we have divided tars, we decide which of them elaborate",
    )
    parser.add_argument(
        "--out_offset", type=int, default=0, help="Index of the first shard to save"
    )
    args = parser.parse_args()

    run_dinov2_extraction(
        args.model,
        args.data_dir,
        args.ann_path,
        args.batch_size,
        args.resize_dim,
        args.crop_dim,
        args.out_path,
        args.write_as_wds,
        args.n_shards,
        args.n_in_split,
        args.in_batch_offset,
        args.out_offset,
        args.extract_cls,
        args.extract_avg_self_attn,
        args.extract_second_last_out,
        args.extract_patch_tokens,
        args.extract_self_attn_maps,
        args.extract_disentangled_self_attn,
        args.blip_model,
    )


if __name__ == "__main__":
    main()


"""
python dino_extraction_v2.py --ann_path ../coco/captions_val2014.json --out_path ../coco2014_b14/vision_val.pth --model dinov2_vitb14_reg --resize_dim 448 --crop_dim 448 --extract_patch_tokens
python dino_extraction_v2.py --ann_path ../coco/captions_train2014.json --out_path ../coco2014_b14/vision_train.pth --model dinov2_vitb14_reg --resize_dim 448 --crop_dim 448 --extract_patch_tokens

modellist = [samvit_base_patch16, samvit_large_patch16,
             vit_base_patch16_224.mae, vit_large_patch16_224.mae,
             vit_small_patch16_224.dino, vit_base_patch16_224.dino,

             vit_small_patch14_dinov2, vit_base_patch14_dinov2, vit_large_patch14_dinov2, 
             vit_small_patch14_reg4_dinov2, vit_base_patch14_reg4_dinov2, vit_large_patch14_reg4_dinov2
             dinov2_vitb14_reg, dinov2_vitl14_reg
]
"""
