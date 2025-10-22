
# Environment

```bash
conda activate bohnsix_talk2dino
```

CUDA 11.8
```yaml
torch                        1.13.1+cu117
torchvision                  0.14.1
torchaudio                   0.13.1
torchmetrics                 1.5.2

timm                         1.0.17
transformers                 4.37.2

mmcv                         2.2.0
mmcv-full                    1.6.2
mmengine                     0.8.4
mmsegmentation               0.27.0
```


# Preprocess

```bash
bash run_dinov2reg_preprocess.sh
```

# Train 

Using DINOv2-Reg to extract features.

### First Stage

```bash
python train.py --config configs/first_stage/vitb_jscc_infonce.yaml
```

### Second Stage

```bash
python train.py --config configs/second_stage/vitb_jscc_infonce.yaml
```

### Inference
```bash
python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/cityscapes/dinojscc_cityscapes_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/cityscapes/eval_cityscapes.yml --channelk 48 
```