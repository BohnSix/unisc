# nohup bash run_dinov2_preprocess.sh > preprocess_dinov2.log 2>&1 &

# DINOv2
rm -r /data/bohnsix/talking2dino/coco2014_dinov2_b_p14/
mkdir /data/bohnsix/talking2dino/coco2014_dinov2_b_p14/
mkdir /data/bohnsix/talking2dino/coco2014_dinov2_b_p14/cls_token
mkdir /data/bohnsix/talking2dino/coco2014_dinov2_b_p14/patch_tokens
mkdir /data/bohnsix/talking2dino/coco2014_dinov2_b_p14/avg_self_attn_out
mkdir /data/bohnsix/talking2dino/coco2014_dinov2_b_p14/disentangled_self_attn

CUDA_VISIBLE_DEVICES=0 python dino_extraction_v2.py --model dinov2_vitb14 \
                             --out_path /data/bohnsix/talking2dino/coco2014_dinov2_b_p14/val.pth \
                             --ann_path ../coco/captions_val2014.json --crop_dim 448 --resize_dim 448 \
                             --extract_patch_tokens --extract_cls --extract_avg_self_attn --extract_self_attn_maps
CUDA_VISIBLE_DEVICES=0 python dino_extraction_v2.py --model dinov2_vitb14 \
                             --out_path /data/bohnsix/talking2dino/coco2014_dinov2_b_p14/train.pth \
                             --ann_path ../coco/captions_train2014.json --crop_dim 448 --resize_dim 448 \
                             --extract_patch_tokens --extract_cls --extract_avg_self_attn --extract_self_attn_maps
CUDA_VISIBLE_DEVICES=0 python text_features_extraction.py --ann_path /data/bohnsix/talking2dino/coco2014_dinov2_b_p14/val.pth
CUDA_VISIBLE_DEVICES=0 python text_features_extraction.py --ann_path /data/bohnsix/talking2dino/coco2014_dinov2_b_p14/train.pth
