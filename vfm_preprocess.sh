# # DINOv2
# rm -r /data/bohnsix/talking2dino/coco2014_dinov2_l_p14/
# mkdir /data/bohnsix/talking2dino/coco2014_dinov2_l_p14/
# mkdir /data/bohnsix/talking2dino/coco2014_dinov2_l_p14/cls_token
# mkdir /data/bohnsix/talking2dino/coco2014_dinov2_l_p14/patch_tokens
# mkdir /data/bohnsix/talking2dino/coco2014_dinov2_l_p14/avg_self_attn_out
# mkdir /data/bohnsix/talking2dino/coco2014_dinov2_l_p14/disentangled_self_attn

# CUDA_VISIBLE_DEVICES=0 python dino_extraction_v2.py --model dinov2_vitl14 \
#                              --out_path /data/bohnsix/talking2dino/coco2014_dinov2_l_p14/val.pth \
#                              --ann_path ../coco/captions_val2014.json --crop_dim 448 --resize_dim 448 \
#                              --extract_patch_tokens --extract_cls --extract_avg_self_attn --extract_self_attn_maps
# CUDA_VISIBLE_DEVICES=0 python dino_extraction_v2.py --model dinov2_vitl14 \
#                              --out_path /data/bohnsix/talking2dino/coco2014_dinov2_l_p14/train.pth \
#                              --ann_path ../coco/captions_train2014.json --crop_dim 448 --resize_dim 448 \
#                              --extract_patch_tokens --extract_cls --extract_avg_self_attn --extract_self_attn_maps
# python text_features_extraction.py --ann_path /data/bohnsix/talking2dino/coco2014_dinov2_l_p14/val.pth
# python text_features_extraction.py --ann_path /data/bohnsix/talking2dino/coco2014_dinov2_l_p14/train.pth

# # DINOv2 Reg
# rm -r /data/bohnsix/talking2dino/coco2014_dinov2reg_l_p14/
# mkdir /data/bohnsix/talking2dino/coco2014_dinov2reg_l_p14/
# mkdir /data/bohnsix/talking2dino/coco2014_dinov2reg_l_p14/cls_token
# mkdir /data/bohnsix/talking2dino/coco2014_dinov2reg_l_p14/patch_tokens
# mkdir /data/bohnsix/talking2dino/coco2014_dinov2reg_l_p14/avg_self_attn_out
# mkdir /data/bohnsix/talking2dino/coco2014_dinov2reg_l_p14/disentangled_self_attn

# CUDA_VISIBLE_DEVICES=1 python dino_extraction_v2.py --model dinov2_vitl14_reg \
#                              --out_path /data/bohnsix/talking2dino/coco2014_dinov2reg_l_p14/val.pth \
#                              --ann_path ../coco/captions_val2014.json --crop_dim 448 --resize_dim 448 \
#                              --extract_patch_tokens --extract_cls --extract_avg_self_attn --extract_self_attn_maps --extract_disentangled_self_attn
# CUDA_VISIBLE_DEVICES=1 python dino_extraction_v2.py --model dinov2_vitl14_reg \
#                              --out_path /data/bohnsix/talking2dino/coco2014_dinov2reg_l_p14/train.pth \
#                              --ann_path ../coco/captions_train2014.json --crop_dim 448 --resize_dim 448 \
#                              --extract_patch_tokens --extract_cls --extract_avg_self_attn --extract_self_attn_maps --extract_disentangled_self_attn
# python text_features_extraction.py --ann_path /data/bohnsix/talking2dino/coco2014_dinov2reg_l_p14/val.pth
# python text_features_extraction.py --ann_path /data/bohnsix/talking2dino/coco2014_dinov2reg_l_p14/train.pth

# # SAM
# rm -r /data/bohnsix/talking2dino/coco2014_sam_l_p16/
# mkdir /data/bohnsix/talking2dino/coco2014_sam_l_p16/
# mkdir /data/bohnsix/talking2dino/coco2014_sam_l_p16/cls_token
# mkdir /data/bohnsix/talking2dino/coco2014_sam_l_p16/patch_tokens
# mkdir /data/bohnsix/talking2dino/coco2014_sam_l_p16/avg_self_attn_out
# mkdir /data/bohnsix/talking2dino/coco2014_sam_l_p16/disentangled_self_attn

# CUDA_VISIBLE_DEVICES=3 python dino_extraction_v2.py --model samvit_large_patch16 \
#                              --out_path /data/bohnsix/talking2dino/coco2014_sam_l_p16/val.pth \
#                              --ann_path ../coco/captions_val2014.json --crop_dim 448 --resize_dim 448 \
#                              --extract_patch_tokens --extract_cls
# CUDA_VISIBLE_DEVICES=3 python dino_extraction_v2.py --model samvit_large_patch16 \
#                              --out_path /data/bohnsix/talking2dino/coco2014_sam_l_p16/train.pth \
#                              --ann_path ../coco/captions_train2014.json --crop_dim 448 --resize_dim 448 \
#                              --extract_patch_tokens --extract_cls
# python text_features_extraction.py --ann_path /data/bohnsix/talking2dino/coco2014_sam_l_p16/val.pth
# python text_features_extraction.py --ann_path /data/bohnsix/talking2dino/coco2014_sam_l_p16/train.pth


python dino_extraction_v2.py --ann_path ../coco/captions_train2014.json --out_path /data1/bohnsix/talking2dino/coco2014_dinov2reg_l_p14/train.pth --model dinov2_vitl14_reg --resize_dim 448 --crop_dim 448 --extract_avg_self_attn --extract_disentangled_self_attn --extract_patch_tokens
python dino_extraction_v2.py --ann_path ../coco/captions_val2014.json --out_path /data1/bohnsix/talking2dino/coco2014_dinov2reg_l_p14/val.pth --model dinov2_vitl14_reg --resize_dim 448 --crop_dim 448 --extract_avg_self_attn --extract_disentangled_self_attn --extract_patch_tokens

python text_features_extraction.py --ann_path /data1/bohnsix/talking2dino/coco2014_dinov2reg_l_p14/train.pth
python text_features_extraction.py --ann_path /data1/bohnsix/talking2dino/coco2014_dinov2reg_l_p14/val.pth
