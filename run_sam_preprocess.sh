# nohup bash run_sam_preprocess.sh > preprocess_sam.log 2>&1 &

# SAM
rm -r /data/bohnsix/talking2dino/coco2014_sam_b_p16/
mkdir /data/bohnsix/talking2dino/coco2014_sam_b_p16/
mkdir /data/bohnsix/talking2dino/coco2014_sam_b_p16/cls_token
mkdir /data/bohnsix/talking2dino/coco2014_sam_b_p16/patch_tokens
mkdir /data/bohnsix/talking2dino/coco2014_sam_b_p16/avg_self_attn_out
mkdir /data/bohnsix/talking2dino/coco2014_sam_b_p16/disentangled_self_attn

CUDA_VISIBLE_DEVICES=3 python dino_extraction_v2.py --model samvit_base_patch16 \
                             --out_path /data/bohnsix/talking2dino/coco2014_sam_b_p16/val.pth \
                             --ann_path ../coco/captions_val2014.json --crop_dim 448 --resize_dim 448 \
                             --extract_patch_tokens --extract_cls
CUDA_VISIBLE_DEVICES=3 python dino_extraction_v2.py --model samvit_base_patch16 \
                             --out_path /data/bohnsix/talking2dino/coco2014_sam_b_p16/train.pth \
                             --ann_path ../coco/captions_train2014.json --crop_dim 448 --resize_dim 448 \
                             --extract_patch_tokens --extract_cls
CUDA_VISIBLE_DEVICES=3 python text_features_extraction.py --ann_path /data/bohnsix/talking2dino/coco2014_sam_b_p16/val.pth
CUDA_VISIBLE_DEVICES=3 python text_features_extraction.py --ann_path /data/bohnsix/talking2dino/coco2014_sam_b_p16/train.pth
