python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/voc/dinojscc_voc_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/voc/eval_voc.yml --channelk 12  --output inference_log/voc_k12
python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/voc/dinojscc_voc_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/voc/eval_voc.yml --channelk 24  --output inference_log/voc_k24
python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/voc/dinojscc_voc_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/voc/eval_voc.yml --channelk 36  --output inference_log/voc_k36
python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/voc/dinojscc_voc_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/voc/eval_voc.yml --channelk 48  --output inference_log/voc_k48


python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/stuff/dinojscc_stuff_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml --channelk 12  --output inference_log/stuff_k12
python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/stuff/dinojscc_stuff_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml --channelk 24  --output inference_log/stuff_k24
python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/stuff/dinojscc_stuff_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml --channelk 36  --output inference_log/stuff_k36
python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/stuff/dinojscc_stuff_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml --channelk 48  --output inference_log/stuff_k48


# python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/cityscapes/dinojscc_cityscapes_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/cityscapes/eval_cityscapes.yml --channelk 12  --output inference_log/cityscapes_k12
# python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/cityscapes/dinojscc_cityscapes_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/cityscapes/eval_cityscapes.yml --channelk 24  --output inference_log/cityscapes_k24
# python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/cityscapes/dinojscc_cityscapes_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/cityscapes/eval_cityscapes.yml --channelk 36  --output inference_log/cityscapes_k36
# python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/cityscapes/dinojscc_cityscapes_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/cityscapes/eval_cityscapes.yml --channelk 48  --output inference_log/cityscapes_k48

python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/ade/dinojscc_ade_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/ade/eval_ade.yml --channelk 12  --output inference_log/ade_k12
python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/ade/dinojscc_ade_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/ade/eval_ade.yml --channelk 24  --output inference_log/ade_k24
python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/ade/dinojscc_ade_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/ade/eval_ade.yml --channelk 36  --output inference_log/ade_k36
python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/ade/dinojscc_ade_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/ade/eval_ade.yml --channelk 48  --output inference_log/ade_k48



python wx_push.py

# python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/stuff/dinojscc_stuff_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml --channelk 6 --output inference_log/stuff_k06
# python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/stuff/dinojscc_stuff_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml --channelk 12 --output inference_log/stuff_k12
# python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/stuff/dinojscc_stuff_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml --channelk 18 --output inference_log/stuff_k18
# python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/stuff/dinojscc_stuff_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml --channelk 24 --output inference_log/stuff_k24
# python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/stuff/dinojscc_stuff_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml --channelk 29 --output inference_log/stuff_k29
# python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/stuff/dinojscc_stuff_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml --channelk 35 --output inference_log/stuff_k35


# # nohup bash inferencing_batch.sh > /dev/null 2>&1 &