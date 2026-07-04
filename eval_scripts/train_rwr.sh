cd /media/datasets/yumi/hjh/robo/robomimic
CUDA_VISIBLE_DEVICES=7 \
python robomimic/scripts/train_flow_rwr.py \
  --config /media/datasets/yumi/hjh/robo/robomimic/configs/final_square_flow_rwr.json \
  --ckpt_path /media/datasets/yumi/hjh/robo/robomimic/robomimic/square_flow_matching_image_eval_logs/final_square_flow_matching_augmented_data/20260611085408/models/model_epoch_200_fixed_meta.pth