CUDA_VISIBLE_DEVICES=0 nohup python stage2_train.py;
wait
CUDA_VISIBLE_DEVICES=0 nohup python stage3_train.py;
wait
CUDA_VISIBLE_DEVICES=0 nohup python inference.py   --article_data_dir ./data/gaze_stage23   --sentence_data_dir ./data/gaze_stage1   --split test   --model_base_dir ./trained_models/stage3_gaze_qwen_final   --output_dir ./results_stage3_separated   --max_new_tokens 450;
wait 