CUDA_VISIBLE_DEVICES=3 python main.py \
--mode train \
--model_checkpoint "answerdotai/ModernBERT-base" \
--train_path "/home/leczhang/research/prompting/scripts/reward_modeling/two_combinations_20250127.jsonl" \
--output_dir "/shared/3/projects/lechen/reward_model/synthetic_modernbert/" \
--train_batch_size 16 \
--eval_batch_size 16 \
--max_length 512 \

# CUDA_VISIBLE_DEVICES=2,3 accelerate launch llava_model.py --mode train --model_checkpoint "llava-hf/llava-1.5-7b-hf" --train_path "/shared/0/datasets/mastodon/multimodal-data/oct-17-24-jsonl/two-classes/train.jsonl" --dev_path "/shared/0/datasets/mastodon/multimodal-data/oct-17-24-jsonl/two-classes/dev.jsonl" --output_dir "/shared/3/projects/national-culture/cache/imagemodels/LLaVa/oct-7/two-classes/fine_tuned_llava_model"