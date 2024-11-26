python3 evolve_instruction.py \
    --dataset_type huggingface \
    --dataset_name Kendamarron/aya_dataset_ja \
    --split train \
    --column_name inputs \
    --model_name Qwen/Qwen2.5-32B-Instruct-AWQ \
    --prompt_path output/best_prompt.txt \
    --output_dir ./output/