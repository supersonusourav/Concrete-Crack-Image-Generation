@echo off
setlocal
cd /d "%~dp0"

set MODEL_NAME=runwayml/stable-diffusion-v1-5
set MAX_STEPS=2000

accelerate launch ^
 --num_processes=1 ^
 --num_machines=1 ^
 --mixed_precision=fp16 ^
 --dynamo_backend=no ^
 train_crack_lora.py ^
 --pretrained_model_name_or_path=%MODEL_NAME% ^
 --train_data_dir=./training_data ^
 --resolution=512 ^
 --train_batch_size=1 ^
 --gradient_accumulation_steps=4 ^
 --max_train_steps=%MAX_STEPS% ^
 --learning_rate=1e-4 ^
 --output_dir=./sd-concrete-crack-lora ^
 --checkpointing_steps=500 ^
 --checkpoints_total_limit=3 ^
 --seed=42 ^
 --gradient_checkpointing ^
 --use_8bit_adam ^
 --enable_xformers_memory_efficient_attention ^
 --use_mask_weighting ^
 --mask_loss_weight=3.0

echo Crack LoRA Training Complete! Weights saved in sd-concrete-crack-lora
pause