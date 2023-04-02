#!/bin/bash
python3 ./run_glue_no_trainer.py \
    --model_name_or_path "/path/to/bertlarge-sst2/dense_finetuned_model" \
    --task_name "sst2" \
    --max_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --distill_loss_weight 2.0 \
    --num_train_epochs 15 \
    --weight_decay 5e-5   \
    --cooldown_epochs 5 \
    --sparsity_warm_epochs 0 \
    --lr_scheduler_type "constant" \
    --do_prune \
    --output_dir "./sparse_sst2_bertlarge" \
    --target_sparsity 0.9 \
    --pruning_pattern "4x1" \
    --pruning_frequency 500