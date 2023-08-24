#!/bin/bash

models=("resnet18" "resnet34" "resnet50" "resnet101" "resnet152")
batches=(16 32 64 72 128)
epochs=(80 90 100 110 120)
learning_rates=(0.03 0.04 0.05 0.06 0.07)
momentums=(0.8 0.85 0.9 0.95 1.0)

for model in "${models[@]}"; do
    for batch in "${batches[@]}"; do
        for epoch in "${epochs[@]}"; do
            for lr in "${learning_rates[@]}"; do
                for momentum in "${momentums[@]}"; do
                    echo "Model: $model"
                    echo "Batch Size: $batch"
                    echo "Epochs: $epoch"
                    echo "Learning Rate: $lr"
                    echo "Momentum: $momentum"

                    python /kaggle/working/VIPCUP2023_OLIVES_edit/train.py \
                        --batch_size $batch \
                        --model $model \
                        --dataset 'OLIVES' \
                        --epochs $epoch \
                        --device 'cuda:0' \
                        --train_image_path '../input/olives-vip-cup-2023/2023 IEEE SPS Video and Image Processing (VIP) Cup - Ophthalmic Biomarker Detection/TRAIN/OLIVES' \
                        --test_image_path '../input/olives-vip-cup-2023/2023 IEEE SPS Video and Image Processing (VIP) Cup - Ophthalmic Biomarker Detection/TEST/' \
                        --test_csv_path '/kaggle/working/VIPCUP2023_OLIVES_edit/csv_dir/test_set_submission_template.csv' \
                        --train_csv_path '/kaggle/working/VIPCUP2023_OLIVES_edit/csv_dir/Training_Biomarker_Data.csv' \
                        --num_workers 2 \
                        --learning_rate $lr \
                        --momentum $momentum \
                        --temp 0.07
                done
            done
        done
    done
done
