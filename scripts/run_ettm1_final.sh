export CUDA_VISIBLE_DEVICES=0
# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id final_ETTm1_96'_'96 \
  --model BDEOP \
  --data ETTm1 \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --patience 30 \
  --train_epochs 30 \
  --individual \
  --lradj type4 \
  --LegT_Order 256 \
  --mode_Num 32 \
  --mode_type 1 \
  --use_amp \
  --FDL 1 \
  --Decomp 2

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id final_ETTm1_96'_'192 \
  --model BDEOP \
  --data ETTm1 \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 5 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --patience 30 \
  --train_epochs 30 \
  --individual \
  --lradj type4 \
  --LegT_Order 256 \
  --mode_Num 64 \
  --mode_type 1 \
  --use_amp \
  --FDL 1 \
  --Decomp 3

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id final_ETTm1_96'_'336 \
  --model BDEOP \
  --data ETTm1 \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 5 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --patience 30 \
  --train_epochs 30 \
  --individual \
  --lradj type4 \
  --LegT_Order 1024 \
  --mode_Num 128 \
  --mode_type 1 \
  --use_amp \
  --FDL 1 \
  --Decomp 2

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id final_ETTm1_96'_'720 \
  --model BDEOP \
  --data ETTm1 \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 5 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --patience 30 \
  --train_epochs 30 \
  --individual \
  --lradj type4 \
  --LegT_Order 1024 \
  --mode_Num 256 \
  --mode_type 1 \
  --use_amp \
  --FDL 1 \
  --Decomp 3