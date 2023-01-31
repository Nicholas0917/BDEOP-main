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
  --data_path national_illness.csv \
  --model_id final_ili_36'_'24 \
  --model BDEOP \
  --data custom \
  --features M \
  --seq_len 104 \
  --pred_len 24 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 5 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --patience 60 \
  --train_epochs 60 \
  --individual \
  --lradj type4 \
  --LegT_Order 256 \
  --mode_Num 128 \
  --mode_type 1 \
  --use_amp \
  --FEL 1 \
  --Decomp 3 \
  --RevIN

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id final_ili_36'_'36 \
  --model BDEOP \
  --data custom \
  --features M \
  --seq_len 104 \
  --pred_len 36 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 5 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --patience 60 \
  --train_epochs 60 \
  --individual \
  --lradj type4 \
  --LegT_Order 256 \
  --mode_Num 128 \
  --mode_type 1 \
  --use_amp \
  --FEL 1 \
  --Decomp 3 \
  --RevIN

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id final_ili_36'_'48 \
  --model BDEOP \
  --data custom \
  --features M \
  --seq_len 104 \
  --pred_len 48 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 5 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --patience 60 \
  --train_epochs 60 \
  --individual \
  --lradj type4 \
  --LegT_Order 256 \
  --mode_Num 128 \
  --mode_type 1 \
  --use_amp \
  --FEL 1 \
  --Decomp 3 \
  --RevIN

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id final_ili_36'_'60 \
  --model BDEOP \
  --data custom \
  --features M \
  --seq_len 104 \
  --pred_len 60 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 5 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --patience 60 \
  --train_epochs 60 \
  --individual \
  --lradj type4 \
  --LegT_Order 256 \
  --mode_Num 128 \
  --mode_type 1 \
  --use_amp \
  --FEL 1 \
  --Decomp 3 \
  --RevIN