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
  --data_path weather.csv \
  --model_id final_weather_96'_'96 \
  --model BDEOP \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 5 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --patience 10 \
  --train_epochs 20 \
  --individual \
  --lradj type1 \
  --LegT_Order 256 \
  --mode_Num 1032 \
  --mode_type 1 \
  --use_amp \
  --FEL 1 \
  --Decomp 1 \
  --RevIN

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id final_weather_96'_'192 \
  --model BDEOP \
  --data custom \
  --features M \
  --seq_len 192 \
  --pred_len 192 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 5 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --patience 10 \
  --train_epochs 20 \
  --individual \
  --lradj type1 \
  --LegT_Order 256 \
  --mode_Num 1032 \
  --mode_type 1 \
  --use_amp \
  --FEL 1 \
  --Decomp 1 \
  --RevIN

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id final_weather_96'_'336 \
  --model BDEOP \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 5 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --patience 10 \
  --train_epochs 20 \
  --individual \
  --lradj type1 \
  --LegT_Order 256 \
  --mode_Num 1032 \
  --mode_type 1 \
  --use_amp \
  --FEL 1 \
  --Decomp 1 \
  --RevIN

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id final_weather_96'_'720 \
  --model BDEOP \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 5 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --patience 10 \
  --train_epochs 20 \
  --individual \
  --lradj type1 \
  --LegT_Order 256 \
  --mode_Num 1032 \
  --mode_type 1 \
  --use_amp \
  --FEL 1 \
  --Decomp 1 \
  --RevIN