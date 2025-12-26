#!/bin/bash

# 固定参数
dataset="Grocery_and_Gourmet_Food"
gpu_id="0"

echo "=== 开始参数敏感性分析: SimGCL on ${dataset} ==="

# 1. 分析 Lambda (固定 eps=0.1)
echo "--- Group 1: Sensitivity of Lambda (eps=0.1) ---"
for lam in 0.01 0.05 0.2 0.5
do
    echo "Running lambda=${lam}..."
    python src/main.py \
        --model_name SimGCL --dataset ${dataset} \
        --batch_size 2048 --gpu ${gpu_id} \
        --test_all 0 --save_final_results 0 \
        --lr 0.001 --l2 1e-5 --emb_size 64 --n_layers 2 \
        --eps 0.1 --lam ${lam} \
        --early_stop 10
done

# 2. 分析 Epsilon (固定 lam=0.1)
echo "--- Group 2: Sensitivity of Epsilon (lam=0.1) ---"
for eps in 0.05 0.2 0.5
do
    echo "Running eps=${eps}..."
    python src/main.py \
        --model_name SimGCL --dataset ${dataset} \
        --batch_size 2048 --gpu ${gpu_id} \
        --test_all 0 --save_final_results 0 \
        --lr 0.001 --l2 1e-5 --emb_size 64 --n_layers 2 \
        --eps ${eps} --lam 0.1 \
        --early_stop 10
done

# 注意: lam=0.1, eps=0.1 的组合之前跑过了，可以直接复用那个数据，不用重跑。

# --- DirectAU 敏感性分析 (已修正) ---
echo "=== 开始参数敏感性分析: DirectAU on ${dataset} ==="

for gam in 0.5 1.0 2.0 5.0 10.0; do
    echo ">>> Sensitivity: Grocery | DirectAU | Gamma=${gam}"
    python src/main.py \
        --model_name DirectAU --dataset ${dataset} \
        --emb_size 64 --lr 0.001 --l2 1e-4 --gamma ${gam} \
        --batch_size 2048 --epochs 50 --gpu ${gpu_id} \
        --test_all 0 --save_final_results 0
done