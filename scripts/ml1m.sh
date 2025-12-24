python src/main.py \
    --model_name LightGCN \
    --dataset MovieLens_1M/ML_1MTOPK \
    --batch_size 2048 \
    --lr 0.001 \
    --l2 0 \
    --gpu 2 \
    --test_all 0 \
    --save_final_results 0

# python src/main.py \
#     --model_name DirectAU \
#     --dataset MovieLens_1M/ML_1MTOPK \
#     --batch_size 2048 \
#     --gamma 0.3 \
#     --lr 0.001 \
#     --l2 1e-5 \
#     --epochs 500 \
#     --gpu 1 \
#     --test_all 0 \
#     --save_final_results 0


# python src/main.py \
#     --model_name SimGCL \
#     --dataset MovieLens_1M/ML_1MTOPK \
#     --batch_size 2048 \
#     --gpu 1 \
#     --lr 0.001 \
#     --l2 1e-6 \
#     --emb_size 64 \
#     --n_layers 2 \
#     --eps 0.1 \
#     --tau 0.2 \
#     --test_all 0 \
#     --save_final_results 0