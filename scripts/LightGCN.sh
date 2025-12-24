# # export CUDA_VISIBLE_DEVICES=2 

# # python src/main.py --model_name LightGCN --dataset MIND_Large

# python src/main.py \
#     --model_name LightGCN \
#     --dataset MIND_Large/MINDTOPK \
#     --batch_size 2048 \
#     --lr 0.001 \
#     --l2 0 \
#     --gpu 3 \
#     --test_all 1

# python src/main.py \
#     --model_name LightGCN \
#     --dataset Grocery_and_Gourmet_Food \
#     --batch_size 2048 \
#     --lr 0.001 \
#     --l2 0 \
#     --gpu 3 \
#     --test_all 1

python src/main.py \
    --model_name SimGCL \
    --dataset MIND_Large/MINDTOPK \
    --batch_size 2048 \
    --gpu 2 \
    --lr 0.001 \
    --l2 1e-6 \
    --emb_size 64 \
    --n_layers 2 \
    --cl_lambda 0.02 \
    --eps 0.1 \
    --tau 0.1 \
    --early_stop 10 \
    --test_all 0 \
    --save_final_results 0

python src/main.py \
    --model_name SimGCL \
    --dataset MIND_Large/MINDTOPK \
    --batch_size 2048 \
    --gpu 2 \
    --lr 0.001 \
    --l2 1e-6 \
    --emb_size 64 \
    --n_layers 2 \
    --eps 0.1 \
    --tau 0.2 \
    --early_stop 10 \
    --test_all 0 \
    --save_final_results 0

python src/main.py \
    --model_name SimGCL \
    --dataset Grocery_and_Gourmet_Food \
    --batch_size 2048 \
    --gpu 2 \
    --lr 0.001 \
    --l2 1e-5 \
    --emb_size 64 \
    --n_layers 2 \
    --eps 0.1 \
    --tau 0.2 \
    --early_stop 15 \
    --test_all 0 \
    --save_final_results 0