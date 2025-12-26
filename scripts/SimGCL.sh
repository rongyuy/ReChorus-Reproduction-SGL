python src/main.py \
    --model_name SimGCL \
    --dataset MIND_Large/MINDTOPK \
    --batch_size 2048 \
    --gpu 0 \
    --lr 0.001 \
    --l2 1e-6 \
    --emb_size 64 \
    --n_layers 2 \
    --eps 0.1 \
    --tau 0.2 \
    --early_stop 10 \
    --test_all 0

python src/main.py \
    --model_name SimGCL \
    --dataset Grocery_and_Gourmet_Food \
    --batch_size 2048 \
    --gpu 0 \
    --lr 0.001 \
    --l2 1e-5 \
    --emb_size 64 \
    --n_layers 2 \
    --eps 0.1 \
    --tau 0.2 \
    --early_stop 15 \
    --test_all 0

python src/main.py \
    --model_name SimGCL \
    --dataset MovieLens_1M/ML_1MTOPK \
    --batch_size 2048 \
    --gpu 0 \
    --lr 0.001 \
    --l2 1e-6 \
    --emb_size 64 \
    --n_layers 2 \
    --eps 0.1 \
    --tau 0.2 \
    --test_all 0 \
    --save_final_results 0
