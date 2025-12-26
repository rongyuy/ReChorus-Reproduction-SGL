# # export CUDA_VISIBLE_DEVICES=2 

# # python src/main.py --model_name LightGCN --dataset MIND_Large

python src/main.py \
    --model_name LightGCN \
    --dataset MIND_Large/MINDTOPK \
    --batch_size 2048 \
    --lr 0.001 \
    --l2 0 \
    --gpu 0 \
    --test_all 0 \
    --save_final_results 0

python src/main.py \
    --model_name LightGCN \
    --dataset Grocery_and_Gourmet_Food \
    --batch_size 2048 \
    --lr 0.001 \
    --l2 0 \
    --gpu 0 \
    --test_all 0 \
    --save_final_results 0

python src/main.py \
    --model_name LightGCN \
    --dataset MovieLens_1M/ML_1MTOPK \
    --batch_size 2048 \
    --lr 0.001 \
    --l2 0 \
    --gpu 0 \
    --test_all 0 \
    --save_final_results 0

