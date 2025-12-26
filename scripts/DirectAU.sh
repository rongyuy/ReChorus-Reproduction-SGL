python src/main.py \
    --model_name DirectAU \
    --dataset MIND_Large/MINDTOPK \
    --batch_size 2048 \
    --gamma 0.3 \
    --lr 0.001 \
    --l2 1e-6 \
    --epochs 500 \
    --gpu 0 \
    --test_all 0 \
    --save_final_results 0

python src/main.py \
    --model_name DirectAU \
    --dataset Grocery_and_Gourmet_Food \
    --batch_size 2048 \
    --gamma 0.3 \
    --lr 0.001 \
    --l2 1e-4 \
    --epochs 500 \
    --gpu 0 \
    --test_all 0 \
    --save_final_results 0

python src/main.py \
    --model_name DirectAU \
    --dataset MovieLens_1M/ML_1MTOPK \
    --batch_size 2048 \
    --gamma 0.3 \
    --lr 0.001 \
    --l2 1e-5 \
    --epochs 500 \
    --gpu 0 \
    --test_all 0 \
    --save_final_results 0
