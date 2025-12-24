python src/main.py \
    --model_name POP \
    --dataset Grocery_and_Gourmet_Food \
    --batch_size 2048 \
    --gpu 0 \
    --test_all 0 \
    --save_final_results 0 \
    --train 0  # <--- 关键修改：告诉框架不要训练，直接跑测试

python src/main.py \
    --model_name POP \
    --dataset MIND_Large/MINDTOPK \
    --batch_size 2048 \
    --gpu 0 \
    --test_all 0 \
    --save_final_results 0 \
    --train 0  # <--- 关键修改：告诉框架不要训练，直接跑测试