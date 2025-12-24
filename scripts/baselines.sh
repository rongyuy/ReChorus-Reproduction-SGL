# # ==========================================
# # Dataset 1: Grocery_and_Gourmet_Food
# # ==========================================

# # 1. BPRMF (基础矩阵分解)
# python src/main.py \
#     --model_name BPRMF \
#     --dataset Grocery_and_Gourmet_Food \
#     --batch_size 2048 \
#     --gpu 0 \
#     --lr 0.001 \
#     --l2 1e-6 \
#     --emb_size 64 \
#     --test_all 0 \
#     --save_final_results 0

# # 2. BUIR (无负采样自监督)
# python src/main.py \
#     --model_name BUIR \
#     --dataset Grocery_and_Gourmet_Food \
#     --batch_size 2048 \
#     --gpu 0 \
#     --lr 0.001 \
#     --l2 1e-6 \
#     --emb_size 64 \
#     --momentum 0.995 \
#     --test_all 0 \
#     --save_final_results 0

# # 3. NeuMF (神经协同过滤 - 注意lr通常较小)
# python src/main.py \
#     --model_name NeuMF \
#     --dataset Grocery_and_Gourmet_Food \
#     --batch_size 2048 \
#     --gpu 0 \
#     --lr 0.0005 \
#     --l2 1e-7 \
#     --emb_size 64 \
#     --layers '[64]' \
#     --dropout 0.2 \
#     --test_all 0 \
#     --save_final_results 0

# # # 4. CFKG (知识图谱 - 仅Grocery跑这个)
# # python src/main.py \
# #     --model_name CFKG \
# #     --dataset Grocery_and_Gourmet_Food \
# #     --batch_size 2048 \
# #     --gpu 0 \
# #     --lr 0.0001 \
# #     --l2 1e-6 \
# #     --emb_size 64 \
# #     --margin 1 \
# #     --include_attr 1 \
# #     --test_all 0 \
# #     --save_final_results 0

# # 5. POP (流行度基准)
# python src/main.py \
#     --model_name POP \
#     --dataset Grocery_and_Gourmet_Food \
#     --batch_size 2048 \
#     --gpu 0 \
#     --test_all 0 \
#     --save_final_results 0


# # ==========================================
# # Dataset 2: MIND_Large/MINDTOPK
# # ==========================================

# # 1. BPRMF
# python src/main.py \
#     --model_name BPRMF \
#     --dataset MIND_Large/MINDTOPK \
#     --batch_size 2048 \
#     --gpu 0 \
#     --lr 0.001 \
#     --l2 1e-6 \
#     --emb_size 64 \
#     --test_all 0 \
#     --save_final_results 0

# # 2. BUIR
# python src/main.py \
#     --model_name BUIR \
#     --dataset MIND_Large/MINDTOPK \
#     --batch_size 2048 \
#     --gpu 0 \
#     --lr 0.001 \
#     --l2 1e-6 \
#     --emb_size 64 \
#     --momentum 0.995 \
#     --test_all 0 \
#     --save_final_results 0

# # 3. NeuMF
# python src/main.py \
#     --model_name NeuMF \
#     --dataset MIND_Large/MINDTOPK \
#     --batch_size 2048 \
#     --gpu 0 \
#     --lr 0.0005 \
#     --l2 1e-7 \
#     --emb_size 64 \
#     --layers '[64]' \
#     --dropout 0.2 \
#     --test_all 0 \
#     --save_final_results 0

# # 4. POP
# python src/main.py \
#     --model_name POP \
#     --dataset MIND_Large/MINDTOPK \
#     --batch_size 2048 \
#     --gpu 0 \
#     --test_all 0 \
#     --save_final_results 0

# 1. BPRMF
python src/main.py \
    --model_name BPRMF \
    --dataset MovieLens_1M/ML_1MTOPK \
    --batch_size 2048 \
    --gpu 0 \
    --lr 0.001 \
    --l2 1e-6 \
    --emb_size 64 \
    --test_all 0 \
    --save_final_results 0

# 2. BUIR
python src/main.py \
    --model_name BUIR \
    --dataset MovieLens_1M/ML_1MTOPK \
    --batch_size 2048 \
    --gpu 0 \
    --lr 0.001 \
    --l2 1e-6 \
    --emb_size 64 \
    --momentum 0.995 \
    --test_all 0 \
    --save_final_results 0

# 3. NeuMF
python src/main.py \
    --model_name NeuMF \
    --dataset MovieLens_1M/ML_1MTOPK \
    --batch_size 2048 \
    --gpu 0 \
    --lr 0.0005 \
    --l2 1e-7 \
    --emb_size 64 \
    --layers '[64]' \
    --dropout 0.2 \
    --test_all 0 \
    --save_final_results 0

# 4. POP
python src/main.py \
    --model_name POP \
    --dataset MovieLens_1M/ML_1MTOPK \
    --batch_size 2048 \
    --gpu 0 \
    --test_all 0 \
    --save_final_results 0