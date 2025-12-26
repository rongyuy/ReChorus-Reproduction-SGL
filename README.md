# 机器学习课程大作业 - 基于 ReChorus 的 SimGCL 算法复现

本项目基于 [ReChorus 2.0](https://github.com/THUwangcy/ReChorus) 框架，复现了 **SimGCL (Simple Graph Contrastive Learning)** 推荐算法，并与 BPRMF, BUIR, NeuMF, POP 等基准模型进行了对比实验。

---

## 🛠️ 环境依赖 (Environment)

本项目代码基于 Python 3.10 开发。请确保您的运行环境已安装 Python 3.10+。

### 1. 安装依赖
请在项目根目录下运行以下命令安装所需库：

```bash
pip install -r requirements.txt

```

### 2. 数据集准备

所有实验数据已预处理并存放于 `data/` 目录下，无需额外下载或处理：

* `Grocery_and_Gourmet_Food`
* `MIND_Large`
* `MovieLens_1M`

---

## 🚀 运行指南 (How to Run)

**注意**：所有命令请在 **项目根目录** 下执行。

### 1. 复现 SimGCL (主模型)

我们在 `scripts/SimGCL.sh` 中配置了三个数据集的实验参数。您可以直接运行脚本，或单独运行以下命令：

**方式一：运行脚本 (推荐)**

```bash
bash scripts/SimGCL.sh

```

**方式二：手动运行单条命令 (以 Grocery 数据集为例)**

```bash
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

```

> **⚠️ 关于 GPU 的提示**：
> 脚本中默认使用的 GPU0 运行（即 `--gpu 0`），如有需要可进行 GPU 编号的修改
> 
> 

### 2. 复现 Baselines (基准模型)

我们提供了 `scripts/baselines.sh` 脚本来一键运行所有对比模型（BPRMF, BUIR, NeuMF, POP）。

```bash
bash scripts/baselines.sh

```

该脚本会依次在三个数据集上运行各个基准模型。

### 3. 参数敏感性分析 (可选)

如果需要查看 SimGCL模型的`lambda` 和 `epsilon` 参数以及DirectAU模型的`gamma`参数的分析结果，请运行：

```bash
bash scripts/sensitivity.sh

```

---

## 📂 目录结构说明

```text
.
├── data/                 # 预处理后的数据集
├── src/                  # 源代码
│   ├── models/           # 模型定义 (SimGCL 代码位于 src/models/general/SimGCL.py)
│   ├── helpers/          # 数据读取与训练流程控制
│   └── main.py           # 程序入口
├── scripts/              # 实验运行脚本 (.sh 文件)
├── requirements.txt      # 环境依赖
└── README.md             # 说明文档

```

## 📊 实验结果查看

程序运行结束后，关键指标（HitRate@K, NDCG@K）将直接输出在 **控制台终端**。
如果脚本中设置了 `--save_final_results 1`，结果摘要也会保存在 csv 文件中。

---

**助教老师如有任何运行问题，请随时联系我们。感谢您的评阅！**