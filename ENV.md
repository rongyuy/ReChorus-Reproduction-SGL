# ReChorus 项目环境配置指南 (2025 稳定版)

## 📌 核心痛点与解决方案

**问题：** 原项目依赖较旧，若直接安装 `requirements.txt`，极易导致 Numpy 版本与 PyTorch 底层 MKL 库冲突（报错 `undefined symbol: iJIT_NotifyEvent`），或因 Numpy 版本过新导致代码报错（报错 `AttributeError: module 'numpy' has no attribute 'float'`）。

**解决方案：**

1.  **环境层**：放弃强制降级 Numpy，使用 Conda 统一安装 PyTorch 和 Numpy，确保底层 MKL 库一致性。
2.  **代码层**：使用“热补丁”技术，在入口文件强行修复新版 Numpy 移除旧 API 的问题。

-----

## 🛠️ 第一步：创建纯净的 Conda 环境

不要使用 pip 混合安装核心库，请使用以下命令一键创建环境。

*注意：本指南基于 CUDA 11.x 环境（RTX 30系/40系显卡适用）。*

```bash
# 1. 确保在一个纯净的状态
conda deactivate

# 2. 创建名为 rech 的环境 (Python 3.10)
# 关键点：让 Conda 自动解决 PyTorch 1.12 和 Numpy 的版本匹配，不要人工指定 Numpy 版本
conda create -n rech python=3.10 pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 numpy pandas scipy scikit-learn tqdm pyyaml -c pytorch -y

# 3. 激活环境
conda activate rech
```

-----

## 🩹 第二步：注入代码补丁 (One-Time Fix)

由于现在的 Numpy 版本（通常为 1.24+）移除了 `np.int`, `np.float`, `np.object` 等别名，而 ReChorus 源码中大量使用了这些写法，我们需要修改入口文件。

**操作对象：** `src/main.py`
**操作方法：** 打开文件，在头部 `import` 区域下方，`main()` 函数上方，插入以下代码块：

```python
import numpy as np
import logging
# ... 其他 import ...

# ==========================================
# 【环境兼容性补丁】修复 Numpy 1.24+ 移除旧别名的问题
# 防止报错: AttributeError: module 'numpy' has no attribute 'float/int/object'
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'object'):
    np.object = object
# ==========================================

def main():
    # ... 原有代码 ...
```

*原理：这段代码会在程序启动瞬间，将 Python 原生的 `int/float` 赋值给 `np.int/np.float`，从而欺骗旧代码，使其在新环境中也能正常运行。*

-----

## 🚀 第三步：运行验证

配置完成后，使用以下命令启动训练，应该能看到参数打印和进度条：

```bash
# 请根据实际情况修改 GPU 编号
CUDA_VISIBLE_DEVICES=0 python src/main.py --model_name LightGCN --dataset Grocery_and_Gourmet_Food
```

-----

## ❓ 常见问题排查 (Troubleshooting)

| 错误现象 | 核心原因 | 解决方法 |
| :--- | :--- | :--- |
| **ImportError: ... undefined symbol: iJIT\_NotifyEvent** | PyTorch 和 Numpy 使用了不同版本的 MKL 库（通常由 `pip` 混装导致）。 | **重装环境**。严格按照第一步的 `conda create` 命令执行，不要单独 pip install numpy。 |
| **AttributeError: module 'numpy' has no attribute 'float'** | Numpy 版本过新（\>1.20），移除了 `np.float` 等别名。 | **应用补丁**。检查是否严格按照第二步修改了 `src/main.py`。 |
| **RuntimeError: CUDA error: no kernel image is available** | 显卡太新（如 RTX 4090）但安装的 CUDA Toolkit 版本太旧（如 10.x）。 | 确保安装命令中指定了 `cudatoolkit=11.3` 或更高版本。 |




---
正确运行的命令：
python src/main.py --model_name LightGCN --dataset Grocery_and_Gourmet_Food

