# 基于 PyTorch 的 Fashion-MNIST 图像分类项目

这是一个基于 PyTorch 实现的 Fashion-MNIST 图像分类项目。

项目首先构建线性分类器作为基线模型，然后使用一个小型多层感知机（MLP）进行改进。训练阶段采用固定的训练集 / 验证集划分进行模型选择，并最终在官方测试集上评估最佳模型。

最终 MLP 模型根据 **验证集 Macro-F1** 指标选择最佳 epoch。

## 实验结果

最终 MLP 的主要结果如下：

| 数据集 | Accuracy | Macro-F1 |
|---|---:|---:|
| Validation | 85.52% | 85.57% |
| Test | 84.66% | 84.62% |

在最多 20 个训练 epoch 中，最终选择的最佳模型来自 **第 19 个 epoch**。

### 训练与验证 Loss

![训练与验证 Loss](results/mlp_loss_curve.png)

### Validation 指标

![Validation Accuracy 与 Macro-F1](results/mlp_validation_metrics.png)

### Validation 混淆矩阵

![Validation 混淆矩阵](results/mlp_validation_confusion_matrix.png)

### Test 混淆矩阵

![Test 混淆矩阵](results/mlp_test_confusion_matrix.png)

模型最常见的分类错误主要集中在视觉特征较为相似的上衣类别之间，尤其是：

- T-shirt/top
- Pullover
- Coat
- Shirt

## 模型结构

项目中的主要模型是一个小型 MLP：

```text
输入图像：1 × 28 × 28
        ↓
Flatten：784
        ↓
Linear：784 → 128
        ↓
ReLU
        ↓
Linear：128 → 10
        ↓
类别 logits
```

项目中同时保留了一个线性分类器，作为基线模型进行对照。

## 训练设置

- 数据集：Fashion-MNIST
- 训练样本：54,000
- 验证样本：6,000
- 测试样本：10,000
- Batch size：64
- 优化器：SGD
- Learning rate：0.01
- 最大训练轮数：20 epochs
- Loss function：Cross-Entropy Loss
- 模型选择指标：Validation Macro-F1
- Random seed：42
- Apple Silicon 加速：在支持时使用 PyTorch MPS

官方 Test Set 与训练和模型选择流程完全隔离，只在独立的评估脚本中使用。

## 项目结构

```text
pytorch_fashion_mnist/
├── main.py
├── evaluate_checkpoint.py
├── models.py
├── training.py
├── evaluation.py
├── requirements.txt
├── README.md
├── .gitignore
└── results/
    ├── mlp_training_history.json
    ├── mlp_loss_curve.png
    ├── mlp_validation_metrics.png
    ├── mlp_validation_confusion_matrix.png
    └── mlp_test_confusion_matrix.png
```

本地的 `data/` 和 `checkpoints/` 目录不会提交到版本控制系统。

## 各文件职责

### `main.py`

模型训练入口。

主要负责：

- 在本地不存在数据时下载 Fashion-MNIST；
- 创建固定的 54,000 / 6,000 Train / Validation 划分；
- 训练 MLP 模型；
- 记录 Validation Loss、Accuracy 和 Macro-F1；
- 根据 Validation Macro-F1 选择最佳 epoch；
- 恢复最佳 epoch 的模型参数；
- 保存最佳 checkpoint；
- 将完整训练历史保存为 JSON。

### `evaluate_checkpoint.py`

模型评估和可视化入口。

主要负责：

- 加载训练历史；
- 加载对应的最佳 checkpoint；
- 恢复与训练阶段完全相同的 Validation Set；
- 在 Validation 和 Test 数据上进行评估；
- 输出 classification report；
- 分析最常见的误分类方向；
- 生成 confusion matrix；
- 根据训练历史重新生成训练曲线。

### `models.py`

定义模型结构：

- `LinearClassifier`
- `MLPClassifier`

### `training.py`

包含可复用的训练和验证流程，并负责根据 Validation Macro-F1 选择最佳模型。

### `evaluation.py`

包含可复用的模型评估函数和误分类分析函数。

## 环境安装

推荐使用 Python 3.11。

创建并激活 Python 环境后，安装项目依赖：

```bash
pip install -r requirements.txt
```

当前已经验证可运行的开发环境依赖版本为：

```text
torch==2.14.0
torchvision==0.29.0
scikit-learn==1.9.1
matplotlib==3.11.2
```

## 使用方法

### 1. 训练模型

运行：

```bash
python main.py
```

脚本将依次完成：

1. 如果本地不存在数据，则下载 Fashion-MNIST；
2. 训练 MLP；
3. 根据 Validation Macro-F1 选择最佳 epoch；
4. 将最佳模型 checkpoint 保存到 `checkpoints/`；
5. 将训练历史保存到 `results/`。

### 2. 评估最佳模型

训练完成后运行：

```bash
python evaluate_checkpoint.py
```

该脚本**不会自动下载数据**，因此要求本地已经存在训练阶段生成的数据和 checkpoint。

脚本将完成：

- Validation 评估；
- Test 最终评估；
- Classification report；
- Top-5 误分类分析；
- Confusion matrix；
- 训练 Loss 曲线；
- Validation Accuracy / Macro-F1 曲线；
- 将结果图保存到 `results/`。

## 错误分析

最终模型在 Validation Set 上最常见的 5 个误分类方向为：

```text
Shirt -> T-shirt/top : 99
Pullover -> Coat     : 96
T-shirt/top -> Shirt : 74
Pullover -> Shirt    : 62
Shirt -> Coat        : 58
```

在 Test Set 上最常见的 5 个误分类方向为：

```text
Pullover -> Coat     : 179
Shirt -> T-shirt/top : 137
T-shirt/top -> Shirt : 114
Shirt -> Coat        : 108
Pullover -> Shirt    : 102
```

Validation 与 Test 的主要错误方向高度一致。

这说明当前 MLP 的主要能力限制并不是某一次 Validation 划分造成的偶然现象，而是模型在区分视觉特征相似的上衣类别时存在较稳定的困难，尤其是在以下类别之间：

```text
T-shirt/top
Pullover
Coat
Shirt
```

## 可复现性

实验对以下随机过程进行了固定：

- Train / Validation 数据划分；
- 训练数据每个 epoch 的 shuffle；
- 模型初始化。

随机种子统一设置为：

```text
42
```

训练历史文件记录了完整的 20 个 epoch 指标变化，以及最终选择的最佳 epoch。

当前保存实验的主要结果为：

```text
Best epoch: 19

Validation accuracy:
0.8551666667

Validation Macro-F1:
0.8556532774

Test accuracy:
0.8466

Test Macro-F1:
0.8462483930
```

## 实验设计说明

本项目将训练、模型选择与最终测试进行了明确分离：

```text
Train Set
    ↓
模型训练

Validation Set
    ↓
选择最佳 epoch

Test Set
    ↓
最终模型评估
```

Test Set 不参与训练过程，也不参与最佳模型选择。

这种设计可以减少测试集信息泄漏，并使最终测试结果更能够反映模型在未参与模型选择的数据上的泛化能力。