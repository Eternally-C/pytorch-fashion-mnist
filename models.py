import torch.nn as nn


# 线性分类器基线模型：
# 将每张 28×28 的灰度图从 [1, 28, 28] 展平为 784 维向量，
# 再通过线性层映射为 10 个类别的 logits。
class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x


# 小型 MLP 分类器：
# 在线性输入与输出之间加入隐藏层和 ReLU，
# 用于学习更复杂的非线性特征关系。
class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x