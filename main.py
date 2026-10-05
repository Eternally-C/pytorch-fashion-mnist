from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch
from torch.utils.data import random_split, DataLoader
from pathlib import Path
import json

from models import LinearClassifier, MLPClassifier
from training import train_model

# 获取当前项目根目录，即 main.py 所在的位置。
project_root = Path(__file__).resolve().parent

# 定义项目中的数据目录和模型检查点保存目录。
data_dir = project_root / "data"
checkpoint_dir = project_root / "checkpoints"

# 定义实验结果保存目录，用于保存训练历史等实验结果。
results_dir = project_root / "results"

# 如果实验结果目录不存在，则自动创建。
results_dir.mkdir(exist_ok=True)

# 将 Fashion-MNIST 图片转换为 PyTorch Tensor，
# 并把像素值缩放到 [0, 1] 范围。
transform = ToTensor()

# 加载 Fashion-MNIST 官方训练集。
# download=True 只会在本地不存在数据时触发下载。
train_dataset = FashionMNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=transform
)

# 固定随机种子，保证模型初始化过程可复现。
torch.manual_seed(42)

# 将官方 60,000 条训练数据划分为：
# 54,000 条训练数据和 6,000 条验证数据。
train_size = 54000
val_size = 6000

train_subset, val_subset = random_split(
    train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# 训练集 DataLoader：
# shuffle=True 表示每个 epoch 打乱训练样本顺序；
# 固定 generator 可以保证重复实验时数据顺序可复现。
train_loader = DataLoader(
    train_subset,
    batch_size=64,
    shuffle=True,
    generator=torch.Generator().manual_seed(42)
)

# 验证集只用于评估，不需要打乱顺序。
val_loader = DataLoader(
    val_subset,
    batch_size=64,
    shuffle=False
)

# 优先使用 Apple Silicon 的 MPS 加速；
# 如果当前环境不支持 MPS，则自动回退到 CPU。
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# 输出实际使用的计算设备，便于复现实验和排查环境问题。
print("Device:", device)

model = MLPClassifier().to(device)

# 使用交叉熵损失处理 10 分类任务。
criterion = nn.CrossEntropyLoss()

# 使用 SGD 优化器，根据反向传播得到的梯度更新模型参数。
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

num_epochs = 20

# 使用统一训练函数完成模型训练和验证。
training_results = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs
)

# 取出训练历史，用于保存到 JSON 文件。
train_losses = training_results["train_losses"]
val_losses = training_results["val_losses"]
val_accuracies = training_results["val_accuracies"]
val_macro_f1s = training_results["val_macro_f1s"]


# -------------------------
# 保存模型
# -------------------------

# 创建模型保存目录。
checkpoint_dir.mkdir(exist_ok=True)

# 保存验证集 Macro-F1 最佳 epoch 的 MLP 分类器参数。
model_path = checkpoint_dir / f"mlp_classifier_{num_epochs}ep_best.pth"
torch.save(model.state_dict(), model_path)

print("Model saved to:", model_path)

# -------------------------
# 保存训练历史
# -------------------------

# 保存训练过程中的关键指标，
# 便于后续无需重新训练即可恢复训练曲线和实验结果。
training_history = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "val_accuracies": val_accuracies,
    "val_macro_f1s": val_macro_f1s,
    "best_epoch": training_results["best_epoch"],
    "best_val_loss": training_results["best_val_loss"],
    "best_val_accuracy": training_results["best_val_accuracy"],
    "best_val_macro_f1": training_results["best_val_macro_f1"]
}

history_path = results_dir / "mlp_training_history.json"

with open(history_path, "w", encoding="utf-8") as file:
    json.dump(
        training_history,
        file,
        indent=4
    )

print("Training history saved to:", history_path)