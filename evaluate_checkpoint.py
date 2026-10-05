import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from evaluation import (
    evaluate_model,
    get_top_misclassifications
)
from models import MLPClassifier


# -------------------------
# 项目路径
# -------------------------

project_root = Path(__file__).resolve().parent

data_dir = project_root / "data"
checkpoint_dir = project_root / "checkpoints"

results_dir = project_root / "results"

# 如果实验结果目录不存在，则自动创建。
results_dir.mkdir(exist_ok=True)

history_path = results_dir / "mlp_training_history.json"

# -------------------------
# 加载训练历史
# -------------------------

# 从 JSON 中恢复训练阶段保存的历史指标，
# 无需重新训练即可重新生成训练曲线。
with open(history_path, "r", encoding="utf-8") as file:
    training_history = json.load(file)

train_losses = training_history["train_losses"]
val_losses = training_history["val_losses"]
val_accuracies = training_history["val_accuracies"]
val_macro_f1s = training_history["val_macro_f1s"]

best_epoch = training_history["best_epoch"]

# 根据训练历史自动确定本次实验的训练 epoch 数量。
training_epochs = len(train_losses)

checkpoint_path = (
    checkpoint_dir / f"mlp_classifier_{training_epochs}ep_best.pth"
)

print("Training history loaded from:", history_path)
print("Best epoch:", best_epoch)

# Fashion-MNIST 的 10 个类别名称。
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# 根据类别名称列表自动获取类别数量。
num_classes = len(class_names)

# -------------------------
# 数据准备
# -------------------------

transform = ToTensor()

# 数据已经存在，因此评估脚本禁止自动下载。
train_dataset = FashionMNIST(
    root=data_dir,
    train=True,
    download=False,
    transform=transform
)

test_dataset = FashionMNIST(
    root=data_dir,
    train=False,
    download=False,
    transform=transform
)

# 必须使用与训练阶段完全相同的划分方式，
# 才能恢复原来的 Validation Set。
train_size = 54000
val_size = 6000

_, val_subset = random_split(
    train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

val_loader = DataLoader(
    val_subset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# -------------------------
# 设备和模型
# -------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

model = MLPClassifier().to(device)

criterion = nn.CrossEntropyLoss()


# -------------------------
# 加载最佳 checkpoint
# -------------------------

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
)

print("Checkpoint loaded from:", checkpoint_path)


# 根据训练历史长度生成 epoch 编号。
epochs = range(1, training_epochs + 1)


# -------------------------
# Validation 评估
# -------------------------

val_results = evaluate_model(
    model=model,
    data_loader=val_loader,
    criterion=criterion,
    device=device
)

print(
    "Validation loss:",
    val_results["loss"],
    "Validation accuracy:",
    val_results["accuracy"],
    "Validation Macro-F1:",
    val_results["macro_f1"]
)

val_report = classification_report(
    val_results["labels"],
    val_results["predictions"],
    labels=range(num_classes),
    target_names=class_names,
    digits=4
)

print("Validation classification report:")
print(val_report)

# -------------------------
# Validation 错误分析
# -------------------------

val_confusion_matrix = confusion_matrix(
    val_results["labels"],
    val_results["predictions"],
    labels=range(num_classes)
)

val_misclassifications = get_top_misclassifications(
    confusion_matrix=val_confusion_matrix,
    class_names=class_names,
    top_k=5
)

print("Top 5 validation misclassifications:")

for error_count, true_class, predicted_class in val_misclassifications:
    print(
        true_class,
        "->",
        predicted_class,
        ":",
        error_count
    )

# -------------------------
# Validation 混淆矩阵可视化
# -------------------------

val_confusion_display = ConfusionMatrixDisplay(
    confusion_matrix=val_confusion_matrix,
    display_labels=class_names
)

val_confusion_display.plot(
    xticks_rotation=45
)

plt.title("MLP Classifier Validation Confusion Matrix")
plt.tight_layout()

plt.savefig(
    results_dir / "mlp_validation_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

# -------------------------
# Test 评估
# -------------------------

test_results = evaluate_model(
    model=model,
    data_loader=test_loader,
    criterion=criterion,
    device=device
)

print(
    "Test loss:",
    test_results["loss"],
    "Test accuracy:",
    test_results["accuracy"],
    "Test Macro-F1:",
    test_results["macro_f1"]
)

test_report = classification_report(
    test_results["labels"],
    test_results["predictions"],
    labels=range(num_classes),
    target_names=class_names,
    digits=4
)

print("Test classification report:")
print(test_report)

# -------------------------
# Test 错误分析
# -------------------------

test_confusion_matrix = confusion_matrix(
    test_results["labels"],
    test_results["predictions"],
    labels=range(num_classes)
)

test_misclassifications = get_top_misclassifications(
    confusion_matrix=test_confusion_matrix,
    class_names=class_names,
    top_k=5
)

print("Top 5 test misclassifications:")

for error_count, true_class, predicted_class in test_misclassifications:
    print(
        true_class,
        "->",
        predicted_class,
        ":",
        error_count
    )

# -------------------------
# Test 混淆矩阵可视化
# -------------------------

test_confusion_display = ConfusionMatrixDisplay(
    confusion_matrix=test_confusion_matrix,
    display_labels=class_names
)

test_confusion_display.plot(
    xticks_rotation=45
)

plt.title("MLP Classifier Test Confusion Matrix")
plt.tight_layout()

plt.savefig(
    results_dir / "mlp_test_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

# -------------------------
# Loss 曲线
# -------------------------

plt.figure()

plt.plot(
    epochs,
    train_losses,
    label="Training Loss"
)

plt.plot(
    epochs,
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Classifier Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    results_dir / "mlp_loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

# -------------------------
# Validation 指标曲线
# -------------------------

plt.figure()

plt.plot(
    epochs,
    val_accuracies,
    label="Validation Accuracy"
)

plt.plot(
    epochs,
    val_macro_f1s,
    label="Validation Macro-F1"
)

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("MLP Classifier Validation Metrics")
plt.legend()
plt.tight_layout()

plt.savefig(
    results_dir / "mlp_validation_metrics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()