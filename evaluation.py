import torch
from sklearn.metrics import f1_score


def evaluate_model(
    model,
    data_loader,
    criterion,
    device
):
    """
    在给定数据集上评估已经训练好的模型。

    不更新模型参数，返回 loss、accuracy、Macro-F1，
    以及真实标签和预测标签。
    """

    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    # 评估阶段不需要计算梯度。
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)

            # 按样本数累计 loss，保证最后一个不完整 batch 正确计入平均值。
            loss_sum += loss.item() * batch_size

            correct += (predictions == labels).sum().item()
            total += batch_size

            # 保存真实标签和预测标签，用于后续计算指标和错误分析。
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = loss_sum / total
    accuracy = correct / total

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "labels": all_labels,
        "predictions": all_predictions
    }


def get_top_misclassifications(
    confusion_matrix,
    class_names,
    top_k=5
):
    """
    从混淆矩阵中找出数量最多的误分类方向。

    返回格式：
    (错误数量, 真实类别名称, 预测类别名称)
    """

    misclassifications = []

    num_classes = len(class_names)

    for true_label in range(num_classes):
        for predicted_label in range(num_classes):

            # 跳过对角线，因为对角线表示预测正确。
            if true_label == predicted_label:
                continue

            error_count = confusion_matrix[
                true_label,
                predicted_label
            ]

            # 跳过没有实际发生过的误分类方向。
            if error_count == 0:
                continue

            misclassifications.append(
                (
                    error_count,
                    class_names[true_label],
                    class_names[predicted_label]
                )
            )

    # 按错误数量从高到低排序。
    misclassifications.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return misclassifications[:top_k]