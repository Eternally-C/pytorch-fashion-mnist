import copy

import torch
from sklearn.metrics import f1_score


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs
):
    """
    训练结束后恢复验证集 Macro-F1 最佳 epoch 的模型参数，
    并返回完整训练历史以及该 epoch 的验证指标。
    """

    # 保存每个 epoch 的指标，用于后续绘图和模型比较。
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_macro_f1s = []

    # 记录验证集 Macro-F1 最好的模型。
    best_val_macro_f1 = float("-inf")
    best_epoch = 0
    best_model_state = None
    best_val_loss = None
    best_val_accuracy = None

    for epoch in range(num_epochs):

        # -------------------------
        # 训练阶段
        # -------------------------
        model.train()

        train_loss_sum = 0.0
        train_total = 0

        for images, labels in train_loader:
            # 将当前 batch 移动到当前选择的计算设备。
            images = images.to(device)
            labels = labels.to(device)

            # 清空上一次参数更新后残留的梯度。
            optimizer.zero_grad()

            # 前向传播并计算当前 batch 的损失。
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播计算梯度，并更新模型参数。
            loss.backward()
            optimizer.step()

            # 按样本数累计 loss，
            # 保证最后一个不足完整 batch 的数据也能正确计入平均值。
            batch_size = labels.size(0)

            train_loss_sum += loss.item() * batch_size
            train_total += batch_size

        train_loss = train_loss_sum / train_total

        # -------------------------
        # 验证阶段
        # -------------------------
        model.eval()

        val_loss_sum = 0.0
        correct = 0
        total = 0

        # 保存整个验证集的真实标签和预测标签，
        # 用于计算 Macro-F1。
        all_val_labels = []
        all_val_predictions = []

        # 验证阶段不需要梯度，可以减少计算和内存开销。
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                # 选择 logit 最大的类别作为预测结果。
                predictions = outputs.argmax(dim=1)

                # 将 Tensor 移到 CPU，并转换成普通 Python 列表。
                all_val_labels.extend(labels.cpu().tolist())
                all_val_predictions.extend(predictions.cpu().tolist())

                batch_size = labels.size(0)

                val_loss_sum += loss.item() * batch_size
                correct += (predictions == labels).sum().item()
                total += batch_size

        val_loss = val_loss_sum / total
        val_accuracy = correct / total

        # 分别计算所有类别的 F1，再取等权平均。
        val_macro_f1 = f1_score(
            all_val_labels,
            all_val_predictions,
            average="macro"
        )

        # 保存本轮指标。
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_macro_f1s.append(val_macro_f1)

        # 如果当前 Macro-F1 优于历史最佳结果，
        # 保存当前 epoch 和模型参数快照。
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1

            # 保存当前最佳模型的参数快照。
            best_model_state = copy.deepcopy(model.state_dict())

        # 持久保留的训练日志。
        print(
            "Epoch:",
            epoch + 1,
            "Training loss:",
            train_loss,
            "Validation loss:",
            val_loss,
            "Validation accuracy:",
            val_accuracy,
            "Validation Macro-F1:",
            val_macro_f1
        )

    # 训练结束后恢复验证集 Macro-F1 最好的模型参数。
    model.load_state_dict(best_model_state)

    print(
        "Best epoch:",
        best_epoch,
        "Best validation loss:",
        best_val_loss,
        "Best validation accuracy:",
        best_val_accuracy,
        "Best validation Macro-F1:",
        best_val_macro_f1
    )

    # 将完整训练历史和最佳 epoch 的指标返回给 main.py。
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "val_macro_f1s": val_macro_f1s,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "best_val_macro_f1": best_val_macro_f1
    }