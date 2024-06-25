import os

import matplotlib.pyplot as plt


def plot_graphs(
    loss_train_list, loss_val_list, acc_train_list, acc_val_list, contra_loss_train_list
):
    # Plotting the training loss curve
    if not os.path.exists("results"):
        os.makedirs("results")

    plt.figure(figsize=(8, 6))
    plt.plot(loss_train_list, label="Training Loss")
    plt.plot(loss_val_list, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.savefig("results/loss_curve.png")

    # Plotting the training and testing accuracy curves
    plt.figure(figsize=(8, 6))
    plt.plot(acc_train_list, label="Training Accuracy")
    plt.plot(acc_val_list, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy Curves")
    plt.legend()
    plt.savefig("results/accuracy_curve.png")

    # Plotting the contrastive loss
    plt.figure(figsize=(8, 6))
    plt.plot(contra_loss_train_list, label="Contrastive Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Contrastive Loss Curve")
    plt.legend()
    plt.savefig("results/contra_loss_curve.png")
