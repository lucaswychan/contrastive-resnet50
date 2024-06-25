from contra_flow import classifier_train, contrastive_train, inference
from dataloader import get_train_test_set
from plot import plot_graphs


def main():
    batch_size = 32
    loader_train_cropped, loader_train, loader_val, loader_test = get_train_test_set(
        batch_size
    )

    contra_loss_list = contrastive_train(loader_train_cropped)
    (
        loss_train_list,
        auc_train_list,
        acc_train_list,
        loss_val_list,
        auc_val_list,
        acc_val_list,
    ) = classifier_train(loader_train, loader_val)
    test_auc, test_acc = inference(loader_test)

    plot_graphs(
        loss_train_list, loss_val_list, acc_train_list, acc_val_list, contra_loss_list
    )
    print(f"Test AUC: {test_auc}, Test Accuracy: {test_acc}")


if __name__ == "__main__":
    main()
