import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torchvision.ops import focal_loss
from tqdm import tqdm

from contra_loss import SupConLoss
from models import ClassificationModel, SupConResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists("models"):
    os.makedirs("models")


def contrastive_train(loader_train_cropped) -> list:
    # contrastive model
    contrastive_resnet50 = SupConResNet().to(device)
    criterion = SupConLoss()
    optimizer = torch.optim.SGD(
        contrastive_resnet50.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-2
    )

    # define the loss list
    contra_loss_list = []
    best_contra_loss = float("inf")

    max_epoch = 100
    contrastive_resnet50.train()

    # train the contrastive model
    for epoch in range(max_epoch):
        print(" -- Epoch {}/{}".format(epoch + 1, max_epoch))

        ### training the model
        running_loss = 0.0

        for images, labels in tqdm(loader_train_cropped):
            optimizer.zero_grad()
            images = torch.cat([images[0], images[1]], dim=0)
            images, labels = images.to(device), labels.float().to(device)

            bsz = labels.shape[0]

            features = contrastive_resnet50(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            loss = criterion(features)  # contrastive loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters=contrastive_resnet50.parameters(), max_norm=10
            )  # Clip gradients
            optimizer.step()

            running_loss += loss.item()

        ### record the training loss and metrics
        contra_loss = running_loss / len(loader_train_cropped)
        contra_loss_list.append(contra_loss)

        print("Constrastive Train Loss {:.4f}".format(contra_loss))

        if contra_loss < best_contra_loss:
            best_contra_loss = contra_loss
            torch.save(
                contrastive_resnet50.state_dict(), "models/best_contrastive_resnet50.pt"
            )

    return contra_loss_list


def classifier_train(
    loader_train, loader_val
) -> Tuple[list, list, list, list, list, list]:
    # get the best contrastive model
    contrastive_resnet50 = SupConResNet().to(device)
    contrastive_resnet50.load_state_dict(
        torch.load("models/best_contrastive_resnet50.pt")
    )
    contrastive_resnet50.eval()

    # classifier for downstream task
    classifier = ClassificationModel().to(device)
    classifier_optimizer = torch.optim.SGD(
        classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-2
    )

    max_epoch = 10
    best_val_loss = float("inf")

    loss_train_list, loss_val_list = [], []  # record the training loss
    auc_train_list, acc_train_list = [], []  # record the training metrics
    auc_val_list, acc_val_list = [], []  # record the validation metrics

    for epoch in range(max_epoch):
        print(" -- Epoch {}/{}".format(epoch + 1, max_epoch))
        ### evaluate on validation set
        classifier.train()
        running_loss = 0.0
        train_lbl = []
        train_pred = []

        for images, labels in tqdm(loader_train):
            classifier_optimizer.zero_grad()
            images, labels = images.to(device), labels.float().to(device)

            with torch.no_grad():
                visual = contrastive_resnet50.encoder(images)

            y_pred = classifier(visual.detach())[:, 0]

            loss = focal_loss.sigmoid_focal_loss(y_pred, labels, reduction="mean")
            loss.backward()
            classifier_optimizer.step()

            running_loss += loss.item()
            y_pred = list(y_pred.detach().cpu().numpy())
            y_true = list(labels.detach().cpu().numpy())
            train_lbl += y_true
            train_pred += y_pred

        ### record the training loss and metrics
        loss = running_loss / len(loader_train)
        loss_train_list.append(loss)

        train_lbl, train_pred = np.array(train_lbl), np.array(train_pred)
        train_pred_lbl = np.around(
            train_pred
        )  # pred >= 0.5 pred_lbl = 1 else pred_lbl = 0
        train_auc = roc_auc_score(train_lbl, train_pred)
        train_acc = accuracy_score(train_lbl, train_pred_lbl)

        auc_train_list.append(train_auc)
        acc_train_list.append(train_acc)

        print(" -- Validation")
        classifier.eval()

        val_lbl, val_pred = [], []
        val_loss = 0.0

        for val_images, val_labels in loader_val:
            val_images, val_labels = val_images.to(device), val_labels.float().to(
                device
            )

            with torch.no_grad():
                y_pred = classifier(contrastive_resnet50.encoder(val_images).detach())[
                    :, 0
                ]

            v_loss = focal_loss.sigmoid_focal_loss(y_pred, val_labels, reduction="mean")
            val_loss += v_loss.item()

            y_pred = list(y_pred.detach().cpu().numpy())
            y_true = list(val_labels.detach().cpu().numpy())
            val_lbl += y_true
            val_pred += y_pred

        ### record the validation loss and metrics, save the best checkpoint
        val_loss = val_loss / len(loader_val)
        loss_val_list.append(val_loss)

        val_lbl, val_pred = np.array(val_lbl), np.array(val_pred)
        val_pred_lbl = np.around(val_pred)  # pred >= 0.5 pred_lbl = 1 else pred_lbl = 0
        val_auc = roc_auc_score(val_lbl, val_pred)
        val_acc = accuracy_score(val_lbl, val_pred_lbl)

        auc_val_list.append(val_auc)
        acc_val_list.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(classifier.state_dict(), "models/best_classifier.pt")

        print(
            "Train Loss {:.4f}, Train Accuracy {:.4f}%, Validation Accuracy {:.4f}%, Train AUC {:.4f}%, Validation AUC {:.4f}%".format(
                loss,
                train_acc * 100,
                val_acc * 100,
                train_auc * 100,
                val_auc * 100,
            )
        )

    return (
        loss_train_list,
        auc_train_list,
        acc_train_list,
        loss_val_list,
        auc_val_list,
        acc_val_list,
    )


def inference(loader_test) -> Tuple[float, float]:
    ### evaluate on test set
    contrastive_resnet50 = SupConResNet().to(device)
    contrastive_resnet50.load_state_dict(
        torch.load("models/best_contrastive_resnet50.pt")
    )
    contrastive_resnet50.eval()

    classifier = ClassificationModel().to(device)
    classifier.load_state_dict(torch.load("models/best_classifier.pt"))
    classifier.eval()

    test_lbl, test_pred = [], []

    for test_images, test_labels in loader_test:
        test_images, test_labels = test_images.to(device), test_labels.float().to(
            device
        )

        with torch.no_grad():
            y_pred = classifier(contrastive_resnet50.encoder(test_images).detach())[
                :, 0
            ]
        y_pred = torch.sigmoid(y_pred)

        y_pred = list(y_pred.detach().cpu().numpy())
        y_true = list(test_labels.detach().cpu().numpy())
        test_lbl += y_true
        test_pred += y_pred

    ### compute and print the metrics on test set
    test_lbl, test_pred = np.array(test_lbl), np.array(test_pred)
    test_pred_lbl = np.around(test_pred)  # pred >= 0.5 pred_lbl = 1 else pred_lbl = 0
    test_auc = roc_auc_score(test_lbl, test_pred)
    test_acc = accuracy_score(test_lbl, test_pred_lbl)

    return test_auc, test_acc
