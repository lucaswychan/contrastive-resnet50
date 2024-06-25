import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


root_train, root_val, root_test = "./train", "./val", "./test"

contrastive_train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomVerticalFlip(),  # using stronger augmentation functions to enhance performance or solve LT problem
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_sampler(dataset):
    np.random.shuffle(dataset.samples)

    # solve the class imbalance problem
    count = [0, 0]
    for _, label in dataset:
        count[label] += 1

    class_sample_count = np.array(count)
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[label] for _, label in dataset])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        samples_weight.type("torch.DoubleTensor"), len(samples_weight)
    )

    return sampler


def get_train_test_set(batch_size):
    train_dataset_cropped = ImageFolder(
        root_train, transform=TwoCropTransform(contrastive_train_transform)
    )

    ### weighted sampling to tackle class imbalance
    sampler_train_cropped = get_sampler(train_dataset_cropped)
    loader_train_cropped = torch.utils.data.DataLoader(
        train_dataset_cropped,
        sampler=sampler_train_cropped,
        batch_size=batch_size,
    )

    train_dataset = ImageFolder(root_train, transform=train_transform)

    sampler_train = get_sampler(train_dataset)
    loader_train = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler_train,
        batch_size=batch_size,
    )

    val_dataset = ImageFolder(root_val, transform=test_transform)
    loader_val = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False
    )

    test_dataset = ImageFolder(root_test, transform=test_transform)
    loader_test = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    return loader_train_cropped, loader_train, loader_val, loader_test
