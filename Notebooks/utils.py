# =====================================================
#  utils.py - helper code shared by all project notebooks
#  (data, models, training loop, evaluation, checkpoints, plots)
#
#  The notebooks are run from the Notebooks/ folder, so all paths are relative to it.
# =====================================================

import json
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Subset

# -----------------------------------------------------
# 0. Paths and constants
# -----------------------------------------------------
IN_COLAB = "google.colab" in sys.modules

# on Colab the dataset is stored on the local disk of the machine (much faster than Google Drive),
# checkpoints and results are stored in the project folder on Drive so they are not lost
DATA_DIR = "/content/data" if IN_COLAB else "../data"
CHECKPOINT_DIR = "../Checkpoints"
RESULTS_DIR = "../results"
FIGURES_DIR = "../results/figures"

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# CIFAR-10 mean / std per channel (Method 2 of Lab 9: dataset statistics -> mean 0, std 1)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# ImageNet mean / std: pre-trained models expect inputs normalized like their training data
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed=42):
    """Fix the random seeds so every run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------
# 1. Data transforms (augmentation)
# -----------------------------------------------------
def get_transforms(augmentation="none", img_size=32, imagenet_norm=False):
    """
    augmentation = "none"  : ToTensor + Normalize only (validation / test / baseline)
    augmentation = "basic" : + RandomCrop with padding + RandomHorizontalFlip
    augmentation = "full"  : + RandomAffine (rotation, translation) + ColorJitter
    (MixUp is applied on whole batches inside train_one_epoch, not here)
    """
    if imagenet_norm:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = CIFAR_MEAN, CIFAR_STD

    transform_list = []
    if img_size != 32:
        transform_list.append(transforms.Resize(img_size))  # upscale 32x32 -> img_size for pre-trained models

    if augmentation == "basic" or augmentation == "full":
        # pad the borders (4 px for 32x32) then crop a random img_size x img_size region
        transform_list.append(transforms.RandomCrop(img_size, padding=img_size // 8))
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if augmentation == "full":
        # rotate -15 to +15 degrees and shift up to 10% in x and y
        transform_list.append(transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)))
        transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))

    transform_list.append(transforms.ToTensor())            # [0, 255] -> [0, 1]
    transform_list.append(transforms.Normalize(mean, std))  # (x - mean) / std for each channel
    return transforms.Compose(transform_list)


# -----------------------------------------------------
# 2. Datasets and DataLoaders
# -----------------------------------------------------
def get_dataloaders(batch_size=128, augmentation="none", img_size=32, imagenet_norm=False,
                    train_fraction=1.0, val_size=5000, num_workers=None):
    """
    CIFAR-10 split:
      - train : 45,000 images of the official train set (optionally only train_fraction of them)
      - val   :  5,000 images of the official train set (no augmentation)
      - test  : 10,000 official test images, only used for the final evaluation
    """
    transform_train = get_transforms(augmentation, img_size, imagenet_norm)
    transform_val = get_transforms("none", img_size, imagenet_norm)

    # the train set is loaded twice: with augmentation (train part) and without (val part)
    train_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
    val_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_val)
    test_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_val)

    # fixed random split (same seed -> same 45k / 5k split for every experiment)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # keep only a part of the training images (data-efficiency experiment, Task 3)
    if train_fraction < 1.0:
        train_indices = train_indices[:int(len(train_indices) * train_fraction)]

    # num_workers=0 avoids multiprocessing crash on Windows
    # on Colab (Linux) 2 workers load the next batches in parallel -> faster training
    if num_workers is None:
        num_workers = 2 if IN_COLAB else 0

    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(Subset(val_dataset, val_indices), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def get_test_dataset(img_size=32, imagenet_norm=False):
    """Official test set only (used by the demo notebook)."""
    transform_test = get_transforms("none", img_size, imagenet_norm)
    return torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)


# -----------------------------------------------------
# 3. Models
# -----------------------------------------------------
class Net(nn.Module):
    """Baseline CNN (Task 1): 3 conv layers + 2 fully connected layers, trained from scratch."""

    def __init__(self):
        super(Net, self).__init__()
        # kernel=3x3, padding=1 | input 3x32x32 | output 32x32x32 | h_out = (32 + 2*1 - 3)/1 + 1 = 32
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # input 32x16x16 | output 64x16x16
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # input 64x8x8 | output 128x8x8
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # kernel=2x2 | stride=2 -> height and width are divided by 2
        self.pool = nn.MaxPool2d(2, 2)

        # flatten it to a vector: in_features = channels x height x width = 128 x 4 x 4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32x32 -> 32x16x16
        x = self.pool(F.relu(self.conv2(x)))  # 64x16x16 -> 64x8x8
        x = self.pool(F.relu(self.conv3(x)))  # 128x8x8  -> 128x4x4
        x = x.view(-1, 128 * 4 * 4)           # the size -1 is inferred (= batch size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                       # raw scores (logits), softmax is inside CrossEntropyLoss
        return x


def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    elif name == "gelu":
        return nn.GELU()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError("unknown activation: " + name)


class CustomCNN(nn.Module):
    """
    Flexible version of Net used for the Task 2 experiments.

    conv_channels : output channels of each conv layer -> len(conv_channels) = number of conv layers
    fc_sizes      : hidden fully connected layers (the last Linear(..., 10) is always added)
    kernel_size   : 3, 5 or 7
    padding       : "same" (keeps height/width) or "valid" (no padding, shrinks by kernel_size - 1)
    use_stride    : True -> downsample with a stride-2 conv instead of MaxPool
    use_bn        : add BatchNorm2d after each conv layer
    dropout       : dropout probability in the fully connected part (0 = no dropout)
    activation    : "relu", "leaky_relu", "gelu" or "tanh"

    With the default values it is the same architecture as Net.
    """

    def __init__(self, conv_channels=[32, 64, 128], fc_sizes=[256], kernel_size=3, padding="same",
                 use_stride=False, use_bn=False, dropout=0.0, activation="relu"):
        super(CustomCNN, self).__init__()

        if padding == "same":
            pad = kernel_size // 2
        else:
            pad = 0
        stride = 2 if use_stride else 1

        # ---- convolutional part ----
        conv_layers = []
        in_channels = 3
        size = 32  # height (= width) of the current feature map
        for out_channels in conv_channels:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad))
            if use_bn:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(get_activation(activation))
            size = (size + 2 * pad - kernel_size) // stride + 1  # h_out formula (Lab 9)

            if not use_stride:
                conv_layers.append(nn.MaxPool2d(2, 2))
                size = size // 2

            if size < 1:
                raise ValueError("feature map is too small: use fewer conv layers or a smaller kernel")
            in_channels = out_channels
        self.features = nn.Sequential(*conv_layers)

        # ---- fully connected part ----
        self.num_flat_features = in_channels * size * size
        fc_layers = []
        in_features = self.num_flat_features
        for hidden in fc_sizes:
            fc_layers.append(nn.Linear(in_features, hidden))
            fc_layers.append(get_activation(activation))
            if dropout > 0:
                fc_layers.append(nn.Dropout(dropout))
            in_features = hidden
        fc_layers.append(nn.Linear(in_features, 10))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.num_flat_features)  # flatten
        x = self.classifier(x)
        return x


def get_resnet18(mode="frozen", pretrained=True):
    """
    ResNet-18 pre-trained on ImageNet with a new head for 10 classes.
    mode = "frozen"          : feature extraction, only the new head is trained
    mode = "finetune_layer4" : last residual block (layer4) + head are trained
    mode = "finetune_all"    : the whole network is trained
    """
    if pretrained:
        # download the ImageNet weights from the internet (cached after the first time)
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        # offline alternative (as in the lab):
        # model = torchvision.models.resnet18(weights=None)
        # model.load_state_dict(torch.load("resnet18-f37072fd.pth"))
    else:
        # weights will be loaded from our own checkpoint
        model = torchvision.models.resnet18(weights=None)

    # Freeze feature extractor: none of the original ResNet layers will update during training
    for param in model.parameters():
        param.requires_grad = False

    if mode == "finetune_layer4":
        # Unfreeze last ResNet block
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif mode == "finetune_all":
        for param in model.parameters():
            param.requires_grad = True

    # Replace final fully connected layer with classifier for 10 classes
    # (created after the freezing loop, so its parameters are trainable by default)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),                     # Regularization
        nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes
    )
    return model


def build_model(model_name, model_args={}):
    """Creates an empty model from its name (used when loading checkpoints)."""
    if model_name == "Net":
        return Net()
    elif model_name == "CustomCNN":
        return CustomCNN(**model_args)
    elif model_name == "resnet18":
        return get_resnet18(pretrained=False, **model_args)
    else:
        raise ValueError("unknown model: " + model_name)


def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def receptive_field(model):
    """
    Receptive field (in input pixels) of one value of the last feature map.
    For every conv / pool layer:  rf = rf + (k - 1) * jump   and   jump = jump * stride
    (jump = distance in input pixels between two neighbouring values of the feature map)
    """
    rf, jump = 1, 1
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
            k = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
            s = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
            rf = rf + (k - 1) * jump
            jump = jump * s
    return rf


# -----------------------------------------------------
# 4. Optimizers and learning rate schedulers
# -----------------------------------------------------
def get_optimizer(name, parameters, lr):
    if name == "sgd":
        return optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=5e-4)
    elif name == "adam":
        return optim.Adam(parameters, lr=lr)
    elif name == "adamw":
        # AdamW = Adam with decoupled weight decay
        return optim.AdamW(parameters, lr=lr, weight_decay=0.01)
    else:
        raise ValueError("unknown optimizer: " + name)


def get_scheduler(name, optimizer, num_epochs):
    if name is None:
        return None                                              # constant learning rate
    elif name == "step":
        # lr is divided by 10 every num_epochs // 3 epochs
        return optim.lr_scheduler.StepLR(optimizer, step_size=max(1, num_epochs // 3), gamma=0.1)
    elif name == "cosine":
        # lr decreases smoothly (cosine curve) from lr to ~0 at the last epoch
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        raise ValueError("unknown scheduler: " + name)


# -----------------------------------------------------
# 5. Training & validation loops
# -----------------------------------------------------
def mixup_data(images, labels, alpha=0.2):
    """
    MixUp: each image is blended with another random image of the same batch
        mixed_image = lam * image_i + (1 - lam) * image_j     with lam ~ Beta(alpha, alpha)
    The loss is blended with the same lam (see train_one_epoch).
    """
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0)).to(images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


def train_one_epoch(model, train_loader, criterion, optimizer, device, use_mixup=False, mixup_alpha=0.2):
    model.train()

    # frozen BatchNorm layers (transfer learning) stay in eval mode, otherwise model.train()
    # would still update their running mean/var even if their weights are frozen
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d) and not layer.weight.requires_grad:
            layer.eval()

    correct, total, train_loss = 0, 0, 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)

        # Forward
        outputs = model(images)
        if use_mixup:
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            loss = criterion(outputs, labels)

        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats (with MixUp the accuracy is only approximate: compared to the first label)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        train_loss += loss.item()

    return train_loss / len(train_loader), correct / total


def evaluate(model, data_loader, criterion, device):
    """Returns loss, accuracy, per-class accuracy, all predictions and all labels."""
    model.eval()  # disables dropout and batch normalization updates
    correct, total, eval_loss = 0, 0, 0.0
    class_correct = [0] * 10
    class_total = [0] * 10
    all_preds, all_labels = [], []

    with torch.no_grad():  # no gradients needed for testing
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            eval_loss += loss.item()

            preds, labels = preds.cpu().numpy(), labels.cpu().numpy()
            for i in range(len(labels)):
                class_correct[labels[i]] += int(preds[i] == labels[i])
                class_total[labels[i]] += 1
            all_preds.extend(preds)
            all_labels.extend(labels)

    class_acc = [class_correct[i] / class_total[i] for i in range(10)]
    return eval_loss / len(data_loader), correct / total, class_acc, all_preds, all_labels


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path,
                scheduler=None, use_mixup=False, checkpoint_info=None):
    """
    Trains for num_epochs, logs everything for each epoch and saves the best model (highest val acc).
    checkpoint_info: extra info stored in the checkpoint to rebuild the model later
                     (model_name, model_args, img_size, imagenet_norm)
    """
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_class_acc": [], "lr": [], "epoch_time": []}
    best_acc = 0.0

    for epoch in range(num_epochs):
        start = time.time()

        # Training phase
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_mixup)

        # Validation phase
        val_loss, val_acc, val_class_acc, _, _ = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()  # update the learning rate once per epoch
        epoch_time = time.time() - start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_class_acc"].append(val_class_acc)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | lr: {current_lr:.0e} | {epoch_time:.1f}s")

        # Save best model checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, epoch + 1, val_acc, save_path, checkpoint_info)
            print("   Saved new best model!")

    history["best_epoch"] = int(np.argmax(history["val_acc"])) + 1
    print("Finished training! Best Val Accuracy:", best_acc)
    return history


# -----------------------------------------------------
# 6. Checkpoints and history
# -----------------------------------------------------
def save_checkpoint(model, epoch, val_acc, path, checkpoint_info=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {"epoch": epoch, "val_acc": val_acc, "model_state_dict": model.state_dict()}
    if checkpoint_info is not None:
        checkpoint.update(checkpoint_info)
    torch.save(checkpoint, path)


def load_checkpoint(path, device):
    """Rebuilds the model saved in `path` and loads its weights. Returns (model, checkpoint)."""
    checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint["model_name"], checkpoint.get("model_args", {}))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def save_history(history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def load_history(path):
    with open(path) as f:
        return json.load(f)


# -----------------------------------------------------
# 7. Metrics for the comparison tables
# -----------------------------------------------------
def measure_latency(model, device, img_size=32, n_runs=100):
    """Average inference time (ms) for one image."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size).to(device)
    with torch.no_grad():
        for _ in range(10):  # warm-up
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()  # GPU runs asynchronously -> wait before measuring
        start = time.time()
        for _ in range(n_runs):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.time() - start) / n_runs * 1000


def epochs_to_reach(history, threshold):
    """First epoch where val accuracy >= threshold (None if never reached) -> convergence speed."""
    for epoch, acc in enumerate(history["val_acc"]):
        if acc >= threshold:
            return epoch + 1
    return None


# -----------------------------------------------------
# 8. Plots
# -----------------------------------------------------
def save_figure(save_name):
    if save_name is not None:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(f"{FIGURES_DIR}/{save_name}", dpi=150, bbox_inches="tight")


def imshow(img, imagenet_norm=False):
    """Shows an image grid made with torchvision.utils.make_grid."""
    if imagenet_norm:
        mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)
    else:
        mean, std = np.array(CIFAR_MEAN), np.array(CIFAR_STD)
    npimg = img.numpy().transpose((1, 2, 0))  # C x H x W -> H x W x C
    npimg = std * npimg + mean                # unnormalize
    npimg = np.clip(npimg, 0, 1)
    plt.figure(figsize=(12, 4))
    plt.imshow(npimg)
    plt.axis("off")
    plt.show()


def show_predictions(images, labels, preds, imagenet_norm=False):
    """Images with true (T) and predicted (P) labels: green = correct, red = wrong."""
    if imagenet_norm:
        mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)
    else:
        mean, std = np.array(CIFAR_MEAN), np.array(CIFAR_STD)
    n = len(images)
    plt.figure(figsize=(2 * min(n, 8), 2.4 * ((n + 7) // 8)))
    for i in range(n):
        npimg = np.clip(std * images[i].numpy().transpose((1, 2, 0)) + mean, 0, 1)
        plt.subplot((n + 7) // 8, min(n, 8), i + 1)
        plt.imshow(npimg)
        color = "green" if preds[i] == labels[i] else "red"
        plt.title(f"T: {classes[labels[i]]}\nP: {classes[preds[i]]}", color=color, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_history(history, title="", save_name=None):
    """Loss and accuracy curves (train vs validation)."""
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title + " - Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title + " - Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(save_name)
    plt.show()


def plot_compare(histories, title="", save_name=None):
    """Val loss and val accuracy of several runs. histories = {run name: history}."""
    plt.figure(figsize=(12, 4))
    for name, history in histories.items():
        epochs = range(1, len(history["val_loss"]) + 1)
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history["val_loss"], label=name)
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history["val_acc"], label=name)

    plt.subplot(1, 2, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.title(title + " - Validation Loss")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy")
    plt.title(title + " - Validation Accuracy")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(save_name)
    plt.show()


def plot_class_acc_history(history, title="Per-class validation accuracy", save_name=None):
    """Validation accuracy of each class at each epoch."""
    class_acc = np.array(history["val_class_acc"])  # shape: [epochs, 10]
    epochs = range(1, len(class_acc) + 1)
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.plot(epochs, class_acc[:, i], label=classes[i])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(ncol=5, fontsize=8)
    plt.grid(alpha=0.3)
    save_figure(save_name)
    plt.show()


def plot_class_accuracy(class_acc, title="Per-class accuracy", save_name=None):
    plt.figure(figsize=(10, 4))
    plt.bar(classes, class_acc)
    for i in range(10):
        plt.text(i, class_acc[i] + 0.01, f"{class_acc[i]:.2f}", ha="center", fontsize=8)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title(title)
    save_figure(save_name)
    plt.show()


def plot_confusion_matrix(all_labels, all_preds, title="Normalized confusion matrix", save_name=None):
    """Normalized over the true labels: each row sums to 1 (diagonal = per-class accuracy)."""
    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, values_format=".2f", cmap="Blues", colorbar=False)
    plt.title(title)
    save_figure(save_name)
    plt.show()
