# ============================================================
# train.py
# CAPSICUM CONDITION CLASSIFICATION - RESNET50
#
# DATA FLOW:
#
# data new
#     ↓
# exact duplicate removal
#     ↓
# near duplicate / grouping
#     ↓
# clean 80/10/10 split
#     ↓
# train-only augmentation
#     ↓
# augmented_dataset/train
#     ↓
# ResNet50 training
#     ↓
# Early Stopping
#     ↓
# Best Model
#     ↓
# Internal Test
#
# VALIDATION AND TEST ARE NEVER AUGMENTED
# ============================================================


import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ------------------------------------------------------------
# DATASET DIRECTORIES
# ------------------------------------------------------------

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "augmented_dataset",
    "train"
)

VALID_DIR = os.path.join(
    BASE_DIR,
    "dataset_split",
    "valid"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset_split",
    "test"
)


# ------------------------------------------------------------
# MODEL DIRECTORY
# ------------------------------------------------------------

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# RESULT DIRECTORY
# ------------------------------------------------------------

RESULT_DIR = os.path.join(
    BASE_DIR,
    "training_results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# CLASSES
# ============================================================

CLASSES = [
    "Anthracnose",
    "Larva",
    "Magnesium",
    "bacterial spot",
    "blossom-end rot",
    "down leaf aphid",
    "fruit thrips",
    "healthy",
    "powdery mildew",
    "snail",
    "upperleaf thrips",
    "virus"
]


NUM_CLASSES = len(CLASSES)


# ============================================================
# TRAINING PARAMETERS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 64

NUM_WORKERS = 0

EPOCHS = 10

LEARNING_RATE = 5e-5

WEIGHT_DECAY = 1e-4

DROPOUT = 0.30


# ============================================================
# EARLY STOPPING
# ============================================================

PATIENCE = 3

MIN_DELTA = 0.001


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42


def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("\n")
print("=" * 70)

print("CAPSICUM RESNET50 TRAINING")

print("=" * 70)

print(
    f"\nDevice: {device}"
)

if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

else:

    print(
        "GPU not available - using CPU"
    )


# ============================================================
# CHECK DATASET DIRECTORIES
# ============================================================

print("\n")
print("=" * 70)
print("CHECKING DATASET DIRECTORIES")
print("=" * 70)


required_dirs = [
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR
]


for directory in required_dirs:

    if not os.path.exists(directory):

        raise FileNotFoundError(
            f"\nDataset directory not found:\n"
            f"{directory}"
        )

    print(
        f"OK: {directory}"
    )


# ============================================================
# TRANSFORMS
# ============================================================

# ------------------------------------------------------------
# TRAIN TRANSFORM
#
# Offline augmentation already created 1100/class.
# Here we use only MILD runtime augmentation.
# ------------------------------------------------------------

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ------------------------------------------------------------
# VALIDATION TRANSFORM
#
# NO RANDOM AUGMENTATION
# ------------------------------------------------------------

valid_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ------------------------------------------------------------
# TEST TRANSFORM
#
# NO RANDOM AUGMENTATION
# ------------------------------------------------------------

test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD DATASETS
# ============================================================

print("\n")
print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)


train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)


valid_dataset = datasets.ImageFolder(
    VALID_DIR,
    transform=valid_transform
)


test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)


# ============================================================
# VERIFY CLASS ORDER
# ============================================================

print("\n")
print("Classes detected by ImageFolder:")

print(
    train_dataset.classes
)


if train_dataset.classes != CLASSES:

    print("\nWARNING:")
    print(
        "ImageFolder class order differs from "
        "the expected class list."
    )

    print(
        "\nExpected:"
    )

    print(CLASSES)

    print(
        "\nDetected:"
    )

    print(train_dataset.classes)


# ============================================================
# DATASET COUNTS
# ============================================================

print("\n")
print("=" * 70)
print("DATASET COUNTS")
print("=" * 70)


print(
    f"\nTrain images: "
    f"{len(train_dataset)}"
)

print(
    f"Validation images: "
    f"{len(valid_dataset)}"
)

print(
    f"Test images: "
    f"{len(test_dataset)}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n")
print("TRAIN CLASS DISTRIBUTION")

train_counts = {}


for class_name in train_dataset.classes:

    class_index = train_dataset.class_to_idx[
        class_name
    ]

    count = sum(
        1
        for _, label
        in train_dataset.samples
        if label == class_index
    )

    train_counts[class_name] = count

    print(
        f"{class_name:<25} "
        f"{count}"
    )


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# BUILD RESNET50
# ============================================================

print("\n")
print("=" * 70)
print("BUILDING RESNET50")
print("=" * 70)


# ------------------------------------------------------------
# Load pretrained ResNet50
# ------------------------------------------------------------

try:

    weights = models.ResNet50_Weights.DEFAULT

    model = models.resnet50(
        weights=weights
    )

    print(
        "Loaded ResNet50 with pretrained ImageNet weights."
    )

except Exception:

    print(
        "WARNING: Could not load pretrained weights."
    )

    print(
        "Using ResNet50 without pretrained weights."
    )

    model = models.resnet50(
        weights=None
    )


# ============================================================
# FREEZE EARLY LAYERS
# ============================================================

# Freeze everything first

for parameter in model.parameters():

    parameter.requires_grad = False


# ------------------------------------------------------------
# Fine-tune Layer4
# ------------------------------------------------------------

for parameter in model.layer4.parameters():

    parameter.requires_grad = True


# ============================================================
# REPLACE FINAL CLASSIFIER
# ============================================================

in_features = model.fc.in_features


model.fc = nn.Sequential(

    nn.Dropout(
        p=DROPOUT
    ),

    nn.Linear(
        in_features,
        NUM_CLASSES
    )
)


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(device)


# ============================================================
# PRINT TRAINABLE PARAMETERS
# ============================================================

print("\n")
print("Trainable layers:")

for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        print(
            f"  {name}"
        )


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.AdamW(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=1
)


# ============================================================
# AMP
# ============================================================

use_amp = torch.cuda.is_available()


if use_amp:

    scaler = torch.cuda.amp.GradScaler()

else:

    scaler = None


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {

    "train_loss": [],

    "valid_loss": [],

    "train_accuracy": [],

    "valid_accuracy": []

}


# ============================================================
# EARLY STOPPING VARIABLES
# ============================================================

best_val_loss = float("inf")

best_val_accuracy = 0.0

best_epoch = 0

patience_counter = 0

best_model_state = copy.deepcopy(
    model.state_dict()
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        if use_amp:

            with torch.cuda.amp.autocast():

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )


            scaler.scale(
                loss
            ).backward()


            scaler.step(
                optimizer
            )


            scaler.update()


        else:

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (
            predicted == labels
        ).sum().item()


    epoch_loss = (
        running_loss / total
    )


    epoch_accuracy = (
        100.0 * correct / total
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            if use_amp:

                with torch.cuda.amp.autocast():

                    outputs = model(
                        images
                    )

                    loss = criterion(
                        outputs,
                        labels
                    )

            else:

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )


            running_loss += (
                loss.item()
                * images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)


            correct += (
                predicted == labels
            ).sum().item()


    epoch_loss = (
        running_loss / total
    )


    epoch_accuracy = (
        100.0 * correct / total
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(EPOCHS):

    print("\n")

    print(
        f"Epoch "
        f"[{epoch + 1}/{EPOCHS}]"
    )

    print("-" * 70)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_loss, train_accuracy = (
        train_one_epoch()
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_loss, val_accuracy = (
        validate()
    )


    # --------------------------------------------------------
    # LR SCHEDULER
    # --------------------------------------------------------

    scheduler.step(
        val_loss
    )


    current_lr = optimizer.param_groups[0][
        "lr"
    ]


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history["train_loss"].append(
        train_loss
    )

    history["valid_loss"].append(
        val_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["valid_accuracy"].append(
        val_accuracy
    )


    # --------------------------------------------------------
    # PRINT METRICS
    # --------------------------------------------------------

    print(
        f"Train Loss      : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy  : "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Val Loss        : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy    : "
        f"{val_accuracy:.2f}%"
    )

    print(
        f"Learning Rate   : "
        f"{current_lr:.8f}"
    )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    improvement = (
        best_val_loss - val_loss
    )


    if improvement > MIN_DELTA:

        # ----------------------------------------------------
        # Validation improved
        # ----------------------------------------------------

        best_val_loss = val_loss

        best_val_accuracy = val_accuracy

        best_epoch = epoch + 1

        patience_counter = 0

        best_model_state = copy.deepcopy(
            model.state_dict()
        )


        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        best_model_path = os.path.join(

            MODEL_DIR,

            "condition_resnet50_best.pth"
        )


        torch.save(

            {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    train_dataset.classes,

                "num_classes":
                    NUM_CLASSES,

                "image_size":
                    IMAGE_SIZE,

                "epoch":
                    epoch + 1,

                "val_loss":
                    val_loss,

                "val_accuracy":
                    val_accuracy

            },

            best_model_path
        )


        print(
            "\n✓ Validation loss improved."
        )

        print(
            "✓ Best model saved."
        )


    else:

        # ----------------------------------------------------
        # No improvement
        # ----------------------------------------------------

        patience_counter += 1


        print(
            f"\n⚠ No validation improvement "
            f"({patience_counter}/{PATIENCE})"
        )


    # ========================================================
    # EARLY STOPPING CONDITION
    # ========================================================

    if patience_counter >= PATIENCE:

        print("\n")

        print(
            "=" * 70
        )

        print(
            "EARLY STOPPING TRIGGERED"
        )

        print(
            "=" * 70
        )

        print(
            f"Best Epoch       : "
            f"{best_epoch}"
        )

        print(
            f"Best Val Loss    : "
            f"{best_val_loss:.4f}"
        )

        print(
            f"Best Val Accuracy: "
            f"{best_val_accuracy:.2f}%"
        )

        break


# ============================================================
# RESTORE BEST MODEL
# ============================================================

print("\n")
print("=" * 70)
print("RESTORING BEST MODEL")
print("=" * 70)


model.load_state_dict(
    best_model_state
)


# ============================================================
# SAVE FINAL BEST MODEL
# ============================================================

final_model_path = os.path.join(

    MODEL_DIR,

    "condition_resnet50.pth"
)


torch.save(

    model.state_dict(),

    final_model_path
)


print(
    f"\nBest model copied to:"
)

print(
    final_model_path
)


# ============================================================
# TEST FUNCTION
# ============================================================

def evaluate_test():

    model.eval()

    all_labels = []

    all_predictions = []

    all_probabilities = []


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                device,
                non_blocking=True
            )


            outputs = model(
                images
            )


            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            _, predictions = torch.max(
                outputs,
                1
            )


            all_labels.extend(
                labels.numpy()
            )


            all_predictions.extend(
                predictions.cpu().numpy()
            )


            all_probabilities.extend(
                probabilities.cpu().numpy()
            )


    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )


# ============================================================
# INTERNAL TEST
# ============================================================

print("\n")
print("=" * 70)
print("INTERNAL TEST EVALUATION")
print("=" * 70)


y_true, y_pred, y_prob = evaluate_test()


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n")
print(
    f"Test Accuracy : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision     : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall        : "
    f"{recall * 100:.2f}%"
)

print(
    f"F1 Score      : "
    f"{f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_true,

    y_pred,

    target_names=train_dataset.classes,

    digits=4,

    zero_division=0
)


print("\n")
print("=" * 70)

print(
    "CLASSIFICATION REPORT"
)

print("=" * 70)

print(
    report
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(

    RESULT_DIR,

    "classification_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "CAPSICUM RESNET50 CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"Best Epoch: {best_epoch}\n"
    )

    file.write(
        f"Best Validation Loss: "
        f"{best_val_loss:.6f}\n"
    )

    file.write(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy:.2f}%\n\n"
    )

    file.write(
        f"Test Accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Weighted Precision: "
        f"{precision * 100:.2f}%\n"
    )

    file.write(
        f"Weighted Recall: "
        f"{recall * 100:.2f}%\n"
    )

    file.write(
        f"Weighted F1 Score: "
        f"{f1 * 100:.2f}%\n\n"
    )

    file.write(
        report
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_true,

    y_pred,

    labels=range(NUM_CLASSES)
)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(12, 10)
)


plt.imshow(
    cm,
    interpolation="nearest"
)


plt.title(
    "Capsicum ResNet50 - Confusion Matrix"
)


plt.colorbar()


tick_marks = np.arange(
    NUM_CLASSES
)


plt.xticks(
    tick_marks,
    train_dataset.classes,
    rotation=90
)


plt.yticks(
    tick_marks,
    train_dataset.classes
)


plt.xlabel(
    "Predicted Label"
)


plt.ylabel(
    "True Label"
)


# ------------------------------------------------------------
# Add values inside matrix
# ------------------------------------------------------------

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(

            j,

            i,

            str(cm[i, j]),

            horizontalalignment="center",

            verticalalignment="center"
        )


plt.tight_layout()


cm_path = os.path.join(

    RESULT_DIR,

    "confusion_matrix.png"
)


plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"\nConfusion matrix saved:"
)

print(
    cm_path
)


# ============================================================
# TRAINING LOSS GRAPH
# ============================================================

epochs_completed = range(
    1,
    len(history["train_loss"]) + 1
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(

    epochs_completed,

    history["train_loss"],

    marker="o",

    label="Train Loss"
)


plt.plot(

    epochs_completed,

    history["valid_loss"],

    marker="o",

    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "Training and Validation Loss"
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


loss_path = os.path.join(

    RESULT_DIR,

    "loss_curve.png"
)


plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    epochs_completed,

    history["train_accuracy"],

    marker="o",

    label="Train Accuracy"
)


plt.plot(

    epochs_completed,

    history["valid_accuracy"],

    marker="o",

    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy (%)"
)


plt.title(
    "Training and Validation Accuracy"
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


accuracy_path = os.path.join(

    RESULT_DIR,

    "accuracy_curve.png"
)


plt.savefig(
    accuracy_path,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# SAVE TRAINING SUMMARY
# ============================================================

summary_path = os.path.join(

    RESULT_DIR,

    "training_summary.txt"
)


with open(

    summary_path,

    "w",

    encoding="utf-8"

) as file:

    file.write(
        "CAPSICUM RESNET50 TRAINING SUMMARY\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )


    file.write(
        f"Device: {device}\n"
    )

    file.write(
        f"Image Size: {IMAGE_SIZE}\n"
    )

    file.write(
        f"Batch Size: {BATCH_SIZE}\n"
    )

    file.write(
        f"Maximum Epochs: {EPOCHS}\n"
    )

    file.write(
        f"Learning Rate: {LEARNING_RATE}\n"
    )

    file.write(
        f"Weight Decay: {WEIGHT_DECAY}\n"
    )

    file.write(
        f"Dropout: {DROPOUT}\n"
    )

    file.write(
        f"Patience: {PATIENCE}\n"
    )

    file.write(
        f"Min Delta: {MIN_DELTA}\n\n"
    )


    file.write(
        f"Best Epoch: {best_epoch}\n"
    )

    file.write(
        f"Best Validation Loss: "
        f"{best_val_loss:.6f}\n"
    )

    file.write(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy:.2f}%\n\n"
    )


    file.write(
        f"Test Accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Weighted Precision: "
        f"{precision * 100:.2f}%\n"
    )

    file.write(
        f"Weighted Recall: "
        f"{recall * 100:.2f}%\n"
    )

    file.write(
        f"Weighted F1 Score: "
        f"{f1 * 100:.2f}%\n\n"
    )


    file.write(
        "Classes:\n"
    )


    for index, class_name in enumerate(
        train_dataset.classes
    ):

        file.write(
            f"{index}: {class_name}\n"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)

print(
    "TRAINING COMPLETED"
)

print("=" * 70)


print(
    f"\nBest Epoch:"
    f" {best_epoch}"
)


print(
    f"Best Validation Accuracy:"
    f" {best_val_accuracy:.2f}%"
)


print(
    f"\nFinal Test Accuracy:"
    f" {accuracy * 100:.2f}%"
)


print(
    f"Weighted Precision:"
    f" {precision * 100:.2f}%"
)


print(
    f"Weighted Recall:"
    f" {recall * 100:.2f}%"
)


print(
    f"Weighted F1:"
    f" {f1 * 100:.2f}%"
)


print("\n")
print(
    "Model:"
)

print(
    final_model_path
)


print("\n")
print(
    "Results:"
)

print(
    RESULT_DIR
)


print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)