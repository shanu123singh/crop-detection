import os
import copy
import random
import json
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "training_results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# 2. CLASS CONFIGURATION
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

TRAIN_TARGET = 1100


# ============================================================
# 3. TRAINING CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 64

NUM_WORKERS = 0

EPOCHS = 10

LEARNING_RATE = 5e-5

WEIGHT_DECAY = 1e-4

DROPOUT = 0.30

PATIENCE = 3

MIN_DELTA = 0.001

SEED = 42


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# 5. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 6. PRINT CONFIGURATION
# ============================================================

print("\n")
print("=" * 80)
print("CAPSICUM RESNET50 - FINAL TRAINING")
print("=" * 80)

print(f"Device       : {DEVICE}")
print(f"Image size   : {IMAGE_SIZE}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Epochs       : {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay : {WEIGHT_DECAY}")
print(f"Train target : {TRAIN_TARGET}")
print(f"Num classes  : {NUM_CLASSES}")

print("\nPaths:")
print(f"Train        : {TRAIN_DIR}")
print(f"Validation   : {VALID_DIR}")
print(f"Test         : {TEST_DIR}")
print(f"Models       : {MODEL_DIR}")
print(f"Results      : {RESULT_DIR}")

print("=" * 80)


# ============================================================
# 7. CHECK DIRECTORIES
# ============================================================

for directory in [
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR
]:

    if not os.path.isdir(directory):

        raise FileNotFoundError(
            f"\nRequired directory not found:\n{directory}"
        )


# ============================================================
# 8. TRANSFORMS
# ============================================================
#
# IMPORTANT:
# augmentation2.py already performs OFFLINE augmentation.
#
# Therefore we DO NOT perform random augmentation again here.
#
# This avoids:
#     augmented image
#          +
#     random augmentation
#          =
#     unnecessary double augmentation
#
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


valid_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 9. LOAD DATASETS
# ============================================================

print("\nLoading datasets...")


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
# 10. CLASS ORDER VALIDATION
# ============================================================

print("\nDetected class order:")

print("Train:")
print(train_dataset.classes)

print("\nValidation:")
print(valid_dataset.classes)

print("\nTest:")
print(test_dataset.classes)


expected_classes = CLASSES


if train_dataset.classes != expected_classes:

    raise RuntimeError(
        "\nTRAIN CLASS ORDER MISMATCH!\n"
        f"Expected:\n{expected_classes}\n"
        f"Found:\n{train_dataset.classes}"
    )


if valid_dataset.classes != expected_classes:

    raise RuntimeError(
        "\nVALIDATION CLASS ORDER MISMATCH!\n"
        f"Expected:\n{expected_classes}\n"
        f"Found:\n{valid_dataset.classes}"
    )


if test_dataset.classes != expected_classes:

    raise RuntimeError(
        "\nTEST CLASS ORDER MISMATCH!\n"
        f"Expected:\n{expected_classes}\n"
        f"Found:\n{test_dataset.classes}"
    )


print("\nClass order verified successfully.")


# ============================================================
# 11. DATASET COUNTS
# ============================================================

def get_class_counts(dataset):

    counts = {
        class_name: 0
        for class_name in CLASSES
    }

    for _, label in dataset.samples:

        class_name = dataset.classes[label]

        counts[class_name] += 1

    return counts


train_counts = get_class_counts(train_dataset)

valid_counts = get_class_counts(valid_dataset)

test_counts = get_class_counts(test_dataset)


print("\n")
print("=" * 80)
print("DATASET COUNTS")
print("=" * 80)

print("\nTRAIN:")

for class_name in CLASSES:

    print(
        f"{class_name:<25} : "
        f"{train_counts[class_name]}"
    )


print("\nVALIDATION:")

for class_name in CLASSES:

    print(
        f"{class_name:<25} : "
        f"{valid_counts[class_name]}"
    )


print("\nTEST:")

for class_name in CLASSES:

    print(
        f"{class_name:<25} : "
        f"{test_counts[class_name]}"
    )


# ============================================================
# 12. VERIFY TRAIN = 1100/CLASS
# ============================================================

print("\nChecking training balance...")

for class_name in CLASSES:

    count = train_counts[class_name]

    if count != TRAIN_TARGET:

        raise RuntimeError(
            f"\nTRAIN DATA ERROR:\n"
            f"Class '{class_name}' has {count} images.\n"
            f"Expected exactly {TRAIN_TARGET}."
        )


print(
    f"Training dataset verified: "
    f"{TRAIN_TARGET} images/class."
)


# ============================================================
# 13. DATASET TOTALS
# ============================================================

total_train = len(train_dataset)

total_valid = len(valid_dataset)

total_test = len(test_dataset)


print("\nDataset totals:")

print(f"Train      : {total_train}")

print(f"Validation : {total_valid}")

print(f"Test       : {total_test}")


expected_train_total = TRAIN_TARGET * NUM_CLASSES


if total_train != expected_train_total:

    raise RuntimeError(
        f"\nExpected {expected_train_total} training images "
        f"but found {total_train}."
    )


# ============================================================
# 14. DATA LOADERS
# ============================================================

print("\nCreating DataLoaders...")


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
# 15. LOAD RESNET50
# ============================================================

print("\nLoading ResNet50...")


try:

    weights = models.ResNet50_Weights.DEFAULT

    model = models.resnet50(
        weights=weights
    )

    print("Pretrained ImageNet weights loaded.")


except Exception as e:

    print(
        "\nWARNING: Could not load pretrained weights."
    )

    print(f"Reason: {e}")

    print(
        "Continuing with weights=None."
    )

    model = models.resnet50(
        weights=None
    )


# ============================================================
# 16. FREEZE ALL LAYERS
# ============================================================

for parameter in model.parameters():

    parameter.requires_grad = False


# ============================================================
# 17. UNFREEZE LAYER4
# ============================================================

for parameter in model.layer4.parameters():

    parameter.requires_grad = True


# ============================================================
# 18. REPLACE FINAL CLASSIFIER
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


# Make sure FC is trainable.

for parameter in model.fc.parameters():

    parameter.requires_grad = True


# ============================================================
# 19. MOVE MODEL TO DEVICE
# ============================================================

model = model.to(DEVICE)


# ============================================================
# 20. TRAINABLE PARAMETERS
# ============================================================

trainable_parameters = sum(

    parameter.numel()

    for parameter in model.parameters()

    if parameter.requires_grad
)


total_parameters = sum(

    parameter.numel()

    for parameter in model.parameters()
)


print("\nModel information:")

print(
    f"Total parameters     : "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters : "
    f"{trainable_parameters:,}"
)


# ============================================================
# 21. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 22. OPTIMIZER
# ============================================================

optimizer = optim.AdamW(

    filter(
        lambda parameter:
        parameter.requires_grad,

        model.parameters()
    ),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 23. LR SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=1
)


# ============================================================
# 24. AMP SETUP
# ============================================================

use_amp = DEVICE.type == "cuda"


if use_amp:

    scaler = torch.cuda.amp.GradScaler()

else:

    scaler = None


# ============================================================
# 25. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        if use_amp:

            with torch.cuda.amp.autocast():

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )


            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()


        else:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()


        batch_size = images.size(0)


        running_loss += (
            loss.item() * batch_size
        )


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += batch_size


    epoch_loss = (
        running_loss / total
    )


    epoch_accuracy = (
        correct / total
    )


    return epoch_loss, epoch_accuracy


# ============================================================
# 26. VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in valid_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        if use_amp:

            with torch.cuda.amp.autocast():

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

        else:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


        batch_size = images.size(0)


        running_loss += (
            loss.item() * batch_size
        )


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += batch_size


    epoch_loss = (
        running_loss / total
    )


    epoch_accuracy = (
        correct / total
    )


    return epoch_loss, epoch_accuracy


# ============================================================
# 27. TRAINING VARIABLES
# ============================================================

history = {

    "train_loss": [],

    "train_accuracy": [],

    "valid_loss": [],

    "valid_accuracy": [],

    "learning_rate": []
}


best_val_loss = float("inf")

best_val_accuracy = 0.0

best_epoch = 0

patience_counter = 0

best_model_state = None


# ============================================================
# 28. TRAINING LOOP
# ============================================================

print("\n")
print("=" * 80)
print("STARTING TRAINING")
print("=" * 80)


training_start = time.time()


for epoch in range(
    1,
    EPOCHS + 1
):

    epoch_start = time.time()


    print(
        f"\nEpoch "
        f"{epoch}/{EPOCHS}"
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_loss, train_accuracy = (
        train_one_epoch()
    )


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    valid_loss, valid_accuracy = (
        validate()
    )


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step(
        valid_loss
    )


    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["valid_loss"].append(
        valid_loss
    )

    history["valid_accuracy"].append(
        valid_accuracy
    )

    history["learning_rate"].append(
        current_lr
    )


    # --------------------------------------------------------
    # EPOCH TIME
    # --------------------------------------------------------

    epoch_time = (
        time.time()
        -
        epoch_start
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Acc : "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Valid Loss: "
        f"{valid_loss:.4f}"
    )

    print(
        f"Valid Acc : "
        f"{valid_accuracy * 100:.2f}%"
    )

    print(
        f"LR        : "
        f"{current_lr:.8f}"
    )

    print(
        f"Time      : "
        f"{epoch_time:.1f}s"
    )


    # --------------------------------------------------------
    # CHECK IMPROVEMENT
    # --------------------------------------------------------

    improvement = (
        best_val_loss
        -
        valid_loss
    )


    if improvement > MIN_DELTA:

        best_val_loss = valid_loss

        best_val_accuracy = valid_accuracy

        best_epoch = epoch

        patience_counter = 0


        best_model_state = copy.deepcopy(
            model.state_dict()
        )


        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        best_model_path = os.path.join(

            MODEL_DIR,

            "condition_resnet50_best.pth"
        )


        torch.save(

            {

                "model_state_dict":
                    best_model_state,

                "classes":
                    CLASSES,

                "image_size":
                    IMAGE_SIZE,

                "best_epoch":
                    best_epoch,

                "best_val_loss":
                    best_val_loss,

                "best_val_accuracy":
                    best_val_accuracy,

                "train_target":
                    TRAIN_TARGET

            },

            best_model_path
        )


        print(
            "\n✓ Validation improved."
        )

        print(
            "✓ Best model saved."
        )


    else:

        patience_counter += 1


        print(
            f"\nNo significant improvement."
        )

        print(
            f"Early stopping counter: "
            f"{patience_counter}/{PATIENCE}"
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# 29. TRAINING FINISHED
# ============================================================

training_time = (
    time.time()
    -
    training_start
)


print("\n")
print("=" * 80)
print("TRAINING COMPLETED")
print("=" * 80)


print(
    f"Best epoch      : "
    f"{best_epoch}"
)

print(
    f"Best val loss   : "
    f"{best_val_loss:.4f}"
)

print(
    f"Best val accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Training time   : "
    f"{training_time / 60:.2f} minutes"
)


# ============================================================
# 30. SAFETY CHECK
# ============================================================

if best_model_state is None:

    raise RuntimeError(
        "\nNo best model was saved.\n"
        "Validation loss never improved."
    )


# ============================================================
# 31. RESTORE BEST MODEL
# ============================================================

print("\nRestoring best model...")

model.load_state_dict(
    best_model_state
)

model.eval()

print(
    f"Best epoch {best_epoch} restored."
)


# ============================================================
# 32. SAVE FINAL MODEL
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
    f"\n✓ Final model saved:"
)

print(
    final_model_path
)


# ============================================================
# 33. SAVE MODEL CHECKPOINT
# ============================================================

checkpoint_path = os.path.join(

    MODEL_DIR,

    "condition_resnet50_checkpoint.pth"
)


torch.save(

    {

        "model_state_dict":
            model.state_dict(),

        "classes":
            CLASSES,

        "image_size":
            IMAGE_SIZE,

        "num_classes":
            NUM_CLASSES,

        "best_epoch":
            best_epoch,

        "best_val_loss":
            best_val_loss,

        "best_val_accuracy":
            best_val_accuracy,

        "train_target":
            TRAIN_TARGET,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY

    },

    checkpoint_path
)


print(
    f"✓ Checkpoint saved:"
)

print(
    checkpoint_path
)


# ============================================================
# 34. FINAL TEST EVALUATION
# ============================================================

print("\n")
print("=" * 80)
print("FINAL TEST EVALUATION")
print("=" * 80)


model.eval()


all_true = []

all_predictions = []

all_probabilities = []


test_running_loss = 0.0

test_total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        if use_amp:

            with torch.cuda.amp.autocast():

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

        else:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        predictions = outputs.argmax(
            dim=1
        )


        batch_size = images.size(0)


        test_running_loss += (
            loss.item() * batch_size
        )


        test_total += batch_size


        all_true.extend(
            labels.cpu().numpy()
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )


        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


# ============================================================
# 35. TEST METRICS
# ============================================================

test_loss = (
    test_running_loss
    /
    test_total
)


test_accuracy = accuracy_score(

    all_true,

    all_predictions
)


test_precision = precision_score(

    all_true,

    all_predictions,

    average="weighted",

    zero_division=0
)


test_recall = recall_score(

    all_true,

    all_predictions,

    average="weighted",

    zero_division=0
)


test_f1 = f1_score(

    all_true,

    all_predictions,

    average="weighted",

    zero_division=0
)


# ============================================================
# 36. PRINT TEST RESULTS
# ============================================================

print(
    f"\nTest Loss      : "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy  : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Precision : "
    f"{test_precision * 100:.2f}%"
)

print(
    f"Test Recall    : "
    f"{test_recall * 100:.2f}%"
)

print(
    f"Test F1 Score  : "
    f"{test_f1 * 100:.2f}%"
)


# ============================================================
# 37. CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    all_true,

    all_predictions,

    target_names=CLASSES,

    digits=4,

    zero_division=0
)


print("\n")
print("=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)

print(report)


# ============================================================
# 38. SAVE CLASSIFICATION REPORT
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
        "CAPSICUM RESNET50 - CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 80
        +
        "\n\n"
    )

    file.write(
        report
    )


# ============================================================
# 39. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    all_true,

    all_predictions,

    labels=list(
        range(NUM_CLASSES)
    )
)


plt.figure(
    figsize=(14, 12)
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
    CLASSES,
    rotation=90
)


plt.yticks(
    tick_marks,
    CLASSES
)


plt.xlabel(
    "Predicted Label"
)


plt.ylabel(
    "True Label"
)


# Add values inside cells.

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()


cm_path = os.path.join(

    RESULT_DIR,

    "confusion_matrix.png"
)


plt.savefig(
    cm_path,
    dpi=200,
    bbox_inches="tight"
)


plt.close()


print(
    f"\n✓ Confusion matrix saved:"
)

print(
    cm_path
)


# ============================================================
# 40. TRAINING LOSS GRAPH
# ============================================================

epochs_completed = range(

    1,

    len(
        history["train_loss"]
    ) + 1
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(

    epochs_completed,

    history["train_loss"],

    label="Train Loss"
)


plt.plot(

    epochs_completed,

    history["valid_loss"],

    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "Training vs Validation Loss"
)


plt.legend()


plt.grid(
    True
)


loss_graph_path = os.path.join(

    RESULT_DIR,

    "loss_curve.png"
)


plt.savefig(

    loss_graph_path,

    dpi=200,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 41. ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    epochs_completed,

    np.array(
        history["train_accuracy"]
    ) * 100,

    label="Train Accuracy"
)


plt.plot(

    epochs_completed,

    np.array(
        history["valid_accuracy"]
    ) * 100,

    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy (%)"
)


plt.title(
    "Training vs Validation Accuracy"
)


plt.legend()


plt.grid(
    True
)


accuracy_graph_path = os.path.join(

    RESULT_DIR,

    "accuracy_curve.png"
)


plt.savefig(

    accuracy_graph_path,

    dpi=200,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 42. SAVE HISTORY JSON
# ============================================================

history_path = os.path.join(

    RESULT_DIR,

    "training_history.json"
)


with open(

    history_path,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        history,

        file,

        indent=4
    )


# ============================================================
# 43. SAVE TRAINING SUMMARY
# ============================================================

summary = {

    "model":
        "ResNet50",

    "classes":
        CLASSES,

    "num_classes":
        NUM_CLASSES,

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "epochs_requested":
        EPOCHS,

    "epochs_completed":
        len(
            history["train_loss"]
        ),

    "best_epoch":
        best_epoch,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "dropout":
        DROPOUT,

    "train_target_per_class":
        TRAIN_TARGET,

    "train_total":
        total_train,

    "validation_total":
        total_valid,

    "test_total":
        total_test,

    "best_validation_loss":
        best_val_loss,

    "best_validation_accuracy":
        best_val_accuracy,

    "test_loss":
        test_loss,

    "test_accuracy":
        test_accuracy,

    "test_precision_weighted":
        test_precision,

    "test_recall_weighted":
        test_recall,

    "test_f1_weighted":
        test_f1,

    "device":
        str(DEVICE),

    "trainable_parameters":
        trainable_parameters,

    "total_parameters":
        total_parameters,

    "offline_augmentation":
        True,

    "validation_augmentation":
        False,

    "test_augmentation":
        False

}


summary_path = os.path.join(

    RESULT_DIR,

    "training_summary.json"
)


with open(

    summary_path,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        summary,

        file,

        indent=4
    )


# ============================================================
# 44. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 80)
print("TRAINING PIPELINE FINISHED")
print("=" * 80)

print("\nModel:")
print(
    f"  {final_model_path}"
)

print("\nBest model:")
print(
    os.path.join(
        MODEL_DIR,
        "condition_resnet50_best.pth"
    )
)

print("\nClassification report:")
print(
    report_path
)

print("\nConfusion matrix:")
print(
    cm_path
)

print("\nLoss curve:")
print(
    loss_graph_path
)

print("\nAccuracy curve:")
print(
    accuracy_graph_path
)

print("\nTraining history:")
print(
    history_path
)

print("\nTraining summary:")
print(
    summary_path
)

print("\n")
print("=" * 80)
print("FINAL TEST RESULT")
print("=" * 80)

print(
    f"Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Precision: "
    f"{test_precision * 100:.2f}%"
)

print(
    f"Recall   : "
    f"{test_recall * 100:.2f}%"
)

print(
    f"F1 Score : "
    f"{test_f1 * 100:.2f}%"
)

print("=" * 80)

print("\nDONE.")