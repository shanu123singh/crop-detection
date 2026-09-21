# ============================================================
# test_augu.py
# CAPSICUM RESNET50 - FINAL INTERNAL TESTING
# ============================================================
#
# New Approach:
#
# data new/
#     ↓
# exact duplicate detection
#     ↓
# pHash near duplicate detection
#     ↓
# group-aware split
#     ↓
# train-only augmentation
#     ↓
# ResNet50 training
#     ↓
# BEST MODEL
#     ↓
# CLEAN INTERNAL TEST
#
# IMPORTANT:
# Test set is NEVER augmented.
# Test set is used ONLY for final evaluation.
#
# ============================================================

import os
import json
import csv
import hashlib
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
import pandas as pd

from PIL import Image, UnidentifiedImageError

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# WARNING SETTINGS
# ============================================================

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ------------------------------------------------------------
# TEST DIRECTORY
# ------------------------------------------------------------
#
# This directory must contain ONLY clean/original test images.
# No offline augmentation should be performed here.
#
# If your clean test is in dataset_split/test, change to:
#
# TEST_DIR = os.path.join(BASE_DIR, "dataset_split", "test")
#
# ------------------------------------------------------------

TEST_DIR = os.path.join(
    BASE_DIR,
    "augmented_dataset",
    "test"
)


# ------------------------------------------------------------
# TRAIN DIRECTORY
# ------------------------------------------------------------
#
# Used ONLY for exact duplicate leakage checking.
#
# ------------------------------------------------------------

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "augmented_dataset",
    "train"
)


# ------------------------------------------------------------
# VALIDATION DIRECTORY
# ------------------------------------------------------------

VALID_DIR = os.path.join(
    BASE_DIR,
    "augmented_dataset",
    "valid"
)


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "condition_resnet50.pth"
)


# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "test_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FINAL 12 CLASSES
# ============================================================

EXPECTED_CLASSES = [
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


NUM_CLASSES = len(
    EXPECTED_CLASSES
)


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 64

NUM_WORKERS = 0

RANDOM_SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("\n" + "=" * 80)
print("CAPSICUM RESNET50 - FINAL INTERNAL TEST")
print("=" * 80)

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print(
    f"Test directory: {TEST_DIR}"
)

print(
    f"Model: {MODEL_PATH}"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
)


# ============================================================
# CHECK DIRECTORY
# ============================================================

if not os.path.exists(TEST_DIR):

    raise FileNotFoundError(
        "\n"
        + "=" * 80
        + "\nTEST DIRECTORY NOT FOUND\n"
        + "=" * 80
        + f"\n\nExpected:\n{TEST_DIR}\n\n"
        + "If your clean test set is inside dataset_split/test, "
        + "change TEST_DIR accordingly."
    )


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "\n"
        + "=" * 80
        + "\nMODEL NOT FOUND\n"
        + "=" * 80
        + f"\n\nExpected:\n{MODEL_PATH}"
    )


# ============================================================
# TEST TRANSFORM
# ============================================================
#
# IMPORTANT:
#
# NO RANDOM AUGMENTATION
#
# Resize
# CenterCrop
# ToTensor
# ImageNet Normalize
#
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize(
        256
    ),

    transforms.CenterCrop(
        IMAGE_SIZE
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
# LOAD TEST DATASET
# ============================================================

print("\n" + "=" * 80)
print("LOADING TEST DATASET")
print("=" * 80)

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=test_transform
)


# ============================================================
# IMAGEFOLDER CLASSES
# ============================================================

dataset_classes = test_dataset.classes

dataset_class_to_idx = (
    test_dataset.class_to_idx
)


print("\nImageFolder classes:")

for idx, class_name in enumerate(
    dataset_classes
):

    print(
        f"{idx:2d} -> {class_name}"
    )


# ============================================================
# VALIDATE CLASS COUNT
# ============================================================

if len(dataset_classes) != NUM_CLASSES:

    raise RuntimeError(
        "\n"
        + "=" * 80
        + "\nCLASS COUNT MISMATCH\n"
        + "=" * 80
        + f"\nExpected: {NUM_CLASSES}"
        + f"\nFound: {len(dataset_classes)}"
        + f"\n\nFound classes:\n{dataset_classes}"
    )


# ============================================================
# VALIDATE CLASS NAMES
# ============================================================

expected_set = set(
    EXPECTED_CLASSES
)

dataset_set = set(
    dataset_classes
)


if expected_set != dataset_set:

    missing = sorted(
        expected_set - dataset_set
    )

    extra = sorted(
        dataset_set - expected_set
    )

    raise RuntimeError(
        "\n"
        + "=" * 80
        + "\nCLASS NAME MISMATCH\n"
        + "=" * 80
        + f"\n\nMissing classes:\n{missing}"
        + f"\n\nUnexpected classes:\n{extra}"
    )


# ============================================================
# DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# TEST DATASET COUNTS
# ============================================================

print("\n" + "=" * 80)
print("TEST DATASET COUNTS")
print("=" * 80)

test_counts = {}

for class_name in dataset_classes:

    class_dir = os.path.join(
        TEST_DIR,
        class_name
    )

    count = 0

    if os.path.exists(class_dir):

        for root, dirs, files in os.walk(
            class_dir
        ):

            for file in files:

                if file.lower().endswith(
                    IMAGE_EXTENSIONS
                ):

                    count += 1

    test_counts[class_name] = count

    print(
        f"{class_name:25s}: {count}"
    )


total_test_images = sum(
    test_counts.values()
)

print(
    f"\nTotal test images: {total_test_images}"
)


# ============================================================
# SHA-256 FUNCTION
# ============================================================

def calculate_sha256(
    file_path,
    chunk_size=1024 * 1024
):

    sha256 = hashlib.sha256()

    try:

        with open(
            file_path,
            "rb"
        ) as file:

            while True:

                chunk = file.read(
                    chunk_size
                )

                if not chunk:
                    break

                sha256.update(
                    chunk
                )

        return sha256.hexdigest()

    except Exception:

        return None


# ============================================================
# COLLECT IMAGE FILES
# ============================================================

def collect_image_files(
    directory
):

    files = []

    if not os.path.exists(directory):

        return files

    for root, dirs, filenames in os.walk(
        directory
    ):

        for filename in filenames:

            if filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    return files


# ============================================================
# EXACT DUPLICATE LEAKAGE CHECK
# ============================================================
#
# Checks:
#
# TRAIN ↔ TEST
#
# Exact duplicate image content must NOT exist
# across train and test.
#
# ============================================================

print("\n" + "=" * 80)
print("EXACT DUPLICATE LEAKAGE CHECK")
print("=" * 80)

train_files = collect_image_files(
    TRAIN_DIR
)

test_files = collect_image_files(
    TEST_DIR
)

print(
    f"Train images checked: {len(train_files)}"
)

print(
    f"Test images checked : {len(test_files)}"
)


train_hashes = {}

for file_path in train_files:

    file_hash = calculate_sha256(
        file_path
    )

    if file_hash is not None:

        train_hashes.setdefault(
            file_hash,
            []
        ).append(
            file_path
        )


test_hash_duplicates = []

for file_path in test_files:

    file_hash = calculate_sha256(
        file_path
    )

    if (
        file_hash is not None
        and file_hash in train_hashes
    ):

        test_hash_duplicates.append({
            "test_file": file_path,
            "train_files": train_hashes[
                file_hash
            ]
        })


if len(test_hash_duplicates) == 0:

    print(
        "\nPASS: No exact SHA-256 duplicate "
        "leakage between TRAIN and TEST."
    )

else:

    print(
        "\nWARNING: Exact duplicate leakage detected!"
    )

    print(
        f"Duplicate test files: "
        f"{len(test_hash_duplicates)}"
    )


# ============================================================
# SAVE EXACT DUPLICATE REPORT
# ============================================================

duplicate_rows = []

for item in test_hash_duplicates:

    for train_file in item[
        "train_files"
    ]:

        duplicate_rows.append({

            "test_file": item[
                "test_file"
            ],

            "train_file": train_file

        })


duplicate_report_path = os.path.join(
    OUTPUT_DIR,
    "exact_duplicate_leakage.csv"
)


pd.DataFrame(
    duplicate_rows
).to_csv(
    duplicate_report_path,
    index=False
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\n" + "=" * 80)
print("LOADING MODEL CHECKPOINT")
print("=" * 80)


try:

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

except TypeError:

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


# ============================================================
# EXTRACT STATE DICT
# ============================================================

if isinstance(
    checkpoint,
    dict
):

    if (
        "model_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        "state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        # Check whether the checkpoint itself
        # is a state_dict.

        if any(
            key.endswith(
                ".weight"
            )
            for key in checkpoint.keys()
        ):

            state_dict = checkpoint

        else:

            raise RuntimeError(
                "\n"
                + "=" * 80
                + "\nMODEL STATE DICT NOT FOUND\n"
                + "=" * 80
            )

else:

    raise RuntimeError(
        "\n"
        + "=" * 80
        + "\nINVALID CHECKPOINT FORMAT\n"
        + "=" * 80
    )


# ============================================================
# REMOVE DDP "module." PREFIX
# ============================================================

clean_state_dict = {}

for key, value in state_dict.items():

    new_key = key

    if new_key.startswith(
        "module."
    ):

        new_key = new_key[
            len("module.") :
        ]

    clean_state_dict[
        new_key
    ] = value


state_dict = clean_state_dict


# ============================================================
# PRINT FC LAYERS
# ============================================================

print("\nDetected FC layers:")

for key, value in state_dict.items():

    if (
        key.startswith("fc.")
        and key.endswith(".weight")
    ):

        print(
            f"{key} -> "
            f"{tuple(value.shape)}"
        )


# ============================================================
# CHECKPOINT CLASS METADATA
# ============================================================

checkpoint_classes = None

checkpoint_class_to_idx = None


if isinstance(
    checkpoint,
    dict
):

    if (
        "classes"
        in checkpoint
    ):

        checkpoint_classes = (
            checkpoint["classes"]
        )

    if (
        "class_to_idx"
        in checkpoint
    ):

        checkpoint_class_to_idx = (
            checkpoint["class_to_idx"]
        )


# ============================================================
# DETERMINE NUMBER OF CLASSES
# ============================================================

detected_num_classes = None


# ------------------------------------------------------------
# IMPORTANT FIX FOR YOUR ERROR
#
# Your checkpoint:
#
# fc.1.weight -> (12, 2048)
#
# Therefore:
#
# Dropout -> Linear(2048, 12)
#
# ------------------------------------------------------------

if (
    "fc.1.weight"
    in state_dict
    and
    "fc.1.bias"
    in state_dict
):

    detected_num_classes = (
        state_dict[
            "fc.1.weight"
        ].shape[0]
    )

    fc_input_features = (
        state_dict[
            "fc.1.weight"
        ].shape[1]
    )

    print(
        "\nDetected classifier:"
    )

    print(
        "Dropout(0.30) -> "
        "Linear(2048, num_classes)"
    )

    print(
        f"Input features : "
        f"{fc_input_features}"
    )

    print(
        f"Output classes : "
        f"{detected_num_classes}"
    )


# ------------------------------------------------------------
# SIMPLE LINEAR
# ------------------------------------------------------------

elif (
    "fc.weight"
    in state_dict
    and
    "fc.bias"
    in state_dict
):

    detected_num_classes = (
        state_dict[
            "fc.weight"
        ].shape[0]
    )

    fc_input_features = (
        state_dict[
            "fc.weight"
        ].shape[1]
    )

    print(
        "\nDetected classifier:"
    )

    print(
        "Linear(2048, num_classes)"
    )

    print(
        f"Input features : "
        f"{fc_input_features}"
    )

    print(
        f"Output classes : "
        f"{detected_num_classes}"
    )


# ------------------------------------------------------------
# DEEP FC
# ------------------------------------------------------------

elif (
    "fc.1.weight"
    in state_dict
    and
    "fc.4.weight"
    in state_dict
):

    hidden_features = (
        state_dict[
            "fc.1.weight"
        ].shape[0]
    )

    detected_num_classes = (
        state_dict[
            "fc.4.weight"
        ].shape[0]
    )

    fc_input_features = (
        state_dict[
            "fc.1.weight"
        ].shape[1]
    )

    print(
        "\nDetected classifier:"
    )

    print(
        "Dropout -> "
        "Linear(2048,hidden) -> "
        "ReLU -> "
        "Dropout -> "
        "Linear(hidden,num_classes)"
    )

    print(
        f"Input features : "
        f"{fc_input_features}"
    )

    print(
        f"Hidden features: "
        f"{hidden_features}"
    )

    print(
        f"Output classes : "
        f"{detected_num_classes}"
    )


# ------------------------------------------------------------
# ALTERNATIVE DEEP FC
# ------------------------------------------------------------

elif (
    "fc.0.weight"
    in state_dict
    and
    "fc.3.weight"
    in state_dict
):

    hidden_features = (
        state_dict[
            "fc.0.weight"
        ].shape[0]
    )

    detected_num_classes = (
        state_dict[
            "fc.3.weight"
        ].shape[0]
    )

    print(
        "\nDetected classifier:"
    )

    print(
        "Linear -> ReLU -> "
        "Dropout -> Linear"
    )

    print(
        f"Hidden features: "
        f"{hidden_features}"
    )

    print(
        f"Output classes : "
        f"{detected_num_classes}"
    )


else:

    detected_fc_layers = []

    for key, value in state_dict.items():

        if (
            key.startswith("fc.")
            and
            key.endswith(".weight")
        ):

            detected_fc_layers.append(
                f"{key} -> "
                f"{tuple(value.shape)}"
            )

    raise RuntimeError(
        "\n"
        + "=" * 80
        + "\nUNKNOWN CLASSIFIER ARCHITECTURE\n"
        + "=" * 80
        + "\n\nDetected FC layers:\n"
        + "\n".join(
            detected_fc_layers
        )
    )


# ============================================================
# VALIDATE NUMBER OF CLASSES
# ============================================================

if detected_num_classes != NUM_CLASSES:

    raise RuntimeError(
        "\n"
        + "=" * 80
        + "\nMODEL CLASS COUNT MISMATCH\n"
        + "=" * 80
        + f"\nExpected classes: {NUM_CLASSES}"
        + f"\nModel classes   : {detected_num_classes}"
    )


# ============================================================
# CREATE RESNET50
# ============================================================

print("\n" + "=" * 80)
print("CREATING RESNET50")
print("=" * 80)


try:

    model = models.resnet50(
        weights=None
    )

except TypeError:

    model = models.resnet50(
        pretrained=False
    )


# ============================================================
# CREATE MATCHING FC
# ============================================================

if (
    "fc.1.weight"
    in state_dict
    and
    "fc.4.weight"
    not in state_dict
):

    # --------------------------------------------------------
    # YOUR CURRENT CHECKPOINT
    #
    # fc.1.weight -> (12,2048)
    #
    # --------------------------------------------------------

    model.fc = nn.Sequential(

        nn.Dropout(
            p=0.30
        ),

        nn.Linear(
            2048,
            detected_num_classes
        )
    )


elif (
    "fc.weight"
    in state_dict
):

    model.fc = nn.Linear(
        2048,
        detected_num_classes
    )


elif (
    "fc.1.weight"
    in state_dict
    and
    "fc.4.weight"
    in state_dict
):

    hidden_features = (
        state_dict[
            "fc.1.weight"
        ].shape[0]
    )

    model.fc = nn.Sequential(

        nn.Dropout(
            p=0.30
        ),

        nn.Linear(
            2048,
            hidden_features
        ),

        nn.ReLU(
            inplace=True
        ),

        nn.Dropout(
            p=0.30
        ),

        nn.Linear(
            hidden_features,
            detected_num_classes
        )
    )


elif (
    "fc.0.weight"
    in state_dict
    and
    "fc.3.weight"
    in state_dict
):

    hidden_features = (
        state_dict[
            "fc.0.weight"
        ].shape[0]
    )

    model.fc = nn.Sequential(

        nn.Linear(
            2048,
            hidden_features
        ),

        nn.ReLU(
            inplace=True
        ),

        nn.Dropout(
            p=0.30
        ),

        nn.Linear(
            hidden_features,
            detected_num_classes
        )
    )


else:

    raise RuntimeError(
        "Unable to construct classifier."
    )


# ============================================================
# PRINT MODEL FC
# ============================================================

print("\nFinal model.fc:")

print(
    model.fc
)


# ============================================================
# LOAD STATE DICT
# ============================================================

print("\n" + "=" * 80)
print("LOADING MODEL WEIGHTS")
print("=" * 80)


try:

    model.load_state_dict(
        state_dict,
        strict=True
    )

except RuntimeError as error:

    print(
        "\nModel loading failed."
    )

    print(
        "\nCheckpoint FC:"
    )

    for key, value in state_dict.items():

        if key.startswith("fc."):

            print(
                key,
                tuple(value.shape)
                if hasattr(
                    value,
                    "shape"
                )
                else ""
            )

    raise error


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(
    DEVICE
)


model.eval()


print(
    "\nModel weights loaded successfully."
)


# ============================================================
# DETERMINE FINAL CLASS ORDER
# ============================================================
#
# Priority:
#
# 1. checkpoint classes
# 2. checkpoint class_to_idx
# 3. ImageFolder classes
#
# ============================================================

if (
    checkpoint_classes is not None
    and
    len(checkpoint_classes)
    == detected_num_classes
):

    model_classes = list(
        checkpoint_classes
    )

    print(
        "\nUsing class order from checkpoint."
    )


elif (
    checkpoint_class_to_idx
    is not None
    and
    len(checkpoint_class_to_idx)
    == detected_num_classes
):

    model_classes = [
        name
        for name, index
        in sorted(
            checkpoint_class_to_idx.items(),
            key=lambda x: x[1]
        )
    ]

    print(
        "\nUsing class_to_idx from checkpoint."
    )


else:

    model_classes = list(
        dataset_classes
    )

    print(
        "\nWARNING:"
    )

    print(
        "Checkpoint does not contain class metadata."
    )

    print(
        "Using ImageFolder class order."
    )


# ============================================================
# VALIDATE MODEL CLASSES
# ============================================================

if set(model_classes) != set(
    EXPECTED_CLASSES
):

    raise RuntimeError(
        "\n"
        + "=" * 80
        + "\nMODEL CLASS NAMES DO NOT MATCH EXPECTED CLASSES\n"
        + "=" * 80
        + f"\n\nModel classes:\n{model_classes}"
        + f"\n\nExpected classes:\n{EXPECTED_CLASSES}"
    )


if set(model_classes) != set(
    dataset_classes
):

    raise RuntimeError(
        "\n"
        + "=" * 80
        + "\nMODEL / TEST CLASS MISMATCH\n"
        + "=" * 80
        + f"\n\nModel classes:\n{model_classes}"
        + f"\n\nTest classes:\n{dataset_classes}"
    )


# ============================================================
# CREATE TEST INDEX → MODEL INDEX MAPPING
# ============================================================

dataset_index_to_model_index = {}

for dataset_idx, class_name in enumerate(
    dataset_classes
):

    model_idx = model_classes.index(
        class_name
    )

    dataset_index_to_model_index[
        dataset_idx
    ] = model_idx


print("\n" + "=" * 80)
print("CLASS MAPPING")
print("=" * 80)

for dataset_idx, class_name in enumerate(
    dataset_classes
):

    model_idx = dataset_index_to_model_index[
        dataset_idx
    ]

    print(
        f"ImageFolder {dataset_idx:2d} "
        f"-> Model {model_idx:2d} "
        f"-> {class_name}"
    )


# ============================================================
# SAVE CLASS ORDER
# ============================================================

class_order_data = {

    "expected_classes":
        EXPECTED_CLASSES,

    "dataset_classes":
        dataset_classes,

    "model_classes":
        model_classes,

    "dataset_class_to_idx":
        dataset_class_to_idx,

    "checkpoint_class_to_idx":
        checkpoint_class_to_idx,

    "dataset_index_to_model_index":
        dataset_index_to_model_index

}


with open(
    os.path.join(
        OUTPUT_DIR,
        "model_class_order.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_order_data,
        file,
        indent=4
    )


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# EVALUATION STORAGE
# ============================================================

all_true = []

all_pred = []

all_confidence = []

all_probabilities = []

all_paths = []


total_loss = 0.0

total_samples = 0


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("RUNNING INTERNAL TEST")
print("=" * 80)

model.eval()


with torch.no_grad():

    for batch_idx, (
        images,
        labels
    ) in enumerate(
        test_loader
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        outputs = model(
            images
        )


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )


        batch_size = (
            images.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )


        # ----------------------------------------------------
        # SOFTMAX
        # ----------------------------------------------------

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        confidence, predictions = (
            torch.max(
                probabilities,
                dim=1
            )
        )


        # ----------------------------------------------------
        # CONVERT IMAGEFOLDER LABEL
        # TO MODEL LABEL
        # ----------------------------------------------------

        model_labels = torch.tensor(

            [
                dataset_index_to_model_index[
                    int(label)
                ]

                for label in labels.cpu().numpy()
            ],

            dtype=torch.long,

            device=DEVICE
        )


        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        all_true.extend(
            model_labels.cpu().numpy().tolist()
        )

        all_pred.extend(
            predictions.cpu().numpy().tolist()
        )

        all_confidence.extend(
            confidence.cpu().numpy().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )


        # ----------------------------------------------------
        # IMAGE PATHS
        # ----------------------------------------------------

        start_index = (
            batch_idx
            * BATCH_SIZE
        )

        batch_paths = [

            test_dataset.samples[
                start_index + i
            ][0]

            for i in range(
                batch_size
            )

        ]

        all_paths.extend(
            batch_paths
        )


        if (
            batch_idx + 1
        ) % 10 == 0:

            print(
                f"Processed "
                f"{total_samples} / "
                f"{len(test_dataset)} images"
            )


# ============================================================
# FINAL LOSS
# ============================================================

test_loss = (
    total_loss
    / max(
        total_samples,
        1
    )
)


# ============================================================
# NUMPY
# ============================================================

y_true = np.array(
    all_true
)

y_pred = np.array(
    all_pred
)

confidence_array = np.array(
    all_confidence
)

probabilities_array = np.array(
    all_probabilities
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


precision_macro = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


recall_macro = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


f1_macro = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


precision_weighted = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


recall_weighted = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


f1_weighted = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# PRINT FINAL METRICS
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)

print(
    f"\nTest Loss       : {test_loss:.4f}"
)

print(
    f"Accuracy        : {accuracy * 100:.2f}%"
)

print(
    f"Macro Precision : {precision_macro * 100:.2f}%"
)

print(
    f"Macro Recall    : {recall_macro * 100:.2f}%"
)

print(
    f"Macro F1        : {f1_macro * 100:.2f}%"
)

print(
    f"Weighted Prec.  : {precision_weighted * 100:.2f}%"
)

print(
    f"Weighted Recall : {recall_weighted * 100:.2f}%"
)

print(
    f"Weighted F1     : {f1_weighted * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(

    y_true,

    y_pred,

    labels=list(
        range(
            len(model_classes)
        )
    ),

    target_names=model_classes,

    output_dict=True,

    zero_division=0
)


classification_report_df = (
    pd.DataFrame(
        report_dict
    ).transpose()
)


classification_report_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "classification_report.csv"
    )
)


# ============================================================
# PER-CLASS METRICS
# ============================================================

per_class_rows = []


for class_index, class_name in enumerate(
    model_classes
):

    class_true = (
        y_true == class_index
    )

    class_pred = (
        y_pred == class_index
    )


    tp = np.sum(
        class_true & class_pred
    )

    fp = np.sum(
        (~class_true) & class_pred
    )

    fn = np.sum(
        class_true & (~class_pred)
    )

    support = np.sum(
        class_true
    )


    class_precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )


    class_recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )


    class_f1 = (

        2
        * class_precision
        * class_recall
        /
        (
            class_precision
            + class_recall
        )

        if (
            class_precision
            + class_recall
        ) > 0

        else 0
    )


    class_accuracy = (

        tp / support

        if support > 0

        else 0
    )


    per_class_rows.append({

        "class": class_name,

        "support": int(
            support
        ),

        "true_positive": int(
            tp
        ),

        "false_positive": int(
            fp
        ),

        "false_negative": int(
            fn
        ),

        "accuracy_percent":
            class_accuracy * 100,

        "precision_percent":
            class_precision * 100,

        "recall_percent":
            class_recall * 100,

        "f1_percent":
            class_f1 * 100

    })


per_class_df = pd.DataFrame(
    per_class_rows
)


per_class_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "per_class_metrics.csv"
    ),

    index=False
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_true,

    y_pred,

    labels=list(
        range(
            len(model_classes)
        )
    )
)


# ============================================================
# SAVE RAW CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(

    cm,

    index=model_classes,

    columns=model_classes
)


cm_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    )
)


# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

cm_normalized = (
    cm.astype(float)
    /
    np.maximum(
        cm.sum(axis=1, keepdims=True),
        1
    )
)


cm_normalized_df = pd.DataFrame(

    cm_normalized,

    index=model_classes,

    columns=model_classes
)


cm_normalized_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix_normalized.csv"
    )
)


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

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
    len(model_classes)
)

plt.xticks(
    tick_marks,
    model_classes,
    rotation=90
)

plt.yticks(
    tick_marks,
    model_classes
)


threshold = (
    cm.max() / 2.0
    if cm.size > 0
    else 0
)


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(

            j,

            i,

            str(
                cm[i, j]
            ),

            horizontalalignment="center"

        )


plt.ylabel(
    "Actual Class"
)

plt.xlabel(
    "Predicted Class"
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# NORMALIZED CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(14, 12)
)

plt.imshow(
    cm_normalized,
    interpolation="nearest"
)

plt.title(
    "Capsicum ResNet50 - Normalized Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    tick_marks,
    model_classes,
    rotation=90
)

plt.yticks(
    tick_marks,
    model_classes
)


for i in range(
    cm_normalized.shape[0]
):

    for j in range(
        cm_normalized.shape[1]
    ):

        plt.text(

            j,

            i,

            f"{cm_normalized[i, j]:.2f}",

            horizontalalignment="center"

        )


plt.ylabel(
    "Actual Class"
)

plt.xlabel(
    "Predicted Class"
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix_normalized.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# ALL PREDICTIONS CSV
# ============================================================

prediction_rows = []


for i in range(
    len(y_true)
):

    true_index = int(
        y_true[i]
    )

    predicted_index = int(
        y_pred[i]
    )


    true_class = model_classes[
        true_index
    ]

    predicted_class = model_classes[
        predicted_index
    ]


    prediction_rows.append({

        "image_path":
            all_paths[i],

        "image_name":
            os.path.basename(
                all_paths[i]
            ),

        "true_class":
            true_class,

        "predicted_class":
            predicted_class,

        "correct":
            bool(
                true_index
                == predicted_index
            ),

        "confidence_percent":
            float(
                confidence_array[i]
                * 100
            )

    })


predictions_df = pd.DataFrame(
    prediction_rows
)


predictions_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "all_predictions.csv"
    ),

    index=False
)


# ============================================================
# MISCLASSIFIED IMAGES
# ============================================================

misclassified_df = (
    predictions_df[
        predictions_df["correct"]
        == False
    ]
)


misclassified_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "misclassified.csv"
    ),

    index=False
)


# ============================================================
# HIGH CONFIDENCE WRONG PREDICTIONS
# ============================================================

high_confidence_wrong = (
    misclassified_df[
        misclassified_df[
            "confidence_percent"
        ] >= 90
    ]
)


high_confidence_wrong.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "high_confidence_wrong_predictions.csv"
    ),

    index=False
)


# ============================================================
# CONFIDENCE STATISTICS
# ============================================================

correct_mask = (
    y_true == y_pred
)

wrong_mask = (
    y_true != y_pred
)


mean_confidence = (
    float(
        confidence_array.mean()
        * 100
    )
    if len(confidence_array) > 0
    else 0
)


correct_confidence = (

    float(
        confidence_array[
            correct_mask
        ].mean()
        * 100
    )

    if np.any(
        correct_mask
    )

    else 0
)


wrong_confidence = (

    float(
        confidence_array[
            wrong_mask
        ].mean()
        * 100
    )

    if np.any(
        wrong_mask
    )

    else 0
)


confidence_statistics = {

    "mean_confidence_percent":
        mean_confidence,

    "correct_prediction_mean_confidence_percent":
        correct_confidence,

    "wrong_prediction_mean_confidence_percent":
        wrong_confidence,

    "high_confidence_wrong_count":
        int(
            len(
                high_confidence_wrong
            )
        )

}


with open(

    os.path.join(
        OUTPUT_DIR,
        "confidence_statistics.json"
    ),

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        confidence_statistics,

        file,

        indent=4

    )


# ============================================================
# CLASS COUNTS JSON
# ============================================================

class_counts_output = {

    "test_counts":
        test_counts,

    "total_test_images":
        total_test_images

}


with open(

    os.path.join(
        OUTPUT_DIR,
        "class_counts.json"
    ),

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        class_counts_output,

        file,

        indent=4

    )


# ============================================================
# FINAL METRICS JSON
# ============================================================

final_metrics = {

    "model":
        "ResNet50",

    "num_classes":
        detected_num_classes,

    "classes":
        model_classes,

    "device":
        str(DEVICE),

    "test_directory":
        TEST_DIR,

    "total_test_images":
        int(total_test_images),

    "test_loss":
        float(test_loss),

    "accuracy":
        float(accuracy),

    "accuracy_percent":
        float(
            accuracy * 100
        ),

    "macro_precision":
        float(
            precision_macro
        ),

    "macro_precision_percent":
        float(
            precision_macro * 100
        ),

    "macro_recall":
        float(
            recall_macro
        ),

    "macro_recall_percent":
        float(
            recall_macro * 100
        ),

    "macro_f1":
        float(
            f1_macro
        ),

    "macro_f1_percent":
        float(
            f1_macro * 100
        ),

    "weighted_precision":
        float(
            precision_weighted
        ),

    "weighted_precision_percent":
        float(
            precision_weighted * 100
        ),

    "weighted_recall":
        float(
            recall_weighted
        ),

    "weighted_recall_percent":
        float(
            recall_weighted * 100
        ),

    "weighted_f1":
        float(
            f1_weighted
        ),

    "weighted_f1_percent":
        float(
            f1_weighted * 100
        ),

    "mean_confidence_percent":
        mean_confidence,

    "exact_duplicate_leakage_count":
        len(
            test_hash_duplicates
        )

}


with open(

    os.path.join(
        OUTPUT_DIR,
        "final_metrics.json"
    ),

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        final_metrics,

        file,

        indent=4

    )


# ============================================================
# TOP CONFUSION PAIRS
# ============================================================

confusion_pairs = []


for i in range(
    len(model_classes)
):

    for j in range(
        len(model_classes)
    ):

        if i == j:
            continue

        count = int(
            cm[i, j]
        )

        if count > 0:

            confusion_pairs.append({

                "actual":
                    model_classes[i],

                "predicted":
                    model_classes[j],

                "count":
                    count

            })


confusion_pairs = sorted(

    confusion_pairs,

    key=lambda x: x["count"],

    reverse=True

)


with open(

    os.path.join(
        OUTPUT_DIR,
        "top_confusion_pairs.json"
    ),

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        confusion_pairs,

        file,

        indent=4

    )


# ============================================================
# TEST SUMMARY TXT
# ============================================================

summary_path = os.path.join(

    OUTPUT_DIR,

    "test_summary.txt"

)


with open(

    summary_path,

    "w",

    encoding="utf-8"

) as file:

    file.write(
        "=" * 80
        + "\n"
    )

    file.write(
        "CAPSICUM RESNET50 - FINAL INTERNAL TEST\n"
    )

    file.write(
        "=" * 80
        + "\n\n"
    )

    file.write(
        f"Model: ResNet50\n"
    )

    file.write(
        f"Device: {DEVICE}\n"
    )

    file.write(
        f"Test directory: {TEST_DIR}\n"
    )

    file.write(
        f"Total test images: "
        f"{total_test_images}\n\n"
    )

    file.write(
        "CLASSES\n"
    )

    file.write(
        "-" * 80
        + "\n"
    )

    for idx, class_name in enumerate(
        model_classes
    ):

        file.write(
            f"{idx:2d}. {class_name}\n"
        )


    file.write(
        "\nMETRICS\n"
    )

    file.write(
        "-" * 80
        + "\n"
    )

    file.write(
        f"Test Loss       : "
        f"{test_loss:.4f}\n"
    )

    file.write(
        f"Accuracy        : "
        f"{accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Macro Precision : "
        f"{precision_macro * 100:.2f}%\n"
    )

    file.write(
        f"Macro Recall    : "
        f"{recall_macro * 100:.2f}%\n"
    )

    file.write(
        f"Macro F1        : "
        f"{f1_macro * 100:.2f}%\n"
    )

    file.write(
        f"Weighted F1     : "
        f"{f1_weighted * 100:.2f}%\n"
    )

    file.write(
        f"Mean Confidence: "
        f"{mean_confidence:.2f}%\n"
    )

    file.write(
        f"High-confidence "
        f"wrong predictions: "
        f"{len(high_confidence_wrong)}\n"
    )

    file.write(
        f"Exact duplicate leakage: "
        f"{len(test_hash_duplicates)}\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("TESTING COMPLETED SUCCESSFULLY")
print("=" * 80)

print(
    f"\nAccuracy        : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro Precision : "
    f"{precision_macro * 100:.2f}%"
)

print(
    f"Macro Recall    : "
    f"{recall_macro * 100:.2f}%"
)

print(
    f"Macro F1        : "
    f"{f1_macro * 100:.2f}%"
)

print(
    f"Weighted F1     : "
    f"{f1_weighted * 100:.2f}%"
)

print(
    f"\nMisclassified   : "
    f"{len(misclassified_df)}"
)

print(
    f"High confidence wrong: "
    f"{len(high_confidence_wrong)}"
)

print(
    f"Exact duplicate leakage: "
    f"{len(test_hash_duplicates)}"
)


print("\n" + "=" * 80)
print("REPORTS SAVED")
print("=" * 80)

print(
    f"\n{OUTPUT_DIR}"
)

print(
    "\nFiles:"
)

print(
    "  classification_report.csv"
)

print(
    "  per_class_metrics.csv"
)

print(
    "  confusion_matrix.csv"
)

print(
    "  confusion_matrix_normalized.csv"
)

print(
    "  confusion_matrix.png"
)

print(
    "  confusion_matrix_normalized.png"
)

print(
    "  all_predictions.csv"
)

print(
    "  misclassified.csv"
)

print(
    "  high_confidence_wrong_predictions.csv"
)

print(
    "  exact_duplicate_leakage.csv"
)

print(
    "  confidence_statistics.json"
)

print(
    "  class_counts.json"
)

print(
    "  final_metrics.json"
)

print(
    "  model_class_order.json"
)

print(
    "  top_confusion_pairs.json"
)

print(
    "  test_summary.txt"
)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)