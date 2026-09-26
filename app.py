
import os
import io
from typing import List

import torch
import torch.nn as nn

from PIL import Image, UnidentifiedImageError

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

from torchvision import models
from torchvision import transforms

from class_info import CLASS_INFO


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.environ.get("MODEL_PATH", "")
if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    default_path = os.path.join(BASE_DIR, "models", "condition_resnet50.pth")
    best_path = os.path.join(BASE_DIR, "models", "condition_resnet50_best.pth")
    if os.path.exists(default_path):
        MODEL_PATH = default_path
    elif os.path.exists(best_path):
        MODEL_PATH = best_path
    else:
        MODEL_PATH = default_path


# ============================================================
# DATASET PATHS
#
# IMPORTANT:
#
# train:
#     augmented_dataset/train
#
# validation:
#     dataset_split/valid
#
# test:
#     dataset_split/test
#
# Only TRAIN was augmented.
# Validation and TEST remain untouched.
# ============================================================

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


# ============================================================
# MODEL CONFIGURATION
#
# Matches train_augu.py
# ============================================================

IMAGE_SIZE = 224

NUM_CLASSES = 12

TOP_K = 5

DROPOUT = 0.30


# ============================================================
# EXPECTED CLASS ORDER
#
# This must exactly match train_augu.py
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


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="Capsicum Plant Condition Detection API",

    description="""

ResNet50 based Capsicum Plant Condition Detection API.

Model:
    ResNet50

Classes:
    12

Input:
    Capsicum plant / leaf image

Output:
    Predicted condition
    Confidence
    Top-5 predictions
    Class information

Training pipeline:

    data new
        ↓
    duplicate-safe split
        ↓
    dataset_split
        ├── train
        ├── valid
        └── test
        ↓
    train-only offline augmentation
        ↓
    augmented_dataset/train
        ↓
    ResNet50 training

Validation and test images are NOT augmented.

This API contains only the CNN model.
Soil and crop prediction modules are not included.

""",

    version="4.0.0"

)


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"]

)


# ============================================================
# INFERENCE TRANSFORM
#
# IMPORTANT:
# No random augmentation during prediction.
#
# train_augu.py:
#     Resize -> Tensor -> Normalize
#
# API:
#     Same deterministic preprocessing
# ============================================================

INFERENCE_TRANSFORM = transforms.Compose([

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
# GLOBAL VARIABLES
# ============================================================

model = None

MODEL_CLASSES = []

MODEL_ARCHITECTURE = "Unknown"

CHECKPOINT_INFO = {}


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = (

    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff"

)


# ============================================================
# COUNT IMAGES
# ============================================================

def count_images(directory: str):

    if not os.path.isdir(directory):

        return 0

    total = 0

    for root, _, files in os.walk(directory):

        for filename in files:

            if filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                total += 1

    return total


# ============================================================
# COUNT IMAGES PER CLASS
# ============================================================

def count_images_per_class(directory: str):

    result = {}

    for class_name in EXPECTED_CLASSES:

        class_dir = os.path.join(

            directory,

            class_name

        )

        result[class_name] = count_images(
            class_dir
        )

    return result


# ============================================================
# EXTRACT STATE DICT
# ============================================================

def extract_state_dict(checkpoint):

    if not isinstance(
        checkpoint,
        dict
    ):

        return checkpoint

    if "model_state_dict" in checkpoint:

        return checkpoint[
            "model_state_dict"
        ]

    if "state_dict" in checkpoint:

        return checkpoint[
            "state_dict"
        ]

    return checkpoint


# ============================================================
# CLEAN STATE DICT
# ============================================================

def clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[
                len("module.") :
            ]

        cleaned[key] = value

    return cleaned


# ============================================================
# GET MODEL CLASSES
# ============================================================

def get_model_classes(checkpoint):

    if isinstance(
        checkpoint,
        dict
    ):

        # ----------------------------------------------------
        # Preferred: classes saved inside checkpoint
        # ----------------------------------------------------

        if "classes" in checkpoint:

            return list(
                checkpoint["classes"]
            )

        # ----------------------------------------------------
        # class_to_idx
        # ----------------------------------------------------

        if "class_to_idx" in checkpoint:

            mapping = checkpoint[
                "class_to_idx"
            ]

            return [

                class_name

                for class_name, index

                in sorted(

                    mapping.items(),

                    key=lambda x: x[1]

                )

            ]

    # --------------------------------------------------------
    # Current train_augu.py class order
    # --------------------------------------------------------

    return EXPECTED_CLASSES.copy()


# ============================================================
# VALIDATE CLASSES
# ============================================================

def validate_classes(classes):

    # --------------------------------------------------------
    # Number of classes
    # --------------------------------------------------------

    if len(classes) != NUM_CLASSES:

        raise RuntimeError(

            f"Model has {len(classes)} classes. "

            f"Expected {NUM_CLASSES}."

        )

    # --------------------------------------------------------
    # Exact order
    #
    # IMPORTANT:
    # Class order must match model output indices.
    # --------------------------------------------------------

    if classes != EXPECTED_CLASSES:

        raise RuntimeError(

            "CRITICAL: Model class order does not "
            "match train_augu.py.\n\n"

            f"Expected order:\n"
            f"{EXPECTED_CLASSES}\n\n"

            f"Model order:\n"
            f"{classes}"

        )


# ============================================================
# PRINT FC LAYERS
# ============================================================

def print_fc_layers(state_dict):

    print()

    print(
        "-" * 70
    )

    print(
        "CHECKPOINT FC LAYERS"
    )

    print(
        "-" * 70
    )

    found = False

    for key, value in state_dict.items():

        if key.startswith("fc."):

            found = True

            if hasattr(
                value,
                "shape"
            ):

                print(

                    f"{key:<20}"
                    f"{tuple(value.shape)}"

                )

            else:

                print(
                    f"{key:<20}"
                )

    if not found:

        print(
            "No FC layers found."
        )

    print(
        "-" * 70
    )


# ============================================================
# BUILD RESNET50
#
# EXACT ARCHITECTURE FROM train_augu.py:
#
# ResNet50
#     ↓
# Layer4 trainable
#     ↓
# FC
#     Dropout(0.30)
#     Linear(2048, 12)
#
# During inference dropout is automatically disabled
# because model.eval() is used.
# ============================================================

def build_model(state_dict):

    global MODEL_ARCHITECTURE

    # --------------------------------------------------------
    # Base ResNet50
    # --------------------------------------------------------

    network = models.resnet50(
        weights=None
    )

    num_features = (
        network.fc.in_features
    )

    # --------------------------------------------------------
    # Expected current architecture
    #
    # fc.0 = Dropout
    # fc.1 = Linear
    # --------------------------------------------------------

    if (
        "fc.1.weight" not in state_dict
        or
        "fc.1.bias" not in state_dict
    ):

        print_fc_layers(
            state_dict
        )

        raise RuntimeError(

            "Checkpoint does not match "
            "the current train_augu.py "
            "architecture.\n\n"

            "Expected classifier:\n"

            "Dropout(0.30) + "
            "Linear(2048,12)"

        )

    # --------------------------------------------------------
    # FC weight
    # --------------------------------------------------------

    fc_weight = state_dict[
        "fc.1.weight"
    ]

    fc_bias = state_dict[
        "fc.1.bias"
    ]

    fc_in = fc_weight.shape[1]

    fc_out = fc_weight.shape[0]

    # --------------------------------------------------------
    # Validate input features
    # --------------------------------------------------------

    if fc_in != num_features:

        raise RuntimeError(

            "FC input mismatch.\n"

            f"Expected: {num_features}\n"

            f"Found: {fc_in}"

        )

    # --------------------------------------------------------
    # Validate output classes
    # --------------------------------------------------------

    if fc_out != NUM_CLASSES:

        raise RuntimeError(

            "FC output mismatch.\n"

            f"Expected: {NUM_CLASSES}\n"

            f"Found: {fc_out}"

        )

    # --------------------------------------------------------
    # Validate bias
    # --------------------------------------------------------

    if fc_bias.shape[0] != NUM_CLASSES:

        raise RuntimeError(

            "FC bias dimension mismatch.\n"

            f"Expected: {NUM_CLASSES}\n"

            f"Found: {fc_bias.shape[0]}"

        )

    # --------------------------------------------------------
    # EXACT classifier
    # --------------------------------------------------------

    network.fc = nn.Sequential(

        nn.Dropout(
            p=DROPOUT
        ),

        nn.Linear(

            fc_in,

            fc_out

        )

    )

    MODEL_ARCHITECTURE = (

        "ResNet50 + "
        "Dropout(0.30) + "
        "Linear(2048,12)"

    )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    try:

        network.load_state_dict(

            state_dict,

            strict=True

        )

    except RuntimeError as error:

        raise RuntimeError(

            "Model architecture does not "
            "match the checkpoint.\n\n"

            f"{error}"

        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    network = network.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    network.eval()

    return network


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global model

    global MODEL_CLASSES

    global CHECKPOINT_INFO

    print()

    print(
        "=" * 80
    )

    print(
        "LOADING RESNET50 MODEL"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # Model file check
    # --------------------------------------------------------

    if not os.path.isfile(
        MODEL_PATH
    ):

        raise FileNotFoundError(

            "Model file not found:\n"

            + MODEL_PATH

        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save checkpoint information
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        CHECKPOINT_INFO = checkpoint

    else:

        CHECKPOINT_INFO = {}

    # --------------------------------------------------------
    # Extract state dict
    # --------------------------------------------------------

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    MODEL_CLASSES = get_model_classes(
        checkpoint
    )

    validate_classes(
        MODEL_CLASSES
    )

    # --------------------------------------------------------
    # FC information
    # --------------------------------------------------------

    print_fc_layers(
        state_dict
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model(
        state_dict
    )

    # --------------------------------------------------------
    # Success information
    # --------------------------------------------------------

    print()

    print(
        "=" * 80
    )

    print(
        "CNN MODEL LOADED SUCCESSFULLY"
    )

    print(
        "=" * 80
    )

    print(
        f"Model            : ResNet50"
    )

    print(
        f"Architecture     : "
        f"{MODEL_ARCHITECTURE}"
    )

    print(
        f"Classes          : "
        f"{len(MODEL_CLASSES)}"
    )

    print(
        f"Class order      : "
        f"{MODEL_CLASSES}"
    )

    print(
        f"Input size       : "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Device           : "
        f"{DEVICE}"
    )

    print(
        f"Model path       : "
        f"{MODEL_PATH}"
    )

    print(
        "=" * 80
    )


# ============================================================
# STARTUP
# ============================================================

@app.on_event(
    "startup"
)
def startup():

    load_model()


# ============================================================
# ROOT API
# ============================================================

@app.get("/")
def root():

    return {

        "status":
            "running",

        "application":
            "Capsicum Plant Condition Detection API",

        "model":
            "ResNet50",

        "architecture":
            MODEL_ARCHITECTURE,

        "num_classes":
            NUM_CLASSES,

        "classes":
            MODEL_CLASSES,

        "input_size":
            "224x224",

        "device":
            str(DEVICE),

        "training_pipeline": {

            "train":
                "augmented_dataset/train",

            "validation":
                "dataset_split/valid",

            "test":
                "dataset_split/test",

            "augmentation":
                "train only",

            "runtime_augmentation":
                False

        },

        "soil_model":
            False,

        "crop_prediction":
            False,

        "docs":
            "/docs"

    }


# ============================================================
# HEALTH API
# ============================================================

@app.get(
    "/health"
)
def health():

    return {

        "status":
            "healthy",

        "cnn_model_loaded":
            model is not None,

        "model":
            "ResNet50",

        "architecture":
            MODEL_ARCHITECTURE,

        "num_classes":
            len(MODEL_CLASSES),

        "device":
            str(DEVICE)

    }


# ============================================================
# CLASS LIST
# ============================================================

@app.get(
    "/classes"
)
def classes():

    return {

        "num_classes":
            len(MODEL_CLASSES),

        "classes":
            MODEL_CLASSES

    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get(
    "/model-info"
)
def model_info():

    result = {

        "architecture":
            "ResNet50",

        "classifier_architecture":
            MODEL_ARCHITECTURE,

        "framework":
            "PyTorch",

        "input_size":
            "224x224",

        "num_classes":
            NUM_CLASSES,

        "classes":
            MODEL_CLASSES,

        "fine_tuned_layers":
            "Layer4 + FC",

        "dropout":
            DROPOUT,

        "device":
            str(DEVICE),

        "model_path":
            MODEL_PATH,

        "training_data":
            {

                "train":
                    TRAIN_DIR,

                "validation":
                    VALID_DIR,

                "test":
                    TEST_DIR

            }

    }

    # --------------------------------------------------------
    # Optional checkpoint metadata
    # --------------------------------------------------------

    if isinstance(
        CHECKPOINT_INFO,
        dict
    ):

        if "epoch" in CHECKPOINT_INFO:

            try:

                result[
                    "best_model_epoch"
                ] = (

                    int(
                        CHECKPOINT_INFO[
                            "epoch"
                        ]
                    ) + 1

                )

            except (
                ValueError,
                TypeError
            ):

                result[
                    "best_model_epoch"
                ] = str(

                    CHECKPOINT_INFO[
                        "epoch"
                    ]

                )

        if "best_train_accuracy" in CHECKPOINT_INFO:

            try:

                result[
                    "best_train_accuracy"
                ] = float(

                    CHECKPOINT_INFO[
                        "best_train_accuracy"
                    ]

                )

            except (
                ValueError,
                TypeError
            ):

                result[
                    "best_train_accuracy"
                ] = str(

                    CHECKPOINT_INFO[
                        "best_train_accuracy"
                    ]

                )

        if "best_valid_accuracy" in CHECKPOINT_INFO:

            try:

                result[
                    "best_valid_accuracy"
                ] = float(

                    CHECKPOINT_INFO[
                        "best_valid_accuracy"
                    ]

                )

            except (
                ValueError,
                TypeError
            ):

                result[
                    "best_valid_accuracy"
                ] = str(

                    CHECKPOINT_INFO[
                        "best_valid_accuracy"
                    ]

                )

        if "architecture" in CHECKPOINT_INFO:

            result[
                "checkpoint_architecture"
            ] = str(

                CHECKPOINT_INFO[
                    "architecture"
                ]

            )

        if "num_classes" in CHECKPOINT_INFO:

            try:

                result[
                    "checkpoint_num_classes"
                ] = int(

                    CHECKPOINT_INFO[
                        "num_classes"
                    ]

                )

            except (
                ValueError,
                TypeError
            ):

                result[
                    "checkpoint_num_classes"
                ] = str(

                    CHECKPOINT_INFO[
                        "num_classes"
                    ]

                )

    return result


# ============================================================
# CLASS INFORMATION
# ============================================================

@app.get(
    "/class-info/{class_name}"
)
def get_class_info(
    class_name: str
):

    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    if class_name in CLASS_INFO:

        return {

            "class":
                class_name,

            "information":
                CLASS_INFO[
                    class_name
                ]

        }

    # --------------------------------------------------------
    # Case-insensitive match
    # --------------------------------------------------------

    normalized_name = (
        class_name.strip().lower()
    )

    for existing_name in CLASS_INFO:

        if (
            existing_name.lower()
            ==
            normalized_name
        ):

            return {

                "class":
                    existing_name,

                "information":
                    CLASS_INFO[
                        existing_name
                    ]

            }

    # --------------------------------------------------------
    # Not found
    # --------------------------------------------------------

    raise HTTPException(

        status_code=404,

        detail={

            "message":
                "Class information not found",

            "requested_class":
                class_name,

            "available_classes":
                list(
                    CLASS_INFO.keys()
                )

        }

    )


# ============================================================
# DATASET INFORMATION
#
# IMPORTANT:
#
# Train:
#     augmented_dataset/train
#
# Validation:
#     dataset_split/valid
#
# Test:
#     dataset_split/test
#
# Therefore validation/test remain untouched.
# ============================================================

@app.get(
    "/dataset"
)
def dataset_info():

    train_count = count_images(
        TRAIN_DIR
    )

    valid_count = count_images(
        VALID_DIR
    )

    test_count = count_images(
        TEST_DIR
    )

    return {

        "train_images":
            train_count,

        "validation_images":
            valid_count,

        "test_images":
            test_count,

        "total_images":

            (
                train_count
                +
                valid_count
                +
                test_count
            ),

        "num_classes":
            NUM_CLASSES,

        "classes":
            MODEL_CLASSES,

        "train_directory":
            TRAIN_DIR,

        "validation_directory":
            VALID_DIR,

        "test_directory":
            TEST_DIR,

        "augmentation":
            "Train only",

        "validation_test_augmented":
            False

    }


# ============================================================
# DATASET CLASS-WISE
# ============================================================

@app.get(
    "/dataset/class-wise"
)
def dataset_class_wise():

    train_counts = count_images_per_class(
        TRAIN_DIR
    )

    valid_counts = count_images_per_class(
        VALID_DIR
    )

    test_counts = count_images_per_class(
        TEST_DIR
    )

    result = {}

    for class_name in MODEL_CLASSES:

        result[class_name] = {

            "train":
                train_counts.get(
                    class_name,
                    0
                ),

            "validation":
                valid_counts.get(
                    class_name,
                    0
                ),

            "test":
                test_counts.get(
                    class_name,
                    0
                )

        }

    return {

        "num_classes":
            NUM_CLASSES,

        "classes":
            result

    }


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(
    image: Image.Image
):

    if model is None:

        raise RuntimeError(
            "Model is not loaded."
        )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    image = image.convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    tensor = INFERENCE_TRANSFORM(
        image
    )

    # --------------------------------------------------------
    # Batch dimension
    # --------------------------------------------------------

    tensor = tensor.unsqueeze(
        0
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    tensor = tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.inference_mode():

        output = model(
            tensor
        )

    # --------------------------------------------------------
    # Softmax
    # --------------------------------------------------------

    probabilities = torch.softmax(

        output,

        dim=1

    )

    # --------------------------------------------------------
    # Top K
    # --------------------------------------------------------

    top_k = min(

        TOP_K,

        len(MODEL_CLASSES)

    )

    top_probabilities, top_indices = torch.topk(

        probabilities,

        top_k,

        dim=1

    )

    # --------------------------------------------------------
    # Prediction list
    # --------------------------------------------------------

    predictions = []

    for rank in range(
        top_k
    ):

        index = (

            top_indices[
                0
            ][
                rank
            ].item()

        )

        confidence = (

            top_probabilities[
                0
            ][
                rank
            ].item()

            *

            100.0

        )

        class_name = (

            MODEL_CLASSES[
                index
            ]

        )

        predictions.append({

            "rank":
                rank + 1,

            "class":
                class_name,

            "class_index":
                index,

            "confidence_percent":
                round(
                    confidence,
                    4
                ),

            "class_information":
                CLASS_INFO.get(
                    class_name,
                    {}
                )

        })

    # --------------------------------------------------------
    # Primary prediction
    # --------------------------------------------------------

    primary = predictions[0]

    confidence = primary[
        "confidence_percent"
    ]

    # --------------------------------------------------------
    # Confidence level
    # --------------------------------------------------------

    if confidence >= 90:

        confidence_level = "HIGH"

    elif confidence >= 70:

        confidence_level = "MEDIUM"

    elif confidence >= 50:

        confidence_level = "LOW"

    else:

        confidence_level = "VERY LOW"

    # --------------------------------------------------------
    # Warning
    # --------------------------------------------------------

    warning = None

    if confidence < 50:

        warning = (

            "Low model confidence. "

            "The image may differ from the "

            "training distribution or may not "

            "belong to one of the supported classes."

        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {

        "predicted_class":
            primary[
                "class"
            ],

        "class_index":
            primary[
                "class_index"
            ],

        "confidence_percent":
            primary[
                "confidence_percent"
            ],

        "confidence_level":
            confidence_level,

        "warning":
            warning,

        "class_information":
            primary[
                "class_information"
            ],

        "top_5_predictions":
            predictions

    }


# ============================================================
# READ UPLOADED IMAGE
# ============================================================

async def read_uploaded_image(
    file: UploadFile
):

    allowed_types = {

        "image/jpeg",

        "image/png",

        "image/webp",

        "image/bmp",

        "image/tiff"

    }

    # --------------------------------------------------------
    # Content type
    # --------------------------------------------------------

    if file.content_type not in allowed_types:

        raise HTTPException(

            status_code=400,

            detail=(

                "Unsupported image format. "

                "Allowed formats: "

                "JPG, JPEG, PNG, WEBP, BMP, TIFF."

            )

        )

    try:

        # ----------------------------------------------------
        # Read file
        # ----------------------------------------------------

        contents = await file.read()

        if not contents:

            raise HTTPException(

                status_code=400,

                detail="Uploaded file is empty."

            )

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        image = Image.open(

            io.BytesIO(
                contents
            )

        )

        # Force actual image decoding
        image.load()

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        image = image.convert(
            "RGB"
        )

        return image

    except UnidentifiedImageError:

        raise HTTPException(

            status_code=400,

            detail=(
                "Uploaded file is not "
                "a valid image."
            )

        )

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(

            status_code=400,

            detail=(
                f"Unable to read image: {error}"
            )

        )


# ============================================================
# SINGLE PREDICTION
# ============================================================

@app.post(
    "/predict"
)
async def predict(

    file: UploadFile = File(...)

):

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = await read_uploaded_image(
        file
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        prediction = predict_image(
            image
        )

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=(
                f"Prediction failed: {error}"
            )

        )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "success":
            True,

        "filename":
            file.filename,

        "image_width":
            image.width,

        "image_height":
            image.height,

        "model":
            "ResNet50",

        "architecture":
            MODEL_ARCHITECTURE,

        "num_classes":
            NUM_CLASSES,

        "device":
            str(DEVICE),

        "prediction":
            prediction

    }


# ============================================================
# BATCH PREDICTION
# ============================================================

@app.post(
    "/predict-batch"
)
async def predict_batch(

    files: List[
        UploadFile
    ] = File(...)

):

    if len(files) == 0:

        raise HTTPException(

            status_code=400,

            detail=(
                "No images were uploaded."
            )

        )

    results = []

    for file in files:

        try:

            # ------------------------------------------------
            # Read image
            # ------------------------------------------------

            image = await read_uploaded_image(
                file
            )

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            prediction = predict_image(
                image
            )

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            results.append({

                "filename":
                    file.filename,

                "image_width":
                    image.width,

                "image_height":
                    image.height,

                "success":
                    True,

                "prediction":
                    prediction

            })

        except Exception as error:

            results.append({

                "filename":
                    file.filename,

                "success":
                    False,

                "error":
                    str(error)

            })

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    successful = sum(

        1

        for result in results

        if result["success"]

    )

    failed = (

        len(results)
        -
        successful

    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "success":
            True,

        "total_images":
            len(files),

        "successful_predictions":
            successful,

        "failed_predictions":
            failed,

        "results":
            results

    }


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "app:app",

        host="0.0.0.0",

        port=int(os.environ.get("PORT", 8080)),

        reload=False

    )

