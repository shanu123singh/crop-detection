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
# CNN MODEL PATH
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
# ============================================================

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "augmented_dataset",
    "train"
)

VALID_DIR = os.path.join(
    BASE_DIR,
    "augmented_dataset",
    "valid"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "augmented_dataset",
    "test"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

TOP_K = 5

NUM_CLASSES = 12


# ============================================================
# EXPECTED CNN CLASS ORDER
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

    title="Capsicum Plant Disease Detection API",

    description="""

    ResNet50 based Capsicum Plant
    Disease / Condition Detection API.

    Input:
    - Capsicum plant image

    Output:
    - Predicted disease / condition
    - Confidence
    - Top predictions
    - Class information

    This API contains only the CNN model.
    Soil and crop prediction modules are not included.

    """,

    version="3.0.0"

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
# CNN INFERENCE TRANSFORM
# ============================================================

INFERENCE_TRANSFORM = transforms.Compose([

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
# GLOBAL CNN VARIABLES
# ============================================================

model = None

MODEL_CLASSES = []

CHECKPOINT_INFO = {}

MODEL_ARCHITECTURE = "Unknown"


# ============================================================
# IMAGE EXTENSIONS
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

def count_images(
    directory
):

    if not os.path.isdir(
        directory
    ):

        return 0


    total = 0


    for root, _, files in os.walk(
        directory
    ):

        for filename in files:

            if filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                total += 1


    return total


# ============================================================
# COUNT IMAGES PER CLASS
# ============================================================

def count_images_per_class(
    directory
):

    result = {}


    if not os.path.isdir(
        directory
    ):

        return result


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

def extract_state_dict(
    checkpoint
):

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

def clean_state_dict(
    state_dict
):

    cleaned = {}


    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]


        cleaned[key] = value


    return cleaned


# ============================================================
# GET MODEL CLASSES
# ============================================================

def get_model_classes(
    checkpoint
):

    if isinstance(
        checkpoint,
        dict
    ):

        if "classes" in checkpoint:

            return list(
                checkpoint["classes"]
            )


        if "class_to_idx" in checkpoint:

            mapping = checkpoint[
                "class_to_idx"
            ]


            classes = [

                class_name

                for class_name, index

                in sorted(

                    mapping.items(),

                    key=lambda x: x[1]

                )

            ]


            return classes


    return EXPECTED_CLASSES.copy()


# ============================================================
# VALIDATE CLASSES
# ============================================================

def validate_classes(
    classes
):

    if len(classes) != NUM_CLASSES:

        raise RuntimeError(

            f"Model has {len(classes)} classes. "

            f"Expected {NUM_CLASSES}."

        )


    if set(classes) != set(
        EXPECTED_CLASSES
    ):

        missing = sorted(

            set(EXPECTED_CLASSES)
            -
            set(classes)

        )


        extra = sorted(

            set(classes)
            -
            set(EXPECTED_CLASSES)

        )


        raise RuntimeError(

            "Model classes do not match "
            "the expected 12 classes.\n\n"

            f"Missing classes: {missing}\n"

            f"Extra classes: {extra}\n\n"

            f"Expected order: "
            f"{EXPECTED_CLASSES}\n\n"

            f"Checkpoint order: "
            f"{classes}"

        )


# ============================================================
# PRINT FC LAYERS
# ============================================================

def print_fc_layers(
    state_dict
):

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

        if key.startswith(
            "fc."
        ):

            found = True


            if hasattr(
                value,
                "shape"
            ):

                print(

                    f"{key:<20} "
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
# ============================================================

def build_model(
    state_dict
):

    global MODEL_ARCHITECTURE


    network = models.resnet50(
        weights=None
    )


    num_features = (
        network.fc.in_features
    )


    # ========================================================
    # ARCHITECTURE 1
    #
    # Dropout
    # Linear(2048,512)
    # ReLU
    # Dropout
    # Linear(512,12)
    # ========================================================

    if (

        "fc.1.weight" in state_dict

        and

        "fc.4.weight" in state_dict

    ):

        first_weight = state_dict[
            "fc.1.weight"
        ]

        final_weight = state_dict[
            "fc.4.weight"
        ]


        first_in = (
            first_weight.shape[1]
        )

        hidden_features = (
            first_weight.shape[0]
        )

        final_in = (
            final_weight.shape[1]
        )

        final_out = (
            final_weight.shape[0]
        )


        if first_in != num_features:

            raise RuntimeError(

                "Classifier input mismatch. "

                f"Expected {num_features}, "

                f"found {first_in}."

            )


        if hidden_features != final_in:

            raise RuntimeError(

                "Classifier hidden dimension "
                "mismatch."

            )


        if final_out != NUM_CLASSES:

            raise RuntimeError(

                "Classifier output mismatch. "

                f"Expected {NUM_CLASSES}, "

                f"found {final_out}."

            )


        network.fc = nn.Sequential(

            nn.Dropout(
                p=0.30
            ),

            nn.Linear(
                first_in,
                hidden_features
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                p=0.30
            ),

            nn.Linear(
                final_in,
                final_out
            )

        )


        MODEL_ARCHITECTURE = (

            "ResNet50 + "
            "Dropout + Linear(2048,512) "
            "+ ReLU + Dropout + "
            "Linear(512,12)"

        )


    # ========================================================
    # ARCHITECTURE 2
    #
    # Linear(2048,hidden)
    # ReLU
    # Dropout
    # Linear(hidden,12)
    # ========================================================

    elif (

        "fc.0.weight" in state_dict

        and

        "fc.3.weight" in state_dict

    ):

        first_weight = state_dict[
            "fc.0.weight"
        ]

        final_weight = state_dict[
            "fc.3.weight"
        ]


        first_in = (
            first_weight.shape[1]
        )

        hidden_features = (
            first_weight.shape[0]
        )

        final_in = (
            final_weight.shape[1]
        )

        final_out = (
            final_weight.shape[0]
        )


        if first_in != num_features:

            raise RuntimeError(

                "Classifier input mismatch. "

                f"Expected {num_features}, "

                f"found {first_in}."

            )


        if hidden_features != final_in:

            raise RuntimeError(

                "Classifier hidden dimension "
                "mismatch."

            )


        if final_out != NUM_CLASSES:

            raise RuntimeError(

                "Final classifier output must "

                f"be {NUM_CLASSES}. "

                f"Found {final_out}."

            )


        network.fc = nn.Sequential(

            nn.Linear(
                first_in,
                hidden_features
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                p=0.30
            ),

            nn.Linear(
                final_in,
                final_out
            )

        )


        MODEL_ARCHITECTURE = (

            "ResNet50 + "
            "Linear(2048,hidden) + ReLU "
            "+ Dropout + "
            "Linear(hidden,12)"

        )


    # ========================================================
    # ARCHITECTURE 3
    #
    # Simple Linear(2048,12)
    # ========================================================

    elif (

        "fc.weight" in state_dict

        and

        "fc.bias" in state_dict

    ):

        weight = state_dict[
            "fc.weight"
        ]


        fc_in = (
            weight.shape[1]
        )

        fc_out = (
            weight.shape[0]
        )


        if fc_in != num_features:

            raise RuntimeError(

                "FC input mismatch. "

                f"Expected {num_features}, "

                f"found {fc_in}."

            )


        if fc_out != NUM_CLASSES:

            raise RuntimeError(

                "FC output mismatch. "

                f"Expected {NUM_CLASSES}, "

                f"found {fc_out}."

            )


        network.fc = nn.Linear(

            fc_in,

            fc_out

        )


        MODEL_ARCHITECTURE = (

            "ResNet50 + Linear(2048,12)"

        )


    # ========================================================
    # ARCHITECTURE 4
    #
    # Dropout + Linear(2048,12)
    # ========================================================

    elif (

        "fc.1.weight" in state_dict

        and

        "fc.1.bias" in state_dict

    ):

        weight = state_dict[
            "fc.1.weight"
        ]


        fc_in = (
            weight.shape[1]
        )

        fc_out = (
            weight.shape[0]
        )


        if fc_in != num_features:

            raise RuntimeError(

                "FC input mismatch. "

                f"Expected {num_features}, "

                f"found {fc_in}."

            )


        if fc_out != NUM_CLASSES:

            raise RuntimeError(

                "FC output mismatch. "

                f"Expected {NUM_CLASSES}, "

                f"found {fc_out}."

            )


        network.fc = nn.Sequential(

            nn.Dropout(
                p=0.30
            ),

            nn.Linear(
                fc_in,
                fc_out
            )

        )


        MODEL_ARCHITECTURE = (

            "ResNet50 + Dropout + "
            "Linear(2048,12)"

        )


    else:

        print_fc_layers(
            state_dict
        )


        raise RuntimeError(

            "Unable to detect the classifier "
            "architecture from the checkpoint."

        )


    # ========================================================
    # LOAD STATE DICT
    # ========================================================

    try:

        network.load_state_dict(

            state_dict,

            strict=True

        )

    except RuntimeError as error:

        raise RuntimeError(

            "Model architecture does not match "
            "the checkpoint.\n\n"

            f"{error}"

        )


    network = network.to(
        DEVICE
    )


    network.eval()


    return network


# ============================================================
# LOAD CNN MODEL
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


    # ========================================================
    # CHECK MODEL FILE
    # ========================================================

    if not os.path.isfile(
        MODEL_PATH
    ):

        raise FileNotFoundError(

            "Model file not found:\n"

            + MODEL_PATH

        )


    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

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


    # ========================================================
    # CHECKPOINT INFORMATION
    # ========================================================

    if isinstance(
        checkpoint,
        dict
    ):

        CHECKPOINT_INFO = checkpoint

    else:

        CHECKPOINT_INFO = {}


    # ========================================================
    # STATE DICT
    # ========================================================

    state_dict = extract_state_dict(
        checkpoint
    )


    state_dict = clean_state_dict(
        state_dict
    )


    # ========================================================
    # MODEL CLASSES
    # ========================================================

    MODEL_CLASSES = get_model_classes(
        checkpoint
    )


    validate_classes(
        MODEL_CLASSES
    )


    # ========================================================
    # PRINT FC INFORMATION
    # ========================================================

    print_fc_layers(
        state_dict
    )


    # ========================================================
    # BUILD MODEL
    # ========================================================

    model = build_model(
        state_dict
    )


    # ========================================================
    # PRINT MODEL INFORMATION
    # ========================================================

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
        "Model            : ResNet50"
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

    # Only CNN
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
            "Capsicum Plant Disease Detection API",

        "model":
            "ResNet50",

        "architecture":
            MODEL_ARCHITECTURE,

        "num_classes":
            NUM_CLASSES,

        "classes":
            MODEL_CLASSES,

        "device":
            str(DEVICE),

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

        "num_classes":
            len(MODEL_CLASSES),

        "device":
            str(DEVICE)

    }


# ============================================================
# CNN CLASS LIST
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

        "device":
            str(DEVICE),

        "model_path":
            MODEL_PATH

    }


    # ========================================================
    # CHECKPOINT INFORMATION
    # ========================================================

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
                    )

                    + 1

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
            TEST_DIR

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
# CNN PREDICTION FUNCTION
# ============================================================

def predict_image(
    image: Image.Image
):

    if model is None:

        raise RuntimeError(
            "Model is not loaded."
        )


    # ========================================================
    # RGB
    # ========================================================

    image = image.convert(
        "RGB"
    )


    # ========================================================
    # TRANSFORM
    # ========================================================

    tensor = INFERENCE_TRANSFORM(
        image
    )


    # ========================================================
    # BATCH DIMENSION
    # ========================================================

    tensor = tensor.unsqueeze(
        0
    )


    # ========================================================
    # DEVICE
    # ========================================================

    tensor = tensor.to(
        DEVICE
    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    with torch.inference_mode():

        output = model(
            tensor
        )


    # ========================================================
    # SOFTMAX
    # ========================================================

    probabilities = torch.softmax(

        output,

        dim=1

    )


    # ========================================================
    # TOP K
    # ========================================================

    top_k = min(

        TOP_K,

        len(MODEL_CLASSES)

    )


    top_probabilities, top_indices = torch.topk(

        probabilities,

        top_k,

        dim=1

    )


    # ========================================================
    # PREDICTION LIST
    # ========================================================

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

            100

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


    # ========================================================
    # PRIMARY PREDICTION
    # ========================================================

    primary = predictions[0]


    confidence = (

        primary[
            "confidence_percent"
        ]

    )


    # ========================================================
    # CONFIDENCE LEVEL
    # ========================================================

    if confidence >= 90:

        confidence_level = "HIGH"

    elif confidence >= 70:

        confidence_level = "MEDIUM"

    elif confidence >= 50:

        confidence_level = "LOW"

    else:

        confidence_level = "VERY LOW"


    # ========================================================
    # WARNING
    # ========================================================

    warning = None


    if confidence < 50:

        warning = (

            "Low model confidence. "

            "The image may differ from the "
            "training distribution or may not "
            "belong to one of the supported classes."

        )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "predicted_class":
            primary["class"],

        "class_index":
            primary["class_index"],

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


    # ========================================================
    # CHECK FILE TYPE
    # ========================================================

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
        # READ FILE
        # ----------------------------------------------------

        contents = await file.read()


        if not contents:

            raise HTTPException(

                status_code=400,

                detail="Uploaded file is empty."

            )


        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        image = Image.open(

            io.BytesIO(
                contents
            )

        )


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
                "Uploaded file is not a valid image."
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
# CNN SINGLE PREDICTION
# ============================================================

@app.post(
    "/predict"
)
async def predict(

    file: UploadFile = File(...)

):

    # ========================================================
    # READ IMAGE
    # ========================================================

    image = await read_uploaded_image(
        file
    )


    # ========================================================
    # PREDICT
    # ========================================================

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


    # ========================================================
    # RESPONSE
    # ========================================================

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
# CNN BATCH PREDICTION
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

            detail="No images were uploaded."

        )


    results = []


    for file in files:

        try:

            # ------------------------------------------------
            # READ IMAGE
            # ------------------------------------------------

            image = await read_uploaded_image(
                file
            )


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            prediction = predict_image(
                image
            )


            # ------------------------------------------------
            # RESULT
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


    # ========================================================
    # FINAL BATCH RESPONSE
    # ========================================================

    return {

        "success":
            True,

        "total_images":
            len(files),

        "successful_predictions":

            sum(

                1

                for result in results

                if result["success"]

            ),

        "failed_predictions":

            sum(

                1

                for result in results

                if not result["success"]

            ),

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