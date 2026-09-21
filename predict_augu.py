# ============================================================
# CAPSICUM PLANT CONDITION PREDICTION - RESNET50
# ============================================================
#
# FILE:
# predict_augu.py
#
# MODEL:
# ResNet50
#
# CLASSES:
# 14
#
# IMPORTANT:
# down leaf mites is excluded.
#
# ============================================================


import os
import json

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms, models

from class_info import CLASS_INFO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\singh\OneDrive\Desktop\CNN CROPS"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "condition_resnet50.pth"
)

# ------------------------------------------------------------
# CHANGE ONLY THIS IMAGE PATH
# ------------------------------------------------------------

IMAGE_PATH = (
    r"C:\Users\singh\Downloads\WhatsApp Image 2026-09-09 at 8.10.45 AM.jpeg"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "prediction_results"
)

IMAGE_SIZE = 224

TOP_K = 5


# ============================================================
# FINAL 14 CLASSES
# ============================================================
#
# ImageFolder alphabetical order
#
# ============================================================

EXPECTED_CLASSES = [

    "Anthracnose",

    "Larva",

    "Magnesium",

    "Nitrogen deficiency",

    "bacterial spot",

    "blossom-end rot",

    "down leaf aphid",

    "fruit thrips",

    "healthy",

    "powdery mildew",

    "recover powdery mildew",

    "snail",

    "upperleaf thrips",

    "virus"

]


NUM_CLASSES = len(
    EXPECTED_CLASSES
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)


# ============================================================
# HEADER
# ============================================================

print()

print("=" * 90)

print(
    "CAPSICUM PLANT CONDITION PREDICTION"
)

print("=" * 90)

print(
    "Architecture : ResNet50"
)

print(
    f"Classes      : {NUM_CLASSES}"
)

print(
    f"Device       : {device}"
)


if torch.cuda.is_available():

    print(
        "GPU          : "
        + torch.cuda.get_device_name(0)
    )

else:

    print(
        "GPU          : Not available"
    )


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.isfile(
    MODEL_PATH
):

    raise FileNotFoundError(

        "\nModel file not found:\n"
        + MODEL_PATH

    )


# ============================================================
# CHECK IMAGE
# ============================================================

if not os.path.isfile(
    IMAGE_PATH
):

    raise FileNotFoundError(

        "\nImage file not found:\n"
        + IMAGE_PATH

    )


# ============================================================
# CHECK CLASS_INFO
# ============================================================

missing_class_info = [

    class_name

    for class_name in EXPECTED_CLASSES

    if class_name not in CLASS_INFO

]


if missing_class_info:

    raise ValueError(

        "\nMissing classes in class_info.py:\n"

        + "\n".join(

            f"  - {class_name}"

            for class_name in missing_class_info

        )

    )


# ============================================================
# IMAGE TRANSFORM
# ============================================================
#
# SAME AS VALIDATION / TEST
#
# Resize 256
# CenterCrop 224
# Normalize ImageNet
#
# NO RANDOM AUGMENTATION
#
# ============================================================

transform = transforms.Compose([

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
# LOAD CHECKPOINT
# ============================================================

print()

print("=" * 90)

print(
    "LOADING CHECKPOINT"
)

print("=" * 90)


checkpoint = torch.load(

    MODEL_PATH,

    map_location=device,

    weights_only=False

)


print(
    "✓ Checkpoint loaded"
)


# ============================================================
# EXTRACT STATE DICT
# ============================================================

if (

    isinstance(
        checkpoint,
        dict
    )

    and

    "model_state_dict"
    in checkpoint

):

    state_dict = checkpoint[
        "model_state_dict"
    ]

    print(
        "Checkpoint format : model_state_dict"
    )


elif (

    isinstance(
        checkpoint,
        dict
    )

    and

    "state_dict"
    in checkpoint

):

    state_dict = checkpoint[
        "state_dict"
    ]

    print(
        "Checkpoint format : state_dict"
    )


else:

    state_dict = checkpoint

    print(
        "Checkpoint format : raw state_dict"
    )


# ============================================================
# VALIDATE STATE DICT
# ============================================================

if not isinstance(
    state_dict,
    dict
):

    raise RuntimeError(

        "\nCheckpoint does not contain "
        "a valid state dictionary."

    )


# ============================================================
# REMOVE module. PREFIX
# ============================================================

clean_state_dict = {}


for key, value in state_dict.items():

    if key.startswith(
        "module."
    ):

        key = key[
            len("module.") :
        ]

    clean_state_dict[key] = value


state_dict = clean_state_dict


print(
    f"State dict parameters : {len(state_dict)}"
)


# ============================================================
# CHECKPOINT CLASS INFORMATION
# ============================================================

checkpoint_classes = None

checkpoint_class_to_idx = None


if isinstance(
    checkpoint,
    dict
):

    if "classes" in checkpoint:

        checkpoint_classes = list(
            checkpoint["classes"]
        )


    if "class_to_idx" in checkpoint:

        checkpoint_class_to_idx = dict(
            checkpoint["class_to_idx"]
        )


# ============================================================
# DETERMINE CLASS ORDER
# ============================================================

if checkpoint_classes is not None:

    classes = checkpoint_classes

    print(
        "Class order source : checkpoint 'classes'"
    )


elif checkpoint_class_to_idx is not None:

    classes = [

        class_name

        for class_name, index

        in sorted(

            checkpoint_class_to_idx.items(),

            key=lambda x: x[1]

        )

    ]

    print(
        "Class order source : checkpoint 'class_to_idx'"
    )


else:

    classes = EXPECTED_CLASSES.copy()

    print(
        "Class order source : ImageFolder alphabetical order"
    )


# ============================================================
# VALIDATE CLASS COUNT
# ============================================================

if len(classes) != NUM_CLASSES:

    raise ValueError(

        "\n"
        + "=" * 80
        + "\n"
        + "CLASS COUNT ERROR"
        + "\n"
        + "=" * 80
        + "\n"
        + f"Expected : {NUM_CLASSES}\n"
        + f"Found    : {len(classes)}\n\n"
        + f"Classes  : {classes}\n"

    )


# ============================================================
# VALIDATE CLASS NAMES
# ============================================================

if set(classes) != set(
    EXPECTED_CLASSES
):

    raise ValueError(

        "\n"
        + "=" * 80
        + "\n"
        + "CLASS NAME MISMATCH"
        + "\n"
        + "=" * 80
        + "\n\n"
        + "Expected classes:\n"
        + str(EXPECTED_CLASSES)
        + "\n\n"
        + "Model classes:\n"
        + str(classes)
        + "\n"

    )


# ============================================================
# PRINT CLASS ORDER
# ============================================================

print()

print("=" * 90)

print(
    "MODEL CLASS ORDER"
)

print("=" * 90)


for index, class_name in enumerate(
    classes
):

    print(
        f"{index:2d} : {class_name}"
    )


print()

print(
    f"Total classes : {len(classes)}"
)


# ============================================================
# CREATE RESNET50
# ============================================================

print()

print("=" * 90)

print(
    "CREATING RESNET50"
)

print("=" * 90)


model = models.resnet50(
    weights=None
)


num_features = (
    model.fc.in_features
)


print(
    "Architecture      : ResNet50"
)

print(
    f"Feature dimension : {num_features}"
)

print(
    f"Number of classes : {NUM_CLASSES}"
)


# ============================================================
# DETECT FC ARCHITECTURE
# ============================================================
#
# IMPORTANT:
#
# Complex architecture is checked BEFORE simple fc.1.
#
# Supported:
#
# 1. fc.weight
#
#    Linear(2048, 14)
#
# 2. fc.1 + fc.4
#
#    Dropout
#    Linear(2048, hidden)
#    ReLU
#    Dropout
#    Linear(hidden, 14)
#
# 3. fc.0 + fc.3
#
#    Linear(2048, hidden)
#    ReLU
#    Dropout
#    Linear(hidden, 14)
#
# 4. fc.1 only
#
#    Dropout
#    Linear(2048, 14)
#
# ============================================================

print()

print("=" * 90)

print(
    "DETECTING FC ARCHITECTURE"
)

print("=" * 90)


# ============================================================
# PRINT FC WEIGHT SHAPES
# ============================================================

fc_weight_keys = [

    key

    for key in state_dict.keys()

    if key.startswith("fc.")
    and key.endswith(".weight")

]


print()

print(
    "FC layers found:"
)


for key in fc_weight_keys:

    value = state_dict[key]

    if hasattr(
        value,
        "shape"
    ):

        print(
            f"  {key} -> {tuple(value.shape)}"
        )

    else:

        print(
            f"  {key}"
        )


# ============================================================
# CASE 1
#
# fc.1 + fc.4
#
# MUST COME BEFORE fc.1 ONLY
#
# ============================================================

if (

    "fc.1.weight" in state_dict

    and

    "fc.4.weight" in state_dict

):

    first_weight = state_dict[
        "fc.1.weight"
    ]

    second_weight = state_dict[
        "fc.4.weight"
    ]


    first_in = (
        first_weight.shape[1]
    )

    first_out = (
        first_weight.shape[0]
    )

    second_in = (
        second_weight.shape[1]
    )

    second_out = (
        second_weight.shape[0]
    )


    print()

    print(
        "Detected classifier:"
    )

    print(
        "    fc.0 = Dropout(0.30)"
    )

    print(
        f"    fc.1 = Linear("
        f"{first_in}, {first_out})"
    )

    print(
        "    fc.2 = ReLU"
    )

    print(
        "    fc.3 = Dropout(0.30)"
    )

    print(
        f"    fc.4 = Linear("
        f"{second_in}, {second_out})"
    )


    if first_in != num_features:

        raise RuntimeError(

            "\nFC input mismatch.\n"
            f"Expected : {num_features}\n"
            f"Found    : {first_in}"

        )


    if first_out != second_in:

        raise RuntimeError(

            "\nHidden dimension mismatch.\n"
            f"First output : {first_out}\n"
            f"Second input : {second_in}"

        )


    if second_out != NUM_CLASSES:

        raise RuntimeError(

            "\nFinal classifier output mismatch.\n"
            f"Expected : {NUM_CLASSES}\n"
            f"Found    : {second_out}"

        )


    model.fc = nn.Sequential(

        nn.Dropout(
            p=0.30
        ),

        nn.Linear(

            first_in,

            first_out

        ),

        nn.ReLU(
            inplace=True
        ),

        nn.Dropout(
            p=0.30
        ),

        nn.Linear(

            second_in,

            second_out

        )

    )


# ============================================================
# CASE 2
#
# fc.0 + fc.3
#
# ============================================================

elif (

    "fc.0.weight" in state_dict

    and

    "fc.3.weight" in state_dict

):

    first_weight = state_dict[
        "fc.0.weight"
    ]

    second_weight = state_dict[
        "fc.3.weight"
    ]


    first_in = (
        first_weight.shape[1]
    )

    first_out = (
        first_weight.shape[0]
    )

    second_in = (
        second_weight.shape[1]
    )

    second_out = (
        second_weight.shape[0]
    )


    print()

    print(
        "Detected classifier:"
    )

    print(
        f"    fc.0 = Linear("
        f"{first_in}, {first_out})"
    )

    print(
        "    fc.1 = ReLU"
    )

    print(
        "    fc.2 = Dropout(0.30)"
    )

    print(
        f"    fc.3 = Linear("
        f"{second_in}, {second_out})"
    )


    if first_in != num_features:

        raise RuntimeError(

            "\nFC input mismatch.\n"
            f"Expected : {num_features}\n"
            f"Found    : {first_in}"

        )


    if first_out != second_in:

        raise RuntimeError(

            "\nHidden dimension mismatch.\n"
            f"First output : {first_out}\n"
            f"Second input : {second_in}"

        )


    if second_out != NUM_CLASSES:

        raise RuntimeError(

            "\nFinal classifier output mismatch.\n"
            f"Expected : {NUM_CLASSES}\n"
            f"Found    : {second_out}"

        )


    model.fc = nn.Sequential(

        nn.Linear(

            first_in,

            first_out

        ),

        nn.ReLU(
            inplace=True
        ),

        nn.Dropout(
            p=0.30
        ),

        nn.Linear(

            second_in,

            second_out

        )

    )


# ============================================================
# CASE 3
#
# SIMPLE fc.weight
#
# ============================================================

elif (

    "fc.weight" in state_dict

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


    print()

    print(
        f"Detected classifier : "
        f"Linear({fc_in}, {fc_out})"
    )


    if fc_in != num_features:

        raise RuntimeError(

            "\nFC input mismatch.\n"
            f"Expected : {num_features}\n"
            f"Found    : {fc_in}"

        )


    if fc_out != NUM_CLASSES:

        raise RuntimeError(

            "\nFC output mismatch.\n"
            f"Expected : {NUM_CLASSES}\n"
            f"Found    : {fc_out}"

        )


    model.fc = nn.Linear(

        fc_in,

        fc_out

    )


# ============================================================
# CASE 4
#
# fc.1 ONLY
#
# ============================================================

elif (

    "fc.1.weight" in state_dict

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


    print()

    print(
        "Detected classifier:"
    )

    print(
        "    fc.0 = Dropout(0.30)"
    )

    print(
        f"    fc.1 = Linear("
        f"{fc_in}, {fc_out})"
    )


    if fc_in != num_features:

        raise RuntimeError(

            "\nFC input mismatch.\n"
            f"Expected : {num_features}\n"
            f"Found    : {fc_in}"

        )


    if fc_out != NUM_CLASSES:

        raise RuntimeError(

            "\nFC output mismatch.\n"
            f"Expected : {NUM_CLASSES}\n"
            f"Found    : {fc_out}"

        )


    model.fc = nn.Sequential(

        nn.Dropout(
            p=0.30
        ),

        nn.Linear(

            fc_in,

            fc_out

        )

    )


# ============================================================
# UNKNOWN ARCHITECTURE
# ============================================================

else:

    raise RuntimeError(

        "\n"
        + "=" * 80
        + "\n"
        + "UNKNOWN FC ARCHITECTURE"
        + "\n"
        + "=" * 80
        + "\n"
        + "Could not determine the classifier "
        + "architecture from the checkpoint."

    )


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

print()

print("=" * 90)

print(
    "LOADING TRAINED WEIGHTS"
)

print("=" * 90)


try:

    model.load_state_dict(

        state_dict,

        strict=True

    )

except RuntimeError as error:

    print()

    print(
        "MODEL LOADING ERROR:"
    )

    print(
        error
    )


    raise RuntimeError(

        "\nModel architecture and checkpoint "
        "architecture do not match."

    ) from error


print()

print(
    "✓ Model weights loaded successfully"
)


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(
    device
)

model.eval()


# ============================================================
# TRAINING INFORMATION
# ============================================================

print()

print("=" * 90)

print(
    "TRAINING INFORMATION"
)

print("=" * 90)


if isinstance(
    checkpoint,
    dict
):

    if "architecture" in checkpoint:

        print(

            "Architecture       :",
            checkpoint["architecture"]

        )


    if "trainable_layers" in checkpoint:

        print(

            "Trainable layers   :",
            checkpoint["trainable_layers"]

        )


    # --------------------------------------------------------
    # FIXED SYNTAX
    # --------------------------------------------------------

    if "best_train_accuracy" in checkpoint:

        print(

            "Best train accuracy:",
            f"{float(checkpoint['best_train_accuracy']):.2f}%"

        )


    # --------------------------------------------------------
    # FIXED SYNTAX
    # --------------------------------------------------------

    if "best_valid_accuracy" in checkpoint:

        print(

            "Best valid accuracy:",
            f"{float(checkpoint['best_valid_accuracy']):.2f}%"

        )


    if "epoch" in checkpoint:

        print(

            "Best model epoch   :",
            int(checkpoint["epoch"]) + 1

        )


# ============================================================
# LOAD IMAGE
# ============================================================

print()

print("=" * 90)

print(
    "LOADING IMAGE"
)

print("=" * 90)


image = Image.open(
    IMAGE_PATH
).convert(
    "RGB"
)


print(
    "Image        :",
    IMAGE_PATH
)

print(
    "Original size:",
    image.size
)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

input_tensor = transform(
    image
)


# ============================================================
# ADD BATCH DIMENSION
# ============================================================

input_tensor = input_tensor.unsqueeze(
    0
)


# ============================================================
# MOVE TO DEVICE
# ============================================================

input_tensor = input_tensor.to(
    device
)


# ============================================================
# RUN PREDICTION
# ============================================================

print()

print("=" * 90)

print(
    "RUNNING PREDICTION"
)

print("=" * 90)


with torch.no_grad():

    outputs = model(
        input_tensor
    )


# ============================================================
# CHECK OUTPUT
# ============================================================

if outputs.ndim != 2:

    raise RuntimeError(

        "\nUnexpected model output shape:\n"
        f"{outputs.shape}"

    )


if outputs.shape[1] != NUM_CLASSES:

    raise RuntimeError(

        "\n"
        "MODEL OUTPUT CLASS COUNT MISMATCH\n"
        f"Expected : {NUM_CLASSES}\n"
        f"Found    : {outputs.shape[1]}"

    )


# ============================================================
# SOFTMAX
# ============================================================

probabilities = torch.softmax(

    outputs,

    dim=1

)


# ============================================================
# TOP K
# ============================================================

top_k = min(

    TOP_K,

    NUM_CLASSES

)


top_probabilities, top_indices = torch.topk(

    probabilities,

    top_k,

    dim=1

)


# ============================================================
# FINAL PREDICTION
# ============================================================

predicted_index = (

    top_indices[0][0].item()

)


predicted_class = (

    classes[predicted_index]

)


predicted_confidence = (

    top_probabilities[0][0].item()

    * 100

)


# ============================================================
# CONFIDENCE LEVEL
# ============================================================

if predicted_confidence >= 90:

    confidence_level = "HIGH"

elif predicted_confidence >= 70:

    confidence_level = "MEDIUM"

elif predicted_confidence >= 50:

    confidence_level = "LOW"

else:

    confidence_level = "VERY LOW"


# ============================================================
# FINAL PREDICTION
# ============================================================

print()

print("=" * 90)

print(
    "FINAL PREDICTION"
)

print("=" * 90)

print()

print(
    f"Predicted Class  : "
    f"{predicted_class}"
)

print(
    f"Confidence       : "
    f"{predicted_confidence:.2f}%"
)

print(
    f"Confidence Level : "
    f"{confidence_level}"
)

print(
    f"Class Index      : "
    f"{predicted_index}"
)


# ============================================================
# TOP K PREDICTIONS
# ============================================================

print()

print("=" * 90)

print(
    f"TOP {top_k} PREDICTIONS"
)

print("=" * 90)

print()


top_predictions = []


for rank in range(
    top_k
):

    index = (
        top_indices[0][rank].item()
    )


    probability = (

        top_probabilities[0][rank].item()

        * 100

    )


    class_name = (
        classes[index]
    )


    top_predictions.append({

        "rank":
            rank + 1,

        "class":
            class_name,

        "class_index":
            index,

        "confidence_percent":
            round(
                probability,
                4
            )

    })


    print(

        f"{rank + 1}. "
        f"{class_name:<25} "
        f"{probability:7.2f}%"

    )


# ============================================================
# CLASS INFORMATION
# ============================================================

print()

print("=" * 90)

print(
    "CLASS INFORMATION"
)

print("=" * 90)


info = CLASS_INFO.get(
    predicted_class
)


if info is None:

    print()

    print(
        "No CLASS_INFO available "
        "for this class."
    )

else:

    if "type" in info:

        print()

        print(
            "Type       :",
            info["type"]
        )


    if "insect" in info:

        print(
            "Insect     :",
            info["insect"]
        )


    if "disease" in info:

        print(
            "Disease    :",
            info["disease"]
        )


    if "deficiency" in info:

        print(
            "Deficiency :",
            info["deficiency"]
        )


    # ========================================================
    # SYMPTOMS
    # ========================================================

    if "symptoms" in info:

        print()

        print(
            "-" * 90
        )

        print(
            "SYMPTOMS"
        )

        print(
            "-" * 90
        )


        symptoms = info["symptoms"]


        if isinstance(
            symptoms,
            (list, tuple)
        ):

            for symptom in symptoms:

                print(
                    f"• {symptom}"
                )

        else:

            print(
                symptoms
            )


    # ========================================================
    # CAUSE
    # ========================================================

    if "cause" in info:

        print()

        print(
            "-" * 90
        )

        print(
            "CAUSE"
        )

        print(
            "-" * 90
        )


        cause = info["cause"]


        if isinstance(
            cause,
            (list, tuple)
        ):

            for item in cause:

                print(
                    f"• {item}"
                )

        else:

            print(
                cause
            )


    # ========================================================
    # PRECAUTIONS
    # ========================================================

    if "precaution" in info:

        print()

        print(
            "-" * 90
        )

        print(
            "PRECAUTIONS"
        )

        print(
            "-" * 90
        )


        precautions = info["precaution"]


        if isinstance(
            precautions,
            (list, tuple)
        ):

            for precaution in precautions:

                print(
                    f"• {precaution}"
                )

        else:

            print(
                precautions
            )


# ============================================================
# ALL CLASS PROBABILITIES
# ============================================================

print()

print("=" * 90)

print(
    "ALL CLASS PROBABILITIES"
)

print("=" * 90)

print()


all_probabilities = []


for index, class_name in enumerate(
    classes
):

    probability = (

        probabilities[0][index].item()

        * 100

    )


    all_probabilities.append({

        "class":
            class_name,

        "class_index":
            index,

        "probability_percent":
            round(
                probability,
                6
            )

    })


    print(

        f"{index:2d}. "
        f"{class_name:<25} "
        f"{probability:8.4f}%"

    )


# ============================================================
# CREATE RESULT
# ============================================================

result = {

    "model":
        "ResNet50",

    "model_path":
        MODEL_PATH,

    "image":
        IMAGE_PATH,

    "image_name":
        os.path.basename(
            IMAGE_PATH
        ),

    "image_size":
        list(
            image.size
        ),

    "device":
        str(device),

    "num_classes":
        NUM_CLASSES,

    "classes":
        classes,

    "predicted_class":
        predicted_class,

    "predicted_index":
        int(
            predicted_index
        ),

    "confidence_percent":
        round(
            predicted_confidence,
            4
        ),

    "confidence_level":
        confidence_level,

    "top_predictions":
        top_predictions,

    "all_probabilities":
        all_probabilities

}


# ============================================================
# ADD CHECKPOINT INFORMATION
# ============================================================

if isinstance(
    checkpoint,
    dict
):

    # --------------------------------------------------------
    # EPOCH
    # --------------------------------------------------------

    if "epoch" in checkpoint:

        result[
            "best_model_epoch"
        ] = (

            int(
                checkpoint["epoch"]
            )

            + 1

        )


    # --------------------------------------------------------
    # BEST TRAIN ACCURACY
    # --------------------------------------------------------

    if "best_train_accuracy" in checkpoint:

        result[
            "best_train_accuracy"
        ] = float(

            checkpoint[
                "best_train_accuracy"
            ]

        )


    # --------------------------------------------------------
    # BEST VALID ACCURACY
    # --------------------------------------------------------

    if "best_valid_accuracy" in checkpoint:

        result[
            "best_valid_accuracy"
        ] = float(

            checkpoint[
                "best_valid_accuracy"
            ]

        )


    # --------------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------------

    if "architecture" in checkpoint:

        result[
            "checkpoint_architecture"
        ] = str(
            checkpoint["architecture"]
        )


# ============================================================
# ADD CLASS INFORMATION
# ============================================================

if info is not None:

    result[
        "class_information"
    ] = info


# ============================================================
# SAVE JSON
# ============================================================

result_path = os.path.join(

    OUTPUT_DIR,

    "prediction_result.json"

)


with open(

    result_path,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        result,

        file,

        indent=4,

        ensure_ascii=False

    )


# ============================================================
# SAVE TXT
# ============================================================

txt_path = os.path.join(

    OUTPUT_DIR,

    "prediction_result.txt"

)


with open(

    txt_path,

    "w",

    encoding="utf-8"

) as file:

    file.write(
        "CAPSICUM PLANT CONDITION PREDICTION\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"Image           : {IMAGE_PATH}\n"
    )

    file.write(
        f"Predicted Class : {predicted_class}\n"
    )

    file.write(
        f"Class Index     : {predicted_index}\n"
    )

    file.write(
        f"Confidence      : "
        f"{predicted_confidence:.2f}%\n"
    )

    file.write(
        f"Confidence Level: "
        f"{confidence_level}\n"
    )

    file.write(
        f"Device          : {device}\n"
    )

    file.write(
        f"Number Classes  : {NUM_CLASSES}\n"
    )

    file.write(
        "\nTOP PREDICTIONS\n"
    )

    file.write(
        "=" * 70 + "\n"
    )


    for item in top_predictions:

        file.write(

            f"{item['rank']}. "
            f"{item['class']} - "
            f"{item['confidence_percent']:.2f}%\n"

        )


    if info is not None:

        file.write(
            "\nCLASS INFORMATION\n"
        )

        file.write(
            "=" * 70 + "\n"
        )


        if "type" in info:

            file.write(
                f"Type       : "
                f"{info['type']}\n"
            )


        if "insect" in info:

            file.write(
                f"Insect     : "
                f"{info['insect']}\n"
            )


        if "disease" in info:

            file.write(
                f"Disease    : "
                f"{info['disease']}\n"
            )


        if "deficiency" in info:

            file.write(
                f"Deficiency : "
                f"{info['deficiency']}\n"
            )


        if "symptoms" in info:

            file.write(
                "\nSymptoms:\n"
            )


            symptoms = info["symptoms"]


            if isinstance(
                symptoms,
                (list, tuple)
            ):

                for symptom in symptoms:

                    file.write(
                        f"- {symptom}\n"
                    )

            else:

                file.write(
                    f"- {symptoms}\n"
                )


        if "cause" in info:

            file.write(
                "\nCause:\n"
            )


            cause = info["cause"]


            if isinstance(
                cause,
                (list, tuple)
            ):

                for item in cause:

                    file.write(
                        f"- {item}\n"
                    )

            else:

                file.write(
                    f"- {cause}\n"
                )


        if "precaution" in info:

            file.write(
                "\nPrecautions:\n"
            )


            precautions = info["precaution"]


            if isinstance(
                precautions,
                (list, tuple)
            ):

                for precaution in precautions:

                    file.write(
                        f"- {precaution}\n"
                    )

            else:

                file.write(
                    f"- {precautions}\n"
                )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()

print("=" * 90)

print(
    "PREDICTION COMPLETED SUCCESSFULLY"
)

print("=" * 90)

print()

print(
    f"Image      : "
    f"{os.path.basename(IMAGE_PATH)}"
)

print(
    f"Prediction : "
    f"{predicted_class}"
)

print(
    f"Confidence : "
    f"{predicted_confidence:.2f}%"
)

print(
    f"Level      : "
    f"{confidence_level}"
)

print(
    "Model      : ResNet50"
)

print(
    f"Classes    : {NUM_CLASSES}"
)

print(
    f"Device     : {device}"
)

print()

print(
    "JSON result:"
)

print(
    result_path
)

print()

print(
    "TXT result:"
)

print(
    txt_path
)

print()

print("=" * 90)