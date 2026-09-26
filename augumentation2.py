import os
import shutil
import hashlib
import random
import json
from collections import defaultdict

from PIL import Image, ImageFile
import imagehash

from torchvision import transforms


# ============================================================
# PIL SETTINGS
# ============================================================

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_DIR = os.path.join(BASE_DIR, "data new")

SPLIT_DIR = os.path.join(BASE_DIR, "dataset_split")

OUTPUT_DIR = os.path.join(BASE_DIR, "augmented_dataset")

REPORT_DIR = os.path.join(BASE_DIR, "dataset_reports")


# ============================================================
# DATASET CLASSES
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


# ============================================================
# DATASET CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

TRAIN_TARGET = 1100

RANDOM_SEED = 42

# pHash distance threshold
# Lower = stricter
# 8 is a reasonable starting point
PHASH_THRESHOLD = 8


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
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# SHA-256 HASH
# ============================================================

def calculate_sha256(file_path):

    sha256 = hashlib.sha256()

    try:

        with open(file_path, "rb") as file:

            while True:

                data = file.read(1024 * 1024)

                if not data:
                    break

                sha256.update(data)

        return sha256.hexdigest()

    except Exception:

        return None


# ============================================================
# VALID IMAGE CHECK
# ============================================================

def is_valid_image(file_path):

    try:

        with Image.open(file_path) as img:

            img.verify()

        return True

    except Exception:

        return False


# ============================================================
# COLLECT IMAGES
# ============================================================

def collect_images():

    class_images = defaultdict(list)

    print("\n" + "=" * 70)
    print("COLLECTING DATASET IMAGES")
    print("=" * 70)

    for class_name in CLASSES:

        class_dir = os.path.join(SOURCE_DIR, class_name)

        if not os.path.exists(class_dir):

            print(f"WARNING: Class folder not found: {class_name}")

            continue

        for root, _, files in os.walk(class_dir):

            for file_name in files:

                if file_name.lower().endswith(IMAGE_EXTENSIONS):

                    file_path = os.path.join(root, file_name)

                    class_images[class_name].append(file_path)

        print(
            f"{class_name:<25} : "
            f"{len(class_images[class_name])}"
        )

    return class_images


# ============================================================
# STEP 1
# EXACT DUPLICATE DETECTION
# SHA-256
# ============================================================

def find_exact_duplicates(class_images):

    print("\n" + "=" * 70)
    print("STEP 1: EXACT DUPLICATE DETECTION")
    print("Using SHA-256")
    print("=" * 70)

    hash_to_file = {}

    unique_images = defaultdict(list)

    duplicate_images = []

    invalid_images = []

    total_images = 0

    for class_name in CLASSES:

        images = class_images[class_name]

        for file_path in images:

            total_images += 1

            # --------------------------------------------
            # Validate image
            # --------------------------------------------

            if not is_valid_image(file_path):

                invalid_images.append(file_path)

                continue

            # --------------------------------------------
            # SHA-256
            # --------------------------------------------

            file_hash = calculate_sha256(file_path)

            if file_hash is None:

                invalid_images.append(file_path)

                continue

            # --------------------------------------------
            # Duplicate check
            # --------------------------------------------

            if file_hash in hash_to_file:

                original_file = hash_to_file[file_hash]

                duplicate_images.append({
                    "duplicate": file_path,
                    "original": original_file,
                    "sha256": file_hash,
                    "class": class_name
                })

            else:

                hash_to_file[file_hash] = file_path

                unique_images[class_name].append(file_path)

    print("\nOriginal images :", total_images)

    print(
        "Exact duplicates:",
        len(duplicate_images)
    )

    print(
        "Invalid images  :",
        len(invalid_images)
    )

    print(
        "Unique images   :",
        sum(len(v) for v in unique_images.values())
    )

    # --------------------------------------------
    # Save report
    # --------------------------------------------

    report = {
        "total_images": total_images,
        "exact_duplicates": len(duplicate_images),
        "invalid_images": len(invalid_images),
        "unique_images": sum(
            len(v) for v in unique_images.values()
        ),
        "duplicates": duplicate_images,
        "invalid_files": invalid_images
    }

    report_path = os.path.join(
        REPORT_DIR,
        "exact_duplicate_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print(
        "\nExact duplicate report:",
        report_path
    )

    return unique_images


# ============================================================
# PHASH CALCULATION
# ============================================================

def calculate_phash(file_path):

    try:

        with Image.open(file_path) as img:

            img = img.convert("RGB")

            return imagehash.phash(img)

    except Exception:

        return None


# ============================================================
# STEP 2
# NEAR DUPLICATE DETECTION
# ============================================================

def find_near_duplicates(unique_images):

    print("\n" + "=" * 70)
    print("STEP 2: NEAR-DUPLICATE DETECTION")
    print("Using Perceptual Hash (pHash)")
    print("=" * 70)

    all_images = []

    for class_name in CLASSES:

        for file_path in unique_images[class_name]:

            all_images.append(
                (
                    class_name,
                    file_path
                )
            )

    print(
        "\nImages for pHash analysis:",
        len(all_images)
    )

    # --------------------------------------------------------
    # Calculate pHash
    # --------------------------------------------------------

    hash_data = []

    for index, (class_name, file_path) in enumerate(all_images):

        phash = calculate_phash(file_path)

        if phash is not None:

            hash_data.append({
                "index": index,
                "class": class_name,
                "path": file_path,
                "hash": phash
            })

        if (index + 1) % 500 == 0:

            print(
                f"pHash calculated: "
                f"{index + 1}/{len(all_images)}"
            )

    print(
        "\nStarting near-duplicate comparison..."
    )

    near_duplicates = []

    total_comparisons = 0

    # --------------------------------------------------------
    # Compare pHash
    # --------------------------------------------------------

    for i in range(len(hash_data)):

        item_a = hash_data[i]

        for j in range(i + 1, len(hash_data)):

            item_b = hash_data[j]

            total_comparisons += 1

            distance = item_a["hash"] - item_b["hash"]

            if distance <= PHASH_THRESHOLD:

                near_duplicates.append({
                    "image_1": item_a["path"],
                    "class_1": item_a["class"],
                    "image_2": item_b["path"],
                    "class_2": item_b["class"],
                    "phash_distance": distance,
                    "cross_class": (
                        item_a["class"] != item_b["class"]
                    )
                })

    print(
        "\nTotal comparisons:",
        total_comparisons
    )

    print(
        "Near-duplicate pairs:",
        len(near_duplicates)
    )

    cross_class = [
        item
        for item in near_duplicates
        if item["cross_class"]
    ]

    print(
        "Cross-class near duplicates:",
        len(cross_class)
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report = {
        "phash_threshold": PHASH_THRESHOLD,
        "total_images": len(hash_data),
        "total_comparisons": total_comparisons,
        "near_duplicate_pairs": len(near_duplicates),
        "cross_class_near_duplicates": len(cross_class),
        "pairs": near_duplicates
    }

    report_path = os.path.join(
        REPORT_DIR,
        "near_duplicate_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print(
        "\nNear-duplicate report:",
        report_path
    )

    return hash_data, near_duplicates


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:

    def __init__(self, n):

        self.parent = list(range(n))

    def find(self, x):

        while self.parent[x] != x:

            self.parent[x] = self.parent[
                self.parent[x]
            ]

            x = self.parent[x]

        return x

    def union(self, a, b):

        root_a = self.find(a)

        root_b = self.find(b)

        if root_a != root_b:

            self.parent[root_b] = root_a


# ============================================================
# STEP 3
# CREATE NEAR-DUPLICATE GROUPS
# ============================================================

def create_near_duplicate_groups(
    hash_data,
    near_duplicates
):

    print("\n" + "=" * 70)
    print("STEP 3: CREATING NEAR-DUPLICATE GROUPS")
    print("=" * 70)

    path_to_index = {}

    for index, item in enumerate(hash_data):

        path_to_index[item["path"]] = index

    uf = UnionFind(len(hash_data))

    for pair in near_duplicates:

        image_1 = pair["image_1"]

        image_2 = pair["image_2"]

        index_1 = path_to_index[image_1]

        index_2 = path_to_index[image_2]

        uf.union(
            index_1,
            index_2
        )

    groups = defaultdict(list)

    for index, item in enumerate(hash_data):

        root = uf.find(index)

        groups[root].append(item)

    # --------------------------------------------------------
    # Convert to list
    # --------------------------------------------------------

    final_groups = []

    for group_id, members in groups.items():

        final_groups.append({
            "group_id": len(final_groups),
            "images": members
        })

    grouped_count = sum(
        1
        for group in final_groups
        if len(group["images"]) > 1
    )

    print(
        "\nTotal groups:",
        len(final_groups)
    )

    print(
        "Groups containing near duplicates:",
        grouped_count
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_data = []

    for group in final_groups:

        if len(group["images"]) > 1:

            report_data.append({
                "group_id": group["group_id"],
                "images": [
                    {
                        "path": item["path"],
                        "class": item["class"]
                    }
                    for item in group["images"]
                ]
            })

    report_path = os.path.join(
        REPORT_DIR,
        "near_duplicate_groups.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report_data,
            file,
            indent=4
        )

    print(
        "\nGroup report:",
        report_path
    )

    return final_groups


# ============================================================
# STEP 4
# GROUP-WISE 80/10/10 SPLIT
# ============================================================

def split_groups(
    groups
):

    print("\n" + "=" * 70)
    print("STEP 4: CLEAN 80/10/10 GROUP-WISE SPLIT")
    print("ONE NEAR-DUPLICATE GROUP = ONE SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Remove previous split
    # --------------------------------------------------------

    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    for split_name in [
        "train",
        "valid",
        "test"
    ]:
        for class_name in CLASSES:
            os.makedirs(
                os.path.join(
                    SPLIT_DIR,
                    split_name,
                    class_name
                ),
                exist_ok=True
            )

    # --------------------------------------------------------
    # IMPORTANT:
    # A near-duplicate group must be assigned exactly once.
    #
    # This is especially important for cross-class groups.
    # The complete group stays in the same split, while each
    # image is still copied into its own original class folder.
    # --------------------------------------------------------

    all_groups = list(groups)
    random.shuffle(all_groups)

    total_images = sum(
        len(group["images"])
        for group in all_groups
    )

    target_train = int(total_images * TRAIN_RATIO)
    target_valid = int(total_images * VALID_RATIO)

    train_groups = []
    valid_groups = []
    test_groups = []

    train_count = 0
    valid_count = 0
    test_count = 0

    # --------------------------------------------------------
    # Greedy group-wise 80/10/10 assignment
    # --------------------------------------------------------

    for group in all_groups:

        group_size = len(group["images"])

        # Prefer train until its target is reached.
        if train_count + group_size <= target_train:
            train_groups.append(group)
            train_count += group_size

        # Then validation.
        elif valid_count + group_size <= target_valid:
            valid_groups.append(group)
            valid_count += group_size

        # Remaining groups go to test.
        else:
            test_groups.append(group)
            test_count += group_size

    # --------------------------------------------------------
    # Copy groups
    # --------------------------------------------------------

    def copy_groups(
        group_list,
        split_name
    ):

        copied_by_class = defaultdict(int)

        for group in group_list:

            for item in group["images"]:

                source = item["path"]
                item_class = item["class"]

                destination_dir = os.path.join(
                    SPLIT_DIR,
                    split_name,
                    item_class
                )

                os.makedirs(
                    destination_dir,
                    exist_ok=True
                )

                destination = os.path.join(
                    destination_dir,
                    os.path.basename(source)
                )

                # Avoid filename collision
                if os.path.exists(destination):

                    base, ext = os.path.splitext(
                        os.path.basename(source)
                    )

                    counter = 1
                    while True:

                        candidate = os.path.join(
                            destination_dir,
                            f"{base}_{counter}{ext}"
                        )

                        if not os.path.exists(candidate):
                            destination = candidate
                            break

                        counter += 1

                shutil.copy2(
                    source,
                    destination
                )

                copied_by_class[item_class] += 1

        return dict(copied_by_class)

    train_by_class = copy_groups(
        train_groups,
        "train"
    )

    valid_by_class = copy_groups(
        valid_groups,
        "valid"
    )

    test_by_class = copy_groups(
        test_groups,
        "test"
    )

    # --------------------------------------------------------
    # Per-class summary
    # --------------------------------------------------------

    split_summary = {}

    for class_name in CLASSES:

        class_train = train_by_class.get(
            class_name,
            0
        )

        class_valid = valid_by_class.get(
            class_name,
            0
        )

        class_test = test_by_class.get(
            class_name,
            0
        )

        class_total = (
            class_train +
            class_valid +
            class_test
        )

        split_summary[class_name] = {
            "total": class_total,
            "train": class_train,
            "valid": class_valid,
            "test": class_test
        }

        print(
            f"{class_name:<25} "
            f"Total={class_total:<5} "
            f"Train={class_train:<5} "
            f"Valid={class_valid:<5} "
            f"Test={class_test:<5}"
        )

    print(
        "\nGlobal split totals: "
        f"Train={train_count}, "
        f"Valid={valid_count}, "
        f"Test={test_count}, "
        f"Total={total_images}"
    )

    # --------------------------------------------------------
    # Save split report
    # --------------------------------------------------------

    report_path = os.path.join(
        REPORT_DIR,
        "split_summary.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "global": {
                    "total": total_images,
                    "train": train_count,
                    "valid": valid_count,
                    "test": test_count,
                    "train_ratio_actual": (
                        train_count / total_images
                        if total_images else 0
                    ),
                    "valid_ratio_actual": (
                        valid_count / total_images
                        if total_images else 0
                    ),
                    "test_ratio_actual": (
                        test_count / total_images
                        if total_images else 0
                    )
                },
                "per_class": split_summary
            },
            file,
            indent=4
        )

    return split_summary


# ============================================================
# AUGMENTATION TRANSFORMS
# ============================================================

def get_strong_transform():

    return transforms.Compose([

        transforms.RandomResizedCrop(
            224,
            scale=(0.65, 1.0),
            ratio=(0.85, 1.15)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomVerticalFlip(
            p=0.3
        ),

        transforms.RandomRotation(
            degrees=30
        ),

        transforms.ColorJitter(
            brightness=0.30,
            contrast=0.30,
            saturation=0.30,
            hue=0.08
        )
    ])


def get_moderate_transform():

    return transforms.Compose([

        transforms.RandomResizedCrop(
            224,
            scale=(0.75, 1.0),
            ratio=(0.90, 1.10)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=20
        ),

        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.20,
            hue=0.05
        )
    ])


def get_light_transform():

    return transforms.Compose([

        transforms.Resize(
            (224, 224)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=10
        ),

        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.10,
            hue=0.02
        )
    ])


# ============================================================
# AUGMENTATION STRATEGY
# ============================================================

def get_augmentation_transform(
    train_count
):

    if train_count < 300:

        return (
            get_strong_transform(),
            "STRONG"
        )

    elif train_count <= 800:

        return (
            get_moderate_transform(),
            "MODERATE"
        )

    else:

        return (
            get_light_transform(),
            "LIGHT"
        )


# ============================================================
# SAVE AUGMENTED IMAGE
# ============================================================

def save_augmented_image(
    image,
    destination
):

    image = image.convert("RGB")

    image.save(
        destination,
        quality=95
    )


# ============================================================
# STEP 5
# TRAIN-ONLY AUGMENTATION
# ============================================================

def augment_training_data():

    print("\n" + "=" * 70)
    print("STEP 5: TRAIN-ONLY AUGMENTATION")
    print(f"TARGET = {TRAIN_TARGET} IMAGES / CLASS")
    print("=" * 70)

    train_source = os.path.join(
        SPLIT_DIR,
        "train"
    )

    train_output = os.path.join(
        OUTPUT_DIR,
        "train"
    )

    # --------------------------------------------------------
    # Remove previous augmented train
    # --------------------------------------------------------

    if os.path.exists(train_output):

        shutil.rmtree(train_output)

    # --------------------------------------------------------
    # Create class directories
    # --------------------------------------------------------

    for class_name in CLASSES:

        os.makedirs(
            os.path.join(
                train_output,
                class_name
            ),
            exist_ok=True
        )

    augmentation_summary = {}

    # --------------------------------------------------------
    # Process classes
    # --------------------------------------------------------

    for class_name in CLASSES:

        source_class_dir = os.path.join(
            train_source,
            class_name
        )

        output_class_dir = os.path.join(
            train_output,
            class_name
        )

        images = []

        if os.path.exists(
            source_class_dir
        ):

            for file_name in os.listdir(
                source_class_dir
            ):

                if file_name.lower().endswith(
                    IMAGE_EXTENSIONS
                ):

                    images.append(
                        os.path.join(
                            source_class_dir,
                            file_name
                        )
                    )

        original_count = len(images)

        # ----------------------------------------------------
        # Select augmentation strength
        # ----------------------------------------------------

        transform, strength = (
            get_augmentation_transform(
                original_count
            )
        )

        print(
            f"\n{class_name}"
        )

        print(
            "Original train:",
            original_count
        )

        print(
            "Augmentation:",
            strength
        )

        # ----------------------------------------------------
        # Copy ORIGINAL train images
        # ----------------------------------------------------

        for index, source in enumerate(images):

            destination = os.path.join(
                output_class_dir,
                f"original_{index:05d}"
                + os.path.splitext(source)[1]
            )

            shutil.copy2(
                source,
                destination
            )

        current_count = original_count

        # ----------------------------------------------------
        # No augmentation needed
        # ----------------------------------------------------

        if current_count >= TRAIN_TARGET:

            augmentation_summary[class_name] = {
                "original_train": original_count,
                "augmentation": "NONE",
                "generated": 0,
                "final_train": current_count
            }

            continue

        # ----------------------------------------------------
        # Generate augmented images
        # ----------------------------------------------------

        if not images:

            print(
                "WARNING: No valid training images found "
                f"for class: {class_name}"
            )

            augmentation_summary[class_name] = {
                "original_train": 0,
                "augmentation": strength,
                "generated": 0,
                "final_train": 0,
                "status": "FAILED_NO_SOURCE_IMAGES"
            }

            continue

        generated = 0
        failed_attempts = 0
        max_failed_attempts = max(
            100,
            (TRAIN_TARGET - original_count) * 10
        )

        while current_count < TRAIN_TARGET:

            source = random.choice(
                images
            )

            try:

                with Image.open(source) as img:

                    img = img.convert(
                        "RGB"
                    )

                    augmented = transform(
                        img
                    )

                    output_name = (
                        f"augmented_"
                        f"{generated:05d}.jpg"
                    )

                    output_path = os.path.join(
                        output_class_dir,
                        output_name
                    )

                    save_augmented_image(
                        augmented,
                        output_path
                    )

                    generated += 1

                    current_count += 1

            except Exception as error:

                failed_attempts += 1

                print(
                    "Augmentation error:",
                    source,
                    error
                )

                if failed_attempts >= max_failed_attempts:

                    print(
                        "WARNING: Too many augmentation failures "
                        f"for class: {class_name}"
                    )

                    break

        print(
            "Generated:",
            generated
        )

        print(
            "Final train:",
            current_count
        )

        augmentation_summary[class_name] = {
            "original_train": original_count,
            "augmentation": strength,
            "generated": generated,
            "final_train": current_count,
            "status": (
                "PASS"
                if current_count == TRAIN_TARGET
                else "INCOMPLETE"
            )
        }

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_path = os.path.join(
        REPORT_DIR,
        "augmentation_summary.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            augmentation_summary,
            file,
            indent=4
        )

    return augmentation_summary


# ============================================================
# VALIDATION / TEST
# NO AUGMENTATION
# ============================================================

def prepare_validation_and_test():

    print("\n" + "=" * 70)
    print("PREPARING VALIDATION AND TEST")
    print("NO RANDOM AUGMENTATION")
    print("=" * 70)

    for split_name in [
        "valid",
        "test"
    ]:

        source_dir = os.path.join(
            SPLIT_DIR,
            split_name
        )

        output_dir = os.path.join(
            OUTPUT_DIR,
            split_name
        )

        if os.path.exists(output_dir):

            shutil.rmtree(
                output_dir
            )

        shutil.copytree(
            source_dir,
            output_dir
        )

        print(
            f"{split_name.upper()} copied unchanged."
        )


# ============================================================
# SHA-256 SPLIT LEAKAGE CHECK
# ============================================================

def verify_final_split_leakage():

    print("\n" + "=" * 70)
    print("FINAL EXACT DUPLICATE LEAKAGE CHECK")
    print("=" * 70)

    split_hashes = {
        "train": {},
        "valid": {},
        "test": {}
    }

    for split_name in [
        "train",
        "valid",
        "test"
    ]:

        split_dir = os.path.join(
            OUTPUT_DIR,
            split_name
        )

        for class_name in CLASSES:

            class_dir = os.path.join(
                split_dir,
                class_name
            )

            if not os.path.exists(
                class_dir
            ):

                continue

            for file_name in os.listdir(
                class_dir
            ):

                file_path = os.path.join(
                    class_dir,
                    file_name
                )

                if not file_name.lower().endswith(
                    IMAGE_EXTENSIONS
                ):

                    continue

                file_hash = calculate_sha256(
                    file_path
                )

                if file_hash:

                    split_hashes[
                        split_name
                    ][file_hash] = file_path

    train_hashes = set(
        split_hashes["train"]
    )

    valid_hashes = set(
        split_hashes["valid"]
    )

    test_hashes = set(
        split_hashes["test"]
    )

    train_valid = (
        train_hashes &
        valid_hashes
    )

    train_test = (
        train_hashes &
        test_hashes
    )

    valid_test = (
        valid_hashes &
        test_hashes
    )

    print(
        "Train ∩ Valid:",
        len(train_valid)
    )

    print(
        "Train ∩ Test :",
        len(train_test)
    )

    print(
        "Valid ∩ Test :",
        len(valid_test)
    )

    if (
        len(train_valid) == 0
        and
        len(train_test) == 0
        and
        len(valid_test) == 0
    ):

        print(
            "\nPASS: No exact duplicate leakage."
        )

        return True

    print(
        "\nWARNING: Exact duplicate leakage found."
    )

    return False


# ============================================================
# FINAL DATASET COUNT
# ============================================================

def verify_final_dataset():

    print("\n" + "=" * 70)
    print("FINAL DATASET VERIFICATION")
    print("=" * 70)

    all_pass = True

    for split_name in [
        "train",
        "valid",
        "test"
    ]:

        print(
            f"\n--- {split_name.upper()} ---"
        )

        split_dir = os.path.join(
            OUTPUT_DIR,
            split_name
        )

        for class_name in CLASSES:

            class_dir = os.path.join(
                split_dir,
                class_name
            )

            if not os.path.exists(
                class_dir
            ):

                count = 0

            else:

                count = sum(
                    1
                    for file_name
                    in os.listdir(class_dir)
                    if file_name.lower().endswith(
                        IMAGE_EXTENSIONS
                    )
                )

            print(
                f"{class_name:<25}: {count}"
            )

            if (
                split_name == "train"
                and count != TRAIN_TARGET
            ):

                all_pass = False

    if all_pass:

        print(
            "\nPASS: Every training class has 1100 images."
        )

    else:

        print(
            "\nWARNING: Training count mismatch."
        )

    return all_pass


# ============================================================
# TRAIN EXACT DUPLICATE CHECK
# ============================================================

def check_train_duplicates():

    print("\n" + "=" * 70)
    print("TRAIN INTERNAL EXACT DUPLICATE CHECK")
    print("=" * 70)

    train_dir = os.path.join(
        OUTPUT_DIR,
        "train"
    )

    for class_name in CLASSES:

        class_dir = os.path.join(
            train_dir,
            class_name
        )

        hashes = {}

        duplicates = 0

        if not os.path.exists(
            class_dir
        ):

            continue

        for file_name in os.listdir(
            class_dir
        ):

            if not file_name.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                continue

            file_path = os.path.join(
                class_dir,
                file_name
            )

            file_hash = calculate_sha256(
                file_path
            )

            if file_hash in hashes:

                duplicates += 1

            else:

                hashes[file_hash] = file_path

        print(
            f"{class_name:<25}: "
            f"duplicates = {duplicates}"
        )


# ============================================================
# CREATE FINAL SUMMARY
# ============================================================

def save_final_summary(
    split_summary,
    augmentation_summary
):

    summary = {

        "dataset": {
            "source": SOURCE_DIR,
            "classes": CLASSES,
            "number_of_classes": len(CLASSES)
        },

        "split": {
            "train_ratio": TRAIN_RATIO,
            "valid_ratio": VALID_RATIO,
            "test_ratio": TEST_RATIO
        },

        "augmentation": {
            "train_target_per_class": TRAIN_TARGET,
            "train_only": True,
            "validation_augmented": False,
            "test_augmented": False,
            "strategy_based_on": "actual_train_count_after_split",
            "strong_if_train_count_below": 300,
            "moderate_if_train_count_300_to_800": True,
            "light_if_train_count_above_800": True
        },

        "duplicate_detection": {
            "exact_duplicate": "SHA-256",
            "near_duplicate": "pHash",
            "phash_threshold": PHASH_THRESHOLD,
            "same_group_same_split": True
        },

        "split_summary": split_summary,

        "augmentation_summary": augmentation_summary
    }

    path = os.path.join(
        REPORT_DIR,
        "FINAL_DATASET_SUMMARY.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print(
        "\nFinal summary:",
        path
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CAPSICUM DATASET CLEANING + AUGMENTATION PIPELINE")
    print("=" * 70)

    print(
        "\nClasses:",
        len(CLASSES)
    )

    print(
        "Training target:",
        TRAIN_TARGET,
        "images/class"
    )

    # ========================================================
    # STEP 1
    # ========================================================

    class_images = collect_images()

    unique_images = find_exact_duplicates(
        class_images
    )

    # ========================================================
    # STEP 2
    # ========================================================

    hash_data, near_duplicates = (
        find_near_duplicates(
            unique_images
        )
    )

    # ========================================================
    # STEP 3
    # ========================================================

    groups = create_near_duplicate_groups(
        hash_data,
        near_duplicates
    )

    # ========================================================
    # STEP 4
    # ========================================================

    split_summary = split_groups(
        groups
    )

    # ========================================================
    # STEP 5
    # ========================================================

    augmentation_summary = (
        augment_training_data()
    )

    # ========================================================
    # VALID / TEST
    # ========================================================

    prepare_validation_and_test()

    # ========================================================
    # FINAL CHECKS
    # ========================================================

    verify_final_split_leakage()

    verify_final_dataset()

    check_train_duplicates()

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    save_final_summary(
        split_summary,
        augmentation_summary
    )

    # ========================================================
    # DONE
    # ========================================================

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)

    print(
        "\nFinal dataset location:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nReports location:"
    )

    print(
        REPORT_DIR
    )

    print("\nFinal structure:")

    print(
        """
augmented_dataset/
│
├── train/
│   ├── Anthracnose/       → 1100
│   ├── Larva/             → 1100
│   ├── Magnesium/         → 1100
│   ├── bacterial spot/    → 1100
│   ├── blossom-end rot/   → 1100
│   ├── down leaf aphid/   → 1100
│   ├── fruit thrips/      → 1100
│   ├── healthy/           → 1100
│   ├── powdery mildew/    → 1100
│   ├── snail/             → 1100
│   ├── upperleaf thrips/  → 1100
│   └── virus/             → 1100
│
├── valid/
│   └── original unique images
│
└── test/
    └── original unique images
"""
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()