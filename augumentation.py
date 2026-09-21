import os


# ============================================================
# CAPSICUM DATASET IMAGE COUNT & VALIDATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "data new"
)


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
# EXPECTED CAPSICUM CLASSES
# ============================================================

EXPECTED_CLASSES = {
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
}


# ============================================================
# EXPECTED DATASET VALUES
# ============================================================

EXPECTED_CLASS_COUNT = 12
EXPECTED_IMAGE_COUNT = 6801


# ============================================================
# INITIALIZE COUNTERS
# ============================================================

total_images = 0
total_classes = 0

class_counts = {}


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 75)
print("CAPSICUM DATASET IMAGE COUNT & VALIDATION")
print("=" * 75)


# ============================================================
# DATASET LOCATION
# ============================================================

print("\nDataset Location:")
print(DATASET_DIR)


# ============================================================
# CHECK DATASET DIRECTORY
# ============================================================

if not os.path.isdir(DATASET_DIR):

    print("\n" + "!" * 75)
    print("ERROR: DATASET FOLDER NOT FOUND")
    print("!" * 75)

    print("\nExpected structure:")
    print("""
CNN CROPS/
│
├── data new/
│   ├── Anthracnose/
│   ├── Larva/
│   ├── Magnesium/
│   ├── bacterial spot/
│   ├── blossom-end rot/
│   ├── down leaf aphid/
│   ├── fruit thrips/
│   ├── healthy/
│   ├── powdery mildew/
│   ├── snail/
│   ├── upperleaf thrips/
│   └── virus/
│
└── count_dataset.py
""")

    raise SystemExit


# ============================================================
# FIND ACTUAL CLASSES
# ============================================================

actual_classes = set()

for class_name in os.listdir(DATASET_DIR):

    class_path = os.path.join(
        DATASET_DIR,
        class_name
    )

    if os.path.isdir(class_path):
        actual_classes.add(class_name)


# ============================================================
# CLASS-WISE IMAGE COUNT
# ============================================================

print("\n" + "-" * 75)
print("CLASS-WISE IMAGE COUNT")
print("-" * 75)

for class_name in sorted(actual_classes):

    class_path = os.path.join(
        DATASET_DIR,
        class_name
    )

    count = 0

    # --------------------------------------------------------
    # Recursively count images
    # --------------------------------------------------------

    for root, dirs, files in os.walk(class_path):

        for file_name in files:

            if file_name.lower().endswith(
                IMAGE_EXTENSIONS
            ):
                count += 1

    class_counts[class_name] = count

    total_classes += 1
    total_images += count

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if count == 0:
        status = "⚠ EMPTY"
    else:
        status = "✓"

    print(
        f"{class_name:<35} : "
        f"{count:>5}   {status}"
    )


# ============================================================
# TOTAL DATASET IMAGES
# ============================================================

print("\n" + "=" * 75)
print("DATASET TOTAL")
print("=" * 75)

print(
    f"{'TOTAL CLASSES':<35} : {total_classes}"
)

print(
    f"{'TOTAL IMAGES':<35} : {total_images}"
)

print(
    f"{'EXPECTED IMAGES':<35} : {EXPECTED_IMAGE_COUNT}"
)

print("=" * 75)


# ============================================================
# DATASET VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("DATASET VALIDATION")
print("=" * 75)


# ============================================================
# CLASS COUNT CHECK
# ============================================================

print("\n1. CLASS COUNT")

if total_classes == EXPECTED_CLASS_COUNT:

    print(
        f"✓ Classes found : {total_classes}"
    )

else:

    print(
        f"⚠ Classes found : {total_classes}"
    )

    print(
        f"  Expected      : {EXPECTED_CLASS_COUNT}"
    )


# ============================================================
# IMAGE COUNT CHECK
# ============================================================

print("\n2. TOTAL IMAGE COUNT")

if total_images == EXPECTED_IMAGE_COUNT:

    print(
        f"✓ Total images found : {total_images}"
    )

    print(
        f"✓ Expected total     : {EXPECTED_IMAGE_COUNT}"
    )

    print(
        "✓ Image count is correct."
    )

else:

    difference = total_images - EXPECTED_IMAGE_COUNT

    print(
        f"⚠ Total images found : {total_images}"
    )

    print(
        f"  Expected total     : {EXPECTED_IMAGE_COUNT}"
    )

    print(
        f"  Difference         : {difference:+d}"
    )


# ============================================================
# MISSING CLASSES
# ============================================================

missing_classes = EXPECTED_CLASSES - actual_classes

print("\n3. MISSING CLASSES")

if not missing_classes:

    print("✓ No expected classes are missing.")

else:

    for class_name in sorted(missing_classes):

        print(
            f"✗ {class_name}"
        )


# ============================================================
# UNEXPECTED CLASSES
# ============================================================

unexpected_classes = actual_classes - EXPECTED_CLASSES

print("\n4. UNEXPECTED CLASSES")

if not unexpected_classes:

    print("✓ No unexpected classes found.")

else:

    for class_name in sorted(unexpected_classes):

        print(
            f"⚠ {class_name}"
        )


# ============================================================
# EMPTY CLASSES
# ============================================================

empty_classes = [
    class_name
    for class_name, count in class_counts.items()
    if count == 0
]

print("\n5. EMPTY CLASSES")

if not empty_classes:

    print("✓ No empty classes found.")

else:

    for class_name in sorted(empty_classes):

        print(
            f"⚠ {class_name}"
        )


# ============================================================
# CLASS-WISE DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("CLASS-WISE DATASET DISTRIBUTION")
print("=" * 75)

if total_images > 0:

    print(
        f"\n{'Class':<35}"
        f"{'Images':>10}"
        f"{'Percentage':>15}"
    )

    print("-" * 65)

    for class_name in sorted(class_counts):

        count = class_counts[class_name]

        percentage = (
            count / total_images
        ) * 100

        print(
            f"{class_name:<35}"
            f"{count:>10}"
            f"{percentage:>14.2f}%"
        )

    print("-" * 65)

    # --------------------------------------------------------
    # FINAL TOTAL
    # --------------------------------------------------------

    print(
        f"{'TOTAL':<35}"
        f"{total_images:>10}"
        f"{100.00:>14.2f}%"
    )


# ============================================================
# FINAL DATASET STATUS
# ============================================================

dataset_valid = (
    total_classes == EXPECTED_CLASS_COUNT
    and total_images == EXPECTED_IMAGE_COUNT
    and not missing_classes
    and not unexpected_classes
    and not empty_classes
)


print("\n" + "=" * 75)
print("FINAL DATASET STATUS")
print("=" * 75)

if dataset_valid:

    print("\n✓ DATASET VALIDATION PASSED")

    print(
        f"✓ Total Classes : {total_classes}"
    )

    print(
        f"✓ Total Images  : {total_images}"
    )

    print(
        "✓ All expected classes are present"
    )

    print(
        "✓ No unexpected classes found"
    )

    print(
        "✓ No empty classes found"
    )

else:

    print("\n⚠ DATASET VALIDATION FAILED")

    print(
        "Please check the class folders and image counts."
    )


# ============================================================
# FINAL CLASS LIST
# ============================================================

print("\n" + "=" * 75)
print("FINAL CAPSICUM CLASSES")
print("=" * 75)

for index, class_name in enumerate(
    sorted(actual_classes),
    start=1
):

    print(
        f"{index:02d}. {class_name}"
    )


# ============================================================
# REMOVED CLASSES
# ============================================================

print("\n" + "=" * 75)
print("REMOVED CLASSES")
print("=" * 75)

print("01. Nitrogen deficiency")
print("02. recover powdery mildew")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL SUMMARY")
print("=" * 75)

print(f"Total Classes : {total_classes}")
print(f"Total Images  : {total_images}")
print(f"Expected      : {EXPECTED_IMAGE_COUNT}")

if total_images == EXPECTED_IMAGE_COUNT:
    print("Status        : ✓ EXACT IMAGE COUNT MATCH")
else:
    print("Status        : ⚠ IMAGE COUNT MISMATCH")

print("=" * 75)

print("\nDATASET COUNTING & VALIDATION COMPLETED")