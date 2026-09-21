# ============================================================
# CAPSICUM CLASS INFORMATION
# 12 SUPPORTED CLASSES
# ============================================================

CLASS_INFO = {

    # ========================================================
    # 1. ANTHRACNOSE
    # ========================================================

    "Anthracnose": {
        "type": "Disease",
        "disease": "Anthracnose",

        "symptoms": [
            "Fruit par dark aur dhanse hue lesions (daag) dikh sakte hain.",
            "Samay ke saath ye lesions dheere-dheere badh sakte hain."
        ],

        "cause": [
            "Ye ek fungal infection ki wajah se hota hai."
        ],

        "precaution": [
            "Bahut zyada infected plant parts ko remove karein.",
            "Field aur crop ki proper safai rakhein.",
            "Leaves aur fruits par lambe samay tak moisture na rehne dein.",
            "Plants ko regularly monitor karein."
        ]
    },


    # ========================================================
    # 2. LARVA
    # ========================================================

    "Larva": {
        "type": "Insect Pest",
        "insect": "Larva",

        "symptoms": [
            "Plant par visible larvae (keede ke larvae) dikh sakte hain.",
            "Leaves par feeding ke wajah se holes ya damage dikh sakta hai.",
            "Leaves par irregular feeding marks ho sakte hain."
        ],

        "cause": [
            "Insect larvae ke infestation ki wajah se damage hota hai."
        ],

        "precaution": [
            "Plants ko regularly check karein.",
            "Visible larvae ko manually remove karein.",
            "Field ko clean aur hygienic rakhein.",
            "Suitable integrated pest management ka use karein."
        ]
    },


    # ========================================================
    # 3. MAGNESIUM
    # ========================================================

    "Magnesium": {
        "type": "Nutrient Deficiency",
        "deficiency": "Magnesium deficiency",

        "symptoms": [
            "Purane leaves par veins ke beech ka area yellow ya pale ho sakta hai.",
            "Leaf veins comparatively green reh sakti hain."
        ],

        "cause": [
            "Plant ko sufficient magnesium available nahi hone ki wajah se deficiency hoti hai.",
            "Soil mein nutrient imbalance bhi magnesium uptake ko affect kar sakta hai."
        ],

        "precaution": [
            "Soil ka nutrient status check karein.",
            "Crop ke liye balanced nutrition maintain karein.",
            "Soil test ke according magnesium apply karein.",
            "Purane leaves ko regularly monitor karein."
        ]
    },


    # ========================================================
    # 4. BACTERIAL SPOT
    # ========================================================

    "bacterial spot": {
        "type": "Disease",
        "disease": "Bacterial spot",

        "symptoms": [
            "Leaves par chhote dark spots ya daag dikh sakte hain.",
            "Favorable conditions mein spots dheere-dheere badh sakte hain.",
            "Leaves par irregular spotting दिखाई de sakti hai."
        ],

        "cause": [
            "Ye bacterial infection ki wajah se hota hai."
        ],

        "precaution": [
            "Severely infected plant parts ko remove karein.",
            "Field aur crop ki proper sanitation maintain karein.",
            "Unnecessary overhead irrigation avoid karein.",
            "Plants ko regularly monitor karein."
        ]
    },


    # ========================================================
    # 5. BLOSSOM-END ROT
    # ========================================================

    "blossom-end rot": {
        "type": "Physiological Disorder",
        "disease": "Blossom-end rot",

        "symptoms": [
            "Fruit ke blossom-end yani neeche wale end par dark aur dhansa hua area develop ho sakta hai.",
            "Affected area dry aur damaged ho sakta hai."
        ],

        "cause": [
            "Ye developing fruit mein calcium availability se associated ho sakta hai.",
            "Soil moisture mein irregularity bhi is condition ko badha sakti hai."
        ],

        "precaution": [
            "Soil moisture ko consistent rakhein.",
            "Soil ko bahut dry ya bahut wet hone se bachayein.",
            "Soil nutrient status monitor karein.",
            "Balanced crop nutrition maintain karein."
        ]
    },


    # ========================================================
    # 6. DOWN LEAF APHID
    # ========================================================

    "down leaf aphid": {
        "type": "Insect Pest",
        "insect": "Aphid",

        "symptoms": [
            "Leaves par chhote aphids dikh sakte hain.",
            "Leaves curl ya distort ho sakte hain.",
            "Affected leaves ki growth abnormal ho sakti hai."
        ],

        "cause": [
            "Aphid infestation ki wajah se problem hoti hai."
        ],

        "precaution": [
            "Leaves ko regularly inspect karein.",
            "New plant growth ko frequently check karein.",
            "Field ko clean rakhein.",
            "Suitable integrated pest management practices follow karein."
        ]
    },


    # ========================================================
    # 7. FRUIT THRIPS
    # ========================================================

    "fruit thrips": {
        "type": "Insect Pest",
        "insect": "Fruit thrips",

        "symptoms": [
            "Developing fruits par feeding damage dikh sakta hai.",
            "Fruit surface par scars ya discoloration ho sakta hai.",
            "Fruit par feeding marks दिखाई de sakte hain."
        ],

        "cause": [
            "Thrips infestation ki wajah se damage hota hai."
        ],

        "precaution": [
            "Flowers aur developing fruits ko regularly check karein.",
            "Early stage par thrips activity monitor karein.",
            "Field hygiene maintain karein.",
            "Suitable integrated pest management ka use karein."
        ]
    },


    # ========================================================
    # 8. HEALTHY
    # ========================================================

    "healthy": {
        "type": "Healthy",

        "symptoms": [
            "Plant par koi obvious disease ya pest symptom दिखाई nahi de raha hai.",
            "Leaves aur plant parts generally healthy dikh rahe hain."
        ],

        "cause": [
            "Plant ki condition healthy hai aur target disease ya pest detect nahi hua."
        ],

        "precaution": [
            "Crop ko regularly monitor karte rahein.",
            "Proper irrigation maintain karein.",
            "Balanced plant nutrition maintain karein.",
            "Disease aur pest ke early symptoms ko regularly check karein."
        ]
    },


    # ========================================================
    # 9. POWDERY MILDEW
    # ========================================================

    "powdery mildew": {
        "type": "Disease",
        "disease": "Powdery mildew",

        "symptoms": [
            "Leaves ki surface par white powder jaisa fungal growth dikh sakta hai.",
            "Affected leaves par discoloration ya weak growth ho sakti hai."
        ],

        "cause": [
            "Ye fungal infection ki wajah se hota hai."
        ],

        "precaution": [
            "Plants ke beech proper air circulation maintain karein.",
            "Severely affected leaves ya plant material ko remove karein.",
            "Leaves ke around excessive humidity ko avoid karein.",
            "New growth ko regularly monitor karein."
        ]
    },


    # ========================================================
    # 10. SNAIL
    # ========================================================

    "snail": {
        "type": "Pest",
        "insect": "Snail",

        "symptoms": [
            "Leaves par irregular feeding holes dikh sakte hain.",
            "Young plant parts par chewing damage ho sakta hai.",
            "Plant ke aas-paas mucus/slime trails dikh sakte hain."
        ],

        "cause": [
            "Snail infestation ki wajah se plant damage hota hai."
        ],

        "precaution": [
            "Plants aur surrounding area ko regularly inspect karein.",
            "Visible snails ko manually remove karein.",
            "Snails ke hiding places ko reduce karein.",
            "Field sanitation maintain karein."
        ]
    },


    # ========================================================
    # 11. UPPER LEAF THRIPS
    # ========================================================

    "upperleaf thrips": {
        "type": "Insect Pest",
        "insect": "Upper leaf thrips",

        "symptoms": [
            "Young ya upper leaves par feeding damage dikh sakta hai.",
            "Leaf surface par discoloration ya feeding marks ho sakte hain.",
            "Severe infestation mein new growth distort ho sakti hai."
        ],

        "cause": [
            "Thrips infestation ki wajah se damage hota hai."
        ],

        "precaution": [
            "Young aur upper leaves ko regularly monitor karein.",
            "New growth ko carefully inspect karein.",
            "Crop hygiene maintain karein.",
            "Suitable integrated pest management practices follow karein."
        ]
    },


    # ========================================================
    # 12. VIRUS
    # ========================================================

    "virus": {
        "type": "Disease",
        "disease": "Viral infection",

        "symptoms": [
            "Leaves par mosaic-like patterns ya patches dikh sakte hain.",
            "Leaves distort ya curl ho sakte hain.",
            "Plant ki growth reduced ya uneven ho sakti hai."
        ],

        "cause": [
            "Ye viral infection ki wajah se hota hai."
        ],

        "precaution": [
            "Severely infected plants ko remove karein.",
            "Virus spread karne wale insect vectors ko monitor aur control karein.",
            "Field sanitation maintain karein.",
            "Infected plant material ko propagation ke liye use na karein."
        ]
    }

}


# ============================================================
# SUPPORTED 12 CLASSES
# ============================================================

SUPPORTED_CLASSES = [
    "Anthracnose",
    "bacterial spot",
    "blossom-end rot",
    "down leaf aphid",
    "fruit thrips",
    "healthy",
    "Larva",
    "Magnesium",
    "powdery mildew",
    "snail",
    "upperleaf thrips",
    "virus"
]


# ============================================================
# VALIDATION
# ============================================================

assert len(SUPPORTED_CLASSES) == 12, \
    "Expected exactly 12 Capsicum classes."

assert set(SUPPORTED_CLASSES) == set(CLASS_INFO.keys()), \
    "SUPPORTED_CLASSES and CLASS_INFO do not match."