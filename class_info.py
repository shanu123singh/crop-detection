# ============================================================
# CAPSICUM CLASS INFORMATION
# Scientific + Agricultural Reference Information
# 12 CNN-supported classes
#
# IMPORTANT:
# - Class names are kept EXACTLY compatible with the trained model.
# - This information is for image-based agricultural screening.
# - It should not be treated as laboratory-level diagnosis.
# - Chemical control must always follow locally registered labels,
#   PHI/REI requirements and resistance-management guidance.
# ============================================================


CLASS_INFO = {

    # ========================================================
    # 1. ANTHRACNOSE
    # ========================================================

    "Anthracnose": {

        "type": "Disease",
        "category": "Fungal Disease",

        "common_name": "Anthracnose / Anthracnose Fruit Rot",

        "scientific_cause": "Colletotrichum spp.",

        "causal_agent":
            "Plant-pathogenic fungi belonging to the genus Colletotrichum.",

        "affected_parts": [
            "Fruit",
            "Leaves",
            "Stems"
        ],

        "symptoms": [
            "Circular, dark and sunken lesions may develop on fruit.",
            "Lesions generally enlarge as infection progresses.",
            "Affected fruit tissue may become necrotic and deteriorate.",
            "Under favorable conditions, orange to salmon-colored fungal spore masses may develop on lesions.",
            "Severe infection can reduce fruit quality and marketability."
        ],

        "favorable_conditions": [
            "Warm and humid environmental conditions.",
            "Extended periods of moisture on plant surfaces.",
            "Frequent rainfall or overhead irrigation.",
            "Presence of infected plant debris or planting material."
        ],

        "cause_and_biology": [
            "Pepper anthracnose is associated with several Colletotrichum species.",
            "The pathogen infects susceptible plant tissue and produces lesions.",
            "Water splash and contact with infected plant material can contribute to disease spread.",
            "Infected fruit can act as an important source of inoculum."
        ],

        "agricultural_impact": [
            "Reduces marketable fruit yield.",
            "Causes visible fruit quality defects.",
            "Severely infected fruit can become unmarketable.",
            "Infection may contribute to postharvest deterioration."
        ],

        "diagnostic_notes": [
            "Sunken circular fruit lesions are an important visual characteristic.",
            "Several fruit-rotting organisms can produce similar lesions.",
            "Visual classification should therefore be considered screening rather than laboratory confirmation."
        ],

        "management": [
            "Use healthy seed and disease-free planting material.",
            "Remove severely infected fruit and plant debris.",
            "Maintain good field sanitation.",
            "Avoid prolonged wetness of foliage and fruit.",
            "Use suitable crop rotation.",
            "Use locally registered fungicides only when required."
        ],

        "ipm": [
            "Regularly scout fruit for early lesions.",
            "Remove infected fruit before extensive disease development.",
            "Reduce splash movement of contaminated water.",
            "Maintain field sanitation.",
            "Use resistant or tolerant varieties where available.",
            "Rotate fungicide modes of action when chemical control is necessary."
        ],

        "precautions": [
            "Do not use infected fruit as planting material.",
            "Sanitize tools and equipment after contact with infected plant material.",
            "Follow local fungicide labels and resistance-management recommendations."
        ]
    },


    # ========================================================
    # 2. LARVA
    # ========================================================

    "Larva": {

        "type": "Pest",
        "category": "Insect Pest - Larval Feeding",

        "common_name": "Larval Insect Infestation",

        "scientific_cause":
            "Phytophagous insect larvae; exact species is not determined by this CNN class.",

        "causal_agent":
            "Plant-feeding insect larvae.",

        "affected_parts": [
            "Leaves",
            "Shoots",
            "Flowers",
            "Fruit"
        ],

        "symptoms": [
            "Visible insect larvae may occur on plant surfaces.",
            "Irregular chewing holes may appear on leaves.",
            "Leaf margins may show feeding damage.",
            "Young shoots and tender plant parts may be consumed.",
            "Flowers or fruit may show feeding injury depending on larval species.",
            "Frass or insect excreta may sometimes be visible around feeding sites."
        ],

        "favorable_conditions": [
            "Environmental conditions favorable to the specific insect species.",
            "Dense vegetation providing shelter.",
            "Presence of weeds or alternate host plants.",
            "Availability of young tender plant tissue."
        ],

        "cause_and_biology": [
            "This CNN class represents larval feeding damage rather than one specific insect species.",
            "Larvae generally cause direct mechanical feeding injury.",
            "Different larval species have different feeding behavior and life cycles.",
            "Species identification is important for targeted pest management."
        ],

        "agricultural_impact": [
            "Defoliation can reduce photosynthetic capacity.",
            "Flower feeding can reduce fruit set.",
            "Fruit feeding can reduce marketable quality.",
            "Severe infestation can weaken young plants."
        ],

        "diagnostic_notes": [
            "The generic Larva class should not be interpreted as a specific pest species.",
            "Inspect the damaged plant for actual larvae, eggs, frass or webbing.",
            "Species identification should be performed before pest-specific treatment."
        ],

        "management": [
            "Regularly inspect leaves, shoots and fruit.",
            "Hand-pick visible larvae in small-scale production where practical.",
            "Remove heavily infested plant material when appropriate.",
            "Manage weed hosts around the crop.",
            "Use biological control where suitable.",
            "Use registered insecticides only when monitoring justifies treatment."
        ],

        "ipm": [
            "Conduct regular field scouting.",
            "Use pest-specific monitoring tools where available.",
            "Conserve natural enemies.",
            "Avoid unnecessary broad-spectrum insecticide applications.",
            "Rotate insecticide modes of action when chemical control is required."
        ],

        "precautions": [
            "Do not select a pesticide solely from this generic image class.",
            "Identify the pest species whenever possible.",
            "Follow product label, PHI and resistance-management requirements."
        ]
    },


    # ========================================================
    # 3. MAGNESIUM DEFICIENCY
    # ========================================================

    "Magnesium": {

        "type": "Nutrient Deficiency",
        "category": "Secondary Macronutrient Deficiency",

        "common_name": "Magnesium Deficiency",

        "scientific_cause":
            "Insufficient plant-available magnesium (Mg).",

        "causal_agent":
            "Nutritional imbalance, inadequate Mg availability or restricted Mg uptake.",

        "affected_parts": [
            "Older Leaves"
        ],

        "symptoms": [
            "Interveinal chlorosis commonly develops on older leaves.",
            "The tissue between leaf veins becomes yellow or pale.",
            "Leaf veins may remain comparatively green.",
            "Symptoms may become progressively more severe if Mg deficiency continues.",
            "Severe deficiency can lead to necrosis and premature leaf loss."
        ],

        "favorable_conditions": [
            "Low available soil magnesium.",
            "Nutrient imbalance.",
            "Root-zone conditions that reduce Mg availability or uptake.",
            "High crop demand during vigorous growth."
        ],

        "cause_and_biology": [
            "Magnesium is an essential mineral nutrient.",
            "Mg is the central atom of the chlorophyll molecule and is therefore important for photosynthesis.",
            "Magnesium is relatively mobile within the plant.",
            "Consequently, deficiency symptoms commonly appear first on older leaves."
        ],

        "agricultural_impact": [
            "Reduced chlorophyll can decrease photosynthetic efficiency.",
            "Severe deficiency can reduce plant vigor.",
            "Persistent deficiency can negatively affect yield potential.",
            "Visual symptoms may overlap with other nutrient deficiencies."
        ],

        "diagnostic_notes": [
            "Interveinal chlorosis of older leaves is a key diagnostic characteristic.",
            "Yellowing alone should not be used to confirm magnesium deficiency.",
            "Soil and/or plant tissue testing is recommended for confirmation."
        ],

        "management": [
            "Conduct soil testing where possible.",
            "Use plant tissue analysis when required.",
            "Apply an appropriate magnesium source according to soil-test results and crop requirements.",
            "Maintain balanced crop nutrition.",
            "Correct root-zone and irrigation problems affecting nutrient uptake."
        ],

        "ipm": [
            "Monitor older leaves during crop growth.",
            "Base fertilizer applications on soil and crop requirements.",
            "Avoid unnecessary over-application of competing nutrients.",
            "Maintain good root health."
        ],

        "precautions": [
            "Do not assume every interveinal chlorosis is Mg deficiency.",
            "Confirm the nutritional cause before applying large quantities of fertilizer.",
            "Use locally recommended fertilizer rates."
        ]
    },


    # ========================================================
    # 4. BACTERIAL SPOT
    # ========================================================

    "bacterial spot": {

        "type": "Disease",
        "category": "Bacterial Disease",

        "common_name": "Bacterial Spot of Pepper",

        "scientific_cause": "Xanthomonas spp.",

        "causal_agent":
            "Several pathogenic Xanthomonas species can cause bacterial spot in pepper.",

        "affected_parts": [
            "Leaves",
            "Petiole",
            "Stem",
            "Fruit"
        ],

        "symptoms": [
            "Small brown circular spots may develop on leaves.",
            "Leaf spots may be more common or severe on lower foliage.",
            "Brown lesions can occur on stems and petioles.",
            "Fruit may develop raised brown, corky or scabby lesions.",
            "Severe foliar infection may cause leaf loss."
        ],

        "favorable_conditions": [
            "Warm temperatures.",
            "High relative humidity.",
            "Frequent rainfall.",
            "Prolonged leaf wetness.",
            "Overhead irrigation can facilitate bacterial spread."
        ],

        "cause_and_biology": [
            "Bacterial spot is caused by pathogenic Xanthomonas species.",
            "Infected seed and transplants can introduce the pathogen.",
            "Water splash can move bacteria between plants.",
            "Handling and contaminated plant material can contribute to spread.",
            "Infected plant debris can act as an inoculum source."
        ],

        "agricultural_impact": [
            "Reduces functional leaf area.",
            "Can reduce plant vigor under severe infection.",
            "Fruit lesions can make peppers unmarketable.",
            "Disease can develop rapidly under warm and wet conditions."
        ],

        "diagnostic_notes": [
            "Pepper bacterial spot can resemble other leaf-spot diseases.",
            "Raised corky brown fruit lesions are an important diagnostic clue.",
            "Laboratory confirmation may be necessary when symptoms overlap."
        ],

        "management": [
            "Use disease-free seed.",
            "Use healthy transplants.",
            "Avoid unnecessary overhead irrigation.",
            "Avoid working through wet foliage when practical.",
            "Remove severely infected plant material.",
            "Maintain crop sanitation.",
            "Practice suitable crop rotation.",
            "Use locally registered bactericidal products according to label."
        ],

        "ipm": [
            "Scout lower foliage and fruit regularly.",
            "Use clean planting material.",
            "Reduce water-splash movement.",
            "Maintain good field sanitation.",
            "Use tolerant or resistant cultivars where available.",
            "Rotate chemical modes of action where applicable."
        ],

        "precautions": [
            "Do not save seed from symptomatic plants.",
            "Sanitize tools and equipment.",
            "Follow local pesticide labels and resistance-management recommendations."
        ]
    },


    # ========================================================
    # 5. BLOSSOM-END ROT
    # ========================================================

    "blossom-end rot": {

        "type": "Physiological Disorder",
        "category": "Calcium-Related Physiological Disorder",

        "common_name": "Blossom-End Rot",

        "scientific_cause":
            "Localized calcium deficiency in developing fruit tissue.",

        "causal_agent":
            "Physiological disorder associated with inadequate calcium delivery to rapidly developing fruit.",

        "affected_parts": [
            "Fruit"
        ],

        "symptoms": [
            "Tan, brown or dark lesions develop at the blossom end of fruit.",
            "The affected area becomes sunken as the disorder progresses.",
            "The tissue becomes necrotic and may appear leathery.",
            "Secondary fungal or bacterial decay may occur.",
            "Affected fruit may fail to develop normally."
        ],

        "favorable_conditions": [
            "Irregular soil moisture.",
            "Drought followed by rapid irrigation or rainfall.",
            "Rapid vegetative growth.",
            "High fruit demand for calcium.",
            "Root damage or poor root function."
        ],

        "cause_and_biology": [
            "Blossom-end rot is a physiological disorder rather than a contagious disease.",
            "It is associated with insufficient calcium delivery to developing fruit tissue.",
            "The problem does not necessarily mean that total soil calcium is low.",
            "Calcium movement is closely linked to water movement and plant transpiration.",
            "Irregular moisture can interfere with calcium delivery."
        ],

        "agricultural_impact": [
            "Affected fruit lose market value.",
            "Severe incidence can reduce marketable yield.",
            "Secondary microbial decay can further damage fruit."
        ],

        "diagnostic_notes": [
            "The blossom-end location is an important diagnostic feature.",
            "Do not automatically classify the condition as simple soil calcium deficiency.",
            "Check irrigation, root health and overall nutrient status."
        ],

        "management": [
            "Maintain uniform soil moisture.",
            "Avoid severe drought followed by excessive irrigation.",
            "Maintain healthy roots.",
            "Use appropriate irrigation scheduling.",
            "Maintain balanced crop nutrition.",
            "Use soil testing to guide calcium management."
        ],

        "ipm": [
            "Monitor developing fruit regularly.",
            "Use mulching where agronomically suitable to stabilize soil moisture.",
            "Avoid excessive vegetative growth caused by unbalanced fertilization.",
            "Investigate persistent cases for root-zone or irrigation problems."
        ],

        "precautions": [
            "Do not assume soil calcium is always deficient.",
            "Do not rely on foliar calcium sprays as a substitute for proper water management.",
            "Avoid unnecessary fertilizer application without agronomic justification."
        ]
    },


    # ========================================================
    # 6. APHID
    # ========================================================

    "down leaf aphid": {

        "type": "Pest",
        "category": "Insect Pest - Aphid",

        "common_name": "Aphid Infestation",

        "scientific_cause":
            "Aphid species; exact species is not determined by this CNN class.",

        "causal_agent":
            "Sap-feeding insects belonging to the family Aphididae.",

        "affected_parts": [
            "Young Leaves",
            "Leaf Undersides",
            "Shoots",
            "Flowers"
        ],

        "symptoms": [
            "Small soft-bodied aphids may occur on young leaves and shoots.",
            "Leaves may curl or become distorted.",
            "Young shoots may show reduced growth.",
            "Sticky honeydew may accumulate on plant surfaces.",
            "Sooty mold may develop secondarily on honeydew."
        ],

        "favorable_conditions": [
            "Abundant tender plant growth.",
            "Conditions favorable to rapid aphid reproduction.",
            "Presence of alternative host weeds.",
            "Low natural-enemy pressure."
        ],

        "cause_and_biology": [
            "Aphids are piercing-sucking insects that remove plant sap.",
            "Some aphid species reproduce rapidly under favorable conditions.",
            "Certain aphid species are important vectors of plant viruses.",
            "Aphids commonly colonize young, actively growing tissues."
        ],

        "agricultural_impact": [
            "Heavy infestations reduce plant vigor.",
            "Leaf curling and distortion can interfere with growth.",
            "Honeydew can support sooty mold.",
            "Virus transmission can be more important economically than direct feeding."
        ],

        "diagnostic_notes": [
            "Inspect leaf undersides and young shoots.",
            "Species-level identification is important for precise pest management.",
            "Distinguish aphids from other small sap-feeding insects."
        ],

        "management": [
            "Scout young leaves and shoots regularly.",
            "Conserve natural predators and parasitoids.",
            "Manage weeds serving as alternative hosts.",
            "Use selective registered insecticides only when treatment is justified."
        ],

        "ipm": [
            "Use regular field scouting.",
            "Use sticky traps where appropriate.",
            "Encourage lady beetles, lacewings and parasitoids.",
            "Avoid unnecessary broad-spectrum insecticides.",
            "Rotate insecticide modes of action."
        ],

        "precautions": [
            "Do not identify a particular aphid species from this CNN class alone.",
            "Monitor simultaneously for virus symptoms.",
            "Follow local pesticide label and PHI requirements."
        ]
    },


    # ========================================================
    # 7. FRUIT THRIPS
    # ========================================================

    "fruit thrips": {

        "type": "Pest",
        "category": "Insect Pest - Thrips",

        "common_name": "Thrips Feeding Damage on Fruit",

        "scientific_cause":
            "Thrips species; exact species is not determined by this CNN class.",

        "causal_agent":
            "Small thysanopteran insects commonly known as thrips.",

        "affected_parts": [
            "Flowers",
            "Young Fruit",
            "Fruit Surface",
            "Young Leaves"
        ],

        "symptoms": [
            "Silvery or bronzed feeding scars may develop.",
            "Developing fruit may show superficial scars or russeting.",
            "Discoloration can occur around feeding sites.",
            "Severe feeding can cause fruit distortion or reduced quality."
        ],

        "favorable_conditions": [
            "Warm conditions.",
            "Flowering and abundant young plant tissue.",
            "Presence of alternative host plants.",
            "Protected crop environments can sometimes favor population buildup."
        ],

        "cause_and_biology": [
            "Thrips feed by rasping plant tissue and consuming released cell contents.",
            "Adults and immature stages can feed on flowers, leaves and developing fruit.",
            "Some thrips species can transmit plant viruses."
        ],

        "agricultural_impact": [
            "Fruit scarring reduces cosmetic quality.",
            "Severe flower feeding can affect fruit development.",
            "Some species can contribute to virus transmission."
        ],

        "diagnostic_notes": [
            "Thrips are very small and may require close inspection or magnification.",
            "Inspect flowers and young fruit.",
            "Visual scars alone are not sufficient to establish species identity."
        ],

        "management": [
            "Monitor flowers and developing fruit regularly.",
            "Use sticky traps where appropriate.",
            "Manage alternative weed hosts.",
            "Conserve beneficial predatory insects and mites.",
            "Use selective registered insecticides when justified."
        ],

        "ipm": [
            "Combine sticky-trap monitoring with direct flower inspection.",
            "Use biological control where appropriate.",
            "Avoid repeated use of the same insecticide mode of action.",
            "Base chemical treatment on crop stage and monitoring."
        ],

        "precautions": [
            "Do not choose a pesticide solely from the image classification.",
            "Species identification is useful for precise management.",
            "Follow product label, PHI and resistance-management requirements."
        ]
    },


    # ========================================================
    # 8. HEALTHY
    # ========================================================

    "healthy": {

        "type": "Healthy",
        "category": "Normal Plant Condition",

        "common_name": "Healthy Capsicum Plant/Tissue",

        "scientific_cause":
            "No target disease, pest or nutrient-deficiency symptom detected in the image.",

        "causal_agent": None,

        "affected_parts": [
            "Leaves",
            "Stems",
            "Flowers",
            "Fruit"
        ],

        "symptoms": [
            "No obvious target disease symptoms.",
            "No obvious pest feeding injury.",
            "Leaves generally have normal green coloration.",
            "Plant tissue appears intact and normally developed."
        ],

        "favorable_conditions": [
            "Adequate irrigation.",
            "Good drainage.",
            "Balanced crop nutrition.",
            "Healthy root system.",
            "Effective pest and disease monitoring."
        ],

        "cause_and_biology": [
            "The CNN classifies the submitted image as healthy relative to the 12 target classes.",
            "A healthy prediction does not guarantee that the complete plant is free from every possible pest, disease or nutritional disorder."
        ],

        "agricultural_impact": [
            "Healthy foliage supports photosynthesis.",
            "Healthy plants generally have better potential for normal growth and fruit development.",
            "Preventive monitoring helps maintain crop health."
        ],

        "diagnostic_notes": [
            "The model only recognizes patterns represented by its training classes.",
            "Symptoms outside the 12 target classes may not be recognized."
        ],

        "management": [
            "Continue regular crop scouting.",
            "Maintain appropriate irrigation.",
            "Maintain balanced nutrition.",
            "Monitor new growth and fruit regularly.",
            "Maintain field sanitation."
        ],

        "ipm": [
            "Continue preventive monitoring.",
            "Use integrated pest management.",
            "Avoid unnecessary pesticide applications.",
            "Maintain appropriate crop hygiene."
        ],

        "precautions": [
            "A healthy CNN prediction is not a laboratory or whole-field certification.",
            "Continue normal field-level agricultural monitoring."
        ]
    },


    # ========================================================
    # 9. POWDERY MILDEW
    # ========================================================

    "powdery mildew": {

        "type": "Disease",
        "category": "Fungal Disease",

        "common_name": "Pepper Powdery Mildew",

        "scientific_cause": "Leveillula taurica",

        "causal_agent":
            "Leveillula taurica, a powdery mildew fungus affecting pepper.",

        "affected_parts": [
            "Leaves"
        ],

        "symptoms": [
            "Patchy white powdery growth may develop on leaves.",
            "Powdery growth is commonly prominent on the lower leaf surface.",
            "Upper leaf surfaces may show yellow or brown discoloration.",
            "Leaf edges may roll upward.",
            "Severely affected leaves can die and fall."
        ],

        "favorable_conditions": [
            "Warm crop-growing conditions.",
            "High humidity can favor spore germination.",
            "Dense crop canopy can make management more difficult.",
            "Disease can develop over a relatively broad temperature range."
        ],

        "cause_and_biology": [
            "Pepper powdery mildew is commonly caused by Leveillula taurica.",
            "The pathogen develops largely within infected leaf tissue and produces powdery conidia externally.",
            "Wind can disperse conidia to new leaves and plants.",
            "Repeated secondary infections can occur under favorable conditions."
        ],

        "agricultural_impact": [
            "Reduces functional leaf area.",
            "Can cause premature leaf loss.",
            "Defoliation can expose fruit to sunburn.",
            "Severe disease can reduce yield potential."
        ],

        "diagnostic_notes": [
            "Inspect both sides of leaves.",
            "White powdery growth on the lower leaf surface is an important clue.",
            "Leaf yellowing alone is not enough to diagnose powdery mildew."
        ],

        "management": [
            "Scout regularly, especially during favorable weather.",
            "Maintain good crop airflow.",
            "Use tolerant or resistant varieties where available.",
            "Avoid excessive nitrogen fertilization.",
            "Remove severely affected plant material where practical.",
            "Use locally registered fungicides when necessary."
        ],

        "ipm": [
            "Early detection is important.",
            "Combine crop management with approved fungicide programs.",
            "Rotate fungicide modes of action.",
            "Maintain appropriate canopy management."
        ],

        "precautions": [
            "Do not diagnose solely from leaf yellowing.",
            "Follow fungicide labels and resistance-management guidance."
        ]
    },


    # ========================================================
    # 10. SNAIL
    # ========================================================

    "snail": {

        "type": "Pest",
        "category": "Molluscan Pest",

        "common_name": "Snail Feeding Damage",

        "scientific_cause":
            "Snail species; exact species is not determined by this CNN class.",

        "causal_agent":
            "Terrestrial gastropods belonging to the class Gastropoda.",

        "affected_parts": [
            "Leaves",
            "Young Shoots",
            "Seedlings",
            "Fruit Surface"
        ],

        "symptoms": [
            "Irregular holes appear in leaves.",
            "Leaf margins may be chewed.",
            "Young tender tissue can be heavily damaged.",
            "Silvery mucus/slime trails may occur.",
            "Seedlings may suffer severe feeding injury."
        ],

        "favorable_conditions": [
            "Moist soil and plant surfaces.",
            "High humidity.",
            "Sheltered areas containing organic debris.",
            "Night-time or early-morning conditions."
        ],

        "cause_and_biology": [
            "Snails are molluscan pests and are not insects.",
            "They feed mainly when environmental conditions support surface activity.",
            "Crop debris, weeds and sheltered moist areas can provide hiding places."
        ],

        "agricultural_impact": [
            "Can damage seedlings and reduce plant establishment.",
            "Leaf feeding reduces photosynthetic area.",
            "Fruit surface feeding can reduce market quality."
        ],

        "diagnostic_notes": [
            "Look for characteristic mucus trails.",
            "Inspect plants during evening or early morning.",
            "Distinguish snail feeding from caterpillar damage by checking for the pest or slime trails."
        ],

        "management": [
            "Reduce unnecessary hiding places around crop rows.",
            "Manage weeds and excess debris.",
            "Hand-pick snails where practical.",
            "Use suitable physical barriers or traps.",
            "Use registered mollusc-management products where justified."
        ],

        "ipm": [
            "Monitor at night or early morning.",
            "Reduce favorable shelters.",
            "Use physical and cultural controls first where practical.",
            "Use chemical control only when monitoring indicates a need."
        ],

        "precautions": [
            "Snails require mollusc-specific management rather than insect-specific control.",
            "Follow product labels carefully around edible crops."
        ]
    },


    # ========================================================
    # 11. UPPER LEAF THRIPS
    # ========================================================

    "upperleaf thrips": {

        "type": "Pest",
        "category": "Insect Pest - Thrips",

        "common_name": "Thrips Feeding Damage on Young/Upper Leaves",

        "scientific_cause":
            "Thrips species; exact species is not determined by this CNN class.",

        "causal_agent":
            "Thysanopteran insects commonly known as thrips.",

        "affected_parts": [
            "Young Leaves",
            "Upper Foliage",
            "Shoots",
            "Flowers"
        ],

        "symptoms": [
            "Fine silvery or pale feeding marks may occur on young leaves.",
            "Upper leaves may become discolored.",
            "Severe feeding can cause leaf curling or distortion.",
            "New growth can become malformed under heavy infestation.",
            "Small thrips may be visible on tender plant tissue."
        ],

        "favorable_conditions": [
            "Warm conditions.",
            "Abundant tender plant growth.",
            "Presence of alternative host plants.",
            "Conditions favorable for rapid thrips reproduction."
        ],

        "cause_and_biology": [
            "Thrips scrape plant tissue and consume cellular contents.",
            "Adults and immature stages may feed on tender foliage.",
            "Some thrips species are important vectors of plant viruses."
        ],

        "agricultural_impact": [
            "Damage to young leaves can reduce effective photosynthetic area.",
            "Severe feeding can distort new growth.",
            "Virus transmission may increase the economic significance of some thrips populations."
        ],

        "diagnostic_notes": [
            "Inspect the newest leaves carefully.",
            "Use magnification when necessary because thrips are very small.",
            "Species identification is recommended for precise pest management."
        ],

        "management": [
            "Regularly scout young leaves and flowers.",
            "Use sticky traps where appropriate.",
            "Conserve beneficial predators and parasitoids.",
            "Manage alternative weed hosts.",
            "Use selective registered insecticides when treatment is justified."
        ],

        "ipm": [
            "Combine monitoring, sanitation and biological control.",
            "Avoid unnecessary broad-spectrum insecticide applications.",
            "Rotate insecticide modes of action when chemical control is necessary."
        ],

        "precautions": [
            "Do not infer a specific thrips species from the CNN class.",
            "Monitor for associated viral symptoms.",
            "Follow local pesticide labels and resistance-management requirements."
        ]
    },


    # ========================================================
    # 12. VIRUS
    # ========================================================

    "virus": {

        "type": "Disease",
        "category": "Viral Disease",

        "common_name": "Viral Disease / Virus-Associated Symptoms",

        "scientific_cause":
            "Plant viruses; exact virus species is not established by this CNN class.",

        "causal_agent":
            "Plant pathogenic viruses.",

        "affected_parts": [
            "Leaves",
            "Shoots",
            "Fruit",
            "Whole Plant"
        ],

        "symptoms": [
            "Mosaic patterns of light and dark green may develop on leaves.",
            "Leaves may curl or become distorted.",
            "Leaves may become unusually narrow or malformed.",
            "Plants may show stunting or uneven growth.",
            "Fruit may develop mottling, discoloration, rings or deformation depending on the virus."
        ],

        "favorable_conditions": [
            "Use of infected planting material.",
            "Presence of insect vectors such as aphids or thrips for vector-transmitted viruses.",
            "Mechanical transmission through contaminated tools or plant handling for viruses capable of such spread.",
            "Presence of infected weeds or alternative host plants."
        ],

        "cause_and_biology": [
            "This CNN class represents virus-associated visual symptoms rather than one specific virus.",
            "Different pepper viruses can produce overlapping symptoms.",
            "Transmission depends on the particular virus and may involve insects, seed, planting material or mechanical contact.",
            "Exact virus identification requires appropriate diagnostic testing."
        ],

        "agricultural_impact": [
            "Can reduce plant vigor.",
            "May reduce fruit set and yield.",
            "Can reduce fruit quality.",
            "Early infection may produce greater economic losses.",
            "Vector-mediated spread can allow disease to move rapidly through a crop."
        ],

        "diagnostic_notes": [
            "Visual symptoms alone cannot reliably identify the exact virus species.",
            "Nutrient deficiency, herbicide injury and environmental stress can sometimes produce similar symptoms.",
            "Laboratory testing is recommended when species-level diagnosis is economically important."
        ],

        "management": [
            "Use healthy and certified planting material.",
            "Remove severely symptomatic plants where recommended.",
            "Manage important insect vectors using IPM.",
            "Control alternative weed hosts.",
            "Sanitize tools where mechanical transmission is possible.",
            "Do not propagate plants from symptomatic material.",
            "Use resistant or tolerant varieties when available for the relevant virus."
        ],

        "ipm": [
            "Monitor virus symptoms together with vector populations.",
            "Use clean nursery material.",
            "Maintain nursery and field sanitation.",
            "Integrate vector management with removal of infection sources.",
            "Use resistant cultivars where appropriate."
        ],

        "precautions": [
            "Do not assign a specific virus species from this generic CNN prediction.",
            "Do not use symptomatic plants for propagation.",
            "Obtain laboratory confirmation when a specific viral diagnosis is required."
        ]
    }
}


# ============================================================
# SUPPORTED 12 CLASSES
#
# IMPORTANT:
# Keep these EXACTLY the same as the trained ResNet labels.
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