

# 🚨 Emergency — interrupts everything, plays alarm
DANGER_OBJECTS = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "fire hydrant",
}

# ⚠️ Caution — speaks "Caution. X ahead." and beeps
TRIP_HAZARDS = {
    "chair",
    "bench",
    "potted plant",
    "suitcase",
    "backpack",
    "box",
    "person",
    "dog",
    "cat",
    "stairs",
    "step",
    "shopping cart",
    "stroller",
}

# 📏 Detection distances (metres) — edit these to make Luma more/less sensitive
DANGER_DISTANCE = 4.0    # emergency trigger distance
HAZARD_DISTANCE = 1.8    # caution trigger distance
HAZARD_COOLDOWN = 4.0    # seconds before repeating same warning
CROWD_THRESHOLD = 5      # number of people to trigger crowd alert