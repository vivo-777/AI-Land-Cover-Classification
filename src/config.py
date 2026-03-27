"""
Configuration file for the Land Cover Classification project.
Contains all hyperparameters, file paths, and class mappings.
"""

import os

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "Dataset_Delhi_NCR")
RGB_DIR = os.path.join(DATA_DIR, "rgb")
GEOJSON_PATH = os.path.join(DATA_DIR, "delhi_ncr_region.geojson")
RASTER_PATH = os.path.join(DATA_DIR, "worldcover_bbox_delhi_ncr_2021.tif")

# ============================================================
# Spatial Parameters
# ============================================================
GRID_SIZE_METERS = 60_000       # 60 km grid cells
TARGET_CRS = "EPSG:32644"       # UTM Zone 44N (metric CRS for Delhi NCR)
RASTER_WINDOW_SIZE = 128        # Patch size for label extraction (pixels)

# ============================================================
# ESA WorldCover Class Mapping
# ============================================================
ESA_TO_LABEL = {
    10: "Tree_cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare_sparse_veg",
    70: "Snow_ice",
    80: "Water",
    90: "Herbaceous_wetland",
    95: "Mangroves",
    100: "Other",
}

# Simplified category mapping (5 classes)
CATEGORY_TO_ID = {
    # Vegetation (ID: 1)
    "Tree_cover": 1,
    "Shrubland": 1,
    "Grassland": 1,
    # Built-up (ID: 0)
    "Built-up": 0,
    # Water (ID: 2)
    "Water": 2,
    # Cropland (ID: 3)
    "Cropland": 3,
    # Other (ID: 4)
    "Bare_sparse_veg": 4,
    "Snow_ice": 4,
    "Herbaceous_wetland": 4,
    "Mangroves": 4,
    "Other": 4,
}

NUM_CLASSES = 5
CLASS_NAMES = ["Built-up", "Vegetation", "Water", "Cropland", "Other"]

# ============================================================
# Training Hyperparameters
# ============================================================
BATCH_SIZE = 32
LEARNING_RATE = 0.002
NUM_EPOCHS = 10
TEST_SIZE = 0.4
RANDOM_STATE = 42
