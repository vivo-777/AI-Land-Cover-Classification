"""
Data Preprocessing Module
=========================
Handles geospatial filtering, label extraction from ESA WorldCover raster,
and train/test splitting for the Delhi NCR land cover dataset.
"""

import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point
from sklearn.model_selection import train_test_split

from config import (
    RGB_DIR, GEOJSON_PATH, RASTER_PATH,
    ESA_TO_LABEL, RASTER_WINDOW_SIZE,
    TEST_SIZE, RANDOM_STATE,
)


def load_boundary(geojson_path: str) -> gpd.GeoDataFrame:
    """Load the NCR region boundary from a GeoJSON file."""
    return gpd.read_file(geojson_path)


def filter_images_by_boundary(
    image_dir: str,
    boundary: gpd.GeoDataFrame,
) -> tuple[list[str], list[Point]]:
    """
    Filter satellite image patches to keep only those whose
    center coordinate falls within the Delhi NCR boundary.

    Returns:
        filtered_images: list of filenames that are inside the boundary
        center_coords:   list of corresponding Shapely Point objects
    """
    image_list = os.listdir(image_dir)
    filtered_images = []
    center_coords = []

    for name in image_list:
        try:
            clean_name = name.replace(".png", "")
            parts = clean_name.split("_")
            lat, lon = float(parts[0]), float(parts[1])

            if boundary.contains(Point(lon, lat)).any():
                filtered_images.append(name)
                center_coords.append(Point(lon, lat))
        except (ValueError, IndexError):
            continue

    return filtered_images, center_coords


def extract_labels(
    image_filenames: list[str],
    raster_path: str,
    window_size: int = RASTER_WINDOW_SIZE,
) -> list[str]:
    """
    For each image, extract the dominant ESA WorldCover class
    from a window centered on the image coordinates.

    Returns:
        labels: list of human-readable land cover category strings
    """
    landcover = rasterio.open(raster_path)
    labels = []

    for img in image_filenames:
        lat, lon = map(float, img.replace(".png", "").split("_"))

        # Convert geographic coordinates → raster pixel coordinates
        row, col = landcover.index(lon, lat)

        # Extract a window around the center pixel
        half = window_size // 2
        window = Window(col - half, row - half, window_size, window_size)
        patch = landcover.read(1, window=window)

        # Dominant class via majority voting
        unique_classes, counts = np.unique(patch, return_counts=True)
        dominant_class = unique_classes[np.argmax(counts)]
        labels.append(ESA_TO_LABEL.get(dominant_class, "Other"))

    landcover.close()
    return labels


def prepare_data(
    image_dir: str = RGB_DIR,
    geojson_path: str = GEOJSON_PATH,
    raster_path: str = RASTER_PATH,
) -> tuple:
    """
    Full preprocessing pipeline:
    1. Filter images spatially
    2. Extract labels from raster
    3. Split into train / test

    Returns:
        X_train, X_test, y_train, y_test
    """
    boundary = load_boundary(geojson_path)

    print("[1/3] Filtering images by NCR boundary...")
    image_filenames = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    # Note: using all images (unfiltered) for label extraction
    # Filtering can be toggled here as needed

    print("[2/3] Extracting labels from WorldCover raster...")
    labels = extract_labels(image_filenames, raster_path)

    print("[3/3] Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        image_filenames, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
