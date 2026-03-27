# 🌍 AI-Powered Land Cover Classification — Delhi NCR Region

> Deep learning-based satellite image classification system for sustainable urban planning, leveraging ESA WorldCover data and transfer learning with ResNet-18.

---

## 📌 Project Overview

This project uses **AI and geospatial analysis** to classify satellite imagery of the **Delhi National Capital Region (NCR)** into distinct land cover categories. By combining **ESA WorldCover 2021** raster data with **RGB satellite patches**, we train a **ResNet-18 CNN** to predict land cover types — enabling insights for sustainable urban development, environmental monitoring, and green space preservation.

### Key Highlights
- **9,216 satellite image patches** processed, **8,015** retained after spatial filtering
- **5 consolidated land cover classes**: Built-up, Vegetation, Water, Cropland, Other
- **Transfer learning** with pretrained ResNet-18 for efficient classification
- **Geospatial preprocessing** using GeoPandas, Shapely, and Rasterio

---

## 🏗️ Project Structure

```
AI-for-Sustainability/
│
├── data/                          # Dataset directory
│   └── Dataset_Delhi_NCR/
│       ├── delhi_ncr_region.geojson      # NCR boundary polygons
│       ├── delhi_airshed.geojson         # Delhi airshed region
│       ├── worldcover_bbox_delhi_ncr_2021.tif  # ESA WorldCover raster
│       └── rgb/                          # Satellite image patches (PNG)
│
├── src/                           # Source code modules
│   ├── data_preprocessing.py      # Spatial filtering & label extraction
│   ├── dataset.py                 # PyTorch Dataset class
│   ├── model.py                   # Model architecture & configuration
│   ├── train.py                   # Training loop
│   └── config.py                  # Hyperparameters & paths
│
├── notebooks/
│   ├── 01_spatial_analysis.ipynb  # Geospatial data exploration
│   └── 02_model_training.ipynb    # End-to-end training pipeline
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules
```

---

## 🔬 Methodology

### 1. Spatial Data Processing
- Load **Delhi NCR boundary** from GeoJSON (30 districts)
- Reproject to **UTM Zone 44N (EPSG:32644)** for metric-accurate analysis
- Generate **60×60 km uniform grid mesh** for spatial partitioning
- Filter satellite patches to retain only those **within NCR boundaries**

### 2. Label Generation
- Extract **128×128 pixel windows** from ESA WorldCover raster at each image center
- Compute **dominant land cover class** per patch using majority voting
- Map ESA codes to **5 simplified categories**:

| ID | Category   | ESA Codes       |
|----|------------|-----------------|
| 0  | Built-up   | 50              |
| 1  | Vegetation | 10, 20, 30      |
| 2  | Water      | 80              |
| 3  | Cropland   | 40              |
| 4  | Other      | 60, 70, 90, 95+ |

### 3. Model Training
- **Architecture**: ResNet-18 (pretrained on ImageNet)
- **Transfer Learning**: Replace final FC layer → 5 output classes
- **Optimizer**: Adam (lr=0.002)
- **Loss**: CrossEntropyLoss
- **Split**: 60% train / 40% test

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Quick Start
```python
# Train the model
python src/train.py

# Or run the notebook
jupyter notebook notebooks/02_model_training.ipynb
```

---

## 📊 Dataset Statistics

| Metric              | Value   |
|---------------------|---------|
| Total raw patches   | 9,216   |
| Filtered patches    | 8,015   |
| Image size          | 256×256 |
| Raster source       | ESA WorldCover 2021 |
| Spatial resolution  | 10m     |
| Region of interest  | Delhi NCR (~30 districts) |

---

## 🛠️ Tech Stack

| Category        | Tools                                      |
|-----------------|---------------------------------------------|
| Deep Learning   | PyTorch, torchvision                        |
| Geospatial      | GeoPandas, Shapely, Rasterio                |
| Data Science    | NumPy, Pandas, scikit-learn                 |
| Visualization   | Matplotlib                                  |
| Image Processing| Pillow (PIL)                                |

---

## 📜 License

This project is for academic and research purposes.

---

## 👤 Author

**Kinjal Chavda**

---

*Built with ❤️ for a sustainable future*
