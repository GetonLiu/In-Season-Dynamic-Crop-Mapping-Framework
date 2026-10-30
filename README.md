# In-Season Dynamic Crop Mapping Framework

## 📌 Introduction
This repository implements a spatiotemporal adaptive classification framework that couples **High-Order Markov Prior** with a **densely supervised (DS) Transformer**. It is specifically designed for the accurate, in-season dynamic crop mapping of cropping systems, particularly in regions lacking historical crop map products. 

By integrating time-series spectral features with transition probabilities derived from historical rotation patterns, this framework dynamically updates crop classifications throughout the growing season, effectively mitigating early-season uncertainties.

## ⚙️ Requirements & Environment
The framework is built on PyTorch and utilizes several geospatial libraries for large-scale raster and vector processing. 

**Core Dependencies:**
- `python >= 3.8`
- `torch` (CUDA recommended for training and inference)
- `numpy` & `pandas`
- `geopandas` & `shapely` (for vector/ROI processing)
- `rasterio` (for large-scale TIFF block-by-block processing)
- `scikit-learn` (for validation metrics)
- `tqdm` (for progress tracking)

## 📁 Repository Structure

The workflow is divided into prior generation, data preprocessing, model training, dynamic mapping, and accuracy assessment. Please update the DATA_ROOT and OUTPUT_DIR in each script to match your local environment.

* **`Markov_Prior_Generator.py`**
  Calculates high-order Markov transition probabilities from historical crop maps (e.g., a 4-year history predicting the 5th year). It incorporates zone-based frequency analysis and Dirichlet/Beta smoothing to handle complex crop rotations.
* **`PointFeature_Processor.py`**
  Extracts multi-temporal indices (e.g., EVI, NDVI, RVI) from raster data based on training shapefiles. Performs data filtering (anomaly removal), normalization, and exports `.npy` tensors ready for PyTorch ingestion.
* **`Interannual_Weight.py`**
  Calculates and generates interannual weight matrices based on the complexity of historical crop rotation patterns. 
* **`DSTransformer_Tuning.py`**
  A hyperparameter tuning script for the DS-Transformer. Performs grid search across hidden dimensions, learning rates, and dropout rates to find optimal architecture configurations.
* **`DSTransformer_Train.py`**
  The main training script for the DS-Transformer. It features Focal Loss, time-decay weights for early-stage supervision, and dynamic step-by-step evaluation to monitor in-season classification accuracy.
* **`In_Season_Dynamic_Crop_Mapping.py`**
  The core mapping engine. It performs large-scale inference by coupling the trained DS-Transformer outputs with the high-order Markov priors. It uses a linear fusion strategy and a `CropStateTracker` to dynamically update crop distributions.
* **`Validation.py`**
  Evaluates the mapping results against ground-truth validation shapefiles. It computes area-weighted metrics including Overall Accuracy (OA), Precision, Recall, F1-Score, Kappa coefficient, coverage and generates step-by-step confusion matrices.

---

## 🚀 Workflow

**Step 1: Prior & Data Preparation**
1. Run `Markov_Prior_Generator.py` to generate the transition probability CSVs based on historical maps.
2. Prepare your time-series spectral rasters and sample shapefiles, then run `PointFeature_Processor.py` and `Interannual_Weight.py`.

**Step 2: Model Training**
1. Run `DSTransformer_Tuning.py` to find the best hyperparameters.
2. Run `DSTransformer_Train.py` to train the model. The best weights will be saved automatically.

**Step 3: Dynamic Mapping**
1. Configure the prior CSV paths and paths to your target year rasters in `In_Season_Dynamic_Crop_Mapping.py`. 
2. Run the script to generate step-by-step predicted crop maps. The script supports resume-from-checkpoint for interrupted large-scale processing.

**Step 4: Accuracy Assessment**
1. Run `Validation.py` to assess the TIFFs. 

## 📊 Expected Outputs
- **Mapping Results**: A series of `.tif` files representing the in-season crop distributions.
- **Accuracy Report**: A `.csv` containing OA, F1-Score, Kappa coefficient, and Coverage for each tif.


*You can install the main dependencies via pip:*
```bash
pip install torch numpy pandas geopandas shapely rasterio scikit-learn tqdm 
