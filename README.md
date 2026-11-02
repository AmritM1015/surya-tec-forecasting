# Surya-TEC Forecasting

### Solar Foundation Models for Ionospheric Total Electron Content Prediction

This project investigates the use of **Surya**, a pretrained solar foundation model, to predict changes in Earth's ionospheric Total Electron Content (TEC) associated with solar flare activity.

The research explores whether representations learned from solar observations can be adapted to ionospheric forecasting through transfer learning and parameter-efficient fine-tuning.

## Research Objectives

The primary objectives are:

1. Investigate the relationship between solar observations and ionospheric TEC variations.
2. Evaluate the predictive capabilities of pretrained Surya representations.
3. Explore Low-Rank Adaptation (LoRA) for fine-tuning Surya on TEC forecasting tasks.
4. Compare different feature representations and prediction architectures.

## Dataset

The current dataset consists of:

| Component          | Description                                     |
| ------------------ | ----------------------------------------------- |
| Solar flares       | 30 events: 15 M-class and 15 X-class            |
| Solar observations | 60 observations, two per flare                  |
| TEC observations   | 210 TEC maps, seven per flare                   |
| Solar inputs       | Multichannel observations compatible with Surya |
| TEC targets        | Gridded ionospheric TEC measurements            |

Solar observations are selected at approximately four and five hours before each flare event.

TEC observations cover seven temporal offsets, from three hours before to three hours after each flare.

Dataset metadata, timestamps, and file references are maintained in `data/flare_manifest.csv`.

## Repository Structure

```text
surya-tec-forecasting/
├── data/
│   ├── raw/
│   │   ├── solar_observations/
│   │   └── tec_hdf5/
│   ├── processed/
│   │   └── tec_grids/
│   └── flare_manifest.csv
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   └── 02_model_training.ipynb
│
├── outputs/
│   ├── features/
│   ├── predictions/
│   ├── figures/
│   └── results/
│
├── src/
├── pyproject.toml
├── uv.lock
└── README.md
```

The pretrained Surya implementation and checkpoints are maintained separately from this repository.

## Methodology

### 1. Data Preparation

The data preparation notebook handles:

* Solar flare event selection.
* Solar observation retrieval.
* TEC data extraction from HDF5 files.
* Temporal alignment of observations.
* TEC gridding and visualization.
* Dataset manifest construction.

### 2. Feature Extraction

Surya processes multichannel solar observations to generate latent representations.

These representations can be cached to disk, avoiding repeated forward passes during downstream experiments.

### 3. TEC Prediction

The project investigates different approaches to mapping Surya representations to global TEC predictions.

Current experiments include:

* Global average-pooled representations.
* Spatially pooled representations.
* Lightweight TEC prediction heads.
* Leave-One-Out Cross-Validation (LOOCV).

### 4. Parameter-Efficient Fine-Tuning

The project also explores LoRA-based adaptation of Surya's attention layers.

This is intended to enable task-specific fine-tuning while updating only a small subset of the model's parameters.

**Status:** LoRA configuration is implemented; end-to-end fine-tuning and evaluation remain in development.

## Development Environment

The project uses Python and PyTorch, with dependencies managed through `uv`.

Install dependencies using:

```bash
uv sync
```

The pretrained Surya repository and model weights must be obtained separately.

Notebook paths currently assume the Surya repository is located alongside `surya-tec-forecasting`.

## Current Status

* [x] Solar flare dataset selection
* [x] Solar observation retrieval
* [x] TEC data extraction and gridding
* [x] Dataset manifest generation
* [x] Pretrained Surya checkpoint loading code
* [x] Cached-feature experiment implementation
* [x] LOOCV experiment implementation
* [ ] Validate end-to-end LoRA fine-tuning
* [ ] Complete experimental comparisons
* [ ] Analyze forecasting performance

## Research Context

This project is being developed as part of graduate research in Artificial Intelligence at the New Jersey Institute of Technology (NJIT).

The long-term objective is to investigate the applicability of solar foundation models to downstream space-weather prediction tasks.
