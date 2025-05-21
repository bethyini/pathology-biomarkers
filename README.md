# pathology-biomarkers

**Digital Pathology Toolbox for Biomarker Prediction from Whole-Slide Images**

This repository provides a modular pipeline for processing H&E-stained whole-slide images (WSIs) and training models to predict key genetic and clinical biomarkers, including MSI, HER2, BRAF, and others. The workflow is optimized for colorectal cancer datasets such as TCGA-COAD and supports both patch-level feature extraction and slide-level aggregation using attention-based models.

## Key Features

- Patch extraction and filtering from SVS files using OpenSlide and custom tiling logic
- Feature extraction using pretrained foundation models such as UNI2-h and BiomedCLIP
- Slide-level aggregation using attention-based multiple instance learning (CLAM)
- Interpretation via heatmaps derived from attention scores
- Tools for label matching with clinical and genomic data (e.g., from GDC and cBioPortal)
- Modular notebook-based data processing and experiment setup

## Repository Structure

