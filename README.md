# d4pdf-front-detection-model-analysis

## Overview

This repository contains the source code used to develop, apply, and evaluate machine-learning models for detecting surface weather fronts and identifying Baiu-front days from d4PDF global climate model (GCM) simulations. It also includes analysis scripts used to calculate statistics, bootstrap uncertainty, and generate figures for the associated manuscript.

The workflow includes:

1. development and evaluation of a U-Net model for frontal-location detection,
2. development and evaluation of an XGBoost model for Baiu-front-day identification,
3. application of the trained models to d4PDF experiments,
4. generation of derived front datasets,
5. statistical analysis of historical and future experiments, and
6. production of manuscript figures.

## Related manuscript

This repository supports the manuscript:

**[Insert manuscript title here]**

Submitted to **Earth's Future (AGU)**.

## Authors

- Yiwen Mao, Hokkaido University
- Tomohito J. Yamada, Hokkaido University

## Purpose of this repository

The code in this repository is provided to support transparency, peer review, and reproducibility of the analysis presented in the manuscript. It includes scripts for model development, prediction, evaluation, and downstream climate analysis based on d4PDF simulations.

## Repository contents

The repository contains Python scripts for the following main tasks:

### 1. Model development and evaluation

- `develop_model1_unet.py`  
  Develops model 1, the U-Net model used to detect frontal locations.

- `evaluate_unet.py`  
  Evaluates the U-Net model using F1 score, Cohen's kappa, and Matthews correlation coefficient (MCC); used to produce Fig. 2(a–f).

- `evaluate_unet_additional.py`  
  Calculates additional evaluation metrics, including CSI, POD, SR, and bias; used to produce Figs. S7–S8.

- `develop_evaluate_model2_xgboost.py`  
  Develops and evaluates model 2, the XGBoost model used to identify Baiu-front days; used to produce Fig. 2(g–h).

- `XGBoost_predictors.py`  
  Calculates the XGBoost predictors described in Text S3 from ERA5 data.

### 2. Application of trained models to d4PDF data

- `apply_model1.py`  
  Applies the trained U-Net model to d4PDF GCM output to obtain locations of fronts.

- `apply_model2.py`  
  Applies the trained XGBoost model to identify Baiu-front days and collect the spatial locations of all Baiu fronts.

- `HPB_get_monFt.py`  
  Extracts temporal-spatial datasets for Baiu-front and non-Baiu-front conditions in June, July, and August for historical experiments.

### 3. Bootstrap and frequency analysis

- `bootstrap_HFB_bft_rfreq.py`  
  Calculates relative frequency for 10,000 bootstrapped samples for various HFB scenarios; used in later analysis for Fig. 4.

- `bootstrap_HPB_bft_rfreq.py`  
  Calculates relative frequency for 10,000 bootstrapped samples for HPB scenarios; used in later analysis for Fig. 4.

- `bstrp_compare.py`  
  Performs bootstrap resampling of all available years with replacement 10,000 times from historical climate (HPB) and future climate (HFB 2K/4K under SST warming scenarios CC, GF, HA, MI, MP, MR), and calculates differences in statistics between HPB and each HFB scenario for each bootstrap sample.

- `plot_rel_freq.py`  
  Calculates relative frequency of Baiu fronts, defined as the ratio of the number of Baiu fronts occurring on a date in JJA to the total number of years used for prediction, using HPB and multimodel means of HFB 2K and 4K scenarios.

### 4. Historical vs future climate analysis

- `HPBvsHFB_temporal.py`  
  Calculates bootstrap differences in the average number of Baiu-front days in June, July, and August for HFB 2K/4K scenarios; used for Fig. 4(a–c).

- `HPBvsHFB_mm_space.py`  
  Calculates the multimodel mean spatial distribution of differences in the average number of Baiu-front days in JJA; used for Fig. 4(a, e).

- `HPBvsHFB_mm_harmonic_anom.py`  
  Calculates the multimodel mean harmonic fitting and anomalies of vorticity and moisture flux convergence (MFC); used for Fig. 4(e–h).

- `HPBvsHFB_ind_anom.py`  
  Calculates averaged anomalies of vorticity or MFC and plots them as in Figs. S4–S5.

- `HPBvsHFB_ind_harmonic.py`  
  Calculates averaged harmonic fitting of vorticity or MFC and plots them as in Figs. S2–S3.

- `HPBvsHFB_indm_space_month.py`  
  Calculates monthly spatial differences (HFB − HPB) for individual models; used for Figs. 6–8.

- `HPBvsHFB_indm_space.py`  
  Calculates JJA spatial differences (HFB − HPB) for individual models; used for Fig. 9.

### 5. Historical-climate diagnostics and plotting

- `plot_hpb_boxplot.py`  
  Calculates the average number of Baiu-front days per month in JJA; used for Fig. 3c.

- `plot_hpb_spa_dist_JJA.py`  
  Calculates the spatial distribution of the average number of Baiu-front days in JJA; used for Fig. 3(d–f).

- `plot_hbp_stat.py`  
  Calculates spatial distributions of statistics related to the average number of Baiu-front and non-Baiu-front days in JJA; used for Fig. 3(g–l).

## Experiments analyzed

The scripts in this repository analyze historical and future d4PDF experiments, including:

- HPB
- HFB 2K CC
- HFB 2K GF
- HFB 2K HA
- HFB 2K MI
- HFB 2K MP
- HFB 2K MR
- HFB 4K CC
- HFB 4K GF
- HFB 4K HA
- HFB 4K MI
- HFB 4K MP
- HFB 4K MR

## Typical workflow

A typical analysis workflow is:

1. Calculate ERA5-based predictors for XGBoost using `XGBoost_predictors.py`
2. Develop and evaluate the U-Net model using:
   - `develop_model1_unet.py`
   - `evaluate_unet.py`
   - `evaluate_unet_additional.py`
3. Develop and evaluate the XGBoost model using:
   - `develop_evaluate_model2_xgboost.py`
4. Apply the trained models to d4PDF simulations using:
   - `apply_model1.py`
   - `apply_model2.py`
5. Generate temporal-spatial frontal datasets using:
   - `HPB_get_monFt.py`
6. Perform bootstrap and climate-change analyses using:
   - `bootstrap_HPB_bft_rfreq.py`
   - `bootstrap_HFB_bft_rfreq.py`
   - `bstrp_compare.py`
   - `HPBvsHFB_temporal.py`
   - `HPBvsHFB_mm_space.py`
   - `HPBvsHFB_mm_harmonic_anom.py`
   - `HPBvsHFB_ind_anom.py`
   - `HPBvsHFB_ind_harmonic.py`
   - `HPBvsHFB_indm_space_month.py`
   - `HPBvsHFB_indm_space.py`
7. Produce manuscript figures using the plotting scripts.

## Data requirements

These scripts depend on external datasets and intermediate files, including model output and reanalysis-derived products. Depending on the script, required inputs may include:

- d4PDF GCM outputs
- ERA5-derived predictors
- trained machine-learning model files
- intermediate MATLAB or NetCDF files
- bootstrap outputs and derived analysis products

Because some data files may be too large or subject to separate archive policies, they may not be included directly in this repository. Please refer to the manuscript and associated archived datasets for access details or request them directly from the authors via email

## Software requirements

This repository contains Python scripts.

Typical scientific Python dependencies may include:

- numpy
- pandas
- scipy
- matplotlib

Machine-learning scripts may additionally require packages used for:

- tensorflow
- keras
- xgboost




