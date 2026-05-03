# d4pdf-front-detection-model-analysis

## Overview

This repository contains the source code used in the manuscript:

**Predicting the Spatiotemporal Evolution of Baiu Fronts under Climate Change by Machine Learning**

The code supports the development, evaluation, and application of machine-learning models used to detect the timing and spatial locations of Baiu fronts from historical observations, reanalysis data, and d4PDF global climate model (GCM) simulations.

In this study, a U-shaped convolutional neural network (U-Net) is used to predict frontal locations, and an extreme gradient boosting (XGBoost) model is used to predict the timing of Baiu fronts. The trained models are then applied to past and future d4PDF experiments to investigate how the temporal and spatial evolution of Baiu fronts may change under climate warming.

## Related manuscript

**Title:** Predicting the Spatiotemporal Evolution of Baiu Fronts under Climate Change by Machine Learning  
**Authors:** Yiwen Mao, Tomohito J. Yamada  
**Affiliation:** Department of Engineering, Hokkaido University  
**Journal:** Earth's Future (AGU)  
**Corresponding author:** Yiwen Mao (ymaopanda@eng.hokudai.ac.jp)

## Scientific summary

Baiu fronts are quasi-stationary, zonally oriented frontal zones that often bring extreme rainfall as they move from south to north during summer in East Asia. Their occurrence is intermittent and highly variable in both time and space.

This repository contains the code used to develop AI-based front-detection models and apply them to d4PDF GCM simulations under historical and future climate conditions. The associated manuscript shows that days with Baiu fronts tend to increase overall in a warmer climate, by up to about one week during JJA, but the changes vary substantially by month, region, and warming level.

The study suggests that under stronger warming, Baiu-front days tend to decrease between 30°N and 40°N and increase south of 30°N in June and July, while under moderate warming, front days generally increase south of 40°N in July. August shows a small increase in front days in both warming scenarios, suggesting a slightly longer rainy period.

## Key points

- Machine learning models trained using historical observations and reanalysis data are used to automatically detect Baiu fronts.
- The trained models are applied to detect Baiu fronts in past and future d4PDF GCM experiments.
- Days with Baiu fronts tend to increase both temporally and spatially, but strong regional and monthly variability is expected.

## Purpose of this repository

This repository is intended to support transparency, reproducibility, and peer review of the analysis presented in the manuscript. It includes scripts for:

- development of the U-Net and XGBoost models,
- model evaluation,
- application of trained models to d4PDF simulations,
- generation of Baiu-front and non-Baiu-front datasets,
- bootstrap and multimodel statistical analysis,
- and production of manuscript figures.

## Repository contents

### Model development and evaluation

- `develop_model1_unet.py`  
  Script used to develop model 1 (U-Net) for frontal-location detection.

- `evaluate_unet.py`  
  Calculates F1 score, Cohen's kappa, and Matthews correlation coefficient (MCC) to evaluate the U-Net model; used to produce Fig. 2(a–f).

- `evaluate_unet_additional.py`  
  Calculates additional evaluation metrics, including CSI, POD, SR, and bias; used to produce Figs. S7–S8.

- `develop_evaluate_model2_xgboost.py`  
  Develops and evaluates model 2, the XGBoost model used to identify Baiu-front days; used to produce Fig. 2(g–h).

- `XGBoost_predictors.py`  
  Calculates the XGBoost predictors described in Text S3 from ERA5 data.

### Application of trained models

- `apply_model1.py`  
  Applies the trained U-Net model to d4PDF GCM simulations to obtain frontal locations.

- `apply_model2.py`  
  Applies the trained XGBoost model to identify Baiu-front days and collect the spatial locations of all Baiu fronts.

- `HPB_get_monFt.py`  
  Extracts temporal-spatial datasets for Baiu-front and non-Baiu-front conditions in June, July, and August for use in subsequent analysis.

### Bootstrap and frequency analysis

- `bootstrap_HFB_bft_rfreq.py`  
  Calculates relative frequency for 10,000 bootstrapped samples for various HFB scenarios; used in later analysis for Fig. 4.

- `bootstrap_HPB_bft_rfreq.py`  
  Calculates relative frequency for 10,000 bootstrapped samples for HPB scenarios; used in later analysis for Fig. 4.

- `bstrp_compare.py`  
  Resamples all available years with replacement 10,000 times from historical climate (HPB) and future climate (HFB 2K/4K under SST warming scenarios CC, GF, HA, MI, MP, MR), and calculates differences in statistics for each bootstrap sample.

- `plot_rel_freq.py`  
  Calculates the relative frequency of Baiu fronts, defined as the ratio of the number of Baiu fronts occurring on a date in JJA to the total number of years used for prediction, based on HPB and multimodel means of HFB 2K and 4K scenarios.

### Historical vs future climate analysis

- `HPBvsHFB_temporal.py`  
  Calculates bootstrap differences in the average number of Baiu-front days in June, July, and August for HFB 2K/4K scenarios; used for Fig. 4(a–c).

- `HPBvsHFB_mm_space.py`  
  Calculates the multimodel mean spatial distribution of differences in the average number of Baiu-front days in JJA.

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

### Historical-climate diagnostics and plotting

- `plot_hpb_boxplot.py`  
  Calculates the average number of Baiu-front days per month in JJA; used for Fig. 3c.

- `plot_hpb_spa_dist_JJA.py`  
  Calculates the spatial distribution of the average number of Baiu-front days in JJA; used for Fig. 3(d–f).

- `plot_hbp_stat.py`  
  Calculates spatial distributions of statistics related to the average number of Baiu-front and non-Baiu-front days in JJA; used for Fig. 3(g–l).

## Experiments analyzed

This repository includes analysis of the following d4PDF experiments:

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

A typical workflow is:

1. Prepare ERA5-based predictors for XGBoost using `XGBoost_predictors.py`
2. Develop the frontal-location model using `develop_model1_unet.py`
3. Evaluate the U-Net model using `evaluate_unet.py` and `evaluate_unet_additional.py`
4. Develop and evaluate the Baiu-day classifier using `develop_evaluate_model2_xgboost.py`
5. Apply the trained models to d4PDF simulations using `apply_model1.py` and `apply_model2.py`
6. Build temporal-spatial front datasets using `HPB_get_monFt.py`
7. Perform bootstrap and climate-change analyses using the bootstrap and `HPBvsHFB_*` scripts
8. Generate manuscript figures using the corresponding plotting scripts

## Data requirements

The scripts depend on external datasets and intermediate files, which may include:

- d4PDF GCM outputs
- ERA5-based predictors
- trained machine-learning model files
- MATLAB intermediate files
- NetCDF analysis products
- derived bootstrap outputs

These data may not all be included in this repository. Users should refer to the manuscript data/software availability statements and associated archived datasets for access details.

## Software requirements

This project uses Python and scientific Python libraries. Depending on the workflow stage, required packages may include:

- numpy
- pandas
- scipy
- xarray
- matplotlib

Additional machine-learning dependencies may be required for:
- tensorflow
- keras
- xgboost




