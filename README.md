# Metal analysis pipeline for patient and tissue state prediction
This project presents a pipeline for quantitative analysis of spatial metal distributions, given as 2D maps of patients' tissue, within the context of predicting the state of the patient and the tissue after treatment.

This pipeline was exemplified on the TNBC dataset of the Delta Tissue project (currently unpublished).
The dataset contains 2D images reconstructed from LA-ICP-MS imaging of core biopsy tissue samples of TNBC patients before getting chemotherapy (NACT) treatment. The state of each patient was reexamined after 5 years, and the samples were annotated according to the patient response to treatment:

Non responder (NR) - state became worse

Responder (R) - pathologically completely recovered - pCR.

The code is suited for the data formats supplied by Delta tissue and contains the analysis of metals and model predictions of the tissue state.

## Pipeline overview
This pipeline has different variants but the shared steps among all of them are as follows:
1. Cleaning: Removing outliers and background
2. Input: A vector representation based on the tissue's histogram.
3. Model: Training Adaboost classifier to return the probability for a non-responder sample.
4. Patient prediction: Compute patient's probability to be non responder using its sample predictions.

# Getting started
A virtual environment for the project can be set up via:
```sh
conda create --name metalspy python=3.11.5 -y
conda activate metalspy

pip install -r requirements.txt
```
## TNBC Dataset and research result download
Members at BGU lab can access the data and the trained models at BGU SISE cluster in the following path: `/sise/assafzar-group/assafzar/metalspy/` TNBC-metals-data and TNBC-metals-model, respectively.
Otherwise, ask Leor Rose (leorro@post.bgu.ac.il) or Assaf Zaritsky (assafza@bgu.ac.il) for this data.

## TNBC Dataset setup
The path to the data of this project is defined in the config file under `data_root_directory`.
This project expects to find the following files in `<data_root_directory>/la-icp-ms` directory:
1. `240129 DeltaLeap.xlsx` - imaging date for each core tissue sample.
7. `LEAP code response data 05122023.xlsx` - metadata of each core sample. Specifically:
    * matching between core and resection
    * medium of the tissue: FFPE, Frozen or fresh
    * force trial patient
    * is the patient extreme responder (died within 2 years)
6. `cores-with-outlier-distribution-tissue-median.csv` - details on extreme cores. There is an option in the code to specify which cires to ignore. In our analysis we removed only `Leap095a`. For more details refer to `./histogram_representation/extreme_cores.py`.
9. `tnbc_dataset.xlsx` - `analysis_id` and the samples response (responder or non responder). We used `analysis_id` to split the data into groups and apply cross validation, where instances refer to the tissue samples, and groups refer to the patients.
2. `aq_cores_1.h5` - contains 5 channels (Magnesium, Manganese, Iron, Copper, Zinc) 2d images of core tissue samples. Data is calibrated. Tissue medium is FFPE. No imaging issues.
3. `aq_cores_2_FF.h5` - contains 5 channels (Magnesium, Manganese, Iron, Copper, Zinc) 2d images of core tissue samples. The Manganese channel in all images is corrupted and unreliable. Data is calibrated. Tissue medium is Frozen.
4. `aq_cores_2_FFPE.h5` - contains 5 channels (Magnesium, Manganese, Iron, Copper, Zinc) 2d images of core tissue samples. The Manganese channel in all images is corrupted and unreliable. Data is calibrated. Tissue medium is FFPE.
5. `aq.h5` - this is the first version of the `aq_cores_1.h5` file, they contain the same core tissue samples. `aq_cores_1.h5` has better image reconstruction and calibration. This file was used to read which channels exist.
8. `resection_aq.h5` - contains 5 channels (Magnesium, Manganese, Iron, Copper, Zinc) 2d images of resection tissue samples. A patient that didn't respond to a treatment underwent a surgery. The resection is the tumor that had been cut during the surgery. Data is calibrated. Tissue medium is FFPE. No imaging issues. This data is a part of the Delta Tissue Project but wasn't used here.

## Pipline configurations:

### Single metal classification :
The input of each of these piplines is a single metal channel:
1. Baseline: background and outlier removal (keeping hotspots), creating tissue histogram representation, training Adaboost classifier.
2. Validation (Hotspots excluded): same as Baseline but outliers and hotspots are removed together.
3. Positional encoding - same as Baseline but with positional encoding.
4. Yeo Johnson - same as Positional encoding but with Yeo-Johanson histogram representation.
5. Yeo Johnson permutation test - permutation test of Yeo Johson pipeline.

Each pipeline execution can be configured with different histogram representation size, include/exclude outlier cores, specify the classification metal, and `p` (percentile) threshold used for outlier (or hotspots) removal.

### 4 Metals classifier:
This pipeline uses the output probabilities of the model above on 4 non corrupted metals in our data: Magnesium, Iron, Copper and Zinc.

## Run:
### Baseline
```sh
python baseline_cv_train_eval.py \
    --config ./config/default_config.yml \
    --hist-size 20 \
    --exclude_outlier_cores False \
    --metal iron \
    --p 0.8
```
### Hotspots excluded
```sh
python hotspots_excluded_cv_train_eval.py \
    --config ./config/default_config.yml \
    --hist-size 20 \ 
    --exclude_outlier_cores False \
    --metal iron \
    --p 0.8
```
### Positional encoding
```sh
python positonal_encoding_cv_train_eval.py \
    --config ./config/default_config.yml \
    --hist-size 20 \
    --exclude_outlier_cores False \
    --metal iron \
    --p 0.8
```
### Yeo Johnson
```sh
python yeo_johnson_cv_train_eval.py \
    --config ./config/default_config.yml \
    --hist-size 20 \
    --exclude_outlier_cores False \
    --metal iron \
    --p 0.8
```
### Yeo Johnson permutation test
`percentile` and `model_seed` are configurable via the config file. In the research work the percentile is 0.8 (choosen because of experiements of baseline and hotspots excluded) and `model_seed` is 11 (choosen arbitrarly). Note: `--seed` passed as argument controls the seed of label permutation and `model_seed` controls the random seed of the ai model, `model_seed` is always fixed and only `--seed` was repeated 1,000 times with different seeds.
```sh
python yeo_johnson_permutation_test_cv_train_eval.py \
    --config ./config/default_permutation_test_config.yml \
    --hist-size 20 \
    --exclude_outlier_cores False \
    --metal iron \
    --seed 11
```
### 4 Metals classifier
```sh
python classifier_4_metal_yeo_johnson.py
```
## Slurm job execution
Slurm job definitions can be found in `./sbtach/*.batch`. And they can be executed by running the corresponding shell script

# Figures
Each pipeline creates a results directory in `./results`. And these results are analysed in `figures.ipynb`.
