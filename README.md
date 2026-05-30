## About

AI-CAC is a deep learning model that segments and scores coronary artery calcium (CAC) on routine non-gated, non-contrast chest CT scans. The model has been developed and internally validated in the U.S. Department of Veterans Affairs (VA) and benchmarked against paired gated CAC studies.

This repository contains:

* Trained model weights
* Inference code
* Training code

## Requirements

If you don't have `pyenv` already, get it [here](https://github.com/pyenv/pyenv).
Then navigate to the root directory of this project and set up the environment as follows:
```
pyenv install 3.9.19
pyenv virtualenv 3.9.19 ai-cac
pyenv local ai-cac
pip install --upgrade pip
pip install -r requirements.txt
```

If you get `undefined symbol` errors when trying to run `main_inference.py`,
it is possible that `torch` is attempting to access libraries from a different version
of `python` (e.g. `3.12`). Inspect your environment closely, including the
`PYTHONPATH` variable.

## Instructions for Inference (running AI-CAC scoring on non-gated, non-contrast chest CTs)

1. Create a folder for each CT chest scan study and place DICOM files from that study within the folder. The folder name will be used as the name for that study.
2. Modify the following hardcoded variables in `main_inference.py`:
   * `DICOM_ROOT_DIR` – Set to the path of the parent folder that contains the non-gated study subfolders as described above.
   * `MODEL_CHECKPOINT_FILE` – Set to the path of the model weights (download: [va_non_gated_ai_cac_model.pth](https://github.com/Raffi-Hagopian/AI-CAC/releases/download/v1.0.0/va_non_gated_ai_cac_model.pth)).
   * `SCORE_FILE` – Set to the path where the final CSV table containing the study-level AI-CAC generated calcium scores will be saved.
3. OPTIONAL:
   * `VISUALIZE_RESULTS` – Flag whether to display segmentation masks during inference (`default=False`; setting to `True` will slow inference).
   * `SAVE_MASKS` – Flag whether to save AI-CAC segmentations into PNG files (`default=False`; setting to `True` will slow inference).
   * `MASK_FOLDER` – Directory in which to save PNG masks.
4. Run `main_inference.py` to generate the AI-CAC scores for your studies.

The code will select a single non-contrast chest series per study that is most suitable for our CAC model using DICOM metadata. The script internally creates a metadata table across all the suitable imaging series, where each row represents a single DICOM file from a selected series, and has the following columns: `StudyName, DICOMFilePath, AxialPosition`. This table will be used by the inference code to run the model on each slice/DICOM file from the series and aggregate the results into a CAC score.

Addendum: This model was designed to run on non-ECG gated non-contrast chest CT scans and expects full FOV series. If you attempt to use it on a gated CT scan, filter the input to the full FOV series, and keep in mind it was not directly trained on gated slices.

## Citation

Please cite our NEJM AI paper: [doi.org/10.1056/AIoa2400937](https://doi.org/10.1056/AIoa2400937)
