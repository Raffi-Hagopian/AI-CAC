# AI-CAC Project Code
# Creator: Raffi Hagopian MD

import os
import random
import pydicom
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai 
from monai.networks.nets import SwinUNETR
 
from filter_series import * 
from dataset_generator_inference import * 
from processing import *
from visualization import *

NUM_WORKERS = 4
# Number of slices to process per model batch. Can be overridden with env var `INFERENCE_BATCH_SIZE`.
BATCH_SIZE = int(os.environ.get('INFERENCE_BATCH_SIZE', '8'))
RESAMPLE_IMAGE_SIZE = (512, 512)
RESAMPLE_SHAPE = (512, 512, 64) 
ZOOM_FACTORS = (1, 1, 1) 

SAVE_MASKS = False # Save PNGs of AI-CAC segmentations
VISUALIZE_RESULTS = False # Display segmentation masks during inference

DICOM_ROOT_DIR = 'studies2'
MODEL_CHECKPOINT_FILE = os.path.join('model', 'va_non_gated_ai_cac_model.pth')

SCORE_FILE = 'results2\scores.csv'
MASK_FOLDER = 'results2\predicted_masks'

def main():
  score_data = []
  dicom_df = create_dicom_df(DICOM_ROOT_DIR)
  print('DCM DF Created', dicom_df.shape)
  # Save full dicom dataframe for debugging
  os.makedirs('results', exist_ok=True)
  dicom_df.to_csv(os.path.join('results', 'dicom_all.csv'), index=False)
  try:
    print('Modality counts:', dicom_df['Modality'].astype(str).value_counts().to_dict())
  except Exception:
    print('Modality column missing or unreadable')
  print('DCM DF Created')
  one_series_per_study_df = filter_dicom_df(dicom_df)
  print('DCM DF Filtered')
  out_csv = os.path.join('results', 'dicom_input_one_series.csv')
  os.makedirs(os.path.dirname(out_csv), exist_ok=True)
  one_series_per_study_df.to_csv(out_csv, index=False)

  study_files = {}
  for index, row in one_series_per_study_df.iterrows():
    study = row['StudyName']
    file_path = row['DICOMFilePath']
    axial_cord = float(row['AxialPosition'])
    if study not in study_files:
      study_files[study] = [(file_path, axial_cord)]
    else:
      study_files[study].append((file_path, axial_cord))

  study_ids = []
  study_paths = []
  study_labels = []

  for study in study_files.keys():
    study_ids.append(study)
    study_paths.append(study_files[study])
    study_labels.append(-1) #None for now

  input_volume_data = CTChestDataset_nongated(study_ids, study_paths, study_labels, new_shape=RESAMPLE_SHAPE, zoom_factors=ZOOM_FACTORS) # transform was unused, took out
  input_loader = DataLoader(input_volume_data, batch_size=1, shuffle=False, num_workers=NUM_WORKERS) #Batch 1 volume at a time for volume loader, will batch slices within volume later

  model = SwinUNETR(
    spatial_dims=2,
    img_size=RESAMPLE_IMAGE_SIZE,
    in_channels=1,
    out_channels=1,
    feature_size= 96,
    use_checkpoint=True,
    drop_rate=0.2,
  )

  model = nn.DataParallel(model)

  checkpoint = torch.load(MODEL_CHECKPOINT_FILE)
  model.load_state_dict(checkpoint['model_state_dict'])

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)
  model.to(device)

  i=0
  model.eval()
  torch.cuda.empty_cache()
  with torch.no_grad():
    for study_id, inputs, targets, hu_vols, vox_dims in input_loader:
      i += 1
      study_id = study_id[0] # size one batch, first id is only id
      inputs = inputs.to(device)
      num_slices = inputs.shape[4]

      # Try processing with the configured batch size; on OOM halve the batch size and retry
      cur_batch_size = BATCH_SIZE
      while True:
        try:
          pred_vol = torch.zeros(inputs.shape, dtype=torch.float, device=device)
          for start_idx in range(0, num_slices, cur_batch_size):
            end_idx = min(start_idx + cur_batch_size, num_slices)
            batch = inputs[..., start_idx:end_idx]
            batch = batch.squeeze(0).permute(3,0,1,2)
            batch_out = model(batch.float()) #might need to move batch dimension
            batch_out = batch_out.unsqueeze(0).permute(0,2,3,4,1)
            pred_vol[..., start_idx:end_idx] = batch_out
          break
        except RuntimeError as e:
          if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            if cur_batch_size == 1:
              print('ERROR: OOM even with batch size 1; re-raising')
              raise
            cur_batch_size = max(1, cur_batch_size // 2)
            print(f'WARNING: CUDA OOM encountered, reducing slice batch size to {cur_batch_size} and retrying')
          else:
            raise

      # free intermediate cache
      torch.cuda.empty_cache()

      pred_cacs = compute_agatston_for_batch(inputs.cpu(), pred_vol.cpu(), vox_dims)

      print(i, study_id, pred_cacs[0])
      row = {'StudyName': study_id, 'AI-CAC': pred_cacs[0]}
      score_data.append(row)

      # Save predicted masks when enabled or when a positive CAC score is predicted
      try:
        if SAVE_MASKS or (isinstance(pred_cacs, (list, tuple, np.ndarray)) and pred_cacs[0] > 0) or (not isinstance(pred_cacs, (list, tuple, np.ndarray)) and pred_cacs > 0):
          save_path = os.path.join(MASK_FOLDER, study_id)
          os.makedirs(save_path, exist_ok=True)
          save_vol_masks(inputs.cpu().squeeze(), pred_vol.cpu().squeeze(), save_path)
      except Exception as e:
        print('WARNING: failed to save masks:', e)

      if VISUALIZE_RESULTS:
        draw_first_positive(inputs.cpu(), pred_vol.cpu(), pred_vol.cpu(),0)

  score_df = pd.DataFrame(score_data)
  score_df.to_csv(SCORE_FILE, index=False)


if __name__ == '__main__':
  from multiprocessing import freeze_support
  freeze_support()
  main()
