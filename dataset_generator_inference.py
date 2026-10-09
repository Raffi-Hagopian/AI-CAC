# AI-CAC Project Code
# Creator: Raffi Hagopian MD
# CT Chest Dataset Full Volume Loader

import os
import pydicom
import torch
from scipy.ndimage import zoom
from matplotlib.path import Path
from torch.utils.data import Dataset

from processing import * 
from array_utils import center_crop_or_pad

class CTChestDataset_nongated(Dataset):
    def __init__(self, study_ids, study_files, study_labels, transform=None, new_shape=(512, 512, 64), zoom_factors=(1, 1, 1)):
        self.study_ids = study_ids
        self.study_files = study_files
        self.study_labels = study_labels
        self.transform = transform
        self.new_shape = new_shape 
        self.zoom_factors = zoom_factors

    def __len__(self):
        return len(self.study_ids)
    
    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        files = self.study_files[idx] 
        
        try:
          volume, voxel_resolution = get_pixels_hu(load_pydicom_slices_by_axial_cord(files)) # make sure order of slices matches that for segmentations 
        except Exception as e:
          print(f"Error loading study{study_id}: {e}")
          return study_id+'_corrupt', torch.zeros(512, 512, 64), torch.zeros(1), torch.zeros(512,512,64), np.array([0,0,0]) #dummy variables to skip for corrupt data 
        
        h, w, z_length = volume.shape
        new_shape = self.new_shape 
        zoom_factors = self.zoom_factors
        tmp = volume
        tmp = zoom(tmp, zoom_factors) 
        volume = center_crop_or_pad(tmp, new_shape[:2], dtype=float) # Preserve Z-axis
        volume = np.expand_dims(volume, axis=0)
        
        hu_zoom_vol = volume 
        zoom_voxel = np.array([voxel_resolution[0]/zoom_factors[0], voxel_resolution[1]/zoom_factors[1], voxel_resolution[2]/zoom_factors[2]])
        
        label = self.study_labels[idx]
        
        return study_id, volume, torch.tensor(label, dtype=torch.float), hu_zoom_vol, zoom_voxel #if a transformation/normalization is applied to volume, then volume and hu_vol are different 

