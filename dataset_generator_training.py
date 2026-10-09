# AI-CAC Project Code
# Creator: Raffi Hagopian MD

import os
import pydicom
import nibabel as nib

from scipy.ndimage import zoom
from matplotlib.path import Path

from torch.utils.data import Dataset
from monai.utils import set_determinism

from processing import * 
from visualization import * 
from array_utils import center_crop_or_pad

# Dynamic training of negative and positive slices 
class CTChestDataset_dynamic(Dataset):
    def __init__(self, study_files, study_ids, new_shape=(512, 512), zoom_factors=(1, 1), pos_slice_only=False, pos_slice_upsample_factor=1):
        self.study_files = study_files
        self.study_ids = study_ids
        self.training_slices = []
        self.negative_slices = [] 
        self.new_shape = new_shape 
        self.zoom_factors = zoom_factors
        self.pos_slice_only = pos_slice_only
        self.pos_slice_upsample_factor = pos_slice_upsample_factor

        #append slices into patient/slice/path format 
        for study_idx, study_id in enumerate(self.study_ids): 
            study_dict = study_files[study_id] # dictionary of segmentation_dir, (file_path, axial_cord) tuples
            seg_dir, slice_tuples = study_dict['seg_dir'], study_dict['slices']
            slice_tuples.sort(key=lambda x: x[1], reverse = True)   #sort by axial coordinate/reverse ordering         
            
            seg_vol_nifti = nib.load(os.path.join(seg_dir, study_id + '.nii.gz'))
            seg_vol = seg_vol_nifti.get_fdata()
            seg_vol = np.flip(seg_vol.swapaxes(0,1), axis=2) # adjust segmentation vol to match CT vol 
            
            #skip studies with mismatching segmentation and dicom slice counts  
            if seg_vol.shape[2] != len(slice_tuples):
                print(f'Skipping Study with mismatching segmentation and DCM slice counts - study: {study_id}, vol shape: {seg_vol.shape}, num slices: {len(slice_tuples)}')
                continue 
            
            for slice_iidx, (slice_file, axial_cord) in enumerate(slice_tuples):
                #print(f'vol shape: {seg_vol.shape}, num slices: {len(all_slice_files)}, slice index: {slice_iidx}')
                label_slice = seg_vol[:,:,slice_iidx] # Double check if this is correctly aligned 
                if np.max(label_slice)>0: 
                    for _ in range(self.pos_slice_upsample_factor):
                        self.training_slices.append((study_id, slice_iidx, slice_file, seg_dir))
                else:
                    self.negative_slices.append((study_id, slice_iidx, slice_file, seg_dir))
                    if not self.pos_slice_only:
                        self.training_slices.append((study_id, slice_iidx, slice_file, seg_dir))
                

                
    def add_incorrect_negative_slices(self, model, batch_size=16):
        device = next(model.parameters()).device  
        model.eval() 

        total_negs = len(self.negative_slices)

        if total_negs == 0:
            print('No neg slices')
            return 
        
        count = 0 
        added_count = 0 
        print(f'total neg slices to process: {total_negs}')
            
        for i in range(0, total_negs, batch_size):
            neg_slice_batch = self.negative_slices[i:i+batch_size]
            batch_volumes = np.zeros((batch_size, 1, *self.new_shape), dtype=float)
            for index, (study_id, slice_iidx, slice_path, slice_seg_dir) in enumerate(neg_slice_batch):
                count += 1 
                try:
                    dicom_data = pydicom.read_file(slice_path)
                    volume, _ = get_pixels_hu([dicom_data])
                except:
                    print('ERROR Unable to load study_id:', study_id)
                h, w, z = volume.shape
                tmp = volume.squeeze()
                tmp = zoom(tmp, self.zoom_factors)
                resized_volume = center_crop_or_pad(tmp, self.new_shape, dtype=float)
                resized_volume = np.expand_dims(resized_volume, axis=0)
                batch_volumes[index] = resized_volume
                
                if count % int(total_negs/10) == 0:
                    print(f'Inference count: {count}/{total_negs}, slices added count: {added_count}')
            
            # Convert to tensor and move to the same device as the model
            tensor_vol = torch.tensor(batch_volumes).float().to(device)  

            # Run inference
            with torch.no_grad():
                output = model(tensor_vol)
            
            # Check if the output contains values greater than zero
            for j in range(len(neg_slice_batch)): #range(batch_size):
                if torch.any(output[j] > 0):
                  self.training_slices.append(neg_slice_batch[j])
                  added_count += 1 
    
    def __len__(self):
        return len(self.training_slices)

    def __getitem__(self, idx):
        study_id, slice_iidx, slice_path, slice_seg_dir = self.training_slices[idx]
        dicom_data = pydicom.read_file(slice_path)
        volume, voxel_resolution = get_pixels_hu([dicom_data]) #need to change variable name 'volume' to 'slice' this is a vestigial name of prior 3D testing 

        h, w, z = volume.shape
        zoom_factors = self.zoom_factors
        
        tmp = volume.squeeze()
        tmp = zoom(tmp, zoom_factors)
        resized_volume = center_crop_or_pad(tmp, self.new_shape, dtype=float)

        resized_volume = np.expand_dims(resized_volume, axis=0)
        hu_zoom_vol = resized_volume 

        label_slice = np.zeros((h, w))
        
        seg_vol_nifti = nib.load(os.path.join(slice_seg_dir, study_id + '.nii.gz'))
        seg_vol = seg_vol_nifti.get_fdata()
        seg_vol = np.flip(seg_vol.swapaxes(0,1), axis=2) # adjust segmentation vol to match CT vol 
        label_slice = seg_vol[:,:,slice_iidx] #Need to check if this is correctly aligned     

        tmp2 = zoom(label_slice, zoom_factors, mode='nearest') 
        resized_label = center_crop_or_pad(tmp2, self.new_shape, dtype=int)

        resized_label = np.expand_dims(resized_label, axis=0)

        return resized_volume, resized_label, hu_zoom_vol, voxel_resolution[:2]

