# AI-CAC Project Code
# Creator: Raffi Hagopian MD

import os
import re
import ast 
import pydicom 
import pandas as pd 
import matplotlib.pyplot as plt

# Extract selected attributes using pydicom 
def extract_dicom_attributes(dicom_file):
    attributes = {
        'Modality' : None,
        'SliceThickness' : None,
        'SeriesDescription' : None,
        'StudyDescription' : None,
        'KVP' : None,
        'ConvolutionKernel' : None,
        'ImageOrientationPatient' : None,
        'ImageType': None,
        'ContrastBolusAgent': None, 
        'BodyPartExamined': None, 
        'AcquisitionTime' : None, 
        'SeriesInstanceUID': None, 
        'ImagePositionPatient': None, # Used to get slice axial position 
    }
    try: 
        ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
        for attr in attributes.keys():
            attributes[attr] = getattr(ds, attr, None)
    except Exception as e:
        print(f"Error reading DICOM file {dicom_file}: {e}")
    return attributes 
  
# Create Pandas dataframe of StudyName, DICOMFilePath, and DICOM Attributes 
def create_dicom_df(root_dir):
    data = []
    # If the provided path directly contains DICOM files, treat it as a single study
    try:
        entries = os.listdir(root_dir)
    except Exception:
        return pd.DataFrame(data)

    has_dcm_files = any(
        f.lower().endswith('.dcm') and os.path.isfile(os.path.join(root_dir, f))
        for f in entries
    )

    if has_dcm_files:
        study_name = os.path.basename(os.path.normpath(root_dir))
        for parent_path, _, files in os.walk(root_dir):
            for file_name in files:
                if file_name.lower().endswith('.dcm'):
                    dicom_path = os.path.join(parent_path, file_name)
                    dicom_attr = extract_dicom_attributes(dicom_path)
                    row = {
                        'StudyName': study_name,
                        'DICOMFilePath': dicom_path,
                        **dicom_attr,
                    }
                    data.append(row)
    else:
        # Assume each top-level directory under root_dir is a study
        for study_name in entries:
            study_path = os.path.join(root_dir, study_name)
            if os.path.isdir(study_path):
                for parent_path, _, files in os.walk(study_path):
                    for file_name in files:
                        if file_name.lower().endswith('.dcm'):
                            dicom_path = os.path.join(parent_path, file_name)
                            dicom_attr = extract_dicom_attributes(dicom_path)
                            row = {
                                'StudyName': study_name,
                                'DICOMFilePath': dicom_path,
                                **dicom_attr,
                            }
                            data.append(row)

    df = pd.DataFrame(data)
    return df


# Filter to keep only one series per study
# Criteria:
#  1) At least 15 slices
#  2) Non-contrast 
#  3) Axial in Orientation
#  4) No Study, Series, or Body Part descriptors containing non-chest anatomy words [head, skull, brain, ... etc]
#  5) Of the remaining series, select the first series in order of the following descriptors [calc, casc, cac, ca]
#  6) If none of the above words match series/body part descriptors, select the first remaining series in the dataframe
 
def filter_dicom_df(dicom_df):

    def is_axial(orientation_list):
        if orientation_list in (None, 'None'):
            return True
        try:
            orientation_list = ast.literal_eval(orientation_list)
            # normalize to integers (handle floats like -0.0 / -1.0)
            int_list = [int(round(float(x))) for x in orientation_list]
            if len(int_list) < 6:
                return False
            abs_list = [abs(x) for x in int_list[:6]]
            # Accept axial orientation even if the direction is flipped (sign differences)
            return abs_list == [1,0,0,0,1,0]
        except Exception:
            return False
    
    def keep_series(study_desc, series_desc, body_part): 
        # Allow missing StudyDescription as long as SeriesDescription is present
        if (study_desc in ('None', '')) and (series_desc in ('None', '')):
            # If both study and series descriptions are missing, allow the series
            # if BodyPartExamined indicates chest/lung anatomy; otherwise skip.
            if body_part in (None, 'None', ''):
                return False
            # otherwise continue and evaluate body_part below
        study_desc = study_desc.lower()
        series_desc = series_desc.lower()
        body_part = body_part.lower()
        filter_terms = ['head', 'brain', 'skull', 'sinus', 'maxillofacial', 'neck', 'spine', 'sternum', 'bone', 'abdomen', 'abd', 'adrenal', 'liver', 'kidney', 'colon', 'pelvis', 'femoral', 'leg', 'extremity', 'abscess', 'needle', 'drain', 'circleofwillis', 'tavr', 'mip', 'arterial', 'venous', 'delay', 'runoff', 'enhanced']
        for term in filter_terms:
            if term in study_desc or term in series_desc or term in body_part:
                return False
        pattern = r'(?i)(?<!\S)A/P(?!\S)' #remove any 'A/P' for abdomen/pelvis if no preceeding or following non-whitespace chacerters 
        if re.search(pattern, series_desc):
            return False
        return True 
    
    def filter_row(row):
        study_desc = row['StudyDescription']
        orient = row['ImageOrientationPatient']
        series_desc = row['SeriesDescription']
        body_part = row['BodyPartExamined']
        if row['ContrastBolusAgent'] != 'None': 
            return False
        if not is_axial(orient):
            return False
        if not keep_series(study_desc, series_desc, body_part):
            return False
        return True 
  
    def select_row(group):
        series_desc = group['SeriesDescription'].str.lower() # set to lower case
        body_part = group['BodyPartExamined'].str.lower()
        # Return the first series for this study that satisfies this search
        # Initially searching over calcium terms, then heart anatomy, then lung, then chest, then if none matched, pick first series left
        if any('calc' in s for s in series_desc):
            return group[series_desc.str.contains('calc')].iloc[0]
        if any('cacs' in s for s in series_desc):
            return group[series_desc.str.contains('cacs')].iloc[0]
        if any('ca ' in s for s in series_desc):
            return group[series_desc.str.contains('ca ')].iloc[0]
        if any('cac' in s for s in series_desc):
            return group[series_desc.str.contains('cac')].iloc[0]
        if any('calcium' in s for s in body_part):
            return group[body_part.str.contains('calcium')].iloc[0]
        if any('ca ' in s for s in body_part):
            return group[body_part.str.contains('ca ')].iloc[0]
        if any('heart' in s for s in body_part):
            return group[body_part.str.contains('heart')].iloc[0]
        if any('card' in s for s in series_desc):
            return group[series_desc.str.contains('card')].iloc[0]
        if any('lung' in s for s in series_desc):
            return group[series_desc.str.contains('lung')].iloc[0]
        if any('chest' in s for s in body_part):
            return group[body_part.str.contains('chest')].iloc[0]
        else:
            return group.iloc[0]

    # If a series is repeated within a study (similar series specific description, settings tags etc), keep the one that was acquired later in time (more likely higher quality)
    def keep_latest_series_if_repeated(dicom_df):
        dicom_df['AcquisitionTime'] = pd.to_numeric(dicom_df['AcquisitionTime'], errors='coerce')
        dicom_df['AcquisitionTime'] = dicom_df['AcquisitionTime'].fillna(0) # set NA to 0
        max_time_per_pair = dicom_df.groupby(['StudyName','SeriesInstanceUID'])['AcquisitionTime'].max().reset_index() # Get the latest Study/SeriesID pair timestamp 
        latest_pairs = max_time_per_pair.loc[max_time_per_pair.groupby('StudyName')['AcquisitionTime'].idxmax()] # For each study, keep the series with the latest timestamp
        result_df = pd.merge(dicom_df, latest_pairs[['StudyName','SeriesInstanceUID']], on=['StudyName', 'SeriesInstanceUID'])
        return result_df

    series_specific_columns = ['StudyName','StudyDescription','SeriesDescription', 'SliceThickness', 'ImageType', 'ConvolutionKernel', 'ImageOrientationPatient', 'KVP', 'ContrastBolusAgent', 'BodyPartExamined']

    # Ensure expected columns exist to avoid KeyErrors during filtering/grouping
    expected_cols = set(series_specific_columns + ['DICOMFilePath', 'SeriesInstanceUID', 'Modality', 'ImagePositionPatient', 'AcquisitionTime'])
    for col in expected_cols:
        if col not in dicom_df.columns:
            dicom_df[col] = None

    # Filter by modality if present; match case-insensitively and handle non-string values
    if 'Modality' in dicom_df.columns:
        dicom_df = dicom_df[dicom_df['Modality'].astype(str).str.upper() == 'CT']
    else:
        dicom_df = dicom_df.iloc[0:0]
    dicom_df['SliceThickness'] =  pd.to_numeric(dicom_df['SliceThickness'], errors='coerce')
    # Allow thinner-slice CTs (e.g., 0.625 mm) commonly found in LDCT datasets
    dicom_df = dicom_df[(dicom_df['SliceThickness'] >= 0.5) & (dicom_df['SliceThickness'] <= 5)] # Keep DICOMs with SliceThickness between 0.5 and 5
    dicom_df.fillna('None', inplace = True) # Replace NA with string 'None'
    dicom_df = dicom_df.astype(str) # allows for grouping in certain columns 
    # If dicom_df is empty after filtering, return an empty dataframe with expected columns
    if dicom_df.empty:
        print('DEBUG: dicom_df is empty after modality filter; returning empty dataframe')
        return pd.DataFrame(columns=series_specific_columns + ['size'])

    group_df = dicom_df.groupby(series_specific_columns, as_index = False).size() # Group by these columns to get one row per series in the dataframe 
    group_df = group_df[group_df['size'] > 15] # Only keep series with at least 15 slices
    print('DEBUG: dicom_df shape:', dicom_df.shape)
    print('DEBUG: dicom_df columns:', dicom_df.columns.tolist())
    print('DEBUG: group_df shape:', group_df.shape)
    print('DEBUG: group_df columns:', group_df.columns.tolist())
    try:
        print('DEBUG: group_df head:\n', group_df.head().to_string())
    except Exception:
        pass
    mask = group_df.apply(filter_row, axis=1)
    filt_df = group_df[mask] # Filter out series with contrast, non-axial slices, non-chest body part
    print('DEBUG: filt_df shape:', filt_df.shape)
    print('DEBUG: filt_df columns:', filt_df.columns.tolist())
    try:
        print('DEBUG: filt_df head:\n', filt_df.head().to_string())
    except Exception:
        pass
    select_df = filt_df.groupby('StudyName', group_keys=False).apply(select_row).reset_index(drop=True) # Select series based on search term priorties calcium terms, cardiac terms, lung, chest etc
    one_series_per_study_df = pd.merge(dicom_df, select_df, on = series_specific_columns) # Expand selected series dataframe to now have rows for each slice (still only one series per study selected)

    one_series_per_study_df = keep_latest_series_if_repeated(one_series_per_study_df) # If series specific settings were repeated twice, keep the one with the latest timestamp 
    one_series_per_study_df['AxialPosition'] = one_series_per_study_df['ImagePositionPatient'].apply(lambda coord: coord.split(', ')[-1].replace(']', '')) # Take Axial position out of [#, #, #]
    one_series_per_study_df = one_series_per_study_df.astype(str)
    return one_series_per_study_df

