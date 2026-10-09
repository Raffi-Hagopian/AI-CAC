import os
import uuid
from pathlib import Path as FilePath
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.networks.nets import SwinUNETR


# ---------------------------------------------------------------------
# Project-specific imports
# ---------------------------------------------------------------------
# Avoid importing Path from matplotlib accidentally.
# If your project modules expose these functions/classes directly,
# these explicit imports are safest.

try:
    from filter_series import create_dicom_df, filter_dicom_df
except ImportError:
    from filter_series import *
    # create_dicom_df and filter_dicom_df should now exist in globals.

try:
    from dataset_generator_inference import CTChestDataset_nongated
except ImportError:
    from dataset_generator_inference import *

try:
    from processing import compute_agatston_for_batch
except ImportError:
    from processing import *

try:
    from visualization import save_vol_masks
except ImportError:
    from visualization import *


class CACSInferenceService:
    def __init__(
        self,
        model_checkpoint_file: str,
        mask_root: str = "storage/outputs/masks",
        debug_root: str = "storage/outputs/debug",
        batch_size: int = 8,
        num_workers: int = 0,
        s3_service=None,
    ):
        self.model_checkpoint_file = model_checkpoint_file

        # IMPORTANT:
        # Use FilePath, not Path.
        # visualization.py / matplotlib may import matplotlib.path.Path,
        # which breaks pathlib-style path handling.
        self.mask_root = FilePath(mask_root)
        self.debug_root = FilePath(debug_root)
        
        # S3 service for uploading outputs
        self.s3_service = s3_service
        self.use_s3 = s3_service is not None
        
        # Delete local files after S3 upload (saves disk space)
        self.cleanup_after_s3 = os.environ.get("S3_CLEANUP_LOCAL", "false").lower() in {"true", "1", "yes"}

        self.batch_size = int(os.environ.get("INFERENCE_BATCH_SIZE", batch_size))
        self.num_workers = num_workers

        self.resample_image_size = (512, 512)
        self.resample_shape = (512, 512, 64)
        self.zoom_factors = (1, 1, 1)

        # Create local directories (used as temp storage even with S3)
        self.mask_root.mkdir(parents=True, exist_ok=True)
        self.debug_root.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("Loading AI-CAC model...")
        print("Device:", self.device)
        print("Checkpoint:", self.model_checkpoint_file)

        self.model = self._load_model()

        print("AI-CAC model loaded successfully.")

    # -----------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------

    def _load_model(self):
        model = SwinUNETR(
            spatial_dims=2,
            img_size=self.resample_image_size,
            in_channels=1,
            out_channels=1,
            feature_size=96,
            use_checkpoint=True,
            drop_rate=0.2,
        )

        # Your original inference.py used DataParallel before loading
        # checkpoint['model_state_dict'], so we preserve that.
        model = nn.DataParallel(model)

        checkpoint = torch.load(
            self.model_checkpoint_file,
            map_location=self.device,
        )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain key 'model_state_dict'. "
                "Check that MODEL_CHECKPOINT_FILE points to the correct .pth file."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        return model

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _safe_study_name(self, study_name: str) -> str:
        """
        Make study name safe for folder creation.
        """
        study_name = str(study_name)

        return "".join(
            c if c.isalnum() or c in ("-", "_", ".") else "_"
            for c in study_name
        )

    def _write_debug_csvs(
        self,
        request_id: str,
        dicom_df: pd.DataFrame,
        filtered_df: pd.DataFrame,
    ):
        """
        Save debug CSVs for troubleshooting input filtering.
        Uploads to S3 if s3_service is configured.
        """
        debug_request_dir = self.debug_root / request_id
        debug_request_dir.mkdir(parents=True, exist_ok=True)

        dicom_all_path = debug_request_dir / "dicom_all.csv"
        dicom_filtered_path = debug_request_dir / "dicom_input_one_series.csv"
        
        dicom_df.to_csv(dicom_all_path, index=False)
        filtered_df.to_csv(dicom_filtered_path, index=False)
        
        # Upload to S3 if enabled
        if self.use_s3:
            try:
                self.s3_service.upload_file(
                    str(dicom_all_path),
                    f"debug/{request_id}/dicom_all.csv"
                )
                self.s3_service.upload_file(
                    str(dicom_filtered_path),
                    f"debug/{request_id}/dicom_input_one_series.csv"
                )
            except Exception as e:
                print(f"WARNING: Failed to upload debug CSVs to S3: {e}")

    def _validate_filtered_df(self, filtered_df: pd.DataFrame):
        """
        Validate required columns before building study paths.
        """
        required_columns = ["StudyName", "DICOMFilePath", "AxialPosition"]

        missing = [col for col in required_columns if col not in filtered_df.columns]

        if missing:
            raise ValueError(
                f"Filtered DICOM dataframe is missing required columns: {missing}. "
                f"Available columns: {filtered_df.columns.tolist()}"
            )

        if filtered_df.empty:
            raise ValueError(
                "No usable CT series found after filter_dicom_df(). "
                "Check whether the uploaded zip contains valid DICOM CT files."
            )

        bad_axial = filtered_df[
            pd.to_numeric(filtered_df["AxialPosition"], errors="coerce").isna()
        ]

        if not bad_axial.empty:
            preview_cols = [
                col for col in ["StudyName", "DICOMFilePath", "AxialPosition"]
                if col in bad_axial.columns
            ]

            preview = bad_axial[preview_cols].head(20).to_string(index=False)

            raise ValueError(
                "Invalid non-numeric AxialPosition found in filtered DICOM dataframe. "
                "This usually means a non-DICOM/output folder was scanned as input, "
                "or the dataframe columns became misaligned.\n\n"
                f"Bad rows preview:\n{preview}"
            )

    def _build_study_dataset(self, dicom_root_dir: str, request_id: str):
        """
        Replicates the dataframe and dataset-building portion of inference.py.
        """
        dicom_root = FilePath(dicom_root_dir)

        if not dicom_root.exists():
            raise FileNotFoundError(f"DICOM root directory does not exist: {dicom_root}")

        if not dicom_root.is_dir():
            raise NotADirectoryError(f"DICOM root is not a directory: {dicom_root}")

        print("DICOM root passed to predict_folder:", str(dicom_root))

        dicom_df = create_dicom_df(str(dicom_root))

        if dicom_df is None or dicom_df.empty:
            raise ValueError(
                f"No DICOM records found by create_dicom_df() in: {dicom_root}"
            )

        print("DICOM DF Created:", dicom_df.shape)

        try:
            print(
                "Modality counts:",
                dicom_df["Modality"].astype(str).value_counts().to_dict(),
            )
        except Exception:
            print("Modality column missing or unreadable.")

        filtered_df = filter_dicom_df(dicom_df)

        if filtered_df is None:
            raise ValueError("filter_dicom_df() returned None.")

        print("DICOM DF Filtered:", filtered_df.shape)
        print("Filtered dataframe columns:", filtered_df.columns.tolist())
        print(filtered_df.head(10).to_string())

        self._validate_filtered_df(filtered_df)
        self._write_debug_csvs(request_id, dicom_df, filtered_df)

        study_files = {}

        for _, row in filtered_df.iterrows():
            study = row["StudyName"]
            file_path = row["DICOMFilePath"]

            try:
                axial_coord = float(row["AxialPosition"])
            except Exception as e:
                raise ValueError(
                    f"Could not convert AxialPosition to float. "
                    f"StudyName={row.get('StudyName')}, "
                    f"DICOMFilePath={row.get('DICOMFilePath')}, "
                    f"AxialPosition={row.get('AxialPosition')}"
                ) from e

            if study not in study_files:
                study_files[study] = []

            study_files[study].append((file_path, axial_coord))

        study_ids = []
        study_paths = []
        study_labels = []

        for study, files in study_files.items():
            study_ids.append(study)
            study_paths.append(files)
            study_labels.append(-1)

        if len(study_ids) == 0:
            raise ValueError("No studies were created after DICOM filtering.")

        input_volume_data = CTChestDataset_nongated(
            study_ids,
            study_paths,
            study_labels,
            new_shape=self.resample_shape,
            zoom_factors=self.zoom_factors,
        )

        input_loader = DataLoader(
            input_volume_data,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return input_loader, filtered_df

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    def _predict_volume(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Runs 2D SwinUNETR slice-by-slice across the volume.

        This preserves your original batching logic:
        inputs shape is expected to have slices in dim 4.
        """
        inputs = inputs.to(self.device)
        num_slices = inputs.shape[4]

        cur_batch_size = self.batch_size

        while True:
            try:
                pred_vol = torch.zeros(
                    inputs.shape,
                    dtype=torch.float,
                    device=self.device,
                )

                for start_idx in range(0, num_slices, cur_batch_size):
                    end_idx = min(start_idx + cur_batch_size, num_slices)

                    batch = inputs[..., start_idx:end_idx]

                    # Original logic:
                    # batch = batch.squeeze(0).permute(3, 0, 1, 2)
                    batch = batch.squeeze(0).permute(3, 0, 1, 2)

                    batch_out = self.model(batch.float())

                    # Original logic:
                    # batch_out = batch_out.unsqueeze(0).permute(0, 2, 3, 4, 1)
                    batch_out = batch_out.unsqueeze(0).permute(0, 2, 3, 4, 1)

                    pred_vol[..., start_idx:end_idx] = batch_out

                return pred_vol

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if cur_batch_size == 1:
                        raise RuntimeError(
                            "CUDA out of memory even with slice batch size 1."
                        ) from e

                    cur_batch_size = max(1, cur_batch_size // 2)

                    print(
                        "WARNING: CUDA OOM encountered. "
                        f"Reducing slice batch size to {cur_batch_size} and retrying."
                    )

                else:
                    raise

    def predict_folder(
        self,
        dicom_root_dir: str,
        request_id: str = None,
        save_masks: bool = True,
    ) -> Dict[str, Any]:
        """
        Run AI-CAC inference on a folder containing one or more DICOM studies.

        Parameters
        ----------
        dicom_root_dir:
            Folder containing extracted DICOM files.
            This should be something like:
            storage/incoming/<request_id>/dicoms

        request_id:
            Optional external request id.
            If not provided, one is generated.

        save_masks:
            Whether to save predicted mask PNGs.

        Returns
        -------
        dict:
            JSON-serializable result containing AI-CAC score and mask references.
        """
        if request_id is None:
            request_id = uuid.uuid4().hex

        input_loader, selected_series_df = self._build_study_dataset(
            dicom_root_dir=dicom_root_dir,
            request_id=request_id,
        )

        results = []

        with torch.no_grad():
            for i, batch_data in enumerate(input_loader, start=1):
                study_id, inputs, targets, hu_vols, vox_dims = batch_data

                study_id = study_id[0]
                safe_study_id = self._safe_study_name(study_id)

                print(f"Running AI-CAC inference for study {i}: {study_id}")

                pred_vol = self._predict_volume(inputs)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                pred_cacs = compute_agatston_for_batch(
                    inputs.cpu(),
                    pred_vol.cpu(),
                    vox_dims,
                )

                if isinstance(pred_cacs, (list, tuple, np.ndarray)):
                    ai_cac = float(pred_cacs[0])
                else:
                    ai_cac = float(pred_cacs)

                print(f"AI-CAC result: study={study_id}, score={ai_cac}")

                mask_dir = None
                mask_files: List[str] = []

                if save_masks:
                    # Save masks locally (temporary if using S3)
                    mask_dir_path = self.mask_root / request_id / safe_study_id
                    mask_dir_path.mkdir(parents=True, exist_ok=True)

                    try:
                        save_vol_masks(
                            inputs.cpu().squeeze(),
                            pred_vol.cpu().squeeze(),
                            str(mask_dir_path),
                        )

                        mask_dir = str(mask_dir_path)

                        # Collect local mask files
                        local_mask_files = []
                        for file_path in sorted(mask_dir_path.glob("*")):
                            if file_path.is_file():
                                local_mask_files.append(file_path)

                        # Upload to S3 if enabled
                        if self.use_s3:
                            try:
                                from concurrent.futures import ThreadPoolExecutor, as_completed
                                
                                print(f"Starting S3 upload for {len(local_mask_files)} mask files...")
                                
                                # Parallel upload with thread pool
                                with ThreadPoolExecutor(max_workers=4) as executor:
                                    future_to_file = {
                                        executor.submit(
                                            self.s3_service.upload_file,
                                            str(local_file),
                                            f"masks/{request_id}/{safe_study_id}/{local_file.name}"
                                        ): local_file
                                        for local_file in local_mask_files
                                    }
                                    
                                    upload_count = 0
                                    for future in as_completed(future_to_file):
                                        try:
                                            s3_key = future.result()
                                            mask_files.append(s3_key)
                                            upload_count += 1
                                        except Exception as e:
                                            local_file = future_to_file[future]
                                            print(f"ERROR: Failed to upload {local_file.name}: {type(e).__name__}: {e}")
                                            import traceback
                                            traceback.print_exc()
                                
                                print(f"✅ Uploaded {upload_count}/{len(local_mask_files)} masks to S3 for {study_id}")
                                
                                if upload_count == 0:
                                    print(f"⚠️  WARNING: No masks uploaded to S3, falling back to local storage")
                                    mask_files = [str(f) for f in local_mask_files]
                                
                                # Clean up local files after S3 upload (optional)
                                if self.cleanup_after_s3:
                                    import shutil
                                    try:
                                        shutil.rmtree(mask_dir_path)
                                        print(f"Cleaned up local masks for {study_id}")
                                    except Exception as e:
                                        print(f"WARNING: Failed to clean up local masks: {e}")
                            except Exception as e:
                                print(f"❌ ERROR: S3 upload completely failed for {study_id}: {type(e).__name__}: {e}")
                                import traceback
                                traceback.print_exc()
                                # Fallback to local paths on S3 upload failure
                                print(f"⚠️  Falling back to local storage for {study_id}")
                                mask_files = [str(f) for f in local_mask_files]
                        else:
                            # Use local file paths
                            mask_files = [str(f) for f in local_mask_files]

                    except Exception as e:
                        print(f"WARNING: failed to save masks for {study_id}: {e}")
                        mask_dir = str(mask_dir_path)

                results.append(
                    {
                        "study_id": str(study_id),
                        "ai_cac": ai_cac,
                        "mask_dir": mask_dir,
                        "mask_files": mask_files,
                    }
                )

        return {
            "request_id": request_id,
            "num_studies": len(results),
            "results": results,
        }