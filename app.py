print("app.py was executed")

import os

# Force Matplotlib to use a non-GUI backend.
# Needed because Flask request handlers may run outside the main thread.
os.environ["MPLBACKEND"] = "Agg"

import uuid
import zipfile
import traceback
from pathlib import Path as FilePath

from flask import Flask, request, jsonify, send_from_directory, redirect
from werkzeug.utils import secure_filename

from cac_inference_service import CACSInferenceService
from s3_storage import S3StorageService


# -----------------------------
# Configuration
# -----------------------------

MODEL_CHECKPOINT_FILE = os.environ.get(
    "MODEL_CHECKPOINT_FILE",
    os.path.join("model", "va_non_gated_ai_cac_model.pth"),
)

# S3 Configuration
USE_S3_STORAGE = os.environ.get("USE_S3_STORAGE", "false").lower() in {"true", "1", "yes"}
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "")
S3_PREFIX = os.environ.get("S3_PREFIX", "RADFLOW/outputs/ai-cac-outputs")
S3_URL_EXPIRATION = int(os.environ.get("S3_URL_EXPIRATION", "3600"))  # 1 hour default

BASE_STORAGE = FilePath("storage")

# Incoming DICOM uploads only (always local for processing)
INCOMING_ROOT = BASE_STORAGE / "incoming"

# Outputs - can be local or S3
OUTPUT_ROOT = BASE_STORAGE / "outputs"
MASK_ROOT = OUTPUT_ROOT / "masks"
DEBUG_ROOT = OUTPUT_ROOT / "debug"

# Always create incoming directory (needed for DICOM extraction)
INCOMING_ROOT.mkdir(parents=True, exist_ok=True)

# Create output directories only if not using S3
if not USE_S3_STORAGE:
    MASK_ROOT.mkdir(parents=True, exist_ok=True)
    DEBUG_ROOT.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Flask app
# -----------------------------

app = Flask(__name__)


# -----------------------------
# Initialize S3 Storage (if enabled)
# -----------------------------

s3_service = None

if USE_S3_STORAGE:
    if not S3_BUCKET_NAME:
        raise ValueError(
            "USE_S3_STORAGE is enabled but S3_BUCKET_NAME is not set. "
            "Set S3_BUCKET_NAME environment variable."
        )
    
    print("Initializing S3 storage...")
    print(f"  Bucket: {S3_BUCKET_NAME}")
    print(f"  Prefix: {S3_PREFIX}")
    print(f"  URL Expiration: {S3_URL_EXPIRATION}s")
    
    s3_service = S3StorageService(
        bucket_name=S3_BUCKET_NAME,
        prefix=S3_PREFIX,
        presigned_url_expiration=S3_URL_EXPIRATION,
    )
    
    print("S3 storage initialized successfully.")
else:
    print("Using local file storage (S3 disabled).")


# -----------------------------
# Auto-load AI-CAC model at startup
# -----------------------------

print("Loading AI-CAC service at startup...")
print("Model checkpoint:", MODEL_CHECKPOINT_FILE)
print("Mask root:", MASK_ROOT)
print("Debug root:", DEBUG_ROOT)

cac_service = CACSInferenceService(
    model_checkpoint_file=MODEL_CHECKPOINT_FILE,
    mask_root=str(MASK_ROOT),
    debug_root=str(DEBUG_ROOT),
    batch_size=int(os.environ.get("INFERENCE_BATCH_SIZE", "16")),
    num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", "4")),
    s3_service=s3_service,
)

print("AI-CAC service loaded successfully.")


# -----------------------------
# Helpers
# -----------------------------

def safe_extract_zip(zip_path: FilePath, extract_dir: FilePath):
    """
    Safely extract a zip file while preventing path traversal.
    """
    extract_dir = extract_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            target_path = (extract_dir / member.filename).resolve()

            if not str(target_path).startswith(str(extract_dir)):
                raise ValueError(f"Unsafe zip path detected: {member.filename}")

        zip_ref.extractall(extract_dir)


def list_extracted_files(folder: FilePath, max_files: int = 20):
    """
    Lightweight debug helper.
    """
    files = []

    for path in folder.rglob("*"):
        if path.is_file():
            files.append(str(path))

        if len(files) >= max_files:
            break

    return files


def local_mask_path_to_url(mask_file: str) -> str:
    """
    Convert storage/outputs/masks/... path into /masks/... URL (local mode)
    or S3 URI (S3 mode).
    """
    if USE_S3_STORAGE and s3_service:
        # For S3 mode, mask_file contains the S3 key
        # Return S3 URI format: s3://bucket/key
        return f"s3://{S3_BUCKET_NAME}/{mask_file}"
    else:
        # For local mode, convert file path to local URL
        mask_path = FilePath(mask_file).resolve()
        mask_root_resolved = MASK_ROOT.resolve()
        
        relative_path = mask_path.relative_to(mask_root_resolved)
        
        return f"/masks/{relative_path.as_posix()}"


# -----------------------------
# Routes
# -----------------------------

@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "success": True,
            "message": "AI-CAC Flask server is running.",
            "health_url": "/health",
            "predict_url": "/predict",
        }
    )


@app.route("/health", methods=["GET"])
def health():
    health_data = {
        "success": True,
        "status": "ok",
        "model_loaded": True,
        "device": str(cac_service.device),
        "model_checkpoint_file": MODEL_CHECKPOINT_FILE,
        "incoming_root": str(INCOMING_ROOT),
        "storage_mode": "s3" if USE_S3_STORAGE else "local",
    }
    
    if USE_S3_STORAGE:
        health_data["s3_bucket"] = S3_BUCKET_NAME
        health_data["s3_prefix"] = S3_PREFIX
    else:
        health_data["mask_root"] = str(MASK_ROOT)
        health_data["debug_root"] = str(DEBUG_ROOT)
    
    return jsonify(health_data)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a zipped DICOM study folder.

    Expected form-data:
      study_zip: file
      save_masks: optional true/false
    """
    if "study_zip" not in request.files:
        return jsonify(
            {
                "success": False,
                "error": "Missing file field 'study_zip'. Upload a zipped DICOM study.",
            }
        ), 400

    uploaded_file = request.files["study_zip"]

    if uploaded_file.filename is None or uploaded_file.filename.strip() == "":
        return jsonify(
            {
                "success": False,
                "error": "Empty filename.",
            }
        ), 400

    request_id = uuid.uuid4().hex

    save_masks_raw = request.form.get("save_masks", "true").strip().lower()
    save_masks = save_masks_raw in {"true", "1", "yes", "y"}

    filename = secure_filename(uploaded_file.filename)

    request_dir = INCOMING_ROOT / request_id
    zip_path = request_dir / filename

    # This is the only folder that should be passed to predict_folder().
    dicom_extract_dir = request_dir / "dicoms"

    request_dir.mkdir(parents=True, exist_ok=True)
    dicom_extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        uploaded_file.save(zip_path)

        safe_extract_zip(zip_path, dicom_extract_dir)

        extracted_preview = list_extracted_files(dicom_extract_dir)

        print("REQUEST ID:", request_id)
        print("ZIP PATH:", zip_path)
        print("DICOM EXTRACT DIR:", dicom_extract_dir)
        print("MASK ROOT:", MASK_ROOT)
        print("DEBUG ROOT:", DEBUG_ROOT)
        print("EXTRACTED FILE PREVIEW:", extracted_preview)

        result = cac_service.predict_folder(
            dicom_root_dir=str(dicom_extract_dir),
            request_id=request_id,
            save_masks=save_masks,
        )

        for study_result in result.get("results", []):
            mask_urls = []

            for mask_file in study_result.get("mask_files", []):
                try:
                    mask_urls.append(local_mask_path_to_url(mask_file))
                except Exception:
                    pass

            study_result["mask_urls"] = mask_urls

        return jsonify(
            {
                "success": True,
                "request_id": request_id,
                "dicom_input_dir": str(dicom_extract_dir),
                "mask_output_root": str(MASK_ROOT / request_id),
                **result,
            }
        ), 200

    except zipfile.BadZipFile:
        return jsonify(
            {
                "success": False,
                "request_id": request_id,
                "error": "Uploaded file is not a valid zip file.",
            }
        ), 400

    except Exception as e:
        traceback.print_exc()

        return jsonify(
            {
                "success": False,
                "request_id": request_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "dicom_input_dir": str(dicom_extract_dir),
                "mask_output_root": str(MASK_ROOT / request_id),
            }
        ), 500


@app.route("/masks/<path:filename>", methods=["GET"])
def get_mask(filename):
    """
    Serves saved mask PNGs from storage/outputs/masks (local mode)
    or redirects to S3 (S3 mode).
    """
    if USE_S3_STORAGE and s3_service:
        # In S3 mode, generate public S3 URL and redirect
        s3_key = f"{S3_PREFIX}/masks/{filename}"
        public_url = s3_service.get_public_url(s3_key)
        return redirect(public_url)
    else:
        # In local mode, serve from local directory
        return send_from_directory(MASK_ROOT, filename)


print("Reached bottom of app.py")
print("__name__ is:", __name__)

if __name__ == "__main__":
    print("Starting AI-CAC Flask server...")
    print("Open: http://127.0.0.1:25000/health")

    # Read debug mode from environment variable
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in {"true", "1", "yes"}
    
    if debug_mode:
        print("⚠️  DEBUG MODE ENABLED - Auto-reload on code changes")
    
    app.run(
        host="0.0.0.0",
        port=25000,
        debug=debug_mode,
        use_reloader=debug_mode,
    )