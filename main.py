import io
import json
import os
import zipfile
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from vision_engine import VisionEngine

# Initialize FastAPI application
app = FastAPI(
    title="PhotoCrop Studio - Intelligent Face & Shoulder Cropper",
    description="Automated intelligent face & shoulder photo cropping with Computer Vision (OpenCV & MediaPipe)",
    version="1.0.0"
)

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-Processed-Count", "X-Error-Count"]
)

# Initialize Vision Engine with Face Detector, Selfie Segmenter, and Haar Cascade
engine = VisionEngine(
    model_path=os.path.join("models", "blaze_face_short_range.tflite"),
    segmenter_path=os.path.join("models", "selfie_segmenter.tflite"),
    cascade_path=os.path.join("models", "haarcascade_frontalface_default.xml")
)

# Create required directories
os.makedirs("static", exist_ok=True)
os.makedirs("samples", exist_ok=True)


def decode_image_file(upload_file: UploadFile) -> Tuple[np.ndarray, str]:
    """Helper to decode uploaded image bytes synchronously into OpenCV BGR numpy array."""
    try:
        contents = upload_file.file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file {upload_file.filename}: {e}")

    if not contents:
        raise HTTPException(status_code=400, detail=f"Empty file uploaded: {upload_file.filename}")

    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {upload_file.filename}")

    return img_bgr, upload_file.filename or "image.jpg"


@app.get("/api/health")
def health_check():
    """Health check and model status endpoint."""
    return {
        "status": "online",
        "mediapipe_loaded": engine.mp_detector is not None,
        "segmenter_loaded": engine.mp_segmenter is not None,
        "cascade_loaded": engine.face_cascade is not None,
        "opencv_version": cv2.__version__
    }


@app.get("/api/samples")
def list_samples():
    """List sample images available for instant demo."""
    samples = []
    if os.path.exists("samples"):
        for fname in os.listdir("samples"):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                samples.append({
                    "name": fname,
                    "url": f"/api/samples/{fname}"
                })
    return {"samples": samples}


@app.get("/api/samples/{filename}")
def get_sample_image(filename: str):
    """Serve a sample image."""
    safe_name = os.path.basename(filename)
    path = os.path.join("samples", safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sample not found")
    media_type = "image/png" if safe_name.endswith(".png") else "image/jpeg"
    return FileResponse(path, media_type=media_type)


@app.post("/api/detect")
def detect_face_and_crop(
    file: UploadFile = File(...),
    aspect_ratio: str = Form("3:4"),
    head_margin: float = Form(0.35),
    shoulder_margin: float = Form(1.25),
    side_margin: float = Form(0.65)
):
    """
    Run face detection + silhouette segmentation on image and compute headwear-aware crop box.
    """
    try:
        img_bgr, filename = decode_image_file(file)
        img_h, img_w = img_bgr.shape[:2]
        
        faces = engine.detect_faces(img_bgr)
        sil = engine.detect_silhouette_and_bg(img_bgr)

        primary_face_bbox = faces[0]["bbox"] if faces else None

        crop_info = engine.calculate_crop_box(
            img_w=img_w,
            img_h=img_h,
            face_bbox=primary_face_bbox,
            silhouette_info=sil,
            aspect_ratio_str=aspect_ratio,
            head_margin=head_margin,
            shoulder_margin=shoulder_margin,
            side_margin=side_margin
        )

        return {
            "filename": filename,
            "width": img_w,
            "height": img_h,
            "face_count": len(faces),
            "faces": faces,
            "crop_info": crop_info,
            "silhouette": sil
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/preview-crop")
def preview_crop(
    file: UploadFile = File(...),
    aspect_ratio: str = Form("3:4"),
    head_margin: float = Form(0.35),
    shoulder_margin: float = Form(1.25),
    side_margin: float = Form(0.65),
    pad_mode: str = Form("auto_studio"),
    pad_color: str = Form("#FFFFFF"),
    custom_box: Optional[str] = Form(None)
):
    """
    Generate rendered crop preview with selected background fill and padding.
    """
    try:
        img_bgr, filename = decode_image_file(file)
        img_h, img_w = img_bgr.shape[:2]

        crop_box = None
        if custom_box:
            try:
                parsed = json.loads(custom_box)
                if isinstance(parsed, list) and len(parsed) == 4:
                    crop_box = [int(v) for v in parsed]
            except Exception:
                crop_box = None

        if crop_box is None:
            faces = engine.detect_faces(img_bgr)
            sil = engine.detect_silhouette_and_bg(img_bgr)
            primary_face = faces[0]["bbox"] if faces else None
            crop_info = engine.calculate_crop_box(
                img_w=img_w,
                img_h=img_h,
                face_bbox=primary_face,
                silhouette_info=sil,
                aspect_ratio_str=aspect_ratio,
                head_margin=head_margin,
                shoulder_margin=shoulder_margin,
                side_margin=side_margin
            )
            crop_box = crop_info["crop_box"]
            if pad_mode == "auto_studio" or pad_color == "auto":
                pad_color = crop_info.get("detected_bg_color", "#FFFFFF")

        # Apply crop and padding
        cropped_img = engine.apply_padding_and_crop(
            image_bgr=img_bgr,
            crop_box=crop_box,
            pad_mode=pad_mode,
            pad_color_hex=pad_color
        )

        success, encoded = cv2.imencode(".jpg", cropped_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode preview image")

        return Response(content=encoded.tobytes(), media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/crop-single")
def crop_single_download(
    file: UploadFile = File(...),
    aspect_ratio: str = Form("3:4"),
    head_margin: float = Form(0.35),
    shoulder_margin: float = Form(1.25),
    side_margin: float = Form(0.65),
    pad_mode: str = Form("auto_studio"),
    pad_color: str = Form("#FFFFFF"),
    custom_box: Optional[str] = Form(None),
    output_format: str = Form("original"),
    quality: int = Form(95)
):
    """
    Download a single processed photo with its original filename.
    """
    try:
        img_bgr, filename = decode_image_file(file)
        img_h, img_w = img_bgr.shape[:2]

        crop_box = None
        if custom_box:
            try:
                parsed = json.loads(custom_box)
                if isinstance(parsed, list) and len(parsed) == 4:
                    crop_box = [int(v) for v in parsed]
            except Exception:
                crop_box = None

        if crop_box is None:
            faces = engine.detect_faces(img_bgr)
            sil = engine.detect_silhouette_and_bg(img_bgr)
            primary_face = faces[0]["bbox"] if faces else None
            crop_info = engine.calculate_crop_box(
                img_w=img_w,
                img_h=img_h,
                face_bbox=primary_face,
                silhouette_info=sil,
                aspect_ratio_str=aspect_ratio,
                head_margin=head_margin,
                shoulder_margin=shoulder_margin,
                side_margin=side_margin
            )
            crop_box = crop_info["crop_box"]
            if pad_mode == "auto_studio" or pad_color == "auto":
                pad_color = crop_info.get("detected_bg_color", "#FFFFFF")

        cropped_img = engine.apply_padding_and_crop(
            image_bgr=img_bgr,
            crop_box=crop_box,
            pad_mode=pad_mode,
            pad_color_hex=pad_color
        )

        clean_filename = os.path.basename(filename.replace("\\", "/"))
        base_name, orig_ext = os.path.splitext(clean_filename)
        orig_ext = orig_ext.lower()

        if output_format == "original":
            ext = orig_ext if orig_ext in [".jpg", ".jpeg", ".png", ".webp"] else ".jpg"
        else:
            ext = f".{output_format.lower()}"

        out_filename = f"{base_name}{ext}"

        if ext in [".jpg", ".jpeg"]:
            media_type = "image/jpeg"
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, max(60, min(100, quality))]
        elif ext == ".webp":
            media_type = "image/webp"
            encode_params = [cv2.IMWRITE_WEBP_QUALITY, max(60, min(100, quality))]
        else:
            media_type = "image/png"
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 4]

        success, encoded = cv2.imencode(ext, cropped_img, encode_params)
        if not success:
            raise HTTPException(status_code=500, detail="Encoding failed")

        return Response(
            content=encoded.tobytes(),
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{out_filename}"'}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process-batch")
def process_batch(
    files: List[UploadFile] = File(...),
    aspect_ratio: str = Form("3:4"),
    head_margin: float = Form(0.35),
    shoulder_margin: float = Form(1.25),
    side_margin: float = Form(0.65),
    pad_mode: str = Form("auto_studio"),
    pad_color: str = Form("#FFFFFF"),
    output_format: str = Form("original"),
    quality: int = Form(95),
    custom_overrides: Optional[str] = Form(None)
):
    """
    Batch process uploaded photos, crop to aspect ratio and shoulder level,
    apply padding/background extension, and return a single ZIP file containing all photos named exactly as original filenames.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for batch processing")

    overrides = {}
    if custom_overrides:
        try:
            overrides = json.loads(custom_overrides)
        except Exception:
            overrides = {}

    zip_buffer = io.BytesIO()
    processed_count = 0
    errors = []

    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for uploaded_file in files:
            orig_filename = uploaded_file.filename
            if not orig_filename:
                continue

            clean_filename = os.path.basename(orig_filename.replace("\\", "/"))
            base_name, orig_ext = os.path.splitext(clean_filename)
            orig_ext = orig_ext.lower()

            try:
                contents = uploaded_file.file.read()
                if not contents:
                    continue

                nparr = np.frombuffer(contents, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    errors.append(f"Failed to decode {clean_filename}")
                    continue

                img_h, img_w = img_bgr.shape[:2]

                # Check if this specific file has a crop box override or precomputed box
                crop_box = None
                file_pad_color = pad_color
                if clean_filename in overrides:
                    try:
                        box_data = overrides[clean_filename]
                        if isinstance(box_data, list) and len(box_data) == 4:
                            crop_box = [int(v) for v in box_data]
                    except Exception:
                        crop_box = None

                # Otherwise detect face + silhouette and compute shoulder crop box
                if crop_box is None:
                    faces = engine.detect_faces(img_bgr)
                    sil = engine.detect_silhouette_and_bg(img_bgr)
                    primary_face = faces[0]["bbox"] if faces else None
                    crop_info = engine.calculate_crop_box(
                        img_w=img_w,
                        img_h=img_h,
                        face_bbox=primary_face,
                        silhouette_info=sil,
                        aspect_ratio_str=aspect_ratio,
                        head_margin=head_margin,
                        shoulder_margin=shoulder_margin,
                        side_margin=side_margin
                    )
                    crop_box = crop_info["crop_box"]
                    if pad_mode == "auto_studio" or pad_color == "auto":
                        file_pad_color = crop_info.get("detected_bg_color", "#FFFFFF")

                # Apply padding and crop
                cropped_img = engine.apply_padding_and_crop(
                    image_bgr=img_bgr,
                    crop_box=crop_box,
                    pad_mode=pad_mode,
                    pad_color_hex=file_pad_color
                )

                # Determine output filename & extension
                if output_format == "original":
                    ext = orig_ext if orig_ext in [".jpg", ".jpeg", ".png", ".webp"] else ".jpg"
                else:
                    ext = f".{output_format.lower()}"

                out_filename = f"{base_name}{ext}"

                # Encode image
                if ext in [".jpg", ".jpeg"]:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, max(60, min(100, quality))]
                elif ext == ".webp":
                    encode_params = [cv2.IMWRITE_WEBP_QUALITY, max(60, min(100, quality))]
                else:
                    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 4]

                success, encoded = cv2.imencode(ext, cropped_img, encode_params)
                if success:
                    # Write to ZIP with exact original filename
                    zip_file.writestr(out_filename, encoded.tobytes())
                    processed_count += 1
                else:
                    errors.append(f"Failed to encode {clean_filename}")

            except Exception as e:
                errors.append(f"Error processing {clean_filename}: {str(e)}")

    zip_buffer.seek(0)
    zip_bytes = zip_buffer.getvalue()

    if processed_count == 0:
        raise HTTPException(
            status_code=500,
            detail=f"Could not process any images. Errors: {'; '.join(errors[:5])}"
        )

    headers = {
        "Content-Disposition": 'attachment; filename="cropped_photos.zip"',
        "X-Processed-Count": str(processed_count),
        "X-Error-Count": str(len(errors))
    }

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers=headers
    )


# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    """Serve the single page application."""
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse("<h1>PhotoCrop Studio is starting up... Please refresh in a moment.</h1>")


if __name__ == "__main__":
    import uvicorn
    print("Starting PhotoCrop Studio server at http://localhost:8000 (0.0.0.0:8000) ...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
