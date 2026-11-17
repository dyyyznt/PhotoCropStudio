"""PhotoCrop Studio - Native Windows Application (Minimal Working Version)"""
import flet as ft
import os
import sys
import asyncio
import threading
import time
import json
from pathlib import Path

# Add project root to path (dev mode); frozen exe resolves via PyInstaller tracing
if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def _resource_root():
    """Resolve resource base for dev mode and PyInstaller onefile mode."""
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(__file__)

# CV/ML imports
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

# Project vision engine
from vision_engine import VisionEngine

# ============================================================
# CONFIGURATION
# ============================================================
APP_NAME = "PhotoCrop Studio - Native"
PORT = 8000
ASSETS_DIR = os.path.join(_resource_root(), "assets")
MODELS_DIR = os.path.join(ASSETS_DIR, "models")

# ============================================================
# FASTAPI INTERNAL SERVER
# ============================================================
app = FastAPI(
    title="PhotoCrop Studio - Intelligent Cropper",
    description="Automated intelligent face & shoulder photo cropping",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Vision Engine
engine = VisionEngine(
    model_path=os.path.join(MODELS_DIR, "blaze_face_short_range.tflite"),
    segmenter_path=os.path.join(MODELS_DIR, "selfie_segmenter.tflite"),
    cascade_path=os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
)

# Create required directories
os.makedirs("static", exist_ok=True)
SAMPLES_DIR = os.path.join(os.path.dirname(_resource_root()), "samples")


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def decode_image_file(upload_file):
    """Helper to decode uploaded image bytes."""
    try:
        contents = upload_file.file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")
    return img_bgr, upload_file.filename or "image.jpg"


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "mediapipe_loaded": engine.mp_detector is not None,
        "segmenter_loaded": engine.mp_segmenter is not None,
        "cascade_loaded": engine.face_cascade is not None,
        "opencv_version": cv2.__version__
    }


@app.get("/api/samples")
def list_samples():
    samples = []
    if os.path.exists(SAMPLES_DIR):
        for fname in os.listdir(SAMPLES_DIR):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                samples.append({"name": fname, "url": f"/api/samples/{fname}"})
    return {"samples": samples}


@app.get("/api/samples/{filename}")
def get_sample_image(filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join(SAMPLES_DIR, safe_name)
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
    try:
        img_bgr, filename = decode_image_file(file)
        img_h, img_w = img_bgr.shape[:2]
        faces = engine.detect_faces(img_bgr)
        sil = engine.detect_silhouette_and_bg(img_bgr)
        primary_face_bbox = faces[0]["bbox"] if faces else None
        crop_info = engine.calculate_crop_box(
            img_w=img_w, img_h=img_h,
            face_bbox=primary_face_bbox, silhouette_info=sil,
            aspect_ratio_str=aspect_ratio,
            head_margin=head_margin, shoulder_margin=shoulder_margin,
            side_margin=side_margin
        )
        return {
            "filename": filename, "width": img_w, "height": img_h,
            "face_count": len(faces), "faces": faces,
            "crop_info": crop_info, "silhouette": sil
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
    custom_box: str = Form(None)
):
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
                img_w=img_w, img_h=img_h, face_bbox=primary_face,
                silhouette_info=sil, aspect_ratio_str=aspect_ratio,
                head_margin=head_margin, shoulder_margin=shoulder_margin,
                side_margin=side_margin
            )
            crop_box = crop_info["crop_box"]
            if pad_mode == "auto_studio" or pad_color == "auto":
                pad_color = crop_info.get("detected_bg_color", "#FFFFFF")
        cropped_img = engine.apply_padding_and_crop(
            image_bgr=img_bgr, crop_box=crop_box,
            pad_mode=pad_mode, pad_color_hex=pad_color
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
    custom_box: str = Form(None),
    output_format: str = Form("original"),
    quality: int = Form(95)
):
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
                img_w=img_w, img_h=img_h, face_bbox=primary_face,
                silhouette_info=sil, aspect_ratio_str=aspect_ratio,
                head_margin=head_margin, shoulder_margin=shoulder_margin,
                side_margin=side_margin
            )
            crop_box = crop_info["crop_box"]
            if pad_mode == "auto_studio" or pad_color == "auto":
                pad_color = crop_info.get("detected_bg_color", "#FFFFFF")
        cropped_img = engine.apply_padding_and_crop(
            image_bgr=img_bgr, crop_box=crop_box, pad_mode=pad_mode, pad_color_hex=pad_color
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
    files: list[UploadFile] = File(...),
    aspect_ratio: str = Form("3:4"),
    head_margin: float = Form(0.35),
    shoulder_margin: float = Form(1.25),
    side_margin: float = Form(0.65),
    pad_mode: str = Form("auto_studio"),
    pad_color: str = Form("#FFFFFF"),
    output_format: str = Form("original"),
    quality: int = Form(95),
    custom_overrides: str = Form(None)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    import io, zipfile
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
                crop_box = None
                if clean_filename in overrides:
                    try:
                        box_data = overrides[clean_filename]
                        if isinstance(box_data, list) and len(box_data) == 4:
                            crop_box = [int(v) for v in box_data]
                    except Exception:
                        crop_box = None
                if crop_box is None:
                    faces = engine.detect_faces(img_bgr)
                    sil = engine.detect_silhouette_and_bg(img_bgr)
                    primary_face = faces[0]["bbox"] if faces else None
                    crop_info = engine.calculate_crop_box(
                        img_w=img_w, img_h=img_h, face_bbox=primary_face,
                        silhouette_info=sil, aspect_ratio_str=aspect_ratio,
                        head_margin=head_margin, shoulder_margin=shoulder_margin,
                        side_margin=side_margin
                    )
                    crop_box = crop_info["crop_box"]
                cropped_img = engine.apply_padding_and_crop(
                    image_bgr=img_bgr, crop_box=crop_box,
                    pad_mode="auto_studio", pad_color_hex="auto"
                )
                if output_format == "original":
                    ext = orig_ext if orig_ext in [".jpg", ".jpeg", ".png", ".webp"] else ".jpg"
                else:
                    ext = f".{output_format.lower()}"
                out_filename = f"{base_name}{ext}"
                if ext in [".jpg", ".jpeg"]:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, max(60, min(100, quality))]
                elif ext == ".webp":
                    encode_params = [cv2.IMWRITE_WEBP_QUALITY, max(60, min(100, quality))]
                else:
                    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 4]
                success, encoded = cv2.imencode(ext, cropped_img, encode_params)
                if success:
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
            status_code=500, detail=f"Could not process any images. Errors: {'; '.join(errors[:5])}"
        )
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="cropped_photos.zip"',
            "X-Processed-Count": str(processed_count),
            "X-Error-Count": str(len(errors))
        }
    )


# ============================================================
# FLET APP
# ============================================================
def main_flet(page: ft.Page):
    """Flet main function - builds the UI and starts the internal server."""
    
    # 1. Page config
    page.title = APP_NAME
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH
    
    # 2. Start internal FastAPI server in background thread
    import uvicorn
    
    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)  # wait for server to start
    
    # 3. Check model loading
    models_loaded = engine.mp_detector is not None or engine.face_cascade is not None
    
    # 4. UI Components - minimal
    status_text = ft.Text("Ready" if models_loaded else "Loading models", color=ft.Colors.AMBER_300, size=12)
    page.add(status_text)
    page.update()
    
    # File picker
    file_picker = ft.FilePicker(on_result=lambda e: handle_file_result(e))
    page.overlay.append(file_picker)
    
    # Dropzone
    dropzone = ft.Container(
        content=ft.Column([
            ft.Text("Drag & drop photos or click to upload", color='#94a3b8', size=12, text_align=ft.TextAlign.CENTER),
            ft.Button("Select Photos", on_click=lambda _: file_picker.pick_files(allow_multiple=True)),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
        width=600, height=300,
        border=ft.Border.all(2, '#334155'),
        border_radius=15,
        bgcolor='#0f172a',
        alignment=ft.Alignment.CENTER,
    )
    
    # Queue grid
    queue_container = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
    
    # Aspect ratio buttons
    aspect_buttons = ft.Row([
        ft.FilledButton("1:1", data="1:1", on_click=lambda e: set_aspect_ratio(e)),
        ft.FilledButton("3:4", data="3:4", on_click=lambda e: set_aspect_ratio(e), bgcolor=ft.Colors.BLUE_600, color=ft.Colors.WHITE),
        ft.FilledButton("4:5", data="4:5", on_click=lambda e: set_aspect_ratio(e)),
    ], spacing=8)
    
    # Margin sliders
    head_margin_label = ft.Text("35%", color='#94a3b8', size=9)
    shoulder_margin_label = ft.Text("125%", color='#94a3b8', size=9)
    side_margin_label = ft.Text("65%", color='#94a3b8', size=9)
    head_margin_slider = ft.Slider(min=0.1, max=0.9, value=0.35, divisions=16, round=2,
                                   on_change=lambda e: update_margin(e.control.value, "head"))
    shoulder_margin_slider = ft.Slider(min=0.5, max=2.2, value=1.25, divisions=34, round=2,
                                     on_change=lambda e: update_margin(e.control.value, "shoulder"))
    side_margin_slider = ft.Slider(min=0.2, max=1.5, value=0.65, divisions=26, round=2,
                                   on_change=lambda e: update_margin(e.control.value, "side"))
    
    # Padding mode
    pad_mode_buttons = ft.Row([
        ft.FilledButton("Auto Studio", data="auto_studio", on_click=lambda e: set_pad_mode(e)),
        ft.FilledButton("Edge Fill", data="edge", on_click=lambda e: set_pad_mode(e)),
        ft.FilledButton("Smart Blur", data="blur", on_click=lambda e: set_pad_mode(e)),
        ft.FilledButton("Manual Color", data="solid", on_click=lambda e: set_pad_mode(e)),
    ], spacing=8)
    
    color_presets = ft.Row([
        ft.Container(bgcolor="#DC2626", width=20, height=20, border_radius=10,
                     on_click=lambda _: set_pad_color("#DC2626"), ink=True),
        ft.Container(bgcolor="#2563EB", width=20, height=20, border_radius=10,
                     on_click=lambda _: set_pad_color("#2563EB"), ink=True),
        ft.Container(bgcolor="#FFFFFF", width=20, height=20, border_radius=10,
                     on_click=lambda _: set_pad_color("#FFFFFF"), ink=True),
    ], spacing=6)

    color_presets.visible = False

    # Output format
    output_format_dropdown = ft.Dropdown(
        options=[
            ft.dropdown.Option("original", text="Same as Original"),
            ft.dropdown.Option("jpeg", text="JPEG (.jpg)"),
            ft.dropdown.Option("png", text="PNG (.png)"),
            ft.dropdown.Option("webp", text="WebP (.webp)"),
        ],
        value="original",
        on_change=lambda e: set_output_format(e.control.value),
    )
    
    quality_slider = ft.Slider(min=70, max=100, value=95, divisions=6,
                               on_change=lambda e: set_quality(int(e.control.value)))
    
    # Preview modal
    preview_modal = ft.Container(visible=False, bgcolor='#020617',
                                 content=ft.Column([
                                     ft.Row([
                                         ft.Text("Preview"),
                                         ft.IconButton(icon=ft.Icons.CLOSE, on_click=lambda _: set_preview_visible(False)),
                                     ]),
                                     ft.Text("Cropped Preview will appear here", color='#94a3b8', size=14),
                                     ft.Container(height=200, bgcolor='#0f172a', border=ft.Border.all(1, '#334155'), padding=10),
                                 ], expand=True),
                                 on_click=lambda _: set_preview_visible(False))
    
    # Batch export
    progress_bar = ft.ProgressBar(width=600, visible=False)
    progress_text = ft.Text("Ready", color='#cbd5e1')
    batch_button = ft.FilledButton("Process & Download ZIP",
                                     on_click=lambda _: start_batch_processing(),
                                     disabled=True,
                                     bgcolor=ft.Colors.BLUE_600, color=ft.Colors.WHITE)
    
    # ===================== EVENT HANDLERS =====================
    
    def handle_file_result(e):
        if e.files:
            for f in e.files:
                add_file_to_queue(f, page)
        render_queue_grid(queue_container)
        update_queue_stats(page)
    
    def add_file_to_queue(file_item, page):
        file_path = file_item.path
        file_name = file_item.name
        try:
            with open(file_path, "rb") as img_file:
                img_data = img_file.read()
            nparr = np.frombuffer(img_data, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return
            h, w = img_bgr.shape[:2]
        except:
            h, w = 800, 600
        
        item_id = f"img_{len((page.session.store.get('items') or []))}"
        item = {
            "id": item_id,
            "file": file_path,
            "filename": file_name,
            "url": f"file://{file_path}",
            "width": w, "height": h,
            "faces": [], "cropInfo": None, "customBox": None, "status": "detecting"
        }
        items = (page.session.store.get('items') or [])
        items.append(item)
        page.session.store.set('items', items)
        
        page.run_task(detect_face_async, item, page)
        render_queue_grid(queue_container)
        update_queue_stats(page)
    
    async def detect_face_async(item, page):
        try:
            import httpx
            settings = (page.session.store.get('settings') or {})
            async with httpx.AsyncClient(timeout=60) as client:
                with open(item["file"], "rb") as f:
                    data = f.read()
                resp = await client.post(
                    f"http://localhost:{PORT}/api/detect",
                    files={"file": (item["filename"], data)},
                    data={
                        "aspect_ratio": settings.get("aspectRatio", "3:4"),
                        "head_margin": str(settings.get("headMargin", 0.35)),
                        "shoulder_margin": str(settings.get("shoulderMargin", 1.25)),
                        "side_margin": str(settings.get("sideMargin", 0.65)),
                    },
                )
                if resp.status_code == 200:
                    result = resp.json()
                    item["faces"] = result.get("faces", [])
                    item["cropInfo"] = result.get("crop_info", None)
                    item["status"] = "ready"
        except Exception as ex:
            print(f"Detection error: {ex}")
            item["status"] = "ready"
            item["faces"] = []
            item["cropInfo"] = {"crop_box": [0, 0, item["width"], item["height"]], "overflows": False, "face_detected": False, "detected_bg_color": "#FFFFFF", "crop_w": item["width"], "crop_h": item["height"]}
        render_queue_grid(queue_container)
        update_queue_stats(page)
    
    def render_queue_grid(container):
        items = (page.session.store.get('items') or [])
        container.controls.clear()
        if not items:
            container.content = ft.Text("No images uploaded yet", color='#64748b')
            page.update()
            return
        
        for item in items:
            face_count = len(item.get("faces", []))
            if item["status"] == "detecting":
                badge = ft.Text("Detecting...", color=ft.Colors.AMBER_300, size=10)
            elif item.get("customBox"):
                badge = ft.Text("★ Custom", color=ft.Colors.PURPLE_300, size=10)
            elif face_count > 0:
                badge = ft.Text(f"Face ({face_count})", color=ft.Colors.GREEN_300, size=10)
            else:
                badge = ft.Text("Center Fallback", color=ft.Colors.AMBER_300, size=10)
            
            dims = f"{item['width']} × {item['height']}"
            
            card = ft.Container(
                content=ft.Column([
                    ft.Container(
                        content=ft.Image(src=item["url"], width=150, height=150, fit=ft.BoxFit.COVER),
                        bgcolor='#0f172a',
                        border=ft.Border.all(1, '#334155'),
                        border_radius=8,
                    ),
                    ft.Text(item["filename"], color='#e2e8f0', size=10, overflow=ft.TextOverflow.ELLIPSIS, expand=True),
                    ft.Text(dims, color='#94a3b8', size=9),
                    ft.Row([
                        badge,
                        ft.IconButton(icon=ft.Icons.DELETE, icon_color=ft.Colors.RED_300, tooltip="Remove",
                                     on_click=lambda e, iid=item["id"]: remove_item(iid, page)),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ], spacing=5),
                bgcolor='#1e293b',
                padding=ft.Padding.all(10),
                border=ft.Border.all(1, '#334155'),
                border_radius=8,
                on_click=lambda e, iid=item["id"]: open_preview(iid),
            )
            container.controls.append(card)
        
        page.update()
    
    def remove_item(item_id, page):
        items = (page.session.store.get('items') or [])
        items = [it for it in items if it["id"] != item_id]
        page.session.store.set('items', items)
        render_queue_grid(queue_container)
        update_queue_stats(page)
    
    def update_queue_stats(page):
        items = (page.session.store.get('items') or [])
        count = len(items)
        batch_button.disabled = count == 0
        if count == 0:
            progress_text.value = "Ready"
        elif not progress_bar.visible:
            progress_text.value = f"{count} photo(s) queued"
        page.update()

    def _get_settings():
        defaults = {
            "aspectRatio": "3:4",
            "headMargin": 0.35,
            "shoulderMargin": 1.25,
            "sideMargin": 0.65,
            "padMode": "auto_studio",
            "padColor": "#FFFFFF",
            "outputFormat": "original",
            "quality": 95,
        }
        stored = page.session.store.get('settings') or {}
        defaults.update(stored)
        return defaults

    def _save_settings(partial):
        settings = _get_settings()
        settings.update(partial)
        page.session.store.set('settings', settings)

    def set_aspect_ratio(e):
        ratio = e.control.data
        _save_settings({"aspectRatio": ratio})
        for btn in aspect_buttons.controls:
            selected = btn.data == ratio
            btn.bgcolor = ft.Colors.BLUE_600 if selected else None
            btn.color = ft.Colors.WHITE if selected else None
        page.update()

    def update_margin(value, margin_type):
        key = {"head": "headMargin", "shoulder": "shoulderMargin", "side": "sideMargin"}[margin_type]
        _save_settings({key: value})
        label = {"head": head_margin_label, "shoulder": shoulder_margin_label, "side": side_margin_label}[margin_type]
        label.value = f"{int(round(value * 100))}%"
        page.update()

    def set_pad_mode(e):
        mode = e.control.data
        _save_settings({"padMode": mode})
        color_presets.visible = (mode == "solid")
        for btn in pad_mode_buttons.controls:
            selected = btn.data == mode
            btn.bgcolor = ft.Colors.BLUE_600 if selected else None
            btn.color = ft.Colors.WHITE if selected else None
        page.update()

    def set_pad_color(hex_color):
        _save_settings({"padColor": hex_color})

    def set_output_format(fmt):
        _save_settings({"outputFormat": fmt})

    def set_quality(value):
        _save_settings({"quality": int(value)})
    
    def set_preview_visible(visible):
        preview_modal.visible = visible
        page.update()
    
    def open_preview(item_id):
        item = next((i for i in (page.session.store.get('items') or []) if i["id"] == item_id), None)
        if not item:
            return
        settings = _get_settings()
        preview_modal.content = ft.Column([
            ft.Row([
                ft.Text(item["filename"]),
                ft.IconButton(icon=ft.Icons.CLOSE, on_click=lambda _: set_preview_visible(False)),
            ]),
            ft.Text(f"{item['width']} × {item['height']} px"),
            ft.Container(
                content=ft.Image(src=item["url"], width=400, height=300, fit=ft.BoxFit.CONTAIN),
                bgcolor='#0f172a',
                border=ft.Border.all(1, '#334155'),
                padding=10,
            ),
            ft.Row([
                ft.Text("Settings:"),
                ft.Column([
                    ft.Text(f"Aspect: {settings.get('aspectRatio', '3:4')}"),
                    ft.Text(f"Head: {int(settings.get('headMargin', 0.35) * 100)}%"),
                    ft.Text(f"Shoulder: {int(settings.get('shoulderMargin', 1.25) * 100)}%"),
                    ft.Text(f"Side: {int(settings.get('sideMargin', 0.65) * 100)}%"),
                    ft.Text(f"Pad: {settings.get('padMode', 'auto_studio')}"),
                ], width=200),
            ]),
            ft.Row([
                ft.FilledButton("Apply", on_click=lambda: set_preview_visible(False)),
                ft.OutlinedButton("Close", on_click=lambda _: set_preview_visible(False)),
            ], alignment=ft.MainAxisAlignment.END),
        ], spacing=10, expand=True)
        set_preview_visible(True)
    
    def start_batch_processing():
        items = (page.session.store.get('items') or [])
        if not items:
            page.snack_bar = ft.SnackBar(ft.Text("No images to process"), open=True)
            page.update()
            return

        settings = _get_settings()
        progress_bar.visible = True
        progress_text.visible = True
        progress_text.value = f"Processing {len(items)} photos..."
        page.update()
        
        import io, zipfile, cv2
        zip_buffer = io.BytesIO()
        processed_count = 0
        errors = []
        
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for i, item in enumerate(items):
                progress_bar.value = 0.05 + (0.9 * (i + 1) / len(items))
                progress_text.value = f"Processing {item['filename']}..."
                page.update()
                
                try:
                    with open(item["file"], "rb") as f:
                        data = f.read()
                    nparr = np.frombuffer(data, np.uint8)
                    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        errors.append(f"Failed to decode {item['filename']}")
                        continue
                    
                    img_h, img_w = img_bgr.shape[:2]
                    crop_box = item.get("cropInfo", {}).get("crop_box", None)
                    if crop_box is None:
                        faces = engine.detect_faces(img_bgr)
                        sil = engine.detect_silhouette_and_bg(img_bgr)
                        primary_face = faces[0]["bbox"] if faces else None
                        crop_info = engine.calculate_crop_box(
                            img_w=img_w, img_h=img_h, face_bbox=primary_face,
                            silhouette_info=sil,
                            aspect_ratio_str=settings.get("aspectRatio", "3:4"),
                            head_margin=settings.get("headMargin", 0.35),
                            shoulder_margin=settings.get("shoulderMargin", 1.25),
                            side_margin=settings.get("sideMargin", 0.65)
                        )
                        crop_box = crop_info["crop_box"]

                    cropped_img = engine.apply_padding_and_crop(
                        image_bgr=img_bgr, crop_box=crop_box,
                        pad_mode=settings.get("padMode", "auto_studio"),
                        pad_color_hex=settings.get("padColor", "#FFFFFF")
                    )

                    output_format = settings.get("outputFormat", "original")
                    quality = settings.get("quality", 95)
                    clean_filename = os.path.basename(item["filename"].replace("\\", "/"))
                    base_name, orig_ext = os.path.splitext(clean_filename)
                    orig_ext = orig_ext.lower()
                    ext = orig_ext if orig_ext in [".jpg", ".jpeg", ".png", ".webp"] else ".jpg"
                    out_filename = f"{base_name}{ext}"
                    
                    if ext in [".jpg", ".jpeg"]:
                        encode_params = [cv2.IMWRITE_JPEG_QUALITY, max(60, min(100, quality))]
                    elif ext == ".webp":
                        encode_params = [cv2.IMWRITE_WEBP_QUALITY, max(60, min(100, quality))]
                    else:
                        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 4]
                    
                    success, encoded = cv2.imencode(ext, cropped_img, encode_params)
                    if success:
                        zip_file.writestr(out_filename, encoded.tobytes())
                        processed_count += 1
                    else:
                        errors.append(f"Failed to encode {item['filename']}")
                except Exception as ex:
                    errors.append(f"Error processing {item['filename']}: {str(ex)}")
        
        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()
        
        downloads_dir = os.path.join(os.environ.get("USERPROFILE", ""), "Downloads")
        os.makedirs(downloads_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(downloads_dir, f"cropped_photos_{timestamp}.zip")
        
        with open(out_path, "wb") as f:
            f.write(zip_bytes)
        
        progress_bar.value = 1.0
        progress_text.value = f"✓ Downloaded {processed_count} photos to {out_path}"
        page.snack_bar = ft.SnackBar(ft.Text(f"Success! {processed_count} photos saved."), open=True)
        page.update()
        
        progress_bar.visible = False
        progress_text.visible = False
    
    # ===================== LAYOUT =====================
    
    main_column = ft.Column([
        ft.Text("PhotoCrop Studio - Native", weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE, size=16),
        ft.Container(content=status_text, padding=ft.Padding.only(top=5, bottom=5)),
        ft.Row([
            ft.Container(
                content=ft.Column([
                    ft.Text("Image Processing Queue", color=ft.Colors.WHITE, size=14, weight=ft.FontWeight.BOLD),
                    ft.Container(content=dropzone, padding=10),
                    ft.Container(content=ft.Column([queue_container], scroll=ft.ScrollMode.AUTO, expand=True), expand=True),
                ]),
                expand=1, padding=10,
            ),
            ft.Container(
                content=ft.Column([
                    ft.Text("Crop Settings", color=ft.Colors.WHITE, size=14, weight=ft.FontWeight.BOLD),
                    ft.Divider(),
                    ft.Text("Target Aspect Ratio", color='#cbd5e1', size=11),
                    aspect_buttons,
                    ft.Divider(),
                    ft.Text("Margins & Safe Area", color='#cbd5e1', size=11),
                    ft.Row([ft.Text("Headroom", size=9), head_margin_slider, head_margin_label]),
                    ft.Row([ft.Text("Shoulder Level", size=9), shoulder_margin_slider, shoulder_margin_label]),
                    ft.Row([ft.Text("Shoulder Width", size=9), side_margin_slider, side_margin_label]),
                    ft.Divider(),
                    ft.Text("Padding & Background Fill", color='#cbd5e1', size=11),
                    pad_mode_buttons,
                    color_presets,
                    ft.Divider(),
                    ft.Text("Export Settings", color='#cbd5e1', size=11),
                    output_format_dropdown,
                    ft.Row([ft.Text("Quality"), quality_slider]),
                    ft.Divider(),
                    ft.Column([batch_button, progress_bar, progress_text], spacing=5),
                ], width=300, scroll=ft.ScrollMode.AUTO, expand=True),
            ),
        ], expand=True),
        preview_modal,
    ], expand=True, scroll=ft.ScrollMode.AUTO)
    
    page.add(main_column)
    page.update()
    
    # Enable batch button when images are loaded
    batch_button.disabled = len((page.session.store.get('items') or [])) == 0
    page.update()
    
    # Final status
    status_text.value = "Backend Online" if models_loaded else "Limited functionality"
    page.update()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    import flet.app as fa
    fa.run(main=main_flet, assets_dir=ASSETS_DIR, view=ft.AppView.WEB_BROWSER)