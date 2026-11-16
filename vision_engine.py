import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"MediaPipe tasks not available: {e}")


class VisionEngine:
    def __init__(
        self,
        model_path: str = "models/blaze_face_short_range.tflite",
        segmenter_path: str = "models/selfie_segmenter.tflite",
        cascade_path: str = "models/haarcascade_frontalface_default.xml"
    ):
        self.model_path = model_path
        self.segmenter_path = segmenter_path
        self.cascade_path = cascade_path
        
        self.mp_detector = None
        self.mp_segmenter = None
        self.face_cascade = None

        self._init_mediapipe_detector()
        self._init_mediapipe_segmenter()
        self._init_cascade()

    def _init_mediapipe_detector(self):
        """Initialize MediaPipe Face Detector task."""
        if not MEDIAPIPE_AVAILABLE:
            return

        if os.path.exists(self.model_path):
            try:
                base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
                options = mp_vision.FaceDetectorOptions(
                    base_options=base_options,
                    min_detection_confidence=0.4
                )
                self.mp_detector = mp_vision.FaceDetector.create_from_options(options)
                print(f"[VisionEngine] MediaPipe FaceDetector initialized from {self.model_path}")
            except Exception as e:
                print(f"[VisionEngine] Error loading MediaPipe FaceDetector: {e}")
                self.mp_detector = None
        else:
            print(f"[VisionEngine] Model file {self.model_path} not found.")

    def _init_mediapipe_segmenter(self):
        """Initialize MediaPipe Image Segmenter (Selfie Segmenter) for full person/headwear silhouette."""
        if not MEDIAPIPE_AVAILABLE:
            return

        if os.path.exists(self.segmenter_path):
            try:
                base_options = mp_python.BaseOptions(model_asset_path=self.segmenter_path)
                options = mp_vision.ImageSegmenterOptions(
                    base_options=base_options,
                    output_confidence_masks=True
                )
                self.mp_segmenter = mp_vision.ImageSegmenter.create_from_options(options)
                print(f"[VisionEngine] MediaPipe ImageSegmenter initialized from {self.segmenter_path}")
            except Exception as e:
                print(f"[VisionEngine] Error loading Selfie Segmenter: {e}")
                self.mp_segmenter = None
        else:
            print(f"[VisionEngine] Segmenter file {self.segmenter_path} not found.")

    def _init_cascade(self):
        """Initialize OpenCV Haar Cascade as secondary fallback."""
        if os.path.exists(self.cascade_path):
            try:
                self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
                if not self.face_cascade.empty():
                    print(f"[VisionEngine] OpenCV Haar Cascade initialized from {self.cascade_path}")
                else:
                    self.face_cascade = None
            except Exception as e:
                print(f"[VisionEngine] Error loading Haar Cascade: {e}")
                self.face_cascade = None

    def detect_faces(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in BGR image.
        Returns list of face dicts:
        [{
            'bbox': [x, y, w, h],
            'confidence': float,
            'source': 'mediapipe' | 'opencv',
            'keypoints': [{'x': float, 'y': float, 'name': str}, ...]
        }]
        """
        img_h, img_w = image_bgr.shape[:2]
        faces = []

        # 1. Try MediaPipe first
        if self.mp_detector is not None:
            try:
                image_rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                detection_result = self.mp_detector.detect(mp_image)

                for det in detection_result.detections:
                    bbox = det.bounding_box
                    x = max(0, int(bbox.origin_x))
                    y = max(0, int(bbox.origin_y))
                    w = min(img_w - x, int(bbox.width))
                    h = min(img_h - y, int(bbox.height))

                    score = float(det.categories[0].score) if det.categories else 0.95

                    keypoints = []
                    keypoint_names = ["right_eye", "left_eye", "nose_tip", "mouth_center", "right_ear", "left_ear"]
                    if hasattr(det, "keypoints") and det.keypoints:
                        for idx, kp in enumerate(det.keypoints):
                            name = keypoint_names[idx] if idx < len(keypoint_names) else f"kp_{idx}"
                            keypoints.append({
                                "x": float(kp.x * img_w),
                                "y": float(kp.y * img_h),
                                "name": name
                            })

                    faces.append({
                        "bbox": [x, y, w, h],
                        "confidence": score,
                        "source": "mediapipe",
                        "keypoints": keypoints
                    })
            except Exception as e:
                print(f"[VisionEngine] MediaPipe detection exception: {e}")

        # 2. Fallback to OpenCV Haar Cascade if MediaPipe found nothing
        if not faces and self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                cv_faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                for (x, y, w, h) in cv_faces:
                    faces.append({
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "confidence": 0.85,
                        "source": "opencv",
                        "keypoints": []
                    })
            except Exception as e:
                print(f"[VisionEngine] OpenCV cascade detection exception: {e}")

        # Sort faces by area descending (primary subject first)
        faces.sort(key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True)
        return faces

    def detect_silhouette_and_bg(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Segments the person silhouette using MediaPipe ImageSegmenter (Selfie Segmenter).
        Extracts:
        - person_top_y: exact top pixel of peci / hijab / headwear
        - person_bottom_y: bottom pixel of visible torso
        - person_left_x, person_right_x: lateral width of person (shoulders & hijab)
        - dominant_bg_hex: auto-detected studio backdrop color
        - dominant_bg_bgr: BGR tuple for background fill
        """
        img_h, img_w = image_bgr.shape[:2]

        # 1. Sample corners for background color estimation as baseline
        corner_size = max(10, min(img_w // 20, img_h // 20))
        tl = image_bgr[0:corner_size, 0:corner_size].reshape(-1, 3)
        tr = image_bgr[0:corner_size, img_w - corner_size:img_w].reshape(-1, 3)
        corner_pixels = np.vstack([tl, tr])
        med_bgr = np.median(corner_pixels, axis=0).astype(int)
        dominant_bg_hex = '#{:02X}{:02X}{:02X}'.format(med_bgr[2], med_bgr[1], med_bgr[0])

        info = {
            "person_top_y": 0,
            "person_bottom_y": img_h,
            "person_left_x": 0,
            "person_right_x": img_w,
            "has_silhouette": False,
            "dominant_bg_hex": dominant_bg_hex,
            "dominant_bg_bgr": [int(med_bgr[0]), int(med_bgr[1]), int(med_bgr[2])]
        }

        if self.mp_segmenter is not None:
            try:
                img_rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                seg_res = self.mp_segmenter.segment(mp_img)

                if seg_res.confidence_masks:
                    mask = seg_res.confidence_masks[0].numpy_view()[:, :, 0]
                    
                    # Refine background color from actual background mask
                    bg_mask = mask < 0.2
                    if np.sum(bg_mask) > 100:
                        bg_pixels = image_bgr[bg_mask]
                        med_bgr = np.median(bg_pixels, axis=0).astype(int)
                        dominant_bg_hex = '#{:02X}{:02X}{:02X}'.format(med_bgr[2], med_bgr[1], med_bgr[0])
                        info["dominant_bg_hex"] = dominant_bg_hex
                        info["dominant_bg_bgr"] = [int(med_bgr[0]), int(med_bgr[1]), int(med_bgr[2])]

                    # Person mask (confidence > 0.4)
                    person_binary = mask > 0.4
                    y_indices, x_indices = np.where(person_binary)
                    if len(y_indices) > 0:
                        info["person_top_y"] = int(np.min(y_indices))
                        info["person_bottom_y"] = int(np.max(y_indices))
                        info["person_left_x"] = int(np.min(x_indices))
                        info["person_right_x"] = int(np.max(x_indices))
                        info["has_silhouette"] = True
            except Exception as e:
                print(f"[VisionEngine] Segmentation exception: {e}")

        return info

    def parse_aspect_ratio(self, ratio_str: str) -> Tuple[float, float]:
        """Parse aspect ratio string like '1:1', '3:4', '4:3', '16:9', '9:16', 'custom'."""
        if not ratio_str or ratio_str == "custom" or ratio_str == "free":
            return (1.0, 1.0)

        ratio_str = ratio_str.strip().lower()
        mapping = {
            "square": (1.0, 1.0),
            "1:1": (1.0, 1.0),
            "portrait_3_4": (3.0, 4.0),
            "3:4": (3.0, 4.0),
            "portrait_4_5": (4.0, 5.0),
            "4:5": (4.0, 5.0),
            "portrait_9_16": (9.0, 16.0),
            "9:16": (9.0, 16.0),
            "landscape_4_3": (4.0, 3.0),
            "4:3": (4.0, 3.0),
            "landscape_16_9": (16.0, 9.0),
            "16:9": (16.0, 9.0),
            "2:3": (2.0, 3.0),
            "portrait_2_3": (2.0, 3.0),
        }
        if ratio_str in mapping:
            return mapping[ratio_str]

        if ":" in ratio_str:
            parts = ratio_str.split(":")
            try:
                rw = float(parts[0])
                rh = float(parts[1])
                if rw > 0 and rh > 0:
                    return (rw, rh)
            except ValueError:
                pass

        return (1.0, 1.0)

    def calculate_crop_box(
        self,
        img_w: int,
        img_h: int,
        face_bbox: Optional[List[int]],
        silhouette_info: Optional[Dict[str, Any]] = None,
        aspect_ratio_str: str = "3:4",
        head_margin: float = 0.35,
        shoulder_margin: float = 1.25,
        side_margin: float = 0.65
    ) -> Dict[str, Any]:
        """
        Calculate shoulder-level crop box using face landmarks AND person silhouette.
        Ensures peci, hijab, and hair are fully preserved, and fits strictly to aspect ratio (3:4).
        If the framing needs to be expanded, it permits negative or overflowing coordinates
        which will be seamlessly filled by apply_padding_and_crop.
        """
        ratio_w, ratio_h = self.parse_aspect_ratio(aspect_ratio_str)
        target_aspect = ratio_w / ratio_h

        has_headwear = False
        headwear_top_y = None

        if face_bbox is not None:
            fx, fy, fw, fh = face_bbox
            cx = fx + fw / 2.0
            chin_y = fy + fh

            # Check if silhouette detected headwear / hair above the forehead
            if silhouette_info and silhouette_info.get("has_silhouette"):
                person_top = silhouette_info["person_top_y"]
                # If silhouette extends above face top, we have headwear (peci/hijab) or hair
                if person_top < fy:
                    has_headwear = True
                    headwear_top_y = person_top
                    total_head_h = max(fh, chin_y - person_top)
                    # Headroom placed safely above the top of peci/hijab
                    headroom = head_margin * total_head_h * 0.65
                    top_y = person_top - headroom
                else:
                    total_head_h = fh
                    top_y = fy - (head_margin * fh)
                
                # Torso / shoulder depth
                torso_depth = shoulder_margin * total_head_h * 0.85
                bottom_y = chin_y + torso_depth
                
                # Minimum shoulder width
                min_shoulder_w = max(fw * (1.0 + 2.0 * side_margin), total_head_h * 1.15)
                roi_w = min_shoulder_w
            else:
                top_y = fy - (head_margin * fh)
                bottom_y = fy + fh + (shoulder_margin * fh)
                roi_w = fw + (2.0 * side_margin * fw)

            roi_h = bottom_y - top_y
        else:
            # Fallback if no face detected: centered portrait crop
            roi_w = img_w * 0.75
            roi_h = roi_w / target_aspect
            if roi_h > img_h * 0.9:
                roi_h = img_h * 0.9
                roi_w = roi_h * target_aspect
            cx = img_w / 2.0
            face_center_y = img_h * 0.40
            top_y = face_center_y - (roi_h * 0.35)
            bottom_y = top_y + roi_h
            roi_w = roi_h * target_aspect

        current_aspect = roi_w / roi_h

        # Fit strictly to target aspect ratio:
        if current_aspect < target_aspect:
            # Needs to be wider to match target aspect ratio
            target_w = roi_h * target_aspect
            diff_w = target_w - roi_w
            crop_x1 = (cx - target_w / 2.0)
            crop_x2 = (cx + target_w / 2.0)
            crop_y1 = top_y
            crop_y2 = bottom_y
        else:
            # Needs to be taller to match target aspect ratio
            target_h = roi_w / target_aspect
            diff_h = target_h - roi_h
            crop_y1 = top_y - (diff_h * 0.25)
            crop_y2 = bottom_y + (diff_h * 0.75)
            crop_x1 = (cx - roi_w / 2.0)
            crop_x2 = (cx + roi_w / 2.0)

        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        # Check if crop box overflows image bounds
        overflows = (crop_x1 < 0 or crop_y1 < 0 or crop_x2 > img_w or crop_y2 > img_h)

        dominant_bg_hex = silhouette_info.get("dominant_bg_hex", "#FFFFFF") if silhouette_info else "#FFFFFF"

        return {
            "crop_box": [int(round(crop_x1)), int(round(crop_y1)), int(round(crop_x2)), int(round(crop_y2))],
            "crop_w": int(round(crop_w)),
            "crop_h": int(round(crop_h)),
            "target_aspect": target_aspect,
            "aspect_ratio_str": f"{int(ratio_w)}:{int(ratio_h)}" if ratio_w.is_integer() and ratio_h.is_integer() else f"{ratio_w:.2f}:{ratio_h:.2f}",
            "overflows": overflows,
            "face_detected": face_bbox is not None,
            "has_headwear": has_headwear,
            "headwear_top_y": headwear_top_y,
            "detected_bg_color": dominant_bg_hex
        }

    def apply_padding_and_crop(
        self,
        image_bgr: np.ndarray,
        crop_box: List[int],
        pad_mode: str = "auto_studio",
        pad_color_hex: str = "#FFFFFF"
    ) -> np.ndarray:
        """
        Crops image to crop_box [x1, y1, x2, y2].
        If crop_box exceeds image boundaries, applies specified background fill/extension:
        - 'auto_studio': Automatically samples the studio backdrop color and seamlessly extends it.
        - 'edge': Replicates edge pixels (Border Replicate) seamlessly.
        - 'blur': Artistic blurred enlarged backdrop.
        - 'solid': Solid color (pad_color_hex).
        """
        img_h, img_w = image_bgr.shape[:2]
        x1, y1, x2, y2 = crop_box
        target_w = x2 - x1
        target_h = y2 - y1

        if target_w <= 0 or target_h <= 0:
            return image_bgr

        # If completely inside image bounds, simple slice
        if x1 >= 0 and y1 >= 0 and x2 <= img_w and y2 <= img_h:
            return image_bgr[y1:y2, x1:x2].copy()

        # Determine solid color (auto_studio or user hex)
        if pad_mode == "auto_studio":
            corner_size = max(10, min(img_w // 20, img_h // 20))
            tl = image_bgr[0:corner_size, 0:corner_size].reshape(-1, 3)
            tr = image_bgr[0:corner_size, img_w - corner_size:img_w].reshape(-1, 3)
            med_bgr = np.median(np.vstack([tl, tr]), axis=0).astype(int)
            solid_bgr = (int(med_bgr[0]), int(med_bgr[1]), int(med_bgr[2]))
        else:
            pad_color_hex = pad_color_hex.lstrip("#")
            if len(pad_color_hex) == 6:
                r = int(pad_color_hex[0:2], 16)
                g = int(pad_color_hex[2:4], 16)
                b = int(pad_color_hex[4:6], 16)
            else:
                r, g, b = 255, 255, 255
            solid_bgr = (b, g, r)

        # Region of overlap with original image
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(img_w, x2)
        src_y2 = min(img_h, y2)

        # Region where overlap maps in target canvas
        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        src_crop = image_bgr[src_y1:src_y2, src_x1:src_x2]

        if pad_mode == "blur":
            # Smart Blur:
            scale = max(target_w / img_w, target_h / img_h) * 1.15
            scaled_w = int(round(img_w * scale))
            scaled_h = int(round(img_h * scale))
            scaled_img = cv2.resize(image_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

            start_x = max(0, (scaled_w - target_w) // 2)
            start_y = max(0, (scaled_h - target_h) // 2)
            bg_patch = scaled_img[start_y:start_y + target_h, start_x:start_x + target_w]

            if bg_patch.shape[0] != target_h or bg_patch.shape[1] != target_w:
                bg_patch = cv2.resize(bg_patch, (target_w, target_h))

            ksize = int(max(target_w, target_h) * 0.08)
            if ksize % 2 == 0:
                ksize += 1
            ksize = max(25, min(ksize, 99))

            blurred_bg = cv2.GaussianBlur(bg_patch, (ksize, ksize), 0)
            blurred_bg = cv2.addWeighted(blurred_bg, 0.88, np.zeros_like(blurred_bg), 0, 10)

            result = blurred_bg
            if src_crop.size > 0:
                result[dst_y1:dst_y2, dst_x1:dst_x2] = src_crop
            return result

        elif pad_mode in ["auto_studio", "solid"]:
            # Solid Studio Backdrop Fill:
            result = np.full((target_h, target_w, 3), solid_bgr, dtype=np.uint8)
            if src_crop.size > 0:
                result[dst_y1:dst_y2, dst_x1:dst_x2] = src_crop
            return result

        else:
            # 'edge' -> Auto Edge Fill (Border Replicate)
            top = max(0, -y1)
            bottom = max(0, y2 - img_h)
            left = max(0, -x1)
            right = max(0, x2 - img_w)

            padded = cv2.copyMakeBorder(
                image_bgr,
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                borderType=cv2.BORDER_REPLICATE
            )

            pad_crop_x1 = max(0, x1 + left)
            pad_crop_y1 = max(0, y1 + top)
            pad_crop_x2 = min(padded.shape[1], pad_crop_x1 + target_w)
            pad_crop_y2 = min(padded.shape[0], pad_crop_y1 + target_h)

            result = padded[pad_crop_y1:pad_crop_y2, pad_crop_x1:pad_crop_x2].copy()
            if result.shape[0] != target_h or result.shape[1] != target_w:
                result = cv2.resize(result, (target_w, target_h))
            return result
