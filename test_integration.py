import io
import json
import threading
import time
import urllib.request
import zipfile
import uvicorn
from main import app

def run_tests():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    time.sleep(2)

    try:
        # 1. Test GET /
        with urllib.request.urlopen("http://127.0.0.1:8000/") as resp:
            html = resp.read().decode("utf-8")
            assert resp.status == 200
            assert "PhotoCrop Studio" in html
            print("[PASS] GET / -> 200 OK (Loaded index.html)")

        # 2. Test GET /api/samples
        with urllib.request.urlopen("http://127.0.0.1:8000/api/samples") as resp:
            data = json.loads(resp.read().decode())
            assert resp.status == 200
            assert len(data["samples"]) >= 2
            print(f"[PASS] GET /api/samples -> {len(data['samples'])} demo samples available")

        # 3. Test POST /api/detect
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        with open("samples/portrait_formal.jpg", "rb") as f:
            img_bytes = f.read()

        body = bytearray()
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="aspect_ratio"\r\n\r\n3:4\r\n'.encode())
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="head_margin"\r\n\r\n0.35\r\n'.encode())
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="shoulder_margin"\r\n\r\n1.25\r\n'.encode())
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="side_margin"\r\n\r\n0.65\r\n'.encode())
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="portrait_formal.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode())
        body.extend(img_bytes)
        body.extend(f'\r\n--{boundary}--\r\n'.encode())

        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/detect",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        with urllib.request.urlopen(req) as resp:
            detect_res = json.loads(resp.read().decode())
            assert resp.status == 200
            assert detect_res["face_count"] >= 1
            print(f"[PASS] POST /api/detect -> Face detected: {detect_res['face_count']}, Crop: {detect_res['crop_info']['crop_box']}")

        # 4. Test POST /api/process-batch
        body_batch = bytearray()
        body_batch.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="aspect_ratio"\r\n\r\n3:4\r\n'.encode())
        body_batch.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="pad_mode"\r\n\r\nblur\r\n'.encode())
        body_batch.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="output_format"\r\n\r\noriginal\r\n'.encode())

        for fname in ["portrait_formal.jpg", "portrait_casual.jpg"]:
            with open(f"samples/{fname}", "rb") as f:
                b = f.read()
            body_batch.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{fname}"\r\nContent-Type: image/jpeg\r\n\r\n'.encode())
            body_batch.extend(b)
            body_batch.extend(b"\r\n")
        body_batch.extend(f'--{boundary}--\r\n'.encode())

        req_batch = urllib.request.Request(
            "http://127.0.0.1:8000/api/process-batch",
            data=body_batch,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        with urllib.request.urlopen(req_batch) as resp:
            zip_data = resp.read()
            zf = zipfile.ZipFile(io.BytesIO(zip_data))
            names = zf.namelist()
            assert "portrait_formal.jpg" in names
            assert "portrait_casual.jpg" in names
            print(f"[PASS] POST /api/process-batch -> Generated ZIP with {len(names)} files matching original filenames: {names}")

        print("\n[PASS] ALL INTEGRATION TESTS PASSED 100% SUCCESSFULLY!")

    finally:
        server.should_exit = True

if __name__ == "__main__":
    run_tests()
