# PhotoCrop Studio 📸✨
**Intelligent Face & Shoulder Auto-Cropper berbasis Computer Vision (FastAPI, OpenCV & MediaPipe)**

PhotoCrop Studio adalah aplikasi web full-stack modern untuk melakukan crop otomatis foto profil, pas foto, dan potret formal secara massal (batch processing) dengan mengidentifikasi wajah dan proporsi bahu (*shoulder-level crop*) secara cerdas menggunakan **MediaPipe Face Detection** dan **OpenCV**.

---

## ✨ Fitur Unggulan

### 1. Antarmuka Web & Drag-and-Drop Cerdas
- **Dropzone Interaktif:** Drag-and-drop banyak file gambar sekaligus (`.jpg`, `.png`, `.webp`) atau **seluruh folder** secara rekursif (didukung oleh WebKit Directory API).
- **Tombol Upload Folder & Files:** Pilihan upload folder atau banyak gambar dengan dialog file native.
- **Antrean Thumbnail Dinamis:** Menampilkan status deteksi wajah secara real-time (*Face Detected*, *Multiple Faces*, *Center Fallback*, *Custom Crop*), dimensi resolusi gambar, dan tombol aksi individual.
- **Instant Demo Loader:** Tombol *“Load Demo Photos”* untuk langsung mencoba aplikasi dengan gambar sampel portrait tanpa perlu mencari file terlebih dahulu.

### 2. Live Preview Interaktif (HTML5 Canvas)
- **Modal Preview Visual:** Klik salah satu gambar untuk membuka canvas interaktif resolusi tinggi.
- **Overlay Deteksi:** Menampilkan kotak deteksi wajah MediaPipe (hijau) lengkap dengan facial landmark (mata, hidung, mulut) dan area crop bahu (ungu/indigo).
- **Interactive Drag & Resize:** Pengguna dapat menggeser kotak crop atau mengubah ukuran (*resize*) langsung di canvas secara visual.
- **Visualisasi Padding / Overflow:** Area di luar batas gambar asli divisualisasikan dengan arsiran (*diagonal hatching*) agar pengguna dapat melihat batas pengisian sebelum proses dieksekusi.
- **Dual View Mode:** Beralih instan antara mode panduan kotak (*Overlay & Guides*) dan hasil render nyata (*Rendered Preview*).
- **Download Per-Foto:** Opsi unduh instan untuk foto yang sedang di-preview.

### 3. Kontrol Kustomisasi Lengkap
- **Pilihan Aspek Rasio:**
  - Square (`1:1` - Avatar / Instagram / LinkedIn)
  - Portrait (`3:4` - Pas Foto Ijazah, Visa, Dokumen Resmi)
  - Portrait (`4:5` - Instagram Feed)
  - Portrait (`9:16` - Instagram Stories, TikTok, Reels)
  - Landscape (`4:3` & `16:9` - Banner, Landscape)
  - Custom Ratio (input angka bebas W : H)
- **Slider Safe Area & Margin Bahu:**
  - *Headroom Margin* (ruang di atas kepala: 10% - 90%)
  - *Shoulder Level* (kedalaman torso/bahu di bawah dagu: 50% - 220%)
  - *Shoulder Width* (lebar bahu sisi kiri dan kanan: 20% - 150%)
- **4 Opsi Padding & Background Fill (Backdrop Extension):**
  1. **Auto Studio Match (Default):** Auto-deteksi warna latar studio (merah/biru/putih/dll) dan memperluasnya secara mulus saat crop melampaui batas foto asli agar rasio 3:4 tetap terjaga tanpa memotong peci/hijab.
  2. **Auto Edge Fill:** Mereplikasi piksel tepi gambar (*Border Replicate*) secara mulus tanpa garis batas hitam.
  3. **Smart Blur:** Latar belakang diburamkan secara artistik dengan Gaussian Blur dari gambar yang diperbesar.
  4. **Manual Solid Color:** Pilihan warna solid studio dilengkapi:
     - Preset warna pas foto resmi: Merah (`#DC2626`), Biru (`#2563EB`), Putih Formal (`#FFFFFF`), Studio Dark (`#1E293B`).
     - Color Picker native & input kode HEX.
     - Eyedropper API (ambil warna langsung dari layar).

### 4. Pemrosesan Backend & Ekspor Batch ZIP
- **Preservasi Nama File Asli:** Seluruh file hasil ekspor menggunakan nama file asli persis (*original filename*, e.g., `budi_profile.jpg` tetap `budi_profile.jpg`).
- **Process & Download All (.ZIP):** Tombol batch yang mengompres seluruh hasil foto ke dalam 1 arsip `.zip` dan langsung mengunduhnya ke komputer pengguna.
- **Format Output & Kualitas Fleksibel:** Simpan dalam format asli, JPEG, PNG, atau WebP dengan slider kualitas (70% - 100%).

---

## 📁 Struktur Direktori

```
photocrop/
├── main.py                     # Backend FastAPI, API endpoints, stream ZIP
├── vision_engine.py            # Logika Computer Vision (MediaPipe, OpenCV, Padding)
├── create_samples.py           # Generator sampel foto potret
├── test_integration.py         # Skrip pengujian otomatis end-to-end
├── requirements.txt            # Daftar dependensi Python
├── README.md                   # Dokumentasi instalasi dan panduan
├── models/
│   ├── blaze_face_short_range.tflite   # Model MediaPipe Face Detection
│   └── haarcascade_frontalface_default.xml # Model Fallback OpenCV Haar Cascade
├── samples/
│   ├── portrait_formal.jpg     # Sampel foto potret formal
│   └── portrait_casual.jpg     # Sampel foto potret kasual
└── static/
    ├── index.html              # Antarmuka web modern dengan Tailwind CSS
    ├── css/
    │   └── style.css           # Styling tambahan, kanvas, & animasi
    └── js/
        ├── app.js              # State manager, dropzone, batch zip processing
        └── canvas.js           # Controller canvas interaktif HTML5
```

---

## 🚀 Panduan Menjalankan di Komputer Lokal

### 1. Prasyarat
Pastikan Anda telah menginstal **Python 3.9+** (didukung hingga Python 3.13) di komputer Anda.

### 2. Instalasi Dependensi
Buka terminal / PowerShell di direktori proyek ini (`d:\Projects\photocrop`), lalu jalankan:

```bash
pip install -r requirements.txt
```

Atau instal paket secara manual:
```bash
pip install fastapi uvicorn opencv-python mediapipe numpy Pillow python-multipart
```

### 3. Menjalankan Server Aplikasi
Jalankan server lokal menggunakan Python atau Uvicorn:

```bash
python main.py
```

Atau via CLI Uvicorn langsung:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Mengakses Aplikasi Web
Buka browser favorit Anda (Google Chrome, Microsoft Edge, Mozilla Firefox) dan kunjungi:
```
http://127.0.0.1:8000
```

---

## 🧪 Menjalankan Pengujian Otomatis (Testing)

Untuk memverifikasi seluruh komponen backend, integrasi MediaPipe, dan pembuatan arsip ZIP:
```bash
python test_integration.py
```

Hasil uji:
- `GET /` -> HTTP 200 (Halaman antarmuka berhasil dimuat)
- `GET /api/samples` -> HTTP 200 (Sampel demo tersedia)
- `POST /api/detect` -> HTTP 200 (Deteksi wajah MediaPipe akurat)
- `POST /api/process-batch` -> HTTP 200 (Arsip ZIP valid dengan nama file asli dipertahankan)

---

## 🛠️ Arsitektur & Logika Computer Vision

1. **Deteksi Wajah Primer (MediaPipe Tasks):**
   Model `blaze_face_short_range.tflite` mendeteksi bounding box wajah beserta titik penting mata, hidung, dan mulut secara cepat dan akurat.
2. **Fallback Sekunder (OpenCV Haar Cascade):**
   Jika wajah tidak terdeteksi oleh MediaPipe (misalnya pada sudut ekstrem), sistem otomatis mengaktifkan OpenCV Haar Cascade classifier.
3. **Pusat Komposisi Bahu (*Shoulder-Level Framing*):**
   Posisi bahu diperkirakan secara proporsional dari dagu ke bawah (`1.25x` tinggi wajah), lebar bahu (`0.65x` ke kiri dan kanan), serta headroom di atas kepala (`0.35x`).
4. **Penyesuaian Aspek Rasio & Padding:**
   Kotak crop disesuaikan secara simetris terhadap aspek rasio yang dipilih. Jika kotak crop melampaui resolusi asli, sistem menerapkan algoritma:
   - *Auto Edge Fill*: `cv2.copyMakeBorder(..., cv2.BORDER_REPLICATE)`.
   - *Smart Blur*: Pembesaran gambar sumber dengan `cv2.GaussianBlur` tebal sebagai latar artistik.
   - *Solid Color*: Kanvas warna seragam RGB/HEX murni.
