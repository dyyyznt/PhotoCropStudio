@echo off
title PhotoCrop Studio - AI Face & Shoulder Auto-Cropper
cd /d "%~dp0"

echo ===================================================
echo              PhotoCrop Studio Launcher
echo    AI Face, Peci & Hijab Safe Auto-Cropper
echo ===================================================
echo.

:: 1. Verifikasi instalasi Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python tidak ditemukan di sistem PATH!
    echo Silakan unduh dan pasang Python dari https://www.python.org/
    echo Pastikan centang "Add Python to PATH" saat instalasi.
    echo.
    pause
    exit /b 1
)

:: 2. Verifikasi paket dependensi utama
echo [*] Memeriksa dependensi Python...
python -c "import fastapi, uvicorn, cv2, mediapipe, numpy, PIL" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [!] Dependensi belum lengkap. Menginstal paket otomatis...
    python -m pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Gagal menginstal dependensi dari requirements.txt.
        pause
        exit /b 1
    )
    echo [*] Dependensi berhasil diinstal!
)

:: 3. Buka browser otomatis setelah delay singkat (2 detik agar server siap)
echo [*] Mempersiapkan browser di http://localhost:8000 ...
start "" /b cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8000"

:: 4. Jalankan server FastAPI
echo [*] Menjalankan server PhotoCrop Studio...
echo.
echo ===================================================
echo  Aplikasi Berjalan di: http://localhost:8000
echo  Tekan Ctrl + C di jendela ini untuk mematikan server.
echo ===================================================
echo.

python main.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [!] Server berhenti atau terjadi kesalahan (Exit code: %ERRORLEVEL%).
    pause
)
