/**
 * PhotoCrop Studio - Main Application Controller
 * Handles drag-and-drop, folder parsing, batch queue, live previews, and ZIP export.
 */

// Global Application State
const appState = {
  items: [], // Array of { id, file, filename, url, img, width, height, faces, cropInfo, customBox, status }
  activeId: null,
  isBatchProcessing: false,
  cropCanvas: null,
  previewMode: 'overlay', // 'overlay' | 'rendered'
  
  settings: {
    aspectRatio: '3:4',
    aspectRatioVal: 3 / 4,
    customW: 3,
    customH: 4,
    headMargin: 0.35,
    shoulderMargin: 1.25,
    sideMargin: 0.65,
    padMode: 'auto_studio', // 'auto_studio' | 'edge' | 'blur' | 'solid'
    padColor: '#FFFFFF',
    outputFormat: 'original',
    quality: 95
  }
};

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
  initDropzone();
  initControlPanel();
  initPreviewModal();
  checkBackendHealth();
  setupDemoButton();
});

/**
 * Check backend health & model status
 */
async function checkBackendHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const badge = document.getElementById('backendStatusBadge');
    if (badge && data.status === 'online') {
      const mpStatus = data.mediapipe_loaded ? 'MediaPipe BlazeFace' : 'OpenCV';
      const segStatus = data.segmenter_loaded ? ' + Silhouette (Peci & Hijab Safe)' : '';
      badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 mr-1.5 animate-pulse"></span> ${mpStatus}${segStatus}`;
      badge.className = 'inline-flex items-center text-xs font-medium text-emerald-400 bg-emerald-950/60 border border-emerald-800/60 px-2.5 py-1 rounded-full';
    }
  } catch (err) {
    console.warn('Backend health check error:', err);
  }
}

/**
 * Drag and Drop & Folder Traversal
 */
function initDropzone() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const folderInput = document.getElementById('folderInput');
  const btnSelectFiles = document.getElementById('btnSelectFiles');
  const btnSelectFolder = document.getElementById('btnSelectFolder');

  if (!dropzone) return;

  btnSelectFiles.addEventListener('click', () => fileInput.click());
  btnSelectFolder.addEventListener('click', () => folderInput.click());

  fileInput.addEventListener('change', (e) => {
    handleFilesList(Array.from(e.target.files));
    fileInput.value = '';
  });

  folderInput.addEventListener('change', (e) => {
    handleFilesList(Array.from(e.target.files));
    folderInput.value = '';
  });

  // Drag over effects
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dropzone-active');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dropzone-active');
    }, false);
  });

  // Handle dropped items (supports both files and entire folders)
  dropzone.addEventListener('drop', async (e) => {
    const dt = e.dataTransfer;
    if (!dt) return;

    const files = [];

    // Support recursive folder traversal via WebKit DataTransferItem API
    if (dt.items && dt.items.length > 0) {
      const items = Array.from(dt.items);
      const entryPromises = items.map(item => {
        if (typeof item.webkitGetAsEntry === 'function') {
          const entry = item.webkitGetAsEntry();
          if (entry) return scanFilesFromEntry(entry);
        }
        const file = item.getAsFile();
        return file ? Promise.resolve([file]) : Promise.resolve([]);
      });

      const results = await Promise.all(entryPromises);
      results.forEach(fileGroup => files.push(...fileGroup));
    } else if (dt.files && dt.files.length > 0) {
      files.push(...Array.from(dt.files));
    }

    handleFilesList(files);
  });
}

/**
 * Recursively scan directory entry for image files
 */
function scanFilesFromEntry(entry) {
  return new Promise((resolve) => {
    if (entry.isFile) {
      entry.file((file) => {
        if (isImageFile(file)) {
          resolve([file]);
        } else {
          resolve([]);
        }
      }, () => resolve([]));
    } else if (entry.isDirectory) {
      const dirReader = entry.createReader();
      const collected = [];

      const readEntries = () => {
        dirReader.readEntries(async (entries) => {
          if (entries.length === 0) {
            resolve(collected);
          } else {
            for (const subEntry of entries) {
              const subFiles = await scanFilesFromEntry(subEntry);
              collected.push(...subFiles);
            }
            readEntries();
          }
        }, () => resolve(collected));
      };
      readEntries();
    } else {
      resolve([]);
    }
  });
}

function isImageFile(file) {
  const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
  const ext = file.name.split('.').pop().toLowerCase();
  return validTypes.includes(file.type) || ['jpg', 'jpeg', 'png', 'webp', 'bmp'].includes(ext);
}

/**
 * Load Instant Demo Samples
 */
function setupDemoButton() {
  const btnDemo = document.getElementById('btnDemoSamples');
  if (!btnDemo) return;

  btnDemo.addEventListener('click', async () => {
    btnDemo.disabled = true;
    btnDemo.innerHTML = `<span class="animate-spin inline-block mr-1">⏳</span> Loading Demos...`;

    try {
      const res = await fetch('/api/samples');
      const data = await res.json();
      if (data.samples && data.samples.length > 0) {
        for (const sample of data.samples) {
          const imgRes = await fetch(sample.url);
          const blob = await imgRes.blob();
          const file = new File([blob], sample.name, { type: blob.type || 'image/jpeg' });
          await addFileToQueue(file);
        }
      }
    } catch (e) {
      console.error('Failed to load demo samples:', e);
      alert('Could not load demo samples from server.');
    } finally {
      btnDemo.disabled = false;
      btnDemo.innerHTML = `<span>✨ Load Demo Photos</span>`;
    }
  });
}

/**
 * Handle incoming files list
 */
async function handleFilesList(files) {
  const validImages = files.filter(isImageFile);
  if (validImages.length === 0) {
    if (files.length > 0) {
      alert('No valid image files found (.jpg, .jpeg, .png, .webp).');
    }
    return;
  }

  for (const file of validImages) {
    await addFileToQueue(file);
  }
}

/**
 * Add a file to queue, parse image, and run detection
 */
async function addFileToQueue(file) {
  // Check for duplicate
  const exists = appState.items.some(it => it.filename === file.name && it.file.size === file.size);
  if (exists) return;

  const id = 'img_' + Math.random().toString(36).substring(2, 9);
  const url = URL.createObjectURL(file);

  const item = {
    id,
    file,
    filename: file.name,
    url,
    img: null,
    width: 0,
    height: 0,
    faces: [],
    cropInfo: null,
    customBox: null,
    status: 'detecting' // 'detecting' | 'ready' | 'custom' | 'error'
  };

  appState.items.push(item);
  renderQueueGrid();
  updateQueueStats();

  // Load Image Object
  const img = new Image();
  img.onload = async () => {
    item.img = img;
    item.width = img.naturalWidth || img.width;
    item.height = img.naturalHeight || img.height;

    // Run backend detection
    await detectFaceForItem(item);
    renderQueueGrid();
    updateQueueStats();
  };
  img.onerror = () => {
    item.status = 'error';
    renderQueueGrid();
  };
  img.src = url;
}

/**
 * Detect face and calculate crop box on backend
 */
async function detectFaceForItem(item) {
  const formData = new FormData();
  formData.append('file', item.file);
  formData.append('aspect_ratio', appState.settings.aspectRatio);
  formData.append('head_margin', appState.settings.headMargin);
  formData.append('shoulder_margin', appState.settings.shoulderMargin);
  formData.append('side_margin', appState.settings.sideMargin);

  try {
    const res = await fetch('/api/detect', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error('Detection failed');

    const data = await res.json();
    item.faces = data.faces || [];
    item.cropInfo = data.crop_info || null;
    item.status = 'ready';

  } catch (err) {
    console.warn(`Detection error for ${item.filename}, fallback to center crop`, err);
    // Intelligent fallback calculation
    item.faces = [];
    item.cropInfo = calculateFallbackCrop(item.width, item.height, appState.settings.aspectRatioVal);
    item.status = 'ready';
  }
}

/**
 * Client-side fallback crop calculation
 */
function calculateFallbackCrop(w, h, aspect) {
  let cropW = w * 0.75;
  let cropH = cropW / aspect;
  if (cropH > h * 0.9) {
    cropH = h * 0.9;
    cropW = cropH * aspect;
  }
  const cx = w / 2;
  const cy = h * 0.45;
  const x1 = Math.round(cx - cropW / 2);
  const y1 = Math.round(cy - cropH / 2);
  const x2 = Math.round(x1 + cropW);
  const y2 = Math.round(y1 + cropH);

  return {
    crop_box: [x1, y1, x2, y2],
    crop_w: x2 - x1,
    crop_h: y2 - y1,
    overflows: (x1 < 0 || y1 < 0 || x2 > w || y2 > h),
    face_detected: false
  };
}

/**
 * Render Queue Cards Grid
 */
function renderQueueGrid() {
  const container = document.getElementById('queueContainer');
  const emptyState = document.getElementById('emptyQueueState');

  if (!container) return;

  if (appState.items.length === 0) {
    if (emptyState) emptyState.classList.remove('hidden');
    container.innerHTML = '';
    return;
  }

  if (emptyState) emptyState.classList.add('hidden');

  container.innerHTML = appState.items.map(item => {
    let statusBadge = '';
    if (item.status === 'detecting') {
      statusBadge = `<span class="inline-flex items-center text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700"><span class="w-1.5 h-1.5 rounded-full bg-indigo-400 mr-1.5 animate-pulse"></span> Detecting...</span>`;
    } else if (item.customBox) {
      statusBadge = `<span class="inline-flex items-center text-xs px-2 py-0.5 rounded-full bg-purple-950/80 text-purple-300 border border-purple-800/60">★ Custom Crop</span>`;
    } else if (item.faces && item.faces.length > 0) {
      const faceCount = item.faces.length;
      if (item.cropInfo && item.cropInfo.has_headwear) {
        statusBadge = `<span class="inline-flex items-center text-xs px-2 py-0.5 rounded-full bg-emerald-950/80 text-emerald-300 border border-emerald-800/60">✓ Peci/Hijab (${faceCount})</span>`;
      } else {
        statusBadge = `<span class="inline-flex items-center text-xs px-2 py-0.5 rounded-full bg-emerald-950/80 text-emerald-300 border border-emerald-800/60">✓ Face (${faceCount})</span>`;
      }
    } else {
      statusBadge = `<span class="inline-flex items-center text-xs px-2 py-0.5 rounded-full bg-amber-950/80 text-amber-300 border border-amber-800/60">Center Fallback</span>`;
    }

    const dims = item.width ? `${item.width} × ${item.height}` : 'Loading...';

    return `
      <div class="group relative bg-slate-800/90 hover:bg-slate-800 border border-slate-700/80 hover:border-indigo-500/70 rounded-xl p-3 transition-all duration-200 shadow-lg flex flex-col justify-between">
        <div class="relative overflow-hidden rounded-lg bg-slate-900 aspect-square flex items-center justify-center mb-2.5 cursor-pointer" onclick="openPreviewModal('${item.id}')">
          <img src="${item.url}" class="object-cover w-full h-full group-hover:scale-105 transition-transform duration-300" alt="${item.filename}" />
          
          <div class="absolute inset-0 bg-slate-950/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
            <span class="bg-indigo-600 hover:bg-indigo-500 text-white p-2 rounded-full shadow-lg transition-transform transform hover:scale-110">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg>
            </span>
          </div>

          <div class="absolute top-2 left-2">
            ${statusBadge}
          </div>
        </div>

        <div>
          <div class="text-xs font-semibold text-slate-200 truncate title="${item.filename}">${item.filename}</div>
          <div class="text-[11px] text-slate-400 mt-0.5">${dims}</div>
        </div>

        <div class="mt-2.5 pt-2 border-t border-slate-700/60 flex items-center justify-between">
          <button onclick="openPreviewModal('${item.id}')" class="text-xs text-indigo-400 hover:text-indigo-300 font-medium inline-flex items-center gap-1 hover:underline">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/></svg>
            Adjust
          </button>
          
          <button onclick="removeItem('${item.id}')" class="text-xs text-rose-400 hover:text-rose-300 p-1 rounded hover:bg-rose-950/40" title="Remove image">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
          </button>
        </div>
      </div>
    `;
  }).join('');
}

function updateQueueStats() {
  const total = appState.items.length;
  const countBadge = document.getElementById('queueCountBadge');
  const btnProcess = document.getElementById('btnProcessBatch');

  if (countBadge) {
    countBadge.textContent = `${total} photo${total === 1 ? '' : 's'}`;
  }

  if (btnProcess) {
    btnProcess.disabled = total === 0 || appState.isBatchProcessing;
    if (total === 0) {
      btnProcess.classList.add('opacity-50', 'cursor-not-allowed');
    } else {
      btnProcess.classList.remove('opacity-50', 'cursor-not-allowed');
    }
  }
}

function removeItem(id) {
  const idx = appState.items.findIndex(it => it.id === id);
  if (idx !== -1) {
    URL.revokeObjectURL(appState.items[idx].url);
    appState.items.splice(idx, 1);
    renderQueueGrid();
    updateQueueStats();
  }
}

function clearAllItems() {
  if (appState.items.length === 0) return;
  if (!confirm('Clear all uploaded images?')) return;
  
  appState.items.forEach(it => URL.revokeObjectURL(it.url));
  appState.items = [];
  renderQueueGrid();
  updateQueueStats();
}

/**
 * UI Control Panel (Aspect Ratio, Margins, Fill Modes)
 */
function initControlPanel() {
  // Clear button
  const btnClearAll = document.getElementById('btnClearAll');
  if (btnClearAll) btnClearAll.addEventListener('click', clearAllItems);

  // Aspect Ratio Preset Buttons
  const ratioButtons = document.querySelectorAll('[data-aspect]');
  ratioButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      ratioButtons.forEach(b => b.classList.remove('ring-2', 'ring-indigo-500', 'bg-indigo-600/30', 'border-indigo-500'));
      btn.classList.add('ring-2', 'ring-indigo-500', 'bg-indigo-600/30', 'border-indigo-500');

      const ratioKey = btn.getAttribute('data-aspect');
      appState.settings.aspectRatio = ratioKey;

      const ratios = {
        '1:1': 1.0,
        '3:4': 3 / 4,
        '4:5': 4 / 5,
        '9:16': 9 / 16,
        '4:3': 4 / 3,
        '16:9': 16 / 9,
        'custom': appState.settings.customW / appState.settings.customH
      };

      appState.settings.aspectRatioVal = ratios[ratioKey] || 1.0;

      const customInputs = document.getElementById('customRatioInputs');
      if (customInputs) {
        if (ratioKey === 'custom') {
          customInputs.classList.remove('hidden');
        } else {
          customInputs.classList.add('hidden');
        }
      }

      onGlobalSettingsChanged();
    });
  });

  // Custom Ratio Inputs
  const customW = document.getElementById('customRatioW');
  const customH = document.getElementById('customRatioH');
  if (customW && customH) {
    const updateCustomRatio = () => {
      const w = parseFloat(customW.value) || 1;
      const h = parseFloat(customH.value) || 1;
      appState.settings.customW = w;
      appState.settings.customH = h;
      appState.settings.aspectRatio = `${w}:${h}`;
      appState.settings.aspectRatioVal = w / h;
      onGlobalSettingsChanged();
    };
    customW.addEventListener('input', updateCustomRatio);
    customH.addEventListener('input', updateCustomRatio);
  }

  // Margin Sliders
  bindSlider('sliderHeadMargin', 'labelHeadMargin', (val) => {
    appState.settings.headMargin = parseFloat(val);
    return `${Math.round(val * 100)}%`;
  });

  bindSlider('sliderShoulderMargin', 'labelShoulderMargin', (val) => {
    appState.settings.shoulderMargin = parseFloat(val);
    return `${Math.round(val * 100)}%`;
  });

  bindSlider('sliderSideMargin', 'labelSideMargin', (val) => {
    appState.settings.sideMargin = parseFloat(val);
    return `${Math.round(val * 100)}%`;
  });

  // Padding Fill Modes
  const fillModeButtons = document.querySelectorAll('[data-fill-mode]');
  fillModeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      fillModeButtons.forEach(b => b.classList.remove('ring-2', 'ring-indigo-500', 'border-indigo-500', 'bg-indigo-950/40'));
      btn.classList.add('ring-2', 'ring-indigo-500', 'border-indigo-500', 'bg-indigo-950/40');

      const mode = btn.getAttribute('data-fill-mode');
      appState.settings.padMode = mode;

      const colorSection = document.getElementById('solidColorOptions');
      if (colorSection) {
        if (mode === 'solid') {
          colorSection.classList.remove('hidden');
        } else {
          colorSection.classList.add('hidden');
        }
      }
    });
  });

  // Solid Color Picker & Presets
  const colorPicker = document.getElementById('padColorPicker');
  const colorHexInput = document.getElementById('padColorHex');
  const btnEyedropper = document.getElementById('btnEyedropper');

  if (colorPicker && colorHexInput) {
    colorPicker.addEventListener('input', (e) => {
      colorHexInput.value = e.target.value.toUpperCase();
      appState.settings.padColor = colorHexInput.value;
    });

    colorHexInput.addEventListener('input', (e) => {
      let val = e.target.value;
      if (!val.startsWith('#')) val = '#' + val;
      if (/^#[0-9A-F]{6}$/i.test(val)) {
        colorPicker.value = val;
        appState.settings.padColor = val;
      }
    });

    // Swatches
    document.querySelectorAll('[data-color-swatch]').forEach(swatch => {
      swatch.addEventListener('click', () => {
        const c = swatch.getAttribute('data-color-swatch');
        colorPicker.value = c;
        colorHexInput.value = c.toUpperCase();
        appState.settings.padColor = c;
      });
    });

    // Eyedropper API
    if (btnEyedropper) {
      if (window.EyeDropper) {
        btnEyedropper.addEventListener('click', async () => {
          try {
            const eyeDropper = new EyeDropper();
            const result = await eyeDropper.open();
            if (result && result.sRGBHex) {
              colorPicker.value = result.sRGBHex;
              colorHexInput.value = result.sRGBHex.toUpperCase();
              appState.settings.padColor = result.sRGBHex;
            }
          } catch (e) {
            console.log('Eyedropper cancelled or failed', e);
          }
        });
      } else {
        btnEyedropper.title = 'EyeDropper API is not supported on this browser';
        btnEyedropper.classList.add('opacity-40', 'cursor-not-allowed');
      }
    }
  }

  // Export Settings
  const selectFormat = document.getElementById('selectOutputFormat');
  if (selectFormat) {
    selectFormat.addEventListener('change', (e) => {
      appState.settings.outputFormat = e.target.value;
    });
  }

  const sliderQuality = document.getElementById('sliderQuality');
  const labelQuality = document.getElementById('labelQuality');
  if (sliderQuality && labelQuality) {
    sliderQuality.addEventListener('input', (e) => {
      labelQuality.textContent = `${e.target.value}%`;
      appState.settings.quality = parseInt(e.target.value, 10);
    });
  }

  // Batch Process Button
  const btnProcess = document.getElementById('btnProcessBatch');
  if (btnProcess) {
    btnProcess.addEventListener('click', processBatch);
  }
}

function bindSlider(sliderId, labelId, formatter) {
  const slider = document.getElementById(sliderId);
  const label = document.getElementById(labelId);
  if (!slider || !label) return;

  slider.addEventListener('input', (e) => {
    label.textContent = formatter(e.target.value);
    onGlobalSettingsChanged();
  });
}

/**
 * When global sliders/aspect ratio change, recompute crop bounds for all non-custom items
 */
function onGlobalSettingsChanged() {
  appState.items.forEach(item => {
    if (!item.customBox && item.width && item.height) {
      const primaryFace = item.faces && item.faces.length > 0 ? item.faces[0].bbox : null;
      item.cropInfo = calculateCropGeometry(
        item.width,
        item.height,
        primaryFace,
        appState.settings.aspectRatioVal,
        appState.settings.headMargin,
        appState.settings.shoulderMargin,
        appState.settings.sideMargin
      );
    }
  });

  // If preview modal is currently open, update its canvas as well
  if (appState.activeId && appState.cropCanvas) {
    const activeItem = appState.items.find(it => it.id === appState.activeId);
    if (activeItem) {
      const box = activeItem.customBox || (activeItem.cropInfo ? activeItem.cropInfo.crop_box : [0, 0, activeItem.width, activeItem.height]);
      appState.cropCanvas.setAspectRatio(appState.settings.aspectRatioVal, true);
      appState.cropCanvas.setCropBox(box);
    }
  }

  renderQueueGrid();
}

/**
 * Geometry calculation
 */
function calculateCropGeometry(imgW, imgH, faceBbox, targetAspect, headMargin, shoulderMargin, sideMargin) {
  let leftX, rightX, topY, bottomY;

  if (faceBbox) {
    const [fx, fy, fw, fh] = faceBbox;
    topY = fy - (headMargin * fh);
    bottomY = fy + fh + (shoulderMargin * fh);
    leftX = fx - (sideMargin * fw);
    rightX = fx + fw + (sideMargin * fw);
  } else {
    const roiW = imgW * 0.75;
    const roiH = roiW / targetAspect;
    const cx = imgW / 2;
    const cy = imgH * 0.42;
    topY = cy - roiH * 0.35;
    bottomY = topY + roiH;
    leftX = cx - roiW / 2;
    rightX = cx + roiW / 2;
  }

  const roiW = rightX - leftX;
  const roiH = bottomY - topY;
  const currentAspect = roiW / roiH;

  let cropX1, cropY1, cropX2, cropY2;

  if (currentAspect < targetAspect) {
    const newW = roiH * targetAspect;
    const diffW = newW - roiW;
    cropX1 = leftX - diffW / 2;
    cropX2 = rightX + diffW / 2;
    cropY1 = topY;
    cropY2 = bottomY;
  } else {
    const newH = roiW / targetAspect;
    const diffH = newH - roiH;
    cropY1 = topY - diffH * 0.30;
    cropY2 = bottomY + diffH * 0.70;
    cropX1 = leftX;
    cropX2 = rightX;
  }

  const finalBox = [
    Math.round(cropX1),
    Math.round(cropY1),
    Math.round(cropX2),
    Math.round(cropY2)
  ];

  return {
    crop_box: finalBox,
    crop_w: finalBox[2] - finalBox[0],
    crop_h: finalBox[3] - finalBox[1],
    overflows: (finalBox[0] < 0 || finalBox[1] < 0 || finalBox[2] > imgW || finalBox[3] > imgH),
    face_detected: !!faceBbox
  };
}

/**
 * Interactive Preview Modal (HTML5 Canvas)
 */
function initPreviewModal() {
  const modal = document.getElementById('previewModal');
  const btnClose = document.getElementById('btnClosePreview');
  const canvasElement = document.getElementById('cropPreviewCanvas');

  if (!modal || !canvasElement) return;

  appState.cropCanvas = new InteractiveCropCanvas(canvasElement, {
    aspectRatio: appState.settings.aspectRatioVal,
    lockAspectRatio: true,
    onCropChange: (newBox) => {
      onCanvasCropChanged(newBox);
    }
  });

  if (btnClose) {
    btnClose.addEventListener('click', closePreviewModal);
  }

  // Close on Escape key
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
      closePreviewModal();
    }
  });

  // Zoom buttons
  document.getElementById('btnZoomIn')?.addEventListener('click', () => appState.cropCanvas.zoom(1.2));
  document.getElementById('btnZoomOut')?.addEventListener('click', () => appState.cropCanvas.zoom(0.8));
  document.getElementById('btnZoomFit')?.addEventListener('click', () => appState.cropCanvas.fitToScreen());

  // Toggle Face Box & Guides
  document.getElementById('chkShowFaces')?.addEventListener('change', (e) => {
    appState.cropCanvas.showFaceBox = e.target.checked;
    appState.cropCanvas.render();
  });

  document.getElementById('chkShowGrid')?.addEventListener('change', (e) => {
    appState.cropCanvas.showGrid = e.target.checked;
    appState.cropCanvas.render();
  });

  // View Switcher (Overlay vs Rendered Preview)
  const tabOverlay = document.getElementById('tabViewOverlay');
  const tabRendered = document.getElementById('tabViewRendered');

  if (tabOverlay && tabRendered) {
    tabOverlay.addEventListener('click', () => switchPreviewTab('overlay'));
    tabRendered.addEventListener('click', () => switchPreviewTab('rendered'));
  }

  // Modal Action Buttons
  document.getElementById('btnApplyCustom')?.addEventListener('click', () => {
    const item = appState.items.find(it => it.id === appState.activeId);
    if (item && appState.cropCanvas) {
      item.customBox = appState.cropCanvas.getCropBox();
      renderQueueGrid();
      closePreviewModal();
    }
  });

  document.getElementById('btnResetCustomCrop')?.addEventListener('click', () => {
    const item = appState.items.find(it => it.id === appState.activeId);
    if (item) {
      item.customBox = null;
      const primaryFace = item.faces && item.faces.length > 0 ? item.faces[0].bbox : null;
      item.cropInfo = calculateCropGeometry(
        item.width,
        item.height,
        primaryFace,
        appState.settings.aspectRatioVal,
        appState.settings.headMargin,
        appState.settings.shoulderMargin,
        appState.settings.sideMargin
      );
      appState.cropCanvas.setCropBox(item.cropInfo.crop_box);
      renderQueueGrid();
    }
  });

  document.getElementById('btnDownloadSingle')?.addEventListener('click', () => {
    downloadSinglePhoto(appState.activeId);
  });
}

function openPreviewModal(id) {
  const item = appState.items.find(it => it.id === id);
  if (!item || !item.img) return;

  appState.activeId = id;
  const modal = document.getElementById('previewModal');
  modal.classList.remove('hidden');

  // Set modal header info
  document.getElementById('previewModalTitle').textContent = item.filename;
  document.getElementById('previewModalDims').textContent = `${item.width} × ${item.height} px`;

  const box = item.customBox || (item.cropInfo ? item.cropInfo.crop_box : [0, 0, item.width, item.height]);
  const headwearY = item.cropInfo ? item.cropInfo.headwear_top_y : null;

  appState.cropCanvas.setAspectRatio(appState.settings.aspectRatioVal, true);
  appState.cropCanvas.setImage(item.img, item.faces, box, headwearY);

  switchPreviewTab('overlay');
}

function closePreviewModal() {
  const modal = document.getElementById('previewModal');
  if (modal) modal.classList.add('hidden');
  appState.activeId = null;
}

function onCanvasCropChanged(newBox) {
  const item = appState.items.find(it => it.id === appState.activeId);
  if (item) {
    item.customBox = newBox;
    renderQueueGrid();
  }

  // If in rendered view mode, refresh rendered preview
  if (appState.previewMode === 'rendered') {
    fetchRenderedCropPreview();
  }
}

async function switchPreviewTab(mode) {
  appState.previewMode = mode;
  const tabOverlay = document.getElementById('tabViewOverlay');
  const tabRendered = document.getElementById('tabViewRendered');
  const canvasWrapper = document.getElementById('canvasViewContainer');
  const renderedContainer = document.getElementById('renderedViewContainer');

  if (mode === 'overlay') {
    tabOverlay?.classList.add('bg-indigo-600', 'text-white');
    tabOverlay?.classList.remove('text-slate-400');
    tabRendered?.classList.remove('bg-indigo-600', 'text-white');
    tabRendered?.classList.add('text-slate-400');

    canvasWrapper?.classList.remove('hidden');
    renderedContainer?.classList.add('hidden');
    appState.cropCanvas?.render();
  } else {
    tabRendered?.classList.add('bg-indigo-600', 'text-white');
    tabRendered?.classList.remove('text-slate-400');
    tabOverlay?.classList.remove('bg-indigo-600', 'text-white');
    tabOverlay?.classList.add('text-slate-400');

    canvasWrapper?.classList.add('hidden');
    renderedContainer?.classList.remove('hidden');
    await fetchRenderedCropPreview();
  }
}

async function fetchRenderedCropPreview() {
  const item = appState.items.find(it => it.id === appState.activeId);
  if (!item) return;

  const renderedImg = document.getElementById('renderedPreviewImage');
  const spinner = document.getElementById('renderedPreviewSpinner');

  if (spinner) spinner.classList.remove('hidden');

  const cropBox = item.customBox || (item.cropInfo ? item.cropInfo.crop_box : null);

  const formData = new FormData();
  formData.append('file', item.file);
  formData.append('aspect_ratio', appState.settings.aspectRatio);
  formData.append('head_margin', appState.settings.headMargin);
  formData.append('shoulder_margin', appState.settings.shoulderMargin);
  formData.append('side_margin', appState.settings.sideMargin);
  formData.append('pad_mode', appState.settings.padMode);
  formData.append('pad_color', appState.settings.padColor);
  if (cropBox) {
    formData.append('custom_box', JSON.stringify(cropBox));
  }

  try {
    const res = await fetch('/api/preview-crop', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error('Preview fetch failed');

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    if (renderedImg) {
      renderedImg.onload = () => URL.revokeObjectURL(renderedImg.src);
      renderedImg.src = url;
    }
  } catch (err) {
    console.error('Failed to load rendered preview:', err);
  } finally {
    if (spinner) spinner.classList.add('hidden');
  }
}

/**
 * Download single photo immediately
 */
async function downloadSinglePhoto(id) {
  const item = appState.items.find(it => it.id === id);
  if (!item) return;

  const btn = document.getElementById('btnDownloadSingle');
  const originalText = btn ? btn.innerHTML : '';
  if (btn) btn.innerHTML = `<span class="animate-spin inline-block mr-1">⏳</span> Downloading...`;

  const cropBox = item.customBox || (item.cropInfo ? item.cropInfo.crop_box : null);

  const formData = new FormData();
  formData.append('file', item.file);
  formData.append('aspect_ratio', appState.settings.aspectRatio);
  formData.append('head_margin', appState.settings.headMargin);
  formData.append('shoulder_margin', appState.settings.shoulderMargin);
  formData.append('side_margin', appState.settings.sideMargin);
  formData.append('pad_mode', appState.settings.padMode);
  formData.append('pad_color', appState.settings.padColor);
  formData.append('output_format', appState.settings.outputFormat);
  formData.append('quality', appState.settings.quality);
  if (cropBox) {
    formData.append('custom_box', JSON.stringify(cropBox));
  }

  try {
    const res = await fetch('/api/crop-single', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error('Download failed');

    const blob = await res.blob();
    const disposition = res.headers.get('Content-Disposition');
    let filename = item.filename;
    if (disposition && disposition.includes('filename=')) {
      const match = disposition.match(/filename="?([^"]+)"?/);
      if (match && match[1]) filename = match[1];
    }

    const downloadUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(downloadUrl), 60000);

  } catch (err) {
    alert('Failed to download image: ' + err.message);
  } finally {
    if (btn) btn.innerHTML = originalText;
  }
}

/**
 * Batch Process & Download All (.ZIP)
 */
async function processBatch() {
  if (appState.items.length === 0 || appState.isBatchProcessing) return;

  appState.isBatchProcessing = true;
  const btnProcess = document.getElementById('btnProcessBatch');
  const progressBarContainer = document.getElementById('batchProgressBarContainer');
  const progressBar = document.getElementById('batchProgressBar');
  const progressText = document.getElementById('batchProgressText');

  btnProcess.disabled = true;
  progressBarContainer.classList.remove('hidden');
  progressBar.style.width = '5%';
  progressText.textContent = `Preparing ${appState.items.length} photos...`;

  try {
    // If JSZip is available in browser, process photos item-by-item with live progress
    // This avoids giant multipart request drops, gives real percentage progress, and is resilient
    if (window.JSZip) {
      const zip = new JSZip();
      const total = appState.items.length;
      let completed = 0;
      let errors = [];

      for (let i = 0; i < total; i++) {
        const item = appState.items[i];
        progressText.textContent = `Cropping photo ${i + 1} of ${total}: ${item.filename}...`;
        progressBar.style.width = `${Math.round(((i + 0.3) / total) * 90)}%`;

        const cropBox = item.customBox || (item.cropInfo ? item.cropInfo.crop_box : null);

        const formData = new FormData();
        formData.append('file', item.file, item.filename);
        formData.append('aspect_ratio', appState.settings.aspectRatio);
        formData.append('head_margin', appState.settings.headMargin);
        formData.append('shoulder_margin', appState.settings.shoulderMargin);
        formData.append('side_margin', appState.settings.sideMargin);
        formData.append('pad_mode', appState.settings.padMode);
        formData.append('pad_color', appState.settings.padColor);
        formData.append('output_format', appState.settings.outputFormat);
        formData.append('quality', appState.settings.quality);
        if (cropBox) {
          formData.append('custom_box', JSON.stringify(cropBox));
        }

        try {
          const res = await fetch('/api/crop-single', {
            method: 'POST',
            body: formData
          });

          if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error ${res.status}`);
          }

          const blob = await res.blob();
          const disposition = res.headers.get('Content-Disposition');
          let outName = item.filename;
          if (disposition && disposition.includes('filename=')) {
            const match = disposition.match(/filename="?([^"]+)"?/);
            if (match && match[1]) outName = match[1];
          }

          zip.file(outName, blob);
          completed++;
        } catch (err) {
          console.error(`Error processing ${item.filename}:`, err);
          errors.push(`${item.filename}: ${err.message}`);
        }

        progressBar.style.width = `${Math.round(((i + 1) / total) * 90)}%`;
      }

      if (completed === 0) {
        throw new Error(`Failed to process photos: ${errors.join(', ')}`);
      }

      progressText.textContent = `Packaging ${completed} photos into ZIP archive...`;
      progressBar.style.width = '95%';

      const zipBlob = await zip.generateAsync({
        type: 'blob',
        compression: 'DEFLATE',
        compressionOptions: { level: 6 }
      });

      const downloadUrl = URL.createObjectURL(zipBlob);
      const a = document.createElement('a');
      a.href = downloadUrl;
      a.download = 'cropped_photos.zip';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(downloadUrl), 60000);

      progressBar.style.width = '100%';
      progressText.textContent = `✓ Successfully downloaded ${completed} cropped photos in ZIP archive!`;

      setTimeout(() => {
        progressBarContainer.classList.add('hidden');
      }, 4000);

    } else {
      // Fallback to Backend single-request ZIP generation
      const formData = new FormData();
      appState.items.forEach(item => {
        formData.append('files', item.file, item.filename);
      });
      formData.append('aspect_ratio', appState.settings.aspectRatio);
      formData.append('head_margin', appState.settings.headMargin);
      formData.append('shoulder_margin', appState.settings.shoulderMargin);
      formData.append('side_margin', appState.settings.sideMargin);
      formData.append('pad_mode', appState.settings.padMode);
      formData.append('pad_color', appState.settings.padColor);
      formData.append('output_format', appState.settings.outputFormat);
      formData.append('quality', appState.settings.quality);

      const overrides = {};
      appState.items.forEach(item => {
        if (item.customBox) {
          overrides[item.filename] = item.customBox;
        }
      });
      if (Object.keys(overrides).length > 0) {
        formData.append('custom_overrides', JSON.stringify(overrides));
      }

      progressText.textContent = 'Processing Computer Vision face & shoulder models...';
      progressBar.style.width = '60%';

      const res = await fetch('/api/process-batch', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned status ${res.status}`);
      }

      progressBar.style.width = '95%';
      progressText.textContent = 'Downloading ZIP archive...';

      const blob = await res.blob();
      const downloadUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = downloadUrl;
      a.download = 'cropped_photos.zip';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(downloadUrl), 60000);

      progressBar.style.width = '100%';
      progressText.textContent = `✓ Successfully downloaded ${appState.items.length} cropped photos in ZIP archive!`;

      setTimeout(() => {
        progressBarContainer.classList.add('hidden');
      }, 4000);
    }

  } catch (err) {
    console.error('Batch error:', err);
    alert('Batch Processing Error: ' + err.message);
    progressBarContainer.classList.add('hidden');
  } finally {
    appState.isBatchProcessing = false;
    btnProcess.disabled = false;
  }
}
