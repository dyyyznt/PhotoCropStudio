/**
 * Interactive HTML5 Canvas Controller for PhotoCrop Studio
 * Handles image rendering, face bounding boxes, shoulder crop guides,
 * interactive dragging/resizing, zoom/pan, and padding visualization.
 */

class InteractiveCropCanvas {
  constructor(canvasElement, options = {}) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.options = Object.assign({
      onCropChange: null,
      aspectRatio: 3 / 4,
      lockAspectRatio: true
    }, options);

    this.image = null;
    this.imgW = 0;
    this.imgH = 0;
    this.faces = [];
    
    // Crop box in image pixel coordinates: [x1, y1, x2, y2]
    this.cropBox = [0, 0, 0, 0];

    // Canvas view transforms
    this.scale = 1.0;
    this.offsetX = 0;
    this.offsetY = 0;

    // Interaction state
    this.isDragging = false;
    this.dragMode = null; // 'move', 'nw', 'ne', 'sw', 'se', 'n', 's', 'w', 'e', 'pan'
    this.dragStart = { x: 0, y: 0 };
    this.cropStart = [0, 0, 0, 0];
    this.handleRadius = 8;
    this.headwearTopY = null;

    // Display toggles
    this.showFaceBox = true;
    this.showKeypoints = true;
    this.showGrid = true;
    this.showPaddingArea = true;

    this.bindEvents();
  }

  setImage(imageElement, faces = [], initialCropBox = null, headwearTopY = null) {
    this.image = imageElement;
    this.imgW = imageElement.naturalWidth || imageElement.width;
    this.imgH = imageElement.naturalHeight || imageElement.height;
    this.faces = faces || [];
    this.headwearTopY = headwearTopY;

    if (initialCropBox && initialCropBox.length === 4) {
      this.cropBox = [...initialCropBox];
    } else {
      this.cropBox = [0, 0, this.imgW, this.imgH];
    }

    this.fitToScreen();
    this.render();
  }

  setCropBox(box, notify = false) {
    this.cropBox = [
      Math.round(box[0]),
      Math.round(box[1]),
      Math.round(box[2]),
      Math.round(box[3])
    ];
    this.render();
    if (notify && typeof this.options.onCropChange === 'function') {
      this.options.onCropChange(this.cropBox);
    }
  }

  getCropBox() {
    return [...this.cropBox];
  }

  setAspectRatio(ratio, lock = true) {
    this.options.aspectRatio = ratio;
    this.options.lockAspectRatio = lock;
    if (lock && ratio > 0) {
      this.enforceAspectRatio();
    }
    this.render();
  }

  enforceAspectRatio() {
    const [x1, y1, x2, y2] = this.cropBox;
    const currentW = x2 - x1;
    const currentH = y2 - y1;
    const targetRatio = this.options.aspectRatio;

    if (targetRatio <= 0) return;

    let newW = currentW;
    let newH = currentH;
    const currentRatio = currentW / currentH;

    if (currentRatio < targetRatio) {
      newW = currentH * targetRatio;
    } else {
      newH = currentW / targetRatio;
    }

    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;

    this.cropBox = [
      Math.round(cx - newW / 2),
      Math.round(cy - newH / 2),
      Math.round(cx + newW / 2),
      Math.round(cy + newH / 2)
    ];

    if (typeof this.options.onCropChange === 'function') {
      this.options.onCropChange(this.cropBox);
    }
  }

  fitToScreen() {
    if (!this.image) return;

    const padding = 40;
    const containerW = this.canvas.parentElement.clientWidth || 800;
    const containerH = this.canvas.parentElement.clientHeight || 550;

    this.canvas.width = containerW;
    this.canvas.height = containerH;

    // Determine scale to fit image with padding
    const scaleX = (containerW - padding * 2) / this.imgW;
    const scaleY = (containerH - padding * 2) / this.imgH;
    this.scale = Math.min(scaleX, scaleY, 1.2);

    // Center image
    this.offsetX = (containerW - this.imgW * this.scale) / 2;
    this.offsetY = (containerH - this.imgH * this.scale) / 2;

    this.render();
  }

  zoom(factor, clientX, clientY) {
    const oldScale = this.scale;
    const newScale = Math.max(0.1, Math.min(oldScale * factor, 8.0));

    // Zoom towards point or center
    const rect = this.canvas.getBoundingClientRect();
    const cx = clientX !== undefined ? clientX - rect.left : this.canvas.width / 2;
    const cy = clientY !== undefined ? clientY - rect.top : this.canvas.height / 2;

    this.offsetX = cx - (cx - this.offsetX) * (newScale / oldScale);
    this.offsetY = cy - (cy - this.offsetY) * (newScale / oldScale);
    this.scale = newScale;

    this.render();
  }

  imgToCanvas(x, y) {
    return {
      x: this.offsetX + x * this.scale,
      y: this.offsetY + y * this.scale
    };
  }

  canvasToImg(cx, cy) {
    return {
      x: (cx - this.offsetX) / this.scale,
      y: (cy - this.offsetY) / this.scale
    };
  }

  render() {
    const ctx = this.ctx;
    const cw = this.canvas.width;
    const ch = this.canvas.height;

    ctx.clearRect(0, 0, cw, ch);

    if (!this.image) {
      ctx.fillStyle = '#64748b';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No image loaded', cw / 2, ch / 2);
      return;
    }

    // 1. Draw Image
    const p1 = this.imgToCanvas(0, 0);
    const p2 = this.imgToCanvas(this.imgW, this.imgH);
    ctx.drawImage(this.image, p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);

    // 2. Draw Image Border
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 1;
    ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);

    // 3. Draw Padding Zones (areas of crop box that extend beyond image dimensions)
    const [cx1, cy1, cx2, cy2] = this.cropBox;
    const cp1 = this.imgToCanvas(cx1, cy1);
    const cp2 = this.imgToCanvas(cx2, cy2);
    const cropBoxW = cp2.x - cp1.x;
    const cropBoxH = cp2.y - cp1.y;

    const overflows = (cx1 < 0 || cy1 < 0 || cx2 > this.imgW || cy2 > this.imgH);

    if (overflows && this.showPaddingArea) {
      this.drawPaddingHatching(cp1.x, cp1.y, cropBoxW, cropBoxH, p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
    }

    // 4. Draw Darkened Backdrop outside Crop Box
    ctx.save();
    ctx.fillStyle = 'rgba(15, 23, 42, 0.65)';
    ctx.beginPath();
    // Outer rect
    ctx.rect(0, 0, cw, ch);
    // Inner rect (cut out crop box)
    ctx.rect(cp1.x, cp1.y, cropBoxW, cropBoxH);
    ctx.fill('evenodd');
    ctx.restore();

    // 5. Draw Detected Faces & Headwear Guide
    if (this.showFaceBox && this.faces && this.faces.length > 0) {
      this.drawFaces();
    }

    if (this.headwearTopY !== null && this.headwearTopY !== undefined) {
      const hp = this.imgToCanvas(0, this.headwearTopY);
      ctx.save();
      ctx.strokeStyle = '#10b981'; // emerald
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(cp1.x, hp.y);
      ctx.lineTo(cp2.x, hp.y);
      ctx.stroke();

      ctx.fillStyle = '#10b981';
      ctx.font = '10px sans-serif';
      ctx.fillText('✓ Peci/Hijab Boundary', cp1.x + 8, hp.y - 4);
      ctx.restore();
    }

    // 6. Draw Crop Box Bounds & Rule of Thirds
    this.drawCropBox(cp1.x, cp1.y, cropBoxW, cropBoxH, overflows);

    // 7. Draw Resize & Move Handles
    this.drawHandles(cp1.x, cp1.y, cropBoxW, cropBoxH);
  }

  drawPaddingHatching(cx, cy, cw, ch, ix, iy, iw, ih) {
    const ctx = this.ctx;
    ctx.save();

    // Clip to crop box
    ctx.beginPath();
    ctx.rect(cx, cy, cw, ch);
    ctx.clip();

    // Exclude image area
    ctx.beginPath();
    ctx.rect(cx, cy, cw, ch);
    ctx.rect(ix, iy, iw, ih);
    ctx.fillStyle = 'rgba(239, 68, 68, 0.15)'; // light red / warning tint for padding area
    ctx.fill('evenodd');

    // Draw dashed diagonal lines in overflow area
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.35)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 6]);

    const step = 20;
    for (let x = cx - ch; x < cx + cw + ch; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, cy);
      ctx.lineTo(x + ch, cy + ch);
      ctx.stroke();
    }

    ctx.restore();
  }

  drawFaces() {
    const ctx = this.ctx;

    this.faces.forEach((face, idx) => {
      const [fx, fy, fw, fh] = face.bbox;
      const fp1 = this.imgToCanvas(fx, fy);
      const fp2 = this.imgToCanvas(fx + fw, fy + fh);
      const w = fp2.x - fp1.x;
      const h = fp2.y - fp1.y;

      // Face boundary box
      ctx.save();
      ctx.strokeStyle = '#10b981'; // emerald green
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(fp1.x, fp1.y, w, h);

      // Label badge
      const conf = Math.round((face.confidence || 0.95) * 100);
      const label = `Face #${idx + 1} (${conf}%)`;
      ctx.font = '11px sans-serif';
      const textW = ctx.measureText(label).width;

      ctx.fillStyle = 'rgba(16, 185, 129, 0.9)';
      ctx.fillRect(fp1.x, fp1.y - 18, textW + 12, 18);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, fp1.x + 6, fp1.y - 5);

      // Draw Keypoints (eyes, nose, mouth)
      if (this.showKeypoints && face.keypoints) {
        face.keypoints.forEach(kp => {
          const kpPos = this.imgToCanvas(kp.x, kp.y);
          ctx.beginPath();
          ctx.arc(kpPos.x, kpPos.y, 3, 0, Math.PI * 2);
          ctx.fillStyle = '#38bdf8'; // light sky blue
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1;
          ctx.stroke();
        });
      }

      ctx.restore();
    });
  }

  drawCropBox(x, y, w, h, overflows) {
    const ctx = this.ctx;
    ctx.save();

    // Main Crop Boundary Border
    ctx.strokeStyle = overflows ? '#f59e0b' : '#6366f1'; // Amber if padding needed, Indigo otherwise
    ctx.lineWidth = 2.5;
    ctx.strokeRect(x, y, w, h);

    // Rule of Thirds Grid
    if (this.showGrid && w > 60 && h > 60) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);

      // Vertical thirds
      ctx.beginPath();
      ctx.moveTo(x + w / 3, y);
      ctx.lineTo(x + w / 3, y + h);
      ctx.moveTo(x + (2 * w) / 3, y);
      ctx.lineTo(x + (2 * w) / 3, y + h);

      // Horizontal thirds
      ctx.moveTo(x, y + h / 3);
      ctx.lineTo(x + w, y + h / 3);
      ctx.moveTo(x, y + (2 * h) / 3);
      ctx.lineTo(x + w, y + (2 * h) / 3);
      ctx.stroke();
    }

    // Top Dimension Pill
    const [cx1, cy1, cx2, cy2] = this.cropBox;
    const cropW = Math.round(cx2 - cx1);
    const cropH = Math.round(cy2 - cy1);
    const dimText = `${cropW} × ${cropH} px ${overflows ? '• [Auto Fill Active]' : ''}`;

    ctx.font = '11px sans-serif';
    const badgeW = ctx.measureText(dimText).width + 16;
    ctx.fillStyle = overflows ? 'rgba(245, 158, 11, 0.95)' : 'rgba(99, 102, 241, 0.95)';
    ctx.beginPath();
    ctx.roundRect(x + (w - badgeW) / 2, y + 6, badgeW, 20, 4);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.fillText(dimText, x + w / 2, y + 20);

    ctx.restore();
  }

  drawHandles(x, y, w, h) {
    const ctx = this.ctx;
    const r = this.handleRadius;

    const handles = [
      { x: x, y: y },         // nw
      { x: x + w / 2, y: y }, // n
      { x: x + w, y: y },     // ne
      { x: x + w, y: y + h / 2 }, // e
      { x: x + w, y: y + h }, // se
      { x: x + w / 2, y: y + h }, // s
      { x: x, y: y + h },     // sw
      { x: x, y: y + h / 2 }  // w
    ];

    ctx.save();
    handles.forEach(pos => {
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.strokeStyle = '#6366f1';
      ctx.lineWidth = 2.5;
      ctx.stroke();
    });
    ctx.restore();
  }

  getHandleAt(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const mx = clientX - rect.left;
    const my = clientY - rect.top;

    const [cx1, cy1, cx2, cy2] = this.cropBox;
    const cp1 = this.imgToCanvas(cx1, cy1);
    const cp2 = this.imgToCanvas(cx2, cy2);
    const x = cp1.x;
    const y = cp1.y;
    const w = cp2.x - cp1.x;
    const h = cp2.y - cp1.y;
    const r = this.handleRadius + 4;

    const points = {
      'nw': { x: x, y: y },
      'n':  { x: x + w / 2, y: y },
      'ne': { x: x + w, y: y },
      'e':  { x: x + w, y: y + h / 2 },
      'se': { x: x + w, y: y + h },
      's':  { x: x + w / 2, y: y + h },
      'sw': { x: x, y: y + h },
      'w':  { x: x, y: y + h / 2 }
    };

    for (const [key, pt] of Object.entries(points)) {
      const dist = Math.hypot(mx - pt.x, my - pt.y);
      if (dist <= r) return key;
    }

    // Inside crop box -> move
    if (mx >= x && mx <= x + w && my >= y && my <= y + h) {
      return 'move';
    }

    return null;
  }

  bindEvents() {
    this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
    window.addEventListener('mousemove', (e) => this.onMouseMove(e));
    window.addEventListener('mouseup', (e) => this.onMouseUp(e));
    this.canvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });

    // Touch support
    this.canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        this.onMouseDown(e.touches[0]);
      }
    });
    window.addEventListener('touchmove', (e) => {
      if (this.isDragging && e.touches.length === 1) {
        this.onMouseMove(e.touches[0]);
      }
    });
    window.addEventListener('touchend', (e) => this.onMouseUp(e));
  }

  onMouseDown(e) {
    if (!this.image) return;

    // Middle click or space key pressed -> pan
    if (e.button === 1 || e.spaceKey) {
      this.isDragging = true;
      this.dragMode = 'pan';
      this.dragStart = { x: e.clientX, y: e.clientY };
      return;
    }

    const mode = this.getHandleAt(e.clientX, e.clientY);
    if (mode) {
      this.isDragging = true;
      this.dragMode = mode;
      this.dragStart = { x: e.clientX, y: e.clientY };
      this.cropStart = [...this.cropBox];
    } else {
      // Click outside crop box -> pan view
      this.isDragging = true;
      this.dragMode = 'pan';
      this.dragStart = { x: e.clientX, y: e.clientY };
    }
  }

  onMouseMove(e) {
    if (!this.image) return;

    if (!this.isDragging) {
      // Update cursor based on hover
      const mode = this.getHandleAt(e.clientX, e.clientY);
      const cursors = {
        'nw': 'nwse-resize',
        'se': 'nwse-resize',
        'ne': 'nesw-resize',
        'sw': 'nesw-resize',
        'n':  'ns-resize',
        's':  'ns-resize',
        'e':  'ew-resize',
        'w':  'ew-resize',
        'move': 'move',
        'pan': 'grab'
      };
      this.canvas.style.cursor = cursors[mode] || 'default';
      return;
    }

    const dx = (e.clientX - this.dragStart.x) / this.scale;
    const dy = (e.clientY - this.dragStart.y) / this.scale;

    if (this.dragMode === 'pan') {
      this.offsetX += e.clientX - this.dragStart.x;
      this.offsetY += e.clientY - this.dragStart.y;
      this.dragStart = { x: e.clientX, y: e.clientY };
      this.render();
      return;
    }

    let [x1, y1, x2, y2] = this.cropStart;
    const targetAspect = this.options.aspectRatio;
    const lock = this.options.lockAspectRatio && targetAspect > 0;

    if (this.dragMode === 'move') {
      x1 += dx;
      x2 += dx;
      y1 += dy;
      y2 += dy;
    } else if (this.dragMode === 'se') {
      x2 += dx;
      if (lock) {
        const w = x2 - x1;
        y2 = y1 + w / targetAspect;
      } else {
        y2 += dy;
      }
    } else if (this.dragMode === 'nw') {
      x1 += dx;
      if (lock) {
        const w = x2 - x1;
        y1 = y2 - w / targetAspect;
      } else {
        y1 += dy;
      }
    } else if (this.dragMode === 'ne') {
      x2 += dx;
      if (lock) {
        const w = x2 - x1;
        y1 = y2 - w / targetAspect;
      } else {
        y1 += dy;
      }
    } else if (this.dragMode === 'sw') {
      x1 += dx;
      if (lock) {
        const w = x2 - x1;
        y2 = y1 + w / targetAspect;
      } else {
        y2 += dy;
      }
    } else if (this.dragMode === 'e') {
      x2 += dx;
      if (lock) {
        const w = x2 - x1;
        const h = w / targetAspect;
        const cy = (y1 + y2) / 2;
        y1 = cy - h / 2;
        y2 = cy + h / 2;
      }
    } else if (this.dragMode === 'w') {
      x1 += dx;
      if (lock) {
        const w = x2 - x1;
        const h = w / targetAspect;
        const cy = (y1 + y2) / 2;
        y1 = cy - h / 2;
        y2 = cy + h / 2;
      }
    } else if (this.dragMode === 's') {
      y2 += dy;
      if (lock) {
        const h = y2 - y1;
        const w = h * targetAspect;
        const cx = (x1 + x2) / 2;
        x1 = cx - w / 2;
        x2 = cx + w / 2;
      }
    } else if (this.dragMode === 'n') {
      y1 += dy;
      if (lock) {
        const h = y2 - y1;
        const w = h * targetAspect;
        const cx = (x1 + x2) / 2;
        x1 = cx - w / 2;
        x2 = cx + w / 2;
      }
    }

    // Minimum size guard (at least 40px)
    if (x2 - x1 >= 40 && y2 - y1 >= 40) {
      this.cropBox = [
        Math.round(x1),
        Math.round(y1),
        Math.round(x2),
        Math.round(y2)
      ];
      this.render();

      if (typeof this.options.onCropChange === 'function') {
        this.options.onCropChange(this.cropBox);
      }
    }
  }

  onMouseUp() {
    this.isDragging = false;
    this.dragMode = null;
  }

  onWheel(e) {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    this.zoom(factor, e.clientX, e.clientY);
  }
}
