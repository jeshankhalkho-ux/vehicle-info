/**
 * UltraPlate.js — YOLO-powered license plate detection & OCR
 *
 * A drop-in replacement for Tesseract.js, optimized for vehicle license plates
 * using Ultralytics YOLO object detection models.
 *
 * ─── Features ──────────────────────────────────────────────────────────
 * • YOLO-based license plate detection (TF.js local or Cloud API)
 * • Automatic plate region cropping for higher OCR accuracy
 * • Indian-format plate validation (XX00XX0000)
 * • Image preprocessing pipeline optimized for plates
 * • Camera capture & live scanning support
 *
 * ─── Quick Start ───────────────────────────────────────────────────────
 *   const worker = await UltraPlate.createWorker({
 *     modelPath: '/models/yolo11n_web_model/',   // TF.js model dir
 *     // OR use cloud API:
 *     // apiKey: 'YOUR_ULTRALYTICS_API_KEY',
 *     // modelId: 'YOUR_MODEL_ID',
 *   });
 *   const result = await worker.recognize(imageData);
 *   console.log(result.text); // "MH12DE1433"
 *   worker.terminate();
 *
 * ─── Model Export (Python) ────────────────────────────────────────────
 *   from ultralytics import YOLO
 *   model = YOLO("path/to/license-plate-model.pt")
 *   model.export(format="tfjs")   # creates /model_web_model/
 *
 * ─── CDN Dependencies ─────────────────────────────────────────────────
 *   <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
 *   <script src="https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js"></script>
 *   <script src="lib/ultraplate.js"></script>
 */
(function (global) {
  "use strict";

  // ─── Constants ───────────────────────────────────────────────────────
  const INDIAN_PLATE_RE = /^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$/;
  const PLATE_MIN_CONFIDENCE = 0.25;
  const OCR_MIN_CONFIDENCE = 0.3;
  const MAX_PLATES = 5;

  const STATE = {
    IDLE: 0,
    LOADING: 1,
    READY: 2,
    BUSY: 3,
    ERROR: 4,
  };

  // ─── Utilities ───────────────────────────────────────────────────────
  function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

  function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

  function base64ToCanvas(dataUrl) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement("canvas");
        c.width = img.width;
        c.height = img.height;
        c.getContext("2d").drawImage(img, 0, 0);
        resolve(c);
      };
      img.onerror = reject;
      img.src = dataUrl;
    });
  }

  function canvasToDataURL(canvas, format, quality) {
    return canvas.toDataURL(format || "image/jpeg", quality != null ? quality : 0.92);
  }

  function validateIndianPlate(text) {
    const cleaned = text.replace(/[^A-Z0-9]/gi, "").toUpperCase();
    const match = cleaned.match(INDIAN_PLATE_RE);
    return match ? match[0] : null;
  }

  function levenshtein(a, b) {
    const m = a.length, n = b.length;
    const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;
    for (let i = 1; i <= m; i++)
      for (let j = 1; j <= n; j++)
        dp[i][j] =
          a[i - 1] === b[j - 1]
            ? dp[i - 1][j - 1]
            : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    return dp[m][n];
  }

  function bestPlateMatch(candidates, minSimilarity) {
    minSimilarity = minSimilarity || 0.6;
    let best = null, bestScore = 0;
    for (const c of candidates) {
      const cleaned = c.replace(/[^A-Z0-9]/gi, "").toUpperCase();
      if (cleaned.length < 8 || cleaned.length > 12) continue;
      if (INDIAN_PLATE_RE.test(cleaned)) return { text: cleaned, raw: c, exact: true };
      const sim = cleaned.length >= 8
        ? 1 - levenshtein(cleaned, "MH12AB1234") / 12
        : 0;
      if (sim > bestScore) {
        bestScore = sim;
        best = { text: cleaned, raw: c, exact: false, similarity: sim };
      }
    }
    return best && bestScore >= minSimilarity ? best : null;
  }

  // ─── Image preprocessing pipeline ────────────────────────────────────
  const Preprocess = {
    toGrayscale(canvas) {
      const ctx = canvas.getContext("2d");
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const d = imgData.data;
      for (let i = 0; i < d.length; i += 4) {
        const g = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
        d[i] = d[i + 1] = d[i + 2] = g;
      }
      ctx.putImageData(imgData, 0, 0);
      return canvas;
    },

    adjustContrast(canvas, factor) {
      factor = factor || 1.4;
      const ctx = canvas.getContext("2d");
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const d = imgData.data;
      for (let i = 0; i < d.length; i += 4) {
        d[i] = clamp(Math.round(128 + (d[i] - 128) * factor), 0, 255);
        d[i + 1] = clamp(Math.round(128 + (d[i + 1] - 128) * factor), 0, 255);
        d[i + 2] = clamp(Math.round(128 + (d[i + 2] - 128) * factor), 0, 255);
      }
      ctx.putImageData(imgData, 0, 0);
      return canvas;
    },

    sharpen(canvas) {
      const ctx = canvas.getContext("2d");
      const w = canvas.width, h = canvas.height;
      const src = ctx.getImageData(0, 0, w, h);
      const dst = ctx.createImageData(w, h);
      const kernel = [0, -1, 0, -1, 5, -1, 0, -1, 0];
      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          let r = 0, g = 0, b = 0;
          for (let ky = -1; ky <= 1; ky++) {
            for (let kx = -1; kx <= 1; kx++) {
              const idx = ((y + ky) * w + (x + kx)) * 4;
              const k = kernel[(ky + 1) * 3 + (kx + 1)];
              r += src.data[idx] * k;
              g += src.data[idx + 1] * k;
              b += src.data[idx + 2] * k;
            }
          }
          const idx = (y * w + x) * 4;
          dst.data[idx] = clamp(r, 0, 255);
          dst.data[idx + 1] = clamp(g, 0, 255);
          dst.data[idx + 2] = clamp(b, 0, 255);
          dst.data[idx + 3] = 255;
        }
      }
      ctx.putImageData(dst, 0, 0);
      return canvas;
    },

    enhanceForOCR(canvas) {
      this.toGrayscale(canvas);
      this.adjustContrast(canvas, 1.6);
      this.sharpen(canvas);
      return canvas;
    },

    resize(canvas, maxDim) {
      maxDim = maxDim || 640;
      let { width: w, height: h } = canvas;
      if (w <= maxDim && h <= maxDim) return canvas;
      const scale = maxDim / Math.max(w, h);
      const c = document.createElement("canvas");
      c.width = Math.round(w * scale);
      c.height = Math.round(h * scale);
      c.getContext("2d").drawImage(canvas, 0, 0, c.width, c.height);
      return c;
    },
  };

  // ─── YOLO output parser (for TF.js models) ──────────────────────────
  function parseYOLOOutput(tensor, imgW, imgH, confThresh, iouThresh) {
    confThresh = confThresh || PLATE_MIN_CONFIDENCE;
    iouThresh = iouThresh || 0.5;
    const data = tensor.dataSync ? tensor.dataSync() : tensor;
    const shape = tensor.shape || [1, data.length / 84, 84];
    const numDet = shape[1] || data.length / 84;

    const boxes = [];
    for (let i = 0; i < numDet; i++) {
      const offset = i * 84;
      const cx = data[offset];
      const cy = data[offset + 1];
      const bw = data[offset + 2];
      const bh = data[offset + 3];
      const conf = data[offset + 4];
      if (conf < confThresh) continue;

      let maxClassScore = 0, classId = 0;
      for (let j = 5; j < 84; j++) {
        if (data[offset + j] > maxClassScore) {
          maxClassScore = data[offset + j];
          classId = j - 5;
        }
      }
      const score = conf * maxClassScore;
      if (score < confThresh) continue;

      const x1 = clamp((cx - bw / 2) * imgW, 0, imgW);
      const y1 = clamp((cy - bh / 2) * imgH, 0, imgH);
      const x2 = clamp((cx + bw / 2) * imgW, 0, imgW);
      const y2 = clamp((cy + bh / 2) * imgH, 0, imgH);

      boxes.push({ x1, y1, x2, y2, confidence: score, classId });
    }

    boxes.sort((a, b) => b.confidence - a.confidence);
    const keep = [];
    for (const box of boxes) {
      let overlap = false;
      for (const k of keep) {
        const xi1 = Math.max(box.x1, k.x1);
        const yi1 = Math.max(box.y1, k.y1);
        const xi2 = Math.min(box.x2, k.x2);
        const yi2 = Math.min(box.y2, k.y2);
        const inter = Math.max(0, xi2 - xi1) * Math.max(0, yi2 - yi1);
        const union = (box.x2 - box.x1) * (box.y2 - box.y1)
                    + (k.x2 - k.x1) * (k.y2 - k.y1) - inter;
        if (inter / union > iouThresh) { overlap = true; break; }
      }
      if (!overlap) {
        keep.push(box);
        if (keep.length >= MAX_PLATES) break;
      }
    }
    return keep;
  }

  // ─── OCR Engine ──────────────────────────────────────────────────────
  class OCREngine {
    constructor(config) {
      this.config = config || {};
      this.tesseractWorker = null;
      this.ready = false;
    }

    async init() {
      if (this.ready) return;
      if (typeof Tesseract === "undefined") {
        throw new Error("Tesseract.js not loaded. Include <script src=\"https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js\"></script>");
      }
      this.tesseractWorker = Tesseract.createWorker(
        this.config.lang || "eng",
        this.config.oem || 1,
        { logger: this.config.logger || null }
      );
      await this.tesseractWorker.load();
      this.ready = true;
    }

    async recognize(canvas) {
      if (!this.ready) await this.init();
      const dUrl = canvasToDataURL(canvas, "image/png");
      const { data } = await this.tesseractWorker.recognize(dUrl);
      return {
        text: (data.text || "").trim(),
        confidence: data.confidence ? data.confidence / 100 : 0,
        words: data.words || [],
      };
    }

    async terminate() {
      if (this.tesseractWorker) {
        await this.tesseractWorker.terminate();
        this.tesseractWorker = null;
      }
      this.ready = false;
    }
  }

  // ─── YOLO Inference Engine (TF.js) ───────────────────────────────────
  class TFJSEngine {
    constructor(config) {
      this.modelPath = config.modelPath;
      this.imgsz = config.imgsz || 640;
      this.confidence = config.confidence || PLATE_MIN_CONFIDENCE;
      this.iou = config.iou || 0.5;
      this.model = null;
      this.ready = false;
    }

    async init() {
      if (typeof tf === "undefined") {
        throw new Error("TensorFlow.js not loaded. Include <script src=\"https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js\"></script>");
      }
      this.model = await tf.loadGraphModel(this.modelPath + "model.json");
      this.ready = true;
    }

    async detect(canvas) {
      if (!this.ready) await this.init();
      const resized = Preprocess.resize(canvas, this.imgsz);
      const w = resized.width, h = resized.height;

      let input = tf.browser.fromPixels(resized);
      input = tf.image.resizeBilinear(input, [this.imgsz, this.imgsz]);
      input = input.expandDims(0).div(255.0);

      const output = await this.model.executeAsync(input);
      const boxes = parseYOLOOutput(
        output,
        w, h,
        this.confidence,
        this.iou
      );

      tf.dispose([input, output]);
      return boxes;
    }
  }

  // ─── YOLO Inference Engine (Ultralytics Cloud API) ───────────────────
  class CloudAPIEngine {
    constructor(config) {
      this.apiKey = config.apiKey;
      this.modelId = config.modelId;
      this.confidence = config.confidence || PLATE_MIN_CONFIDENCE;
      this.iou = config.iou || 0.5;
      this.imgsz = config.imgsz || 640;
      this.baseUrl = "https://platform.ultralytics.com/api/models";
      this.ready = true;
    }

    async init() {
      if (!this.apiKey) {
        throw new Error("Ultralytics API key required for cloud API mode.");
      }
    }

    async detect(canvas) {
      const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.92)
      );
      const formData = new FormData();
      formData.append("file", blob, "plate.jpg");
      formData.append("conf", String(this.confidence));
      formData.append("iou", String(this.iou));
      formData.append("imgsz", String(this.imgsz));

      const res = await fetch(
        `${this.baseUrl}/${this.modelId}/predict`,
        {
          method: "POST",
          headers: { Authorization: `Bearer ${this.apiKey}` },
          body: formData,
        }
      );
      if (!res.ok) {
        const err = await res.text();
        throw new Error(`Ultralytics API error (${res.status}): ${err}`);
      }
      const json = await res.json();
      const w = canvas.width, h = canvas.height;
      const boxes = [];
      const imgs = json.images || [];
      for (const img of imgs) {
        const results = img.results || [];
        for (const r of results) {
          const bx = r.box;
          if (!bx) continue;
          boxes.push({
            x1: bx.x1,
            y1: bx.y1,
            x2: bx.x2,
            y2: bx.y2,
            confidence: r.confidence || 0,
            classId: r.class != null ? r.class : 0,
            name: r.name || "",
          });
        }
      }
      return boxes;
    }
  }

  // ─── Main UltraPlate Worker ──────────────────────────────────────────
  class UltraPlateWorker {
    constructor(config) {
      this.config = config || {};
      this.state = STATE.IDLE;
      this.detector = null;
      this.ocr = null;
      this.useTFJS = config.modelPath && !config.apiKey;
      this.useCloudAPI = !!(config.apiKey && config.modelId);
      this.useDetector = this.useTFJS || this.useCloudAPI;
    }

    async init() {
      this.state = STATE.LOADING;

      this.ocr = new OCREngine({
        lang: this.config.lang || "eng",
        oem: this.config.oem || 1,
        logger: this.config.logger || null,
      });
      await this.ocr.init();

      if (this.useTFJS) {
        this.detector = new TFJSEngine({
          modelPath: this.config.modelPath,
          imgsz: this.config.imgsz || 640,
          confidence: this.config.confidence || PLATE_MIN_CONFIDENCE,
          iou: this.config.iou || 0.5,
        });
        await this.detector.init();
      } else if (this.useCloudAPI) {
        this.detector = new CloudAPIEngine({
          apiKey: this.config.apiKey,
          modelId: this.config.modelId,
          imgsz: this.config.imgsz || 640,
          confidence: this.config.confidence || PLATE_MIN_CONFIDENCE,
          iou: this.config.iou || 0.5,
        });
        await this.detector.init();
      }

      this.state = STATE.READY;
    }

    async recognize(input) {
      if (this.state === STATE.IDLE || this.state === STATE.ERROR) {
        await this.init();
      }
      this.state = STATE.BUSY;

      try {
        const canvas =
          typeof input === "string"
            ? await base64ToCanvas(input)
            : input instanceof HTMLCanvasElement
              ? input
              : input instanceof HTMLVideoElement
                ? this._videoToCanvas(input)
                : input instanceof HTMLImageElement
                  ? this._imgToCanvas(input)
                  : null;

        if (!canvas) {
          throw new Error("Unsupported input type. Accepts: dataURL, Canvas, Video, or Image element.");
        }

        let plates = [];

        if (this.useDetector && this.detector) {
          const detections = await this.detector.detect(canvas);
          for (const d of detections) {
            const crop = document.createElement("canvas");
            const pad = 8;
            const cx1 = Math.max(0, d.x1 - pad);
            const cy1 = Math.max(0, d.y1 - pad);
            const cx2 = Math.min(canvas.width, d.x2 + pad);
            const cy2 = Math.min(canvas.height, d.y2 + pad);
            crop.width = cx2 - cx1;
            crop.height = cy2 - cy1;
            crop.getContext("2d").drawImage(
              canvas, cx1, cy1, crop.width, crop.height, 0, 0, crop.width, crop.height
            );
            Preprocess.enhanceForOCR(crop);

            const ocrResult = await this.ocr.recognize(crop);
            const cleaned = ocrResult.text.replace(/[^A-Z0-9]/gi, "").toUpperCase();
            plates.push({
              text: cleaned,
              raw: ocrResult.text,
              confidence: d.confidence,
              ocrConfidence: ocrResult.confidence,
              bbox: { x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2 },
            });
          }
        }

        if (plates.length === 0) {
          const fullOcr = await this.ocr.recognize(Preprocess.enhanceForOCR(canvas));
          const candidates = fullOcr.text.split(/\n/).filter(Boolean);
          const match = bestPlateMatch(candidates);
          if (match) {
            plates.push({
              text: match.text,
              raw: match.raw,
              confidence: match.exact ? 0.95 : match.similarity,
              ocrConfidence: fullOcr.confidence,
              bbox: null,
            });
          } else {
            const cleaned = fullOcr.text.replace(/[^A-Z0-9]/gi, "").toUpperCase();
            plates.push({
              text: cleaned,
              raw: fullOcr.text,
              confidence: fullOcr.confidence,
              ocrConfidence: fullOcr.confidence,
              bbox: null,
            });
          }
        }

        const validated = plates
          .filter((p) => p.text.length >= 4)
          .sort((a, b) => {
            const aValid = INDIAN_PLATE_RE.test(a.text) ? 1 : 0;
            const bValid = INDIAN_PLATE_RE.test(b.text) ? 1 : 0;
            if (aValid !== bValid) return bValid - aValid;
            return b.confidence - a.confidence;
          });

        const best = validated[0] || plates[0];

        this.state = STATE.READY;
        return {
          text: best ? best.text : "",
          raw: best ? best.raw : "",
          confidence: best ? best.confidence : 0,
          ocrConfidence: best ? best.ocrConfidence : 0,
          isIndianPlate: best ? INDIAN_PLATE_RE.test(best.text) : false,
          plate: best ? validateIndianPlate(best.text) : null,
          bbox: best ? best.bbox : null,
          candidates: validated,
          allDetections: plates,
        };
      } catch (err) {
        this.state = STATE.ERROR;
        throw err;
      }
    }

    _videoToCanvas(video) {
      const c = document.createElement("canvas");
      c.width = video.videoWidth;
      c.height = video.videoHeight;
      c.getContext("2d").drawImage(video, 0, 0);
      return c;
    }

    _imgToCanvas(img) {
      const c = document.createElement("canvas");
      c.width = img.naturalWidth || img.width;
      c.height = img.naturalHeight || img.height;
      c.getContext("2d").drawImage(img, 0, 0);
      return c;
    }

    async terminate() {
      if (this.ocr) await this.ocr.terminate();
      if (this.detector && this.detector.model) {
        if (typeof tf !== "undefined" && tf.dispose) {
          tf.dispose(this.detector.model);
        }
        this.detector.model = null;
      }
      this.detector = null;
      this.ocr = null;
      this.state = STATE.IDLE;
    }

    getState() { return this.state; }
    isReady() { return this.state === STATE.READY; }
  }

  // ─── Public API ──────────────────────────────────────────────────────
  const UltraPlate = {
    version: "1.0.0",
    STATE,

    async createWorker(config) {
      config = config || {};
      const worker = new UltraPlateWorker(config);
      await worker.init();
      return worker;
    },

    validatePlate: validateIndianPlate,
    bestPlateMatch,
    INDIAN_PLATE_RE,

    Preprocess,
  };

  global.UltraPlate = UltraPlate;
})(typeof window !== "undefined" ? window : this);
