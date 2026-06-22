/**
 * extension.js — LiteGraph frontend for InteractiveSegmentationMask
 * ==================================================================
 * Registered with ComfyUI via WEB_DIRECTORY in __init__.py.
 *
 * Architecture overview
 * ─────────────────────
 * After each successful queue execution the node:
 *   1. Fetches overlay + id-map PNGs from the Python HTTP route.
 *   2. Renders the overlay image inside the node body.
 *   3. Loads the id-map into an off-screen <canvas> ("the ID canvas").
 *   4. On mousemove — reads one pixel from the ID canvas → resolves
 *      segment ID in O(1) → draws a translucent highlight.
 *   5. On click — toggles the segment's selection state, stores a
 *      representative coordinate, serialises to the hidden widget.
 *
 * Key design decisions
 * ─────────────────────
 * • Off-screen ID canvas: zero JS-side segmentation logic; instant hover.
 * • All interactive state lives on the node object itself, not in globals,
 *   so multiple instances are fully independent.
 * • Every DOM/canvas listener is tracked and torn down in onRemoved() to
 *   prevent memory leaks.
 * • The node resizes itself to fit the image while respecting a max size.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Target width of the in-node image preview (height is computed from aspect). */
const PREVIEW_WIDTH = 400;

/** Hard upper bound on preview height to keep the node manageable. */
const PREVIEW_MAX_HEIGHT = 400;

/** Pixels of top padding before the image starts inside the node body. */
const HEADER_HEIGHT = 30;

/** Pixels of padding below the image preview. */
const FOOTER_PADDING = 10;

// ─────────────────────────────────────────────────────────────────────────────
// Colour constants as RGBA components (used by rendering functions)
// ─────────────────────────────────────────────────────────────────────────────

/** Hover colour: semi-transparent white. */
const HOVER_COLOR = { r: 255, g: 255, b: 255, a: 107 }; // 0.42 * 255 ≈ 107

/** Selection fill colour: bright lime green. */
const SELECTED_COLOR = { r: 50, g: 255, b: 0, a: 89 };  // 0.35 * 255 ≈ 89

/** Selection border colour: bright flashy lime green. */
const SELECTED_BORDER_COLOR = { r: 50, g: 255, b: 0, a: 242 }; // 0.95 * 255 ≈ 242

/** Hover highlight: semi-transparent white fill over the hovered segment. */
const HOVER_FILL = `rgba(${HOVER_COLOR.r}, ${HOVER_COLOR.g}, ${HOVER_COLOR.b}, ${HOVER_COLOR.a})`;

/** Selection fill: bright lime green highlight for selected segments. */
const SELECTED_FILL = `rgba(${SELECTED_COLOR.r}, ${SELECTED_COLOR.g}, ${SELECTED_COLOR.b}, ${SELECTED_COLOR.a})`;

/** Selection border drawn over selected segments — bright flashy green. */
const SELECTED_STROKE = `rgba(${SELECTED_BORDER_COLOR.r}, ${SELECTED_BORDER_COLOR.g}, ${SELECTED_BORDER_COLOR.b}, ${SELECTED_BORDER_COLOR.a})`;

/** Stroke width in canvas pixels for selection borders — thicker for visibility. */
const SELECTION_STROKE_WIDTH = 2;

/** Throttle interval for mousemove pixel reads (ms). Set to 0 for max fps. */
const MOUSEMOVE_THROTTLE_MS = 0; // ~60 fps

// ─────────────────────────────────────────────────────────────────────────────
// Colour parsing helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Encode three uint8 channel values into a single 24-bit integer key.
 * Used as the segment-colour → ID lookup map key.
 *
 * @param {number} r 0-255
 * @param {number} g 0-255
 * @param {number} b 0-255
 * @returns {number} 24-bit integer
 */
function rgbToKey(r, g, b) {
  return (r << 16) | (g << 8) | b;
}

// ─────────────────────────────────────────────────────────────────────────────
// Node extension registration
// ─────────────────────────────────────────────────────────────────────────────

app.registerExtension({
  name: "InteractiveSegmentationMask",

  /**
   * Called by ComfyUI for every node whose type matches our mapping.
   * We augment the node prototype with interactive canvas logic here.
   *
   * @param {Function} nodeType   The LiteGraph node class.
   * @param {object}   nodeData   Node metadata from the Python backend.
   * @param {object}   _app       The ComfyUI app instance (unused, we use the import).
   */
  async beforeRegisterNodeDef(nodeType, nodeData, _app) {
    if (nodeData.name !== "InteractiveSegmentationMask") return;

    // ── onNodeCreated ──────────────────────────────────────────────────────
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      _initNodeState(this);

      // ── Visibilité conditionnelle des paramètres ───────────────────────────
      const SLIC_WIDGETS = ["slic_n_segments", "slic_compactness", "slic_sigma"];
      const SAM_WIDGETS  = ["sam_points_per_side", "sam_pred_iou_thresh", "sam_stability_score_thresh"];

      const updateParamVisibility = (engine) => {
        for (const w of this.widgets ?? []) {
          if (SLIC_WIDGETS.includes(w.name)) {
            w.hidden = engine !== "SLIC";
            w.computeSize = engine === "SLIC" ? undefined : () => [0, -4];
          }
          if (SAM_WIDGETS.includes(w.name)) {
            w.hidden = engine !== "SAM";
            w.computeSize = engine === "SAM" ? undefined : () => [0, -4];
          }
        }
        // Recalculer la taille du nœud
        const size = this.computeSize();
        this.setSize(size);
        this.setDirtyCanvas(true, true);
      };

      // Appliquer au chargement
      const engineWidget = this.widgets?.find(w => w.name === "segmentation_engine");
      if (engineWidget) {
        updateParamVisibility(engineWidget.value);

        // Écouter les changements
        const originalCallback = engineWidget.callback;
        engineWidget.callback = (value) => {
          originalCallback?.call(engineWidget, value);
          updateParamVisibility(value);
        };
      }
    };


    // ── onRemoved ──────────────────────────────────────────────────────────
    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      onRemoved?.apply(this, arguments);
      _teardownNode(this);
    };

    // ── onDrawBackground ───────────────────────────────────────────────────
    // Called by LiteGraph every frame to paint the node body.
    const onDrawBackground = nodeType.prototype.onDrawBackground;
    nodeType.prototype.onDrawBackground = function (ctx) {
      onDrawBackground?.apply(this, arguments);
      _drawNodeBody(this, ctx);
    };

    // ── onMouseMove ────────────────────────────────────────────────────────
    // LiteGraph passes canvas-relative coords; we convert to image-local.
    const onMouseMove = nodeType.prototype.onMouseMove;
    nodeType.prototype.onMouseMove = function (event, localPos) {
      const handled = _handleMouseMove(this, localPos);
      if (!handled) onMouseMove?.apply(this, arguments);
      return handled;
    };

    // ── onMouseDown ────────────────────────────────────────────────────────
    const onMouseDown = nodeType.prototype.onMouseDown;
    nodeType.prototype.onMouseDown = function (event, localPos) {
      const handled = _handleMouseDown(this, localPos);
      if (!handled) onMouseDown?.apply(this, arguments);
      return handled;
    };

    // ── onMouseLeave ───────────────────────────────────────────────────────
    const onMouseLeave = nodeType.prototype.onMouseLeave;
    nodeType.prototype.onMouseLeave = function () {
      if (this.__seg) this.__seg.hoveredSegmentId = null;
      app.graph.setDirtyCanvas(true, false);
      onMouseLeave?.apply(this, arguments);
    };
  },

  // ── Hook into the execution lifecycle ─────────────────────────────────────
  // After the backend executes we fetch fresh segment data.
  async nodeCreated(node) {
    // Intentionally empty — setup happens in onNodeCreated above.
  },
});

// ─────────────────────────────────────────────────────────────────────────────
// WebSocket listener: receive segment data from backend
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Listen for custom WebSocket events from the Python backend containing
 * overlay + id-map data. This replaces the previous HTTP polling approach,
 * enabling real-time push updates via ComfyUI's WebSocket server.
 */
api.addEventListener("interactive_segmask", async (event) => {
  const data = event.detail ?? {};
  const nodeId = String(data.node_id ?? "");
  if (!nodeId) return;

  // Find the matching LiteGraph node on the canvas.
  const node = app.graph._nodes_by_id?.[nodeId];
  if (!node || node.type !== "InteractiveSegmentationMask") return;

  await _applySegmentData(node, nodeId, data);
});

api.addEventListener("sam_model_missing", async ({ detail }) => {
  const { choices, checkpoint_dir } = detail ?? {};
  if (!choices?.length) return;

  // ── 1. Déclencher le confirm natif ComfyUI ─────────────────────────────
  // On l'utilise comme "coquille" : son overlay, ses boutons OK/Annuler,
  // sa gestion du focus. On injecte nos radios dans son corps ensuite.
  const confirmPromise = app.extensionManager.dialog.confirm({
    title: "Télécharger un modèle SAM",
    message: `Aucun modèle SAM trouvé dans :\n${checkpoint_dir}`,
    hint: "Le téléchargement peut prendre plusieurs minutes.",
    type: "default",
  });

  // ── 2. Injecter les radio buttons dans le dialog qui vient d'apparaître
  // On attend le prochain tick pour que le DOM soit rendu
  await new Promise(r => setTimeout(r, 0));

  let selectedFilename = choices[0].filename;

  // Trouver le conteneur du message dans la modale PrimeVue
  const dialogEl = document.querySelector(".p-dialog-content");
  if (dialogEl) {
    // Créer le groupe de radios
    const radioGroup = document.createElement("div");
    radioGroup.style.cssText = `
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-top: 16px;
    `;

    choices.forEach((choice, i) => {
      const label = document.createElement("label");
      label.style.cssText = `
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        border-radius: 6px;
        border: 1px solid var(--p-content-border-color, #444);
        cursor: pointer;
        transition: background 0.15s, border-color 0.15s;
        background: ${i === 0 ? "var(--p-content-hover-background, #2a2a2a)" : "transparent"};
        border-color: ${i === 0 ? "var(--p-primary-color, #4CAF50)" : "var(--p-content-border-color, #444)"};
      `;

      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "sam_model_choice";
      radio.value = choice.filename;
      radio.checked = i === 0;
      radio.style.accentColor = "var(--p-primary-color, #4CAF50)";

      const textWrapper = document.createElement("div");
      textWrapper.style.cssText = `display: flex; flex-direction: column; gap: 2px;`;

      const name = document.createElement("span");
      name.textContent = choice.filename;
      name.style.cssText = `font-size: 13px; font-weight: 600; color: var(--p-text-color, #eee);`;

      const size = document.createElement("span");
      size.textContent = choice.label.match(/\(.*\)/)?.[0] ?? "";
      size.style.cssText = `font-size: 11px; color: var(--p-text-muted-color, #888);`;

      textWrapper.append(name, size);
      label.append(radio, textWrapper);

      // Mise à jour de la sélection + style au clic
      label.addEventListener("click", () => {
        selectedFilename = choice.filename;
        // Reset tous les labels
        radioGroup.querySelectorAll("label").forEach(l => {
          l.style.background = "transparent";
          l.style.borderColor = "var(--p-content-border-color, #444)";
        });
        label.style.background = "var(--p-content-hover-background, #2a2a2a)";
        label.style.borderColor = "var(--p-primary-color, #4CAF50)";
        radio.checked = true;
      });

      radioGroup.appendChild(label);
    });

    dialogEl.appendChild(radioGroup);
  }

  // ── 3. Attendre la réponse de l'utilisateur ────────────────────────────
  const confirmed = await confirmPromise;

  const body = confirmed
    ? { filename: selectedFilename }
    : { cancelled: true };

  fetch("/artishow/sam_download_choice", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).catch(err => console.error("[InteractiveSeg] SAM choice POST failed:", err));
});

// Legacy: Also listen for the "executed" event as a fallback trigger
// (in case WebSocket delivery is delayed). But we don't fetch data anymore;
// we just notify that execution happened.
api.addEventListener("executed", async (event) => {
  // Event fired after execution, but actual data will come via WebSocket.
  // This is kept for potential future use or debugging.
});

// ─────────────────────────────────────────────────────────────────────────────
// Data handling
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Apply segment data received from the backend (via WebSocket) to the node.
 * This is called by the WebSocket listener with data that was pushed from Python.
 *
 * @param {object} node   LiteGraph node.
 * @param {string} nodeId String node UID.
 * @param {object} data   WebSocket payload from backend containing:
 *                        - overlay_b64: base64 PNG
 *                        - id_map_b64: base64 PNG
 *                        - num_segments: number
 *                        - width: number
 *                        - height: number
 */
async function _applySegmentData(node, nodeId, data) {
  const s = node.__seg;
  if (!s || s._fetching) return;
  s._fetching = true;

  try {
    // ── Detect image change ────────────────────────────────────────────
    // Use a cheap hash of width+height+num_segments as a proxy for the image
    // identity.  The Python backend uses a proper tensor hash, but we don't
    // have access to that here.
    const newHash = `${data.width}x${data.height}x${data.num_segments}`;
    if (newHash !== s.lastImageHash) {
      // Image changed → reset all selections and notify the user.
      console.info("[InteractiveSeg] Image changed — resetting selections.");
      s.selectedSegments.clear();
      s.lastImageHash = newHash;
      _writeCoordWidget(node);
    }

    // ── Load overlay image ─────────────────────────────────────────────
    s.originalW = data.width;
    s.originalH = data.height;
    await _loadOverlayImage(node, data.overlay_b64);

    // ── Load ID map ────────────────────────────────────────────────────
    await _loadIdMap(node, data.id_map_b64);

    // Trigger a canvas redraw.
    app.graph.setDirtyCanvas(true, false);
  } catch (err) {
    console.error("[InteractiveSeg] Failed to apply segment data:", err);
  } finally {
    s._fetching = false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// State initialisation & teardown
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Attach the `__seg` state bag to a freshly created node.
 * All interactive state lives here — never on global variables.
 *
 * @param {object} node LiteGraph node instance.
 */
function _initNodeState(node) {
  node.__seg = {
    // Overlay image (with boundary lines drawn) displayed in the node.
    overlayImage: null,        // HTMLImageElement | null

    // Off-screen canvas loaded with the colour-coded ID map.
    idCanvas: null,            // HTMLCanvasElement | null
    idCtx: null,               // CanvasRenderingContext2D | null

    // Lookup table: colour key (int24) → segment ID.
    // Populated when the ID map is loaded.
    colourToSegId: new Map(),  // Map<number, number>

    // Cached ImageData for the full ID canvas (updated when idCanvas changes).
    idImageData: null,         // ImageData | null

    // Current dimensions of the preview inside the node, in canvas pixels.
    previewX: 0,
    previewY: 0,

    originalW: 0,
    originalH: 0,

    previewW: PREVIEW_WIDTH,
    previewH: 200,

    // Hover state.
    hoveredSegmentId: null,    // number | null

    // Selection state.
    // Map: segmentId → {x, y} representative pixel in IMAGE coordinates.
    selectedSegments: new Map(), // Map<number, {x:number, y:number}>

    // Hash of the last image seen; used to auto-reset selections.
    lastImageHash: null,       // string | null

    // Throttle timer id for mousemove.
    _moveTimer: null,

    // Whether we are currently fetching segment data (prevents race).
    _fetching: false,
  };

  // Find the hidden widget and store a direct reference for fast writes.
  node.__seg.coordsWidget = node.widgets?.find(
    (w) => w.name === "selected_coords"
  ) ?? null;

  // AJOUT : Cacher visuellement le widget pour ne pas casser le design du nœud
  if (node.__seg.coordsWidget) {
    node.__seg.coordsWidget.type = "hidden";
    // Force la hauteur à 0 pour ne pas laisser un espace vide
    node.__seg.coordsWidget.computeSize = () => [0, 0]; 
  } else {
    console.warn("[InteractiveSeg] Le widget 'selected_coords' est introuvable. Vérifiez le Python.");
  }

  // Set a reasonable initial size so the node is visible before data loads.
  _resizeNode(node, PREVIEW_WIDTH, 200);
}

/**
 * Clean up all resources when the node is removed from the graph.
 *
 * @param {object} node LiteGraph node instance.
 */
function _teardownNode(node) {
  const s = node.__seg;
  if (!s) return;
  if (s._moveTimer !== null) {
    clearTimeout(s._moveTimer);
    s._moveTimer = null;
  }
  // Release large objects for GC.
  s.overlayImage = null;
  s.idCanvas = null;
  s.idCtx = null;
  s.idImageData = null;
  s.colourToSegId = null;
  s.selectedSegments = null;
  node.__seg = null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Image decoding
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Decode a base-64 PNG string into an HTMLImageElement and store it.
 *
 * @param {object} node
 * @param {string} b64  Raw base-64 (no data-URI prefix).
 * @returns {Promise<void>}
 */
function _loadOverlayImage(node, b64) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      node.__seg.overlayImage = img;

      // Compute scaled preview dimensions.
      const scale = Math.min(
        PREVIEW_WIDTH / img.naturalWidth,
        PREVIEW_MAX_HEIGHT / img.naturalHeight,
        1 // never upscale
      );
      node.__seg.previewW = Math.round(img.naturalWidth * scale);
      node.__seg.previewH = Math.round(img.naturalHeight * scale);

      _resizeNode(node, node.__seg.previewW, node.__seg.previewH);
      resolve();
    };
    img.onerror = () => {
      console.error("[InteractiveSeg] Failed to decode overlay image.");
      resolve();
    };
    img.src = `data:image/png;base64,${b64}`;
  });
}

/**
 * Decode a base-64 PNG ID-map into an off-screen canvas and build the
 * colour→segmentId lookup table from its raw pixel data.
 *
 * @param {object} node
 * @param {string} b64  Raw base-64 (no data-URI prefix).
 * @returns {Promise<void>}
 */
function _loadIdMap(node, b64) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const s = node.__seg;
      const W = img.naturalWidth;
      const H = img.naturalHeight;

      // Create or reuse the off-screen canvas.
      if (!s.idCanvas) {
        s.idCanvas = document.createElement("canvas");
      }
      s.idCanvas.width = W;
      s.idCanvas.height = H;
      s.idCtx = s.idCanvas.getContext("2d", { willReadFrequently: true });
      s.idCtx.drawImage(img, 0, 0);

      // Read ALL pixels once and cache.  This avoids per-frame getImageData.
      s.idImageData = s.idCtx.getImageData(0, 0, W, H);

      // ── Build colour → segmentId lookup ─────────────────────────────
      // We walk every pixel and record unique (r,g,b) → segId entries.
      // Segment ID is derived from the colour using the same deterministic
      // function as Python's _deterministic_colour(), but inverted here via
      // exhaustive scan (we can't easily invert the HSV transform in JS).
      //
      // Strategy: the Python backend guarantees that segment 0 is black
      // (0,0,0) and every other segment has a unique colour produced by
      // _deterministic_colour(id).  We trust the ID map — a unique colour
      // unambiguously identifies a segment.  We build the map dynamically.
      //
      const data32 = new Uint32Array(s.idImageData.data.buffer);
      const colourMap = new Map(); // key → segId
      let nextId = 0;

      for (let i = 0; i < data32.length; i++) {
        const pixel = data32[i];
        // Extract R, G, B (little-endian ABGR in typed array)
        const r = (pixel) & 0xff;
        const g = (pixel >> 8) & 0xff;
        const b = (pixel >> 16) & 0xff;
        const key = rgbToKey(r, g, b);
        if (!colourMap.has(key)) {
          colourMap.set(key, nextId++);
        }
      }

      s.colourToSegId = colourMap;
      console.info(
        `[InteractiveSeg] ID map loaded: ${W}×${H}, ${colourMap.size} unique segments.`
      );
      resolve();
    };
    img.onerror = () => {
      console.error("[InteractiveSeg] Failed to decode ID map.");
      resolve();
    };
    img.src = `data:image/png;base64,${b64}`;
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Node body drawing
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Render the node body: image preview, hover highlight, selection tints.
 * Called by LiteGraph's draw loop via `onDrawBackground`.
 *
 * @param {object} node LiteGraph node.
 * @param {CanvasRenderingContext2D} ctx  The MAIN (screen) canvas context.
 */
function _drawNodeBody(node, ctx) {
  const s = node.__seg;
  if (!s) return;

  const { previewX, previewY, previewW, previewH, overlayImage } = s;

  // ── Draw placeholder if no image yet ───────────────────────────────────
  if (!overlayImage) {
    ctx.save();
    ctx.fillStyle = "rgba(40, 40, 40, 0.8)";
    ctx.fillRect(previewX, previewY, previewW, previewH);
    ctx.fillStyle = "#666";
    ctx.font = "14px monospace";
    ctx.textAlign = "center";
    ctx.fillText(
      "Run the node to load segmentation …",
      previewX + previewW / 2,
      previewY + previewH / 2
    );
    ctx.restore();
    return;
  }

  // ── Draw overlay image ─────────────────────────────────────────────────
  ctx.save();
  ctx.drawImage(overlayImage, previewX, previewY, previewW, previewH);

  // ── Draw selection highlights ──────────────────────────────────────────
  // We blit selected-segment masks using the ID canvas as a stencil.
  if (s.idImageData && s.selectedSegments.size > 0) {
    _drawSegmentHighlights(ctx, s, "selected");
  }

  // ── Draw hover highlight ───────────────────────────────────────────────
  if (s.idImageData && s.hoveredSegmentId !== null) {
    _drawSegmentHighlights(ctx, s, "hover");
  }

  ctx.restore();
}

/**
 * Draw per-pixel tint over segments using an auxiliary ImageData buffer.
 *
 * Rather than tracing polygons (which would require JS-side contour
 * computation), we create a temporary ImageData the same size as the
 * preview region, mark the relevant pixels, and blit it with globalAlpha.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} s   node.__seg state bag.
 * @param {"selected"|"hover"} mode
 */
function _drawSegmentHighlights(ctx, s, mode) {
  const { idImageData, previewX, previewY, previewW, previewH,
          hoveredSegmentId, selectedSegments, idCanvas } = s;

  if (!idImageData) return;

  const srcW = idCanvas.width;
  const srcH = idCanvas.height;

  // Determine which segment IDs to highlight.
  /** @type {Set<number>} */
  const targetIds = new Set();
  if (mode === "hover" && hoveredSegmentId !== null) {
    targetIds.add(hoveredSegmentId);
  } else if (mode === "selected") {
    for (const id of selectedSegments.keys()) targetIds.add(id);
  }
  if (targetIds.size === 0) return;

  // Create a temporary ImageData sized to the ID map (source resolution).
  const tint = new ImageData(srcW, srcH);
  const src = idImageData.data;   // Uint8ClampedArray [R,G,B,A, R,G,B,A, …]
  const dst = tint.data;

  // Determine fill colour using the defined constants.
  let fillColor;
  if (mode === "hover") {
    fillColor = HOVER_COLOR;
  } else {
    fillColor = SELECTED_COLOR;
  }

  for (let i = 0; i < src.length; i += 4) {
    const key = rgbToKey(src[i], src[i + 1], src[i + 2]);
    const segId = s.colourToSegId.get(key);
    if (segId !== undefined && targetIds.has(segId)) {
      dst[i]     = fillColor.r;
      dst[i + 1] = fillColor.g;
      dst[i + 2] = fillColor.b;
      dst[i + 3] = fillColor.a;
    }
  }

  // Blit the tint ImageData onto an auxiliary canvas so we can drawImage it
  // at the scaled preview size.
  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = srcW;
  tmpCanvas.height = srcH;
  tmpCanvas.getContext("2d").putImageData(tint, 0, 0);

  ctx.drawImage(tmpCanvas, previewX, previewY, previewW, previewH);

  // For selected segments draw an additional border stroke.
  if (mode === "selected") {
    _drawSelectionBorders(ctx, s);
  }
}

/**
 * Draw coloured border lines around selected segments by scanning the ID map
 * for boundary pixels (pixels adjacent to a pixel of a different segment ID).
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} s  node.__seg state bag.
 */

function _drawSelectionBorders(ctx, s) {
  const { idImageData, idCanvas, selectedSegments, colourToSegId,
          previewX, previewY, previewW, previewH } = s;
  if (!idImageData) return;

  const srcW = idCanvas.width;
  const srcH = idCanvas.height;
  const data = idImageData.data;

  const scaleX = previewW / srcW;
  const scaleY = previewH / srcH;
  const selectedIds = new Set(selectedSegments.keys());

  function segAtPixel(x, y) {
    if (x < 0 || y < 0 || x >= srcW || y >= srcH) return -1;
    const i = (y * srcW + x) * 4;
    const key = rgbToKey(data[i], data[i + 1], data[i + 2]);
    return colourToSegId.get(key) ?? -1;
  }

  const borderPixels = [];
  for (let y = 0; y < srcH; y++) {
    for (let x = 0; x < srcW; x++) {
      const id = segAtPixel(x, y);
      if (!selectedIds.has(id)) continue;
      
      if (
        segAtPixel(x - 1, y) !== id ||
        segAtPixel(x + 1, y) !== id ||
        segAtPixel(x, y - 1) !== id ||
        segAtPixel(x, y + 1) !== id
      ) {
        borderPixels.push({ x, y });
      }
    }
  }

  ctx.save();
  ctx.fillStyle = SELECTED_STROKE;
  const halfStroke = SELECTION_STROKE_WIDTH / 2;
  
  for (const { x, y } of borderPixels) {
    // Rend simplement un rectangle propre proportionnel à l'épaisseur demandée
    ctx.fillRect(
      previewX + (x * scaleX) - halfStroke,
      previewY + (y * scaleY) - halfStroke,
      SELECTION_STROKE_WIDTH,
      SELECTION_STROKE_WIDTH
    );
  }
  ctx.restore();
}

// ─────────────────────────────────────────────────────────────────────────────
// Interaction handlers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Handle LiteGraph mousemove events.
 *
 * @param {object} node
 * @param {[number, number]} localPos  [x, y] in node-local coordinates.
 * @returns {boolean} true if the event was consumed (cursor is over preview).
 */
function _handleMouseMove(node, localPos) {
  const s = node.__seg;
  if (!s || !s.idImageData) return false;

  const [lx, ly] = localPos;
  if (!_isInsidePreview(s, lx, ly)) {
    if (s.hoveredSegmentId !== null) {
      s.hoveredSegmentId = null;
      app.graph.setDirtyCanvas(true, false);
    }
    return false;
  }

  // Throttle pixel reads to avoid saturating the browser.
  if (s._moveTimer !== null) return true;
  s._moveTimer = setTimeout(() => {
    s._moveTimer = null;
    const segId = _segmentIdAtLocalPos(s, lx, ly);
    if (segId !== s.hoveredSegmentId) {
      s.hoveredSegmentId = segId;
      app.graph.setDirtyCanvas(true, false);
    }
  }, MOUSEMOVE_THROTTLE_MS);

  return true;
}

/**
 * Handle LiteGraph mousedown events (left-click only).
 *
 * @param {object} node
 * @param {[number, number]} localPos
 * @returns {boolean}
 */
function _handleMouseDown(node, localPos) {
  const s = node.__seg;
  if (!s || !s.idImageData) return false;

  const [lx, ly] = localPos;
  if (!_isInsidePreview(s, lx, ly)) return false;

  const segId = _segmentIdAtLocalPos(s, lx, ly);
  if (segId === null) return true; // click in black / no-segment area

  // Toggle selection.
  if (s.selectedSegments.has(segId)) {
    s.selectedSegments.delete(segId);
    console.debug(`[InteractiveSeg] Deselected segment ${segId}.`);
  } else {
    // Store the clicked IMAGE-space coordinate as the representative pixel.
    const imgX = _previewToOriginalImageX(s, lx);
    const imgY = _previewToOriginalImageY(s, ly);
    s.selectedSegments.set(segId, { x: imgX, y: imgY });
    console.debug(`[InteractiveSeg] Selected segment ${segId} @ (${imgX}, ${imgY}).`);
  }

  // Serialise to the hidden widget so Python receives it on the next run.
  _writeCoordWidget(node);

  app.graph.setDirtyCanvas(true, false);
  return true; // consumed; prevent LiteGraph from starting a drag
}

// ─────────────────────────────────────────────────────────────────────────────
// Coordinate helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Test whether a node-local position is inside the image preview area.
 *
 * @param {object} s      node.__seg state bag.
 * @param {number} lx
 * @param {number} ly
 * @returns {boolean}
 */
function _isInsidePreview(s, lx, ly) {
  return (
    lx >= s.previewX &&
    lx < s.previewX + s.previewW &&
    ly >= s.previewY &&
    ly < s.previewY + s.previewH
  );
}

/**
 * Read the segment ID at a node-local position from the cached ID-map data.
 *
 * @param {object} s      node.__seg state bag.
 * @param {number} lx     Node-local x.
 * @param {number} ly     Node-local y.
 * @returns {number|null} Segment ID, or null if outside or no-segment pixel.
 */
function _segmentIdAtLocalPos(s, lx, ly) {
  if (!s.idImageData || !s.idCanvas) return null;

  const imgX = _previewToIdCanvasX(s, lx);
  const imgY = _previewToIdCanvasY(s, ly);
  const srcW = s.idCanvas.width;
  const srcH = s.idCanvas.height;

  if (imgX < 0 || imgY < 0 || imgX >= srcW || imgY >= srcH) return null;

  const i = (imgY * srcW + imgX) * 4;
  const d = s.idImageData.data;
  const key = rgbToKey(d[i], d[i + 1], d[i + 2]);

  return s.colourToSegId.get(key) ?? null;
}

/**
 * Convert node-local x coordinate to ID-map pixel x.
 *
 * @param {object} s
 * @param {number} lx
 * @returns {number} Integer pixel x in ID-map space.
 */
function _previewToImageX(s, lx) {
  const ratio = s.idCanvas ? s.idCanvas.width / s.previewW : 1;
  return Math.floor((lx - s.previewX) * ratio);
}

/**
 * Convert node-local y coordinate to ID-map pixel y.
 *
 * @param {object} s
 * @param {number} ly
 * @returns {number} Integer pixel y in ID-map space.
 */
function _previewToImageY(s, ly) {
  const ratio = s.idCanvas ? s.idCanvas.height / s.previewH : 1;
  return Math.floor((ly - s.previewY) * ratio);
}

// ─────────────────────────────────────────────────────────────────────────────
// Widget serialisation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Serialise the current selection to the hidden `selected_coords` widget.
 *
 * The Python backend reads this string as a JSON array of {x, y} objects.
 * One entry per selected segment (the representative pixel).
 *
 * @param {object} node
 */
function _writeCoordWidget(node) {
  const s = node.__seg;
  if (!s) return;

  const widget = s.coordsWidget;
  if (!widget) {
    console.warn("[InteractiveSeg] selected_coords widget not found.");
    return;
  }

  const coords = [];
  for (const [_segId, { x, y }] of s.selectedSegments) {
    coords.push({ x, y });
  }

  const newValue = JSON.stringify(coords);

  // ON VÉRIFIE SI LA VALEUR A RÉELLEMENT CHANGÉ
  if (widget.value !== newValue) {
    widget.value = newValue;

    // CRITIQUE : Déclenche le callback pour informer ComfyUI 
    // que le graphe est "sale" (dirty) et doit être sérialisé à nouveau
    if (widget.callback) {
      widget.callback(newValue);
    }
    
    // Optionnel mais recommandé : force ComfyUI à savoir que le nœud a changé visuellement
    node.setDirtyCanvas(true, true);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Node sizing
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Adjust the node's size to fit the preview image comfortably.
 * LiteGraph nodes are [width, height] arrays.
 *
 * @param {object} node
 * @param {number} imgW  Preview width in canvas pixels.
 * @param {number} imgH  Preview height in canvas pixels.
 */
function _resizeNode(node, imgW, imgH) {
  const s = node.__seg;
  if (!s) return;

  // Leave room for the header (title + widgets).
  const widgetHeight = (node.widgets?.length ?? 0) * 22 + HEADER_HEIGHT;
  const totalH = widgetHeight + imgH + FOOTER_PADDING;
  const totalW = Math.max(imgW + 16, 260); // minimum comfortable width

  // Update where the preview will be drawn.
  s.previewX = 8;
  s.previewY = widgetHeight;
  s.previewW = imgW;
  s.previewH = imgH;

  node.size = [totalW, totalH];
  node.setDirtyCanvas?.(true, true);
}

/** Conversion pour interagir avec le canvas d'ID sous-échantillonné (hover) */
function _previewToIdCanvasX(s, lx) {
  const ratio = s.idCanvas ? s.idCanvas.width / s.previewW : 1;
  return Math.floor((lx - s.previewX) * ratio);
}

function _previewToIdCanvasY(s, ly) {
  const ratio = s.idCanvas ? s.idCanvas.height / s.previewH : 1;
  return Math.floor((ly - s.previewY) * ratio);
}

/** Conversion pour envoyer les coordonnées absolues au backend (clic) */
function _previewToOriginalImageX(s, lx) {
  const ratio = s.originalW ? s.originalW / s.previewW : 1;
  return Math.floor((lx - s.previewX) * ratio);
}

function _previewToOriginalImageY(s, ly) {
  const ratio = s.originalH ? s.originalH / s.previewH : 1;
  return Math.floor((ly - s.previewY) * ratio);
}