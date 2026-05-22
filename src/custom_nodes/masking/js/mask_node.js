/**
 * mask_node.js — Widget canvas de masquage pour ComfyUI
 * 
 * Ce fichier est automatiquement chargé par ComfyUI au démarrage
 * car il se trouve dans le dossier `js/` de l'extension.
 * 
 * Il enregistre un widget canvas interactif sur le nœud MaskDrawNode
 * permettant à l'utilisateur de dessiner un masque directement dans le graphe.
 */

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// ── Constantes ───────────────────────────────────────────────────────────────

const WIDGET_NAME = "mask_data";
const NODE_TYPE   = "MaskDrawNode";

const COLORS = {
  maskFill:    "rgba(55, 138, 221, 0.45)",
  maskStroke:  "rgba(55, 138, 221, 0.9)",
  eraseFill:   "rgba(255, 80,  80, 0.3)",
  uiSelect:    "rgba(55, 138, 221, 0.15)",
  uiStroke:    "#378ADD",
  toolbar:     "#1a1a1a",
  toolbarBdr:  "#333",
  toolActive:  "#378ADD",
  toolHover:   "#2a2a2a",
  toolText:    "#ccc",
  canvasBg:    "#111",
};

const TOOLS = [
  { id: "rect",    icon: "▭",  label: "Rectangle"  },
  { id: "ellipse", icon: "◯",  label: "Ellipse"    },
  { id: "lasso",   icon: "⌇",  label: "Lasso"      },
  { id: "brush",   icon: "●",  label: "Pinceau"    },
  { id: "eraser",  icon: "◻",  label: "Gomme"      },
];

const CANVAS_W = 512;
const CANVAS_H = 384;
const TOOLBAR_H = 36;
const WIDGET_H  = CANVAS_H + TOOLBAR_H + 4;

// ── Helpers ──────────────────────────────────────────────────────────────────

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function canvasToBase64(canvas) {
  return canvas.toDataURL("image/png");
}

function drawDashedRect(ctx, x, y, w, h) {
  ctx.save();
  ctx.strokeStyle = COLORS.uiStroke;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 3]);
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = COLORS.uiSelect;
  ctx.fillRect(x, y, w, h);
  ctx.restore();
}

function drawDashedEllipse(ctx, cx, cy, rx, ry) {
  ctx.save();
  ctx.strokeStyle = COLORS.uiStroke;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 3]);
  ctx.beginPath();
  ctx.ellipse(cx, cy, Math.abs(rx), Math.abs(ry), 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = COLORS.uiSelect;
  ctx.fill();
  ctx.restore();
}

// ── Création du widget ───────────────────────────────────────────────────────

function createMaskWidget(node, inputName) {
  // État interne du widget
  const state = {
    tool: "rect",
    fusionMode: "add",    // "add" | "sub" | "inter"
    brushSize: 20,
    brushHard: 0.8,
    isDrawing: false,
    startX: 0,
    startY: 0,
    lassoPoints: [],
    history: [],          // pile d'annulation (ImageData)
    sourceImage: null,    // HTMLImageElement de l'image connectée
  };

  // ── Canvases off-screen ──
  const maskCanvas = document.createElement("canvas");
  maskCanvas.width  = CANVAS_W;
  maskCanvas.height = CANVAS_H;
  const maskCtx = maskCanvas.getContext("2d");

  const uiCanvas = document.createElement("canvas");
  uiCanvas.width  = CANVAS_W;
  uiCanvas.height = CANVAS_H;
  const uiCtx = uiCanvas.getContext("2d");

  // Buffer du masque (Uint8 niveaux de gris)
  let maskBuf = new Uint8ClampedArray(CANVAS_W * CANVAS_H);

  function saveMaskHistory() {
    state.history.push(new Uint8ClampedArray(maskBuf));
    if (state.history.length > 20) state.history.shift();
  }

  function flushMask() {
    // Redessine le canvas masque à partir de maskBuf
    const id = maskCtx.createImageData(CANVAS_W, CANVAS_H);
    for (let i = 0; i < maskBuf.length; i++) {
      const a = maskBuf[i];
      id.data[i * 4 + 0] = 55;
      id.data[i * 4 + 1] = 138;
      id.data[i * 4 + 2] = 221;
      id.data[i * 4 + 3] = a;
    }
    maskCtx.putImageData(id, 0, 0);
    // Sérialise et stocke dans le widget value
    exportMask();
    node.setDirtyCanvas(true, true);
  }

  function exportMask() {
    // Exporte le masque en niveaux de gris (blanc = sélectionné)
    const tmp = document.createElement("canvas");
    tmp.width  = CANVAS_W;
    tmp.height = CANVAS_H;
    const tCtx = tmp.getContext("2d");
    const id = tCtx.createImageData(CANVAS_W, CANVAS_H);
    for (let i = 0; i < maskBuf.length; i++) {
      const v = maskBuf[i];
      id.data[i * 4 + 0] = v;
      id.data[i * 4 + 1] = v;
      id.data[i * 4 + 2] = v;
      id.data[i * 4 + 3] = 255;
    }
    tCtx.putImageData(id, 0, 0);
    widget.value = tmp.toDataURL("image/png");
  }

  // ── Opérations masque ────────────────────────────────────────────────────

  function setPixel(x, y, alpha) {
    if (x < 0 || y < 0 || x >= CANVAS_W || y >= CANVAS_H) return;
    const i = y * CANVAS_W + x;
    if (state.fusionMode === "add")   maskBuf[i] = clamp(alpha, 0, 255);
    else if (state.fusionMode === "sub")  maskBuf[i] = 0;
    else if (state.fusionMode === "inter") maskBuf[i] = maskBuf[i] > 0 ? clamp(alpha, 0, 255) : 0;
  }

  function fillRect(x1, y1, x2, y2) {
    const lx = clamp(Math.min(x1, x2), 0, CANVAS_W - 1);
    const rx = clamp(Math.max(x1, x2), 0, CANVAS_W - 1);
    const ly = clamp(Math.min(y1, y2), 0, CANVAS_H - 1);
    const ry = clamp(Math.max(y1, y2), 0, CANVAS_H - 1);
    for (let y = ly; y <= ry; y++)
      for (let x = lx; x <= rx; x++) setPixel(x, y, 255);
  }

  function fillEllipse(cx, cy, rx, ry) {
    const x1 = clamp(Math.floor(cx - Math.abs(rx)), 0, CANVAS_W - 1);
    const x2 = clamp(Math.ceil(cx  + Math.abs(rx)), 0, CANVAS_W - 1);
    const y1 = clamp(Math.floor(cy - Math.abs(ry)), 0, CANVAS_H - 1);
    const y2 = clamp(Math.ceil(cy  + Math.abs(ry)), 0, CANVAS_H - 1);
    for (let y = y1; y <= y2; y++) {
      for (let x = x1; x <= x2; x++) {
        const dx = (x - cx) / (Math.abs(rx) || 1);
        const dy = (y - cy) / (Math.abs(ry) || 1);
        if (dx * dx + dy * dy <= 1) setPixel(x, y, 255);
      }
    }
  }

  function fillLasso(pts) {
    if (pts.length < 3) return;
    const minY = clamp(Math.min(...pts.map(p => p.y)), 0, CANVAS_H - 1);
    const maxY = clamp(Math.max(...pts.map(p => p.y)), 0, CANVAS_H - 1);
    for (let y = minY; y <= maxY; y++) {
      const xs = [];
      for (let i = 0; i < pts.length; i++) {
        const p1 = pts[i], p2 = pts[(i + 1) % pts.length];
        if ((p1.y <= y && p2.y > y) || (p2.y <= y && p1.y > y)) {
          xs.push(p1.x + (y - p1.y) / (p2.y - p1.y) * (p2.x - p1.x));
        }
      }
      xs.sort((a, b) => a - b);
      for (let j = 0; j < xs.length - 1; j += 2) {
        const x1 = clamp(Math.floor(xs[j]), 0, CANVAS_W - 1);
        const x2 = clamp(Math.ceil(xs[j + 1]), 0, CANVAS_W - 1);
        for (let x = x1; x <= x2; x++) setPixel(x, y, 255);
      }
    }
  }

  function brushStroke(x, y, erase = false) {
    const r = Math.floor(state.brushSize / 2);
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist <= r) {
          const alpha = dist < r * state.brushHard
            ? 255
            : Math.round(255 * (1 - (dist - r * state.brushHard) / (r * (1 - state.brushHard) + 0.001)));
          const px = x + dx, py = y + dy;
          if (px >= 0 && py >= 0 && px < CANVAS_W && py < CANVAS_H) {
            const i = py * CANVAS_W + px;
            if (erase) maskBuf[i] = 0;
            else maskBuf[i] = Math.max(maskBuf[i], clamp(alpha, 0, 255));
          }
        }
      }
    }
  }

  function invertMask() {
    for (let i = 0; i < maskBuf.length; i++) maskBuf[i] = 255 - maskBuf[i];
  }

  function selectAll()  { maskBuf.fill(255); }
  function selectNone() { maskBuf.fill(0);   }

  // ── Widget LiteGraph ─────────────────────────────────────────────────────

  const widget = {
    name: inputName,
    type: "mask_draw",
    value: "",
    options: {},

    // Taille demandée au nœud
    computeSize() {
      return [Math.max(node.size[0], CANVAS_W + 20), WIDGET_H + 20];
    },

    draw(ctx, nodeObj, width, posY) {
      const x = 10;
      const y = posY + 2;
      const W = width - 20;
      const scale = W / CANVAS_W;

      ctx.save();

      // Fond du canvas
      ctx.fillStyle = COLORS.canvasBg;
      ctx.fillRect(x, y + TOOLBAR_H, W, CANVAS_H * scale);

      // Image source en fond si disponible
      if (state.sourceImage) {
        ctx.drawImage(state.sourceImage, x, y + TOOLBAR_H, W, CANVAS_H * scale);
      }

      // Overlay masque
      ctx.drawImage(maskCanvas, x, y + TOOLBAR_H, W, CANVAS_H * scale);

      // Overlay UI (sélection en cours)
      ctx.drawImage(uiCanvas, x, y + TOOLBAR_H, W, CANVAS_H * scale);

      // ── Barre d'outils ──
      ctx.fillStyle = COLORS.toolbar;
      ctx.fillRect(x, y, W, TOOLBAR_H);
      ctx.strokeStyle = COLORS.toolbarBdr;
      ctx.lineWidth = 0.5;
      ctx.strokeRect(x, y, W, TOOLBAR_H);

      const btnW = 32, btnH = 26, btnY = y + 5;
      let bx = x + 6;

      TOOLS.forEach(t => {
        const isActive = state.tool === t.id;
        if (isActive) {
          ctx.fillStyle = COLORS.toolActive;
          ctx.beginPath();
          ctx.roundRect(bx, btnY, btnW, btnH, 4);
          ctx.fill();
        }
        ctx.fillStyle = isActive ? "#fff" : COLORS.toolText;
        ctx.font = "14px monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(t.icon, bx + btnW / 2, btnY + btnH / 2);
        bx += btnW + 2;
      });

      // Séparateur
      ctx.strokeStyle = COLORS.toolbarBdr;
      ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(bx, btnY + 2); ctx.lineTo(bx, btnY + btnH - 2); ctx.stroke();
      bx += 8;

      // Boutons utilitaires
      const utils = [["↔", "invert"], ["⬜", "all"], ["⬛", "none"], ["↩", "undo"]];
      utils.forEach(([icon, id]) => {
        ctx.fillStyle = COLORS.toolText;
        ctx.font = "13px monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(icon, bx + 14, btnY + btnH / 2);
        bx += 30;
      });

      // Curseur pinceau
      if ((state.tool === "brush" || state.tool === "eraser") && state._mouseX !== undefined) {
        const mx = x + state._mouseX * scale;
        const my = y + TOOLBAR_H + state._mouseY * scale;
        const r  = (state.brushSize / 2) * scale;
        ctx.strokeStyle = state.tool === "eraser" ? "#f55" : "#378ADD";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 2]);
        ctx.beginPath();
        ctx.arc(mx, my, r, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      ctx.restore();
    },

    mouse(event, pos, nodeObj) {
      const W = nodeObj.size[0] - 20;
      const scale = CANVAS_W / W;

      // Position dans le canvas (hors toolbar)
      const localX = (pos[0] - 10) * scale;
      const localY = (pos[1] - TOOLBAR_H - 2) * scale;

      // ── Clic sur la barre d'outils ──
      if (pos[1] >= 2 && pos[1] <= 2 + TOOLBAR_H) {
        if (event.type === "pointerdown") {
          const bx0 = 6;
          const btnW = 34;
          const relX = pos[0] - 10;

          // Outils principaux
          TOOLS.forEach((t, idx) => {
            const bx = bx0 + idx * (btnW);
            if (relX >= bx && relX <= bx + 32) {
              state.tool = t.id;
              nodeObj.setDirtyCanvas(true);
            }
          });

          // Boutons utilitaires
          let ux = bx0 + TOOLS.length * btnW + 16;
          const utils = ["invert", "all", "none", "undo"];
          utils.forEach(id => {
            if (relX >= ux && relX <= ux + 28) {
              if (id === "invert") { saveMaskHistory(); invertMask(); flushMask(); }
              else if (id === "all")  { saveMaskHistory(); selectAll();  flushMask(); }
              else if (id === "none") { saveMaskHistory(); selectNone(); flushMask(); }
              else if (id === "undo") {
                if (state.history.length > 0) { maskBuf = state.history.pop(); flushMask(); }
              }
            }
            ux += 30;
          });
        }
        return true;
      }

      // ── Zone canvas ──
      const cx = Math.round(localX), cy = Math.round(localY);

      if (state.tool === "brush" || state.tool === "eraser") {
        state._mouseX = cx; state._mouseY = cy;
        nodeObj.setDirtyCanvas(true);
      }

      if (event.type === "pointerdown") {
        state.isDrawing = true;
        state.startX = cx; state.startY = cy;
        if (state.tool === "lasso") state.lassoPoints = [{ x: cx, y: cy }];
        if (state.tool === "brush" || state.tool === "eraser") {
          saveMaskHistory();
          brushStroke(cx, cy, state.tool === "eraser");
          flushMask();
        }
        return true;
      }

      if (event.type === "pointermove" && state.isDrawing) {
        uiCtx.clearRect(0, 0, CANVAS_W, CANVAS_H);
        if (state.tool === "rect") {
          drawDashedRect(uiCtx, Math.min(state.startX, cx), Math.min(state.startY, cy),
            Math.abs(cx - state.startX), Math.abs(cy - state.startY));
        } else if (state.tool === "ellipse") {
          drawDashedEllipse(uiCtx, (state.startX + cx) / 2, (state.startY + cy) / 2,
            (cx - state.startX) / 2, (cy - state.startY) / 2);
        } else if (state.tool === "lasso") {
          state.lassoPoints.push({ x: cx, y: cy });
          uiCtx.strokeStyle = COLORS.uiStroke;
          uiCtx.lineWidth = 1.5;
          uiCtx.setLineDash([4, 2]);
          uiCtx.beginPath();
          state.lassoPoints.forEach((p, i) => i === 0 ? uiCtx.moveTo(p.x, p.y) : uiCtx.lineTo(p.x, p.y));
          uiCtx.stroke();
          uiCtx.setLineDash([]);
        } else if (state.tool === "brush" || state.tool === "eraser") {
          brushStroke(cx, cy, state.tool === "eraser");
          flushMask();
        }
        nodeObj.setDirtyCanvas(true);
        return true;
      }

      if (event.type === "pointerup" && state.isDrawing) {
        state.isDrawing = false;
        uiCtx.clearRect(0, 0, CANVAS_W, CANVAS_H);
        saveMaskHistory();

        if (state.tool === "rect") {
          fillRect(state.startX, state.startY, cx, cy);
        } else if (state.tool === "ellipse") {
          fillEllipse((state.startX + cx) / 2, (state.startY + cy) / 2,
            (cx - state.startX) / 2, (cy - state.startY) / 2);
        } else if (state.tool === "lasso") {
          state.lassoPoints.push({ x: cx, y: cy });
          fillLasso(state.lassoPoints);
          state.lassoPoints = [];
        }
        flushMask();
        return true;
      }

      return false;
    },

    serialize() { return this.value; },
    deserialize(v) { this.value = v || ""; },
  };

  // ── Écoute des changements d'image connectée ──────────────────────────────
  // Quand l'image d'entrée change, on met à jour l'aperçu de fond

  node.onExecuted = function(output) {
    // Si l'exécution retourne une image de prévisualisation
    if (output && output.ui && output.ui.images && output.ui.images.length > 0) {
      const imgInfo = output.ui.images[0];
      const img = new Image();
      img.onload = () => { state.sourceImage = img; node.setDirtyCanvas(true, true); };
      img.src = `/view?filename=${imgInfo.filename}&subfolder=${imgInfo.subfolder}&type=${imgInfo.type}`;
    }
  };

  // Essayer de récupérer l'image depuis le nœud connecté en amont
  const origOnConnectInput = node.onConnectInput;
  node.onConnectInput = function(targetSlot, type, output, originNode, originSlot) {
    if (origOnConnectInput) origOnConnectInput.apply(this, arguments);
    // Récupérer la dernière image exécutée du nœud source
    if (originNode && originNode.imgs && originNode.imgs.length > 0) {
      state.sourceImage = originNode.imgs[0];
      node.setDirtyCanvas(true, true);
    }
  };

  node.widgets = node.widgets || [];
  node.widgets.push(widget);
  node.size = widget.computeSize();

  return widget;
}

// ── Enregistrement de l'extension ────────────────────────────────────────────

app.registerExtension({
  name: "MaskDrawNode",

  async beforeRegisterNodeDef(nodeType, nodeData, _app) {
    if (nodeData.name !== NODE_TYPE) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function() {
      if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);

      // Supprimer le widget texte par défaut de mask_data (on le remplace)
      this.widgets = (this.widgets || []).filter(w => w.name !== WIDGET_NAME);

      // Créer notre widget canvas
      createMaskWidget(this, WIDGET_NAME);

      this.serialize_widgets = true;
    };
  },
});