import { app } from "../../scripts/app.js";

const CANVAS_H = 250;
const PT_RADIUS = 6;
const HIT_RADIUS = 0.07;

const DEFAULT_POINTS = [
  [0, 0],
  [0.25, 0.25],
  [0.5, 0.5],
  [0.75, 0.75],
  [1, 1],
];

// Preview curve
function buildPolynomialWeights(sorted) {
  const logWeights = [];
  const signs = [];

  for (let i = 0; i < sorted.length; i++) {
    let sign = 1;
    let logAbs = 0;

    for (let j = 0; j < sorted.length; j++) {
      if (i === j) continue;
      const delta = sorted[i][0] - sorted[j][0];
      sign *= Math.sign(delta) || 1;
      logAbs += Math.log(Math.abs(delta));
    }

    signs.push(sign);
    logWeights.push(-logAbs);
  }

  const maxLog = Math.max(...logWeights);
  return signs.map((sign, i) => sign * Math.exp(logWeights[i] - maxLog));
}

function previewY(sorted, weights, x) {
  if (sorted.length < 2) return x;

  for (let i = 0; i < sorted.length; i++) {
    if (Math.abs(x - sorted[i][0]) < 1e-12) return sorted[i][1];
  }

  if (sorted.length === 2) {
    const t = (x - sorted[0][0]) / (sorted[1][0] - sorted[0][0]);
    return sorted[0][1] + t * (sorted[1][1] - sorted[0][1]);
  }

  let numerator = 0;
  let denominator = 0;
  for (let i = 0; i < sorted.length; i++) {
    const term = weights[i] / (x - sorted[i][0]);
    numerator += term * sorted[i][1];
    denominator += term;
  }

  return numerator / denominator;
}

// DOM curve editor factory
function makeCurveEditor(initialPoints) {
  const container = document.createElement("div");
  container.style.cssText = `position:relative; width:100%; user-select:none; background:#111;`;

  const canvas = document.createElement("canvas");
  canvas.height = CANVAS_H;
  canvas.style.cssText = `display:block; width:100%; cursor:crosshair;`;
  container.appendChild(canvas);

  const hint = document.createElement("div");
  hint.style.cssText =
    "font-size:10px; color:#555; text-align:center; padding:2px 0 3px;";
  hint.textContent =
    "Click empty space: add point · Drag: move · Double-click point: remove";
  container.appendChild(hint);

  let points = initialPoints.map((p) => [...p]);
  let dragging = null; // index of point being dragged

  function render() {
    const ctx = canvas.getContext("2d");
    const W = canvas.width || canvas.offsetWidth;
    const H = CANVAS_H;
    if (canvas.width !== W) canvas.width = W;

    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = "#2a2a2a";
    ctx.lineWidth = 1;
    // grid lines
    for (let i = 1; i < 4; i++) {
      ctx.beginPath();
      ctx.moveTo((i / 4) * W, 0);
      ctx.lineTo((i / 4) * W, H);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, (i / 4) * H);
      ctx.lineTo(W, (i / 4) * H);
      ctx.stroke();
    }

    // identity reference
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, H);
    ctx.lineTo(W, 0);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = "#3a3a3a";
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, W, H);

    const sorted = [...points].sort((a, b) => a[0] - b[0]);

    // Courbe de Bézier cubique par morceaux (Catmull-Rom → Bézier)
    ctx.strokeStyle = "#e8a000";
    ctx.lineWidth = 2;
    ctx.beginPath();

    if (sorted.length >= 2) {
      // Conversion Catmull-Rom en points de contrôle Bézier cubique
      // La courbe passe par tous les points comme Lagrange, mais avec des tangentes lisses
      const pts = sorted.map(([x, y]) => ({ x: x * W, y: (1 - y) * H }));

      ctx.moveTo(pts[0].x, pts[0].y);

      for (let i = 0; i < pts.length - 1; i++) {
        const p0 = pts[Math.max(i - 1, 0)];
        const p1 = pts[i];
        const p2 = pts[i + 1];
        const p3 = pts[Math.min(i + 2, pts.length - 1)];

        // Tangentes Catmull-Rom converties en handles Bézier (facteur 1/6)
        const cp1x = p1.x + (p2.x - p0.x) / 6;
        const cp1y = p1.y + (p2.y - p0.y) / 6;
        const cp2x = p2.x - (p3.x - p1.x) / 6;
        const cp2y = p2.y - (p3.y - p1.y) / 6;

        ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
      }
    }

    ctx.stroke();

    // Control points
    points.forEach((pt, i) => {
      const px = pt[0] * W;
      const py = (1 - pt[1]) * H;
      ctx.beginPath();
      ctx.arc(px, py, PT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = dragging === i ? "#ffffff" : "#e8a000";
      ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  }

  function toNorm(e) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, 1 - (e.clientY - rect.top) / rect.height)),
    };
  }

  function nearestIdx(nx, ny) {
    let best = -1;
    let bestD = Infinity;
    points.forEach((pt, i) => {
      const d = Math.hypot(pt[0] - nx, pt[1] - ny);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    return bestD < HIT_RADIUS ? best : -1;
  }

  function isEndpoint(idx) {
    const pt = points[idx];
    return pt[0] === 0 || pt[0] === 1;
  }

  // Mouse handlers
  canvas.addEventListener("mousedown", (e) => {
    if (e.button === 2) return;

    e.stopPropagation();
    const { x, y } = toNorm(e);
    const idx = nearestIdx(x, y);

    if (idx !== -1) {
      // clicked on existing point => dragging
      dragging = idx;
    } else {
      // clicked empty space => add point (unless it's too close to an endpoint)
      points.push([
        Math.max(0.01, Math.min(0.99, x)),
        Math.max(0, Math.min(1, y)),
      ]);
      dragging = points.length - 1;
      api.onChange();
    }
    render();
  });

  canvas.addEventListener("dblclick", (e) => {
    e.stopPropagation();
    const { x, y } = toNorm(e);
    const idx = nearestIdx(x, y);

    if (idx !== -1 && points.length > 2 && !isEndpoint(idx)) {
      // double-clicked a non-endpoint => remove it
      points.splice(idx, 1);
      dragging = null;
      api.onChange();
      render();
    }
  });

  canvas.addEventListener("mousemove", (e) => {
    if (dragging === null) return;
    const { x, y } = toNorm(e);

    // enforce x=0 or 1 for endpoints
    const isFirst = points[dragging][0] === 0;
    const isLast = points[dragging][0] === 1;

    // prevent dragging points too close to endpoints
    points[dragging] = [
      isFirst ? 0 : isLast ? 1 : Math.max(0.01, Math.min(0.99, x)),
      Math.max(0, Math.min(1, y)),
    ];
    render();
  });

  const stopDrag = () => {
    if (dragging !== null) {
      dragging = null;
      api.onChange();
      render();
    }
  };
  canvas.addEventListener("mouseup", stopDrag);
  window.addEventListener("mouseup", stopDrag);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  const ro = new ResizeObserver(() => {
    canvas.width = canvas.offsetWidth || 200;
    render();
  });
  ro.observe(canvas);

  // Public API consumed by addDOMWidget
  const api = {
    el: container,
    render,
    onChange: () => {},
    getValue() {
      return JSON.stringify(points);
    },
    setValue(v) {
      try {
        const parsed = JSON.parse(v);
        if (Array.isArray(parsed) && parsed.length >= 2) {
          points = parsed.map((p) => [...p]);
        } else {
          points = DEFAULT_POINTS.map((p) => [...p]);
        }
      } catch {
        points = DEFAULT_POINTS.map((p) => [...p]);
      }
      render();
    },
  };

  return api;
}

// ComfyUI extension
app.registerExtension({
  name: "Artishow.CurveEditor",

  async getCustomWidgets(_app) {
    return {
      CURVE(node, inputName, _inputData, _app) {
        const editor = makeCurveEditor(DEFAULT_POINTS);

        const widget = node.addDOMWidget(inputName, "CURVE", editor.el, {
          getValue() {
            return editor.getValue();
          },
          setValue(v) {
            editor.setValue(v);
          },
          serialize: true,
        });

        editor.onChange = () => {
          widget.value = editor.getValue();
        };

        widget.computeSize = (w) => [w, CANVAS_H + 28];

        // initial render once the element is in the DOM
        requestAnimationFrame(
          () => editor.el.offsetWidth && editor.render && editor.render(),
        );

        return { widget };
      },
    };
  },
});
