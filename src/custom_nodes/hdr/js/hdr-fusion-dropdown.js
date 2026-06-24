/**
 * hdrplus_node.js
 *
 * Adds a dynamic "reference_index" dropdown to the HDRPlusFusion node.
 *
 * Graph walk:
 *   HDRPlusFusion["raw_imgs"] ──link──> BatchReadRawSensorNode["images"]
 *
 * The "images" widget on the loader is a multiline STRING where each line
 * is one file path (e.g. "subfolder/frame0.dng\nsubfolder/frame1.dng").
 * We parse that string, build "[0] frame0.dng" labels, and replace the
 * plain INT widget with a COMBO so the user can pick by filename.
 *
 * Before the prompt is sent to the backend the COMBO display text is
 * converted back to its integer index so Python receives an INT as declared.
 */

import { app } from "../../scripts/app.js";

// ---------------------------------------------------------------------------
// Graph helpers
// ---------------------------------------------------------------------------

/** Follow the link on `inputSlotName` upstream and return the source node. */
function getUpstreamNode(node, inputSlotName) {
  const graph = app.graph;
  if (!graph) return null;
  const slotIdx = node.inputs?.findIndex((i) => i.name === inputSlotName);
  if (slotIdx == null || slotIdx < 0) return null;
  const linkId = node.inputs[slotIdx]?.link;
  if (linkId == null) return null;
  const link = graph.links[linkId];
  if (!link) return null;
  return graph.getNodeById(link.origin_id) ?? null;
}

/**
 * Read the "images" multiline-string widget from the loader node and
 * return a trimmed array of non-empty lines (one per file).
 */
function extractFileList(loaderNode) {
  if (!loaderNode?.widgets) return [];
  const w = loaderNode.widgets.find((w) => w.name === "images");
  if (!w) return [];
  const raw = typeof w.value === "string" ? w.value : String(w.value ?? "");
  return raw
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
}

/** "[0] frame.dng" labels for the dropdown. */
function buildOptions(files) {
  if (!files.length) return [{ value: 0, text: "(no files connected)" }];
  return files.map((f, i) => {
    const base = f.replace(/\\/g, "/").split("/").pop();
    return { value: i, text: `[${i}] ${base}` };
  });
}

// ---------------------------------------------------------------------------
// Dropdown installation
// ---------------------------------------------------------------------------

function installDropdown(node) {
  const loader = getUpstreamNode(node, "raw_imgs");
  const files = extractFileList(loader);
  const options = buildOptions(files);
  const labels = options.map((o) => o.text);

  let widget = node.widgets?.find((w) => w.name === "reference_index");

  if (!widget) {
    // No widget yet — add a fresh COMBO
    widget = node.addWidget("combo", "reference_index", labels[0], () => {}, {
      values: labels,
    });
  } else if (widget.type === "number" || widget.type === "INT") {
    // ComfyUI created a plain number widget from the INT declaration.
    // Convert it in-place to a combo by patching its type and draw method.
    widget.type = "combo";
    widget.options = { values: labels };
    widget.value =
      labels[Math.min(widget.value ?? 0, labels.length - 1)] ?? labels[0];
  } else {
    // Already a combo — just refresh the option list, preserve selection if still valid
    const prevIdx = options.find((o) => o.text === widget.value)?.value ?? 0;
    widget.options = widget.options ?? {};
    widget.options.values = labels;
    widget.value = labels[prevIdx] ?? labels[0];
  }

  widget._hdrplusOptions = options;
  node.setDirtyCanvas(true, true);
}

// ---------------------------------------------------------------------------
// Serialisation — send integer index, not display label, to Python
// ---------------------------------------------------------------------------

function patchSerialize(node) {
  if (node._hdrplusSerializePatched) return;
  node._hdrplusSerializePatched = true;

  const orig = node.serialize.bind(node);
  node.serialize = function () {
    const data = orig();
    const widget = node.widgets?.find((w) => w.name === "reference_index");
    if (!widget?._hdrplusOptions) return data;

    const opt =
      widget._hdrplusOptions.find((o) => o.text === widget.value) ??
      widget._hdrplusOptions[0];
    const intVal = opt?.value ?? 0;

    const wIdx = node.widgets.indexOf(widget);
    if (wIdx >= 0 && Array.isArray(data.widgets_values)) {
      data.widgets_values[wIdx] = intVal;
    }
    return data;
  };
}

// ---------------------------------------------------------------------------
// Per-node setup
// ---------------------------------------------------------------------------

function setupNode(node) {
  if (node.comfyClass !== "HDRPlusFusion") return;

  installDropdown(node);
  patchSerialize(node);

  // Re-populate when connections change
  if (!node._hdrplusConnectionsPatched) {
    node._hdrplusConnectionsPatched = true;
    const orig = node.onConnectionsChange?.bind(node);
    node.onConnectionsChange = function (...args) {
      orig?.(...args);
      setTimeout(() => installDropdown(node), 60);
    };
  }

  // Poll to catch loader widget edits (user types more filenames)
  if (!node._hdrplusPolling) {
    node._hdrplusPolling = true;
    const id = setInterval(() => {
      if (!node.graph) {
        clearInterval(id);
        return;
      }
      installDropdown(node);
    }, 1500);
  }
}

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
  name: "hdrplus.ReferenceFrameDropdown",

  async nodeCreated(node) {
    setupNode(node);
  },

  async loadedGraphNode(node) {
    setupNode(node);
  },
});
