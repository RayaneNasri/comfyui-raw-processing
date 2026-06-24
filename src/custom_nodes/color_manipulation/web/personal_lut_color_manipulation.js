import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ── Helpers (attached to the prototype in beforeRegisterNodeDef) ─────────────

function buildUI() {
    const node = this;

    // Hide the raw STRING widget — driven programmatically
    const hiddenWidget = node.widgets?.find(w => w.name === "lut_path");
    if (hiddenWidget) {
        hiddenWidget.type        = "hidden";
        hiddenWidget.computeSize = () => [0, -4];
        hiddenWidget.draw        = () => {};
    }

    // ── Upload button ─────────────────────────────────────────────────────
    node._uploadWidget = node.addWidget(
        "button",
        "Ouvrir un fichier .cube…",
        null,
        () => node._openFilePicker()
    );

    // ── Label showing the currently loaded file ───────────────────────────
    node._fileLabel = node.addWidget(
        "text",
        "Fichier",
        "(aucun fichier chargé)",
        () => {}
    );
    node._fileLabel.disabled = true;

    // Custom draw so the label is always readable regardless of LiteGraph version
    node._fileLabel.draw = function (ctx, _node, widget_width, y, widget_height) {
        const margin = 15;
        ctx.save();

        // Background
        ctx.fillStyle = "#222";
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(margin, y, widget_width - margin * 2, widget_height, 4);
        } else {
            ctx.rect(margin, y, widget_width - margin * 2, widget_height);
        }
        ctx.fill();

        // Label (left)
        ctx.fillStyle = "#AAA";
        ctx.textAlign = "left";
        ctx.font = "12px Arial";
        ctx.fillText(this.name, margin + 8, y + widget_height * 0.7);

        // Value (right, truncated)
        ctx.fillStyle = "#FFF";
        ctx.textAlign = "right";
        ctx.fillText(
            String(this.value ?? "").substring(0, 40),
            widget_width - margin - 8,
            y + widget_height * 0.7
        );
        ctx.restore();
    };
}

async function openFilePicker() {
    const node  = this;
    const input = document.createElement("input");
    input.type   = "file";
    input.accept = ".cube";

    input.onchange = async () => {
        const file = input.files?.[0];
        if (!file) return;

        const form = new FormData();
        form.append("file", file, file.name);

        try {
            const resp = await api.fetchApi("/personal_lut/upload_cube", {
                method: "POST",
                body:   form,
            });
            const data = await resp.json();

            if (data.error) {
                alert(`Erreur : ${data.error}`);
                return;
            }

            // Update the hidden widget with the resolved server-side path
            const hidden = node.widgets?.find(w => w.name === "lut_path");
            if (hidden) hidden.value = data.path;

            // Update the visible label
            node._fileLabel.value = data.filename;
            node.setDirtyCanvas(true, true);
        } catch (e) {
            console.error("[PersonalLUT] Erreur upload .cube :", e);
            alert("Erreur lors de l'import du fichier .cube");
        }
    };

    input.click();
}

// ── Extension registration ────────────────────────────────────────────────────

app.registerExtension({
    name: "personal_lut.CubeSelector",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PersonalLutColorManipulationNode") return;

        // Attach helpers to the prototype
        nodeType.prototype._buildUI        = buildUI;
        nodeType.prototype._openFilePicker = openFilePicker;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);
            this._buildUI();
        };
    },
});