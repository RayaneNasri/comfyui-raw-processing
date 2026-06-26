import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "hue_saturation_map.DcpSelector",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "HueSaturationMapNode") return;

        // ── Surcharge de onNodeCreated ────────────────────────────────────────
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);

            // Masquer complètement le widget STRING brut — on le pilote
            // programmatiquement, il ne doit ni s'afficher ni occuper d'espace
            // dans le node (certaines versions de LiteGraph ne traitent pas
            // le type "hidden" nativement, donc on neutralise aussi le draw
            // et la taille calculée).
            const hiddenWidget = this.widgets.find(w => w.name === "dcp_selection");
            if (hiddenWidget) {
                hiddenWidget.type        = "hidden";
                hiddenWidget.computeSize = () => [0, -4];
                hiddenWidget.draw        = () => {};
            }

            this._dcpPresets    = [];   // [{label, stem}]
            this._dcpMode       = "preset";   // "preset" | "custom"
            this._selectedPreset = "";
            this._customFilename = "";

            this._buildUI();
            this._loadPresets();
        };
    },

    // ─────────────────────────────────────────────────────────────────────────

    // On étend le prototype après que LiteGraph a construit le type
    async setup() {},
});


// ── Helpers attachés au prototype via la surcharge ───────────────────────────
// (déclarés ici pour lisibilité ; ils sont ajoutés via _buildUI/etc. comme
//  méthodes de l'instance dans onNodeCreated)

function buildUI() {
    const node = this;

    // ── Sélecteur de mode (combo natif LiteGraph → flèches ◀ ▶ intégrées
    //    pour basculer entre "Presets" et "Fichier custom") ────────────────
    node._modeWidget = node.addWidget(
        "combo",
        "Mode",
        "Presets",
        (value) => {
            node._dcpMode = value === "Presets" ? "preset" : "custom";
            node._refreshUI();
        },
        { values: ["Presets", "Fichier custom"] }
    );

    // ── Dropdown presets ──────────────────────────────────────────────────
    node._presetWidget = node.addWidget(
        "combo",
        "Profil caméra",
        "",
        (value) => {
            node._selectedPreset = value;
            node._syncHiddenWidget();
        },
        { values: [] }
    );

    // ── Bouton import fichier custom ──────────────────────────────────────
    node._uploadWidget = node.addWidget(
        "button",
        "Ouvrir un fichier .dcp…",
        null,
        () => node._openFilePicker()
    );

    // ── Label fichier custom sélectionné ─────────────────────────────────
    node._fileLabel = node.addWidget(
        "text",
        "Fichier",
        "(aucun fichier chargé)",
        () => {}
    );
    node._fileLabel.disabled = true;
    // Rendu manuel : on ne dépend plus du comportement interne de LiteGraph
    // pour le type "text" (dont on ne maîtrise pas tous les cas), on dessine
    // nous-mêmes le label + la valeur courante à chaque frame.
    node._fileLabel.draw = function (ctx, _node, widget_width, y, widget_height) {
        const margin = 15;
        ctx.save();
        ctx.fillStyle = "#222";
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(margin, y, widget_width - margin * 2, widget_height, 4);
        } else {
            ctx.rect(margin, y, widget_width - margin * 2, widget_height);
        }
        ctx.fill();

        ctx.fillStyle = "#AAA";
        ctx.textAlign = "left";
        ctx.font = "12px Arial";
        ctx.fillText(this.name, margin + 8, y + widget_height * 0.7);

        ctx.fillStyle = "#FFF";
        ctx.textAlign = "right";
        ctx.fillText(
            String(this.value ?? "").substr(0, 40),
            widget_width - margin - 8,
            y + widget_height * 0.7
        );
        ctx.restore();
    };

    node._refreshUI();
}

async function loadPresets() {
    const node = this;
    try {
        const resp = await api.fetchApi("/hue_sat_map/dcp_presets");
        const data = await resp.json();
        node._dcpPresets = data.presets ?? [];

        const labels = node._dcpPresets.map(p => p.label);
        node._presetWidget.options.values = labels;

        if (labels.length > 0 && !node._selectedPreset) {
            node._selectedPreset = labels[0];
            node._syncHiddenWidget();
        }
        node._presetWidget.value = node._selectedPreset || labels[0] || "";
        node.setDirtyCanvas(true, true);
    } catch (e) {
        console.error("[HueSatMap] Impossible de charger les presets DCP :", e);
    }
}

function refreshUI() {
    const node = this;
    const isPreset = node._dcpMode === "preset";

    node._modeWidget.value    = isPreset ? "Presets" : "Fichier custom";
    node._presetWidget.hidden  = !isPreset;
    node._uploadWidget.hidden  =  isPreset;
    node._fileLabel.hidden     =  isPreset;

    node._syncHiddenWidget();
    node.setSize(node.computeSize());
}

function syncHiddenWidget() {
    const node   = this;
    const hidden = node.widgets?.find(w => w.name === "dcp_selection");
    if (!hidden) return;

    if (node._dcpMode === "preset") {
        // Retrouver le stem depuis le label sélectionné
        const entry = node._dcpPresets.find(p => p.label === node._selectedPreset);
        hidden.value = entry ? `preset:${entry.stem}` : "";
    } else {
        hidden.value = node._customFilename
            ? `custom:${node._customFilename}`
            : "";
    }
}

function openFilePicker() {
    const node  = this;
    const input = document.createElement("input");
    input.type   = "file";
    input.accept = ".dcp";

    input.onchange = async () => {
        const file = input.files?.[0];
        if (!file) return;

        // Upload vers le serveur ComfyUI
        const form = new FormData();
        form.append("file", file, file.name);

        try {
            const resp = await api.fetchApi("/hue_sat_map/upload_dcp", {
                method: "POST",
                body:   form,
            });
            const data = await resp.json();

            if (data.error) {
                alert(`Erreur : ${data.error}`);
                return;
            }

            node._customFilename      = data.filename;
            node._fileLabel.value     = data.filename;
            node._syncHiddenWidget();
            node.setDirtyCanvas(true, true); // force le redraw : le fetch est async,
                                              // donc hors du cycle de rendu déclenché par la souris
        } catch (e) {
            console.error("[HueSatMap] Erreur upload DCP :", e);
            alert("Erreur lors de l'import du fichier .dcp");
        }
    };

    input.click();
}

// Rattachement des méthodes au prototype dans onNodeCreated
app.registerExtension({
    name: "hue_saturation_map.DcpSelector.methods",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "HueSaturationMapNode") return;

        nodeType.prototype._buildUI         = buildUI;
        nodeType.prototype._loadPresets     = loadPresets;
        nodeType.prototype._refreshUI       = refreshUI;
        nodeType.prototype._syncHiddenWidget = syncHiddenWidget;
        nodeType.prototype._openFilePicker  = openFilePicker;
    },
});