import { app } from "../../scripts/app.js";
import { ComfyDialog } from "../../scripts/ui.js";

// Extension de la classe Dialog de ComfyUI pour s'intégrer visuellement
class SAMEditorDialog extends ComfyDialog {
    constructor() {
        super();
        this.element.classList.add("sam-editor-dialog");
        // Structure HTML modulaire
        this.element.innerHTML = `
            <div class="comfy-modal-content" style="display: flex; flex-direction: column; align-items: center; padding: 20px;">
                <h3 style="color: white; margin-top: 0;">SAM Mask Editor</h3>
                <div style="position: relative; border: 2px solid #555; background: #222;">
                    <canvas id="sam-canvas" width="512" height="512" style="cursor: crosshair;"></canvas>
                </div>
                <p style="color: #aaa; font-size: 12px;">Clic Gauche : Inclure (Vert) | Shift + Clic : Exclure (Rouge)</p>
                <div style="display: flex; gap: 10px; margin-top: 15px;">
                    <button id="sam-btn-clear" style="padding: 8px 16px;">Vider</button>
                    <button id="sam-btn-save" style="padding: 8px 16px; background: #28a745; color: white;">Sauvegarder & Fermer</button>
                </div>
            </div>
        `;
        
        this.canvas = this.element.querySelector("#sam-canvas");
        this.ctx = this.canvas.getContext("2d");
        this.points = [];
        this.targetWidget = null; // Le widget texte caché où on sauvegarde
        
        this.bindEvents();
    }

    bindEvents() {
        this.element.querySelector("#sam-btn-clear").addEventListener("click", () => {
            this.points = [];
            this.draw();
        });

        this.element.querySelector("#sam-btn-save").addEventListener("click", () => {
            // Sauvegarde en JSON dans le widget ComfyUI
            if (this.targetWidget) {
                this.targetWidget.value = JSON.stringify(this.points);
            }
            this.close();
        });

        this.canvas.addEventListener("mousedown", (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const isNegative = e.shiftKey || e.button === 2; // Shift ou Clic Droit
            
            this.points.push({ x, y, label: isNegative ? 0 : 1 });
            this.draw();
        });
        
        this.canvas.addEventListener("contextmenu", e => e.preventDefault());
    }

    open(node) {
        // Trouver le widget caché "click_data"
        this.targetWidget = node.widgets.find(w => w.name === "click_data");
        
        // Charger les points existants
        try {
            this.points = JSON.parse(this.targetWidget.value || "[]");
        } catch(e) {
            this.points = [];
        }

        // Tenter de charger l'image du nœud parent (Simplifié pour l'exemple)
        // Dans un cas réel, il faut interroger le serveur via l'API ComfyUI pour récupérer la preview en cache
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = "#333";
        this.ctx.fillText("Exécutez le graphe une fois pour voir l'image ici", 120, 250);
        
        this.draw();
        this.element.style.display = "block";
    }

    draw() {
        // Redessiner l'image (si disponible) ici...
        
        // Dessiner les points
        for (const p of this.points) {
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, 5, 0, 2 * Math.PI);
            this.ctx.fillStyle = p.label === 1 ? "#00ff00" : "#ff0000";
            this.ctx.fill();
            this.ctx.strokeStyle = "white";
            this.ctx.stroke();
        }
    }
}

const samDialog = new SAMEditorDialog();

// Enregistrement de l'extension dans le graphe LiteGraph de ComfyUI
app.registerExtension({
    name: "Comfy.SAMInteractive",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SAMInteractiveNode") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }

                // Cacher le widget texte contenant le JSON laid
                const clickWidget = this.widgets.find(w => w.name === "click_data");
                if (clickWidget) {
                    clickWidget.type = "hidden";
                    clickWidget.computeSize = () => [0, -4];
                }

                // Ajouter le bouton d'interface UX-friendly
                this.addWidget("button", "✏️ Open Mask Editor", "edit", () => {
                    samDialog.open(this);
                });
            };
        }
    }
});