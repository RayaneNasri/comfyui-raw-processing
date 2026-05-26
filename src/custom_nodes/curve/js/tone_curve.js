import { app } from "../../scripts/app.js";

const CHANNELS = ["master", "r", "g", "b"];
const CHANNEL_LABELS = {
    master: "Master",
    r: "Red",
    g: "Green",
    b: "Blue",
};

app.registerExtension({
    name: "Artishow.ToneCurve",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ToneCurveNode") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            this._activeChannel = "master";

            // Minimal theme without background tracks
            const THEME = {
                radioOuter: "#3f3f46",       // Unselected circle border
                radioOuterActive: "#a1a1aa", // Selected circle border
                textMuted: "#71717a",        // Unselected label color
                textActive: "#ffffff",       // Selected label color
                colors: {
                    master: "#e4e4e7",       // Neutral core
                    r: "#ef4444",            // Red core
                    g: "#22c55e",            // Green core
                    b: "#3b82f6"             // Blue core
                }
            };

            const radioWidget = {
                type: "custom_radio",
                name: "channel_selector",
                value: this._activeChannel,
                
                // Increased height slightly (38px) to comfortably breathe with stacked labels
                computeSize: function() {
                    return [0, 38]; 
                },
                
                draw: function(ctx, node, widget_width, y, widget_height) {
                    this.last_width = widget_width;
                    const itemWidth = widget_width / CHANNELS.length;

                    ctx.save();

                    // Loop through each channel slot
                    for (let i = 0; i < CHANNELS.length; i++) {
                        const ch = CHANNELS[i];
                        const isSelected = (this.value === ch);
                        
                        // Horizontal center of the current button slot
                        const itemCenterX = (i * itemWidth) + (itemWidth / 2);
                        
                        // Specific vertical alignments for stacked layout
                        const radioY = y + 12; // Circle near the top half
                        const textY = y + 30;  // Label directly beneath
                        const radioRadius = 6;

                        // --- DRAW RADIO CIRCLE ---
                        ctx.lineWidth = 1.5;
                        ctx.strokeStyle = isSelected ? THEME.radioOuterActive : THEME.radioOuter;
                        ctx.fillStyle = "#1c1c1e"; // Empty core background color
                        ctx.beginPath();
                        ctx.arc(itemCenterX, radioY, radioRadius, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.stroke();

                        // --- DRAW ACTIVE INNER COLOR DOT ---
                        if (isSelected) {
                            ctx.fillStyle = THEME.colors[ch];
                            ctx.beginPath();
                            ctx.arc(itemCenterX, radioY, radioRadius - 3.5, 0, Math.PI * 2);
                            ctx.fill();
                        }

                        // --- DRAW LABEL BELOW ---
                        ctx.fillStyle = isSelected ? THEME.textActive : THEME.textMuted;
                        ctx.textAlign = "center"; // Centered under the circle
                        ctx.textBaseline = "middle";
                        ctx.font = isSelected ? "600 10px sans-serif" : "500 10px sans-serif";
                        
                        ctx.fillText(
                            CHANNEL_LABELS[ch].toUpperCase(), 
                            itemCenterX, 
                            textY
                        );
                    }

                    ctx.restore();
                },
                
                mouse: function(event, pos, node) {
                    if (event.type === "pointerdown" || event.type === "mousedown") {
                        const width = this.last_width || node.size[0];
                        const itemWidth = width / CHANNELS.length;
                        
                        const clickedIndex = Math.floor(pos[0] / itemWidth);
                        
                        if (clickedIndex >= 0 && clickedIndex < CHANNELS.length) {
                            const selectedChannel = CHANNELS[clickedIndex];
                            
                            if (this.value !== selectedChannel) {
                                this.value = selectedChannel;
                                node._activeChannel = selectedChannel;
                                node._updateCurveVisibility();
                                node.setDirtyCanvas(true);
                            }
                            return true; 
                        }
                    }
                    return false;
                }
            };

            this.addCustomWidget(radioWidget);
            this._updateCurveVisibility();
        };

        nodeType.prototype._cycleChannel = function () {
            const idx = CHANNELS.indexOf(this._activeChannel);
            this._activeChannel = CHANNELS[(idx + 1) % CHANNELS.length];

            // Mettre à jour le label du bouton
            this._channelSelectorWidget.value = CHANNEL_LABELS[this._activeChannel];
            this._channelSelectorWidget.name = `Canal : ${CHANNEL_LABELS[this._activeChannel]}`;

            this._updateCurveVisibility();
            this.setDirtyCanvas(true);
        };

        nodeType.prototype._updateCurveVisibility = function () {
            for (const channel of CHANNELS) {
                const widgetName = `curve_${channel}`;
                const widget = this.widgets?.find(w => w.name === widgetName);
                if (!widget) continue;

                const isActive = channel === this._activeChannel;
                widget.hidden = !isActive;

                // Collapse la hauteur du widget si caché
                if (isActive) {
                    widget.computeSize = undefined; // restaure la taille normale
                } else {
                    widget.computeSize = () => [0, -4]; // hauteur nulle
                }
            }
        };
    },
});