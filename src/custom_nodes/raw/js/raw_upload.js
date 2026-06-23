import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const RAW_EXTENSIONS =
  ".dng,.cr2,.cr3,.arw,.nef,.raf,.orf,.rw2,.srw,.tiff,.tif";

app.registerExtension({
  name: "RawSensor.UploadButton",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "ReadRawSensorNode") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);

      const imageWidget = this.widgets.find((w) => w.name === "image");

      const fileInput = document.createElement("input");
      fileInput.type = "file";
      fileInput.accept = RAW_EXTENSIONS;
      fileInput.style.display = "none";
      fileInput.addEventListener("change", async () => {
        const file = fileInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("image", file);
        formData.append("overwrite", "true");

        const resp = await api.fetchApi("/upload/image", {
          method: "POST",
          body: formData,
        });

        if (resp.status === 200) {
          const data = await resp.json();
          const filename = data.subfolder
            ? `${data.subfolder}/${data.name}`
            : data.name;

          if (!imageWidget.options.values.includes(filename)) {
            imageWidget.options.values.push(filename);
          }
          imageWidget.value = filename;
          if (imageWidget.callback) imageWidget.callback(filename);
          app.graph.setDirtyCanvas(true);
        } else {
          alert(`Upload failed: ${resp.status} ${resp.statusText}`);
        }
      });
      document.body.appendChild(fileInput);

      this.addWidget("button", "choose raw file", null, () => {
        fileInput.click();
      });

      this.onRemoved = function () {
        fileInput.remove();
      };
    };
  },
});
