import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const RAW_EXTENSIONS =
  ".dng,.cr2,.cr3,.arw,.nef,.raf,.orf,.rw2,.srw,.tiff,.tif";

app.registerExtension({
  name: "RawSensor.BatchUploadButton",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "BatchReadRawSensorNode") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);

      // Find the multiline string widget we defined in the Python node
      const fileListWidget = this.widgets.find((w) => w.name === "file_list");

      const fileInput = document.createElement("input");
      fileInput.type = "file";
      fileInput.multiple = true; // allows the user to select multiple files
      fileInput.accept = RAW_EXTENSIONS;
      fileInput.style.display = "none";

      fileInput.addEventListener("change", async () => {
        const files = fileInput.files;
        if (!files || files.length === 0) return;

        let uploadedFilenames = [];

        // Upload files one by one to the backend
        for (let i = 0; i < files.length; i++) {
          const file = files[i];
          const formData = new FormData();
          formData.append("image", file);
          formData.append("overwrite", "true");

          try {
            const resp = await api.fetchApi("/upload/image", {
              method: "POST",
              body: formData,
            });

            if (resp.status === 200) {
              const data = await resp.json();
              const filename = data.subfolder
                ? `${data.subfolder}/${data.name}`
                : data.name;
              uploadedFilenames.push(filename);
            } else {
              alert(
                `Upload failed for ${file.name}: ${resp.status} ${resp.statusText}`,
              );
            }
          } catch (error) {
            alert(`Error uploading ${file.name}: ${error}`);
          }
        }

        // update the multiline widget with the uploaded filenames
        if (uploadedFilenames.length > 0) {
          // sort alphabetically to maintain the burst sequence order
          uploadedFilenames.sort();
          fileListWidget.value = uploadedFilenames.join("\n");

          if (fileListWidget.callback)
            fileListWidget.callback(fileListWidget.value);

          // force the node to recompute its size so the textarea
          // actually grows to show the scrollable file list
          const newSize = this.computeSize();
          this.setSize([
            Math.max(this.size[0], newSize[0]),
            Math.max(this.size[1], newSize[1]),
          ]);
          app.graph.setDirtyCanvas(true, true);
        }

        // reset the input so the user can click again if needed
        fileInput.value = "";
      });

      document.body.appendChild(fileInput);

      this.addWidget("button", "Choose RAW files", null, () => {
        fileInput.click();
      });

      this.addWidget("button", "Clear selection", null, () => {
        fileListWidget.value = "";
        if (fileListWidget.callback)
          fileListWidget.callback(fileListWidget.value);

        const newSize = this.computeSize();
        this.setSize([newSize[0], newSize[1]]);
        app.graph.setDirtyCanvas(true, true);
      });

      this.onRemoved = function () {
        fileInput.remove();
      };
    };
  },
});
