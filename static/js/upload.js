/**
 * Enhance the native multipart upload with drag/drop and a local image preview.
 */
(() => {
    "use strict";

    const dropzone = document.querySelector("[data-dropzone]");
    const input = document.querySelector("[data-file-input]");
    const preview = document.querySelector("[data-upload-preview]");
    const previewImage = document.querySelector("[data-preview-image]");
    const title = document.querySelector("[data-drop-title]");
    const metadata = document.querySelector("[data-file-meta]");
    let previewUrl = null;

    if (!dropzone || !input) {
        return;
    }

    function showFile(file) {
        // WHY validate client-side too: operators get immediate feedback; the server remains authoritative.
        if (!file || !["image/jpeg", "image/png"].includes(file.type)) {
            metadata.textContent = "Choose a JPEG or PNG image";
            return;
        }
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
        }
        previewUrl = URL.createObjectURL(file);
        // WHY revoke on load: the browser keeps the decoded bitmap, so the blob
        // URL is safe to release immediately. This prevents a slow memory leak
        // when an operator uploads repeatedly without re-selecting a file.
        previewImage.onload = () => {
            URL.revokeObjectURL(previewUrl);
            previewUrl = null;
        };
        previewImage.src = previewUrl;
        preview.hidden = false;
        title.textContent = file.name;
        metadata.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
    }

    input.addEventListener("change", () => showFile(input.files[0]));
    ["dragenter", "dragover"].forEach((name) => {
        dropzone.addEventListener(name, (event) => {
            event.preventDefault();
            dropzone.classList.add("is-dragging");
        });
    });
    ["dragleave", "drop"].forEach((name) => {
        dropzone.addEventListener(name, (event) => {
            event.preventDefault();
            dropzone.classList.remove("is-dragging");
        });
    });
    dropzone.addEventListener("drop", (event) => {
        const file = event.dataTransfer.files[0];
        if (!file) {
            return;
        }
        const transfer = new DataTransfer();
        transfer.items.add(file);
        input.files = transfer.files;
        showFile(file);
    });
})();
