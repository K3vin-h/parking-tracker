/**
 * Wire shell behavior that is shared by every operator page.
 *
 * Keeping this tiny and framework-free lets HTMX own server interactions while
 * JavaScript handles only presentation details that CSS/HTML cannot express.
 */
(() => {
    "use strict";

    const body = document.body;
    const openButton = document.querySelector("[data-nav-open]");
    const closeTargets = document.querySelectorAll("[data-nav-close], .nav-link");

    function setNavigation(open) {
        // WHY centralize state: aria-expanded and the visual drawer must never drift.
        body.classList.toggle("nav-open", open);
        openButton?.setAttribute("aria-expanded", String(open));
    }

    openButton?.addEventListener("click", () => setNavigation(true));
    closeTargets.forEach((target) => {
        target.addEventListener("click", () => setNavigation(false));
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            setNavigation(false);
        }
    });

    function drawBoundingBox(canvas) {
        // WHY canvas math: normalized CV coordinates must align with the contained image,
        // including any letterbox space introduced by its aspect ratio.
        const image = canvas.parentElement?.querySelector("img");
        if (!image || !image.complete || !image.naturalWidth) {
            image?.addEventListener("load", () => drawBoundingBox(canvas), { once: true });
            return;
        }
        const values = ["x", "y", "width", "height"].map((name) =>
            Number.parseFloat(canvas.dataset[name] || "0")
        );
        if (values.some((value) => !Number.isFinite(value))) {
            canvas.hidden = true;
            return;
        }

        const bounds = canvas.getBoundingClientRect();
        const ratio = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.round(bounds.width * ratio));
        canvas.height = Math.max(1, Math.round(bounds.height * ratio));
        const context = canvas.getContext("2d");
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        context.clearRect(0, 0, bounds.width, bounds.height);

        const scale = Math.min(
            bounds.width / image.naturalWidth,
            bounds.height / image.naturalHeight
        );
        const imageWidth = image.naturalWidth * scale;
        const imageHeight = image.naturalHeight * scale;
        const offsetX = (bounds.width - imageWidth) / 2;
        const offsetY = (bounds.height - imageHeight) / 2;
        const [x, y, width, height] = values.map((value) =>
            Math.min(1, Math.max(0, value))
        );
        const colors = { good: "#22c55e", warning: "#eab308", error: "#ef4444" };
        const color = colors[canvas.dataset.band] || colors.warning;
        const boxX = offsetX + x * imageWidth;
        const boxY = offsetY + y * imageHeight;
        const boxWidth = width * imageWidth;
        const boxHeight = height * imageHeight;

        context.strokeStyle = color;
        context.lineWidth = 2;
        context.strokeRect(boxX, boxY, boxWidth, boxHeight);
        const label = canvas.dataset.label || "PLATE";
        context.font = '10px "JetBrains Mono", monospace';
        const labelWidth = context.measureText(label).width + 10;
        const labelY = Math.max(0, boxY - 17);
        context.fillStyle = color;
        context.fillRect(boxX, labelY, labelWidth, 17);
        context.fillStyle = "#0f1117";
        context.fillText(label, boxX + 5, labelY + 12);
    }

    function drawBoundingBoxes(root = document) {
        root.querySelectorAll("[data-bbox-canvas]").forEach(drawBoundingBox);
    }

    drawBoundingBoxes();
    window.addEventListener("resize", () => drawBoundingBoxes());
    document.body.addEventListener("htmx:afterSwap", (event) => {
        drawBoundingBoxes(event.detail.target);
        const corrected = event.detail.target.querySelector?.("[data-remove-after]");
        if (corrected) {
            const delay = Number.parseInt(corrected.dataset.removeAfter, 10) || 1200;
            window.setTimeout(() => {
                corrected.remove();
                const queue = document.querySelector("#error-queue");
                if (queue && !queue.querySelector("[data-queue-row], [data-remove-after]")) {
                    queue.innerHTML =
                        '<div class="empty-state"><h2>Queue clear</h2><p>No detections need review. Nicely done.</p></div>';
                }
            }, delay);
        }
    });

    document.body.addEventListener("queueCountChanged", (event) => {
        // WHY use the server count: concurrent corrections can make local decrementing wrong.
        const count = event.detail?.count;
        const badge = document.querySelector("#queue-count");
        if (badge && Number.isInteger(count)) {
            badge.textContent = String(count);
        }
    });

    document.querySelectorAll("[data-registration-tab]").forEach((tab) => {
        tab.addEventListener("click", () => {
            // WHY submit the existing form: HTMX then preserves every other filter.
            const form = tab.closest("form");
            const value = form?.querySelector("[data-registration-value]");
            if (!form || !value) {
                return;
            }
            value.value = tab.dataset.registrationTab;
            form.requestSubmit();
        });
    });

    const confidenceRange = document.querySelector("[data-confidence-range]");
    const confidenceOutput = document.querySelector("[data-confidence-output]");
    function updateConfidenceOutput() {
        if (confidenceRange && confidenceOutput) {
            confidenceOutput.value = `${confidenceRange.value}%`;
            confidenceOutput.textContent = `${confidenceRange.value}%`;
        }
    }
    confidenceRange?.addEventListener("input", updateConfidenceOutput);
    updateConfidenceOutput();
})();
