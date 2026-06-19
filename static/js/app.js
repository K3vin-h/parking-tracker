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

    function positionBoundingBoxes(root = document) {
        // WHY percentages: stored boxes are normalized and remain aligned at any image size.
        root.querySelectorAll("[data-bbox]").forEach((box) => {
            const values = ["x", "y", "width", "height"].map((name) =>
                Number.parseFloat(box.dataset[name] || "0")
            );
            if (values.some((value) => !Number.isFinite(value))) {
                box.hidden = true;
                return;
            }
            const [x, y, width, height] = values.map((value) =>
                Math.min(1, Math.max(0, value))
            );
            box.style.left = `${x * 100}%`;
            box.style.top = `${y * 100}%`;
            box.style.width = `${width * 100}%`;
            box.style.height = `${height * 100}%`;
        });
    }

    positionBoundingBoxes();
    document.body.addEventListener("htmx:afterSwap", (event) => {
        positionBoundingBoxes(event.detail.target);
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
})();
