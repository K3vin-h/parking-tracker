/**
 * Resident plate-management browser behavior.
 *
 * Kept in an external file because the production CSP blocks inline handlers.
 */
(() => {
    "use strict";

    document.querySelectorAll("[data-plate-delete]").forEach((form) => {
        form.addEventListener("submit", (event) => {
            const plate = form
                .closest(".plate-list__item")
                ?.querySelector(".plate-list__text")
                ?.textContent.trim();
            const prompt = plate ? `Remove ${plate}?` : "Remove this plate?";
            if (!window.confirm(prompt)) {
                event.preventDefault();
            }
        });
    });
})();
