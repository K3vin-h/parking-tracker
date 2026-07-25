/**
 * Public gate kiosk browser controller.
 *
 * The server decides whether a scan is successful or needs recovery. This file
 * only manages browser-local preview memory, HTMX request states, focus, and the
 * visual rule that success replaces the uploader while recovery keeps it visible.
 */
(() => {
    "use strict";

    const activationForm = document.querySelector("[data-kiosk-activation-form]");
    const activationResult = document.querySelector(
        "[data-kiosk-activation-result]",
    );
    const activationToken = activationForm?.querySelector('input[name="token"]');
    const panel = document.getElementById("kiosk-panel");
    const uploader = document.querySelector("[data-kiosk-uploader]");
    const form = document.getElementById("kiosk-form");
    const processing = document.getElementById("kiosk-processing");
    const result = document.getElementById("kiosk-result");
    const dropzone = document.querySelector("[data-dropzone]");
    const input = document.getElementById("kiosk-image");
    const preview = document.querySelector("[data-dropzone-preview]");
    const prompt = document.querySelector("[data-dropzone-prompt]");
    const reviewCopy = document.querySelector("[data-kiosk-review-copy]");
    const submitButton = document.querySelector("[data-kiosk-submit]");
    const changeButton = document.querySelector("[data-kiosk-change-photo]");
    const chooseButtons = document.querySelectorAll("[data-kiosk-choose-file]");
    let previewUrl = null;

    /**
     * Show a safe operator-facing activation failure without trusting response HTML.
     */
    function showActivationFailure(status = 0) {
        if (!activationResult) {
            return;
        }
        const rateLimited = status === 429;
        const heading = rateLimited
            ? "Too many activation attempts"
            : "Activation failed";
        const detail = rateLimited
            ? "Wait a few minutes, then try again."
            : "Check the activation token, lane, and parking lot, then try again.";

        activationResult.innerHTML = `
            <section class="kiosk-out kiosk-out--error"
                     role="alert"
                     aria-labelledby="kiosk-activation-error-heading">
                <p class="kiosk-out__status"
                   id="kiosk-activation-error-heading"
                   data-kiosk-activation-heading
                   tabindex="-1">${heading}</p>
                <p class="kiosk-out__detail">${detail}</p>
            </section>
        `;
        activationResult
            .querySelector("[data-kiosk-activation-heading]")
            ?.focus();
    }

    /** Restrict activation lifecycle handling to the activation form. */
    function isActivationRequest(event) {
        return (
            activationForm &&
            (event.detail?.requestConfig?.elt === activationForm ||
                event.target === activationForm)
        );
    }

    if (activationForm && activationResult) {
        ["htmx:responseError", "htmx:sendError", "htmx:timeout"].forEach(
            (name) => {
                document.body.addEventListener(name, (event) => {
                    if (!isActivationRequest(event)) {
                        return;
                    }
                    if (activationToken) {
                        activationToken.value = "";
                    }
                    showActivationFailure(event.detail?.xhr?.status || 0);
                });
            },
        );
    }

    // Activation pages load this script but do not expose scan controls.
    if (
        !panel ||
        !uploader ||
        !form ||
        !processing ||
        !result ||
        !dropzone ||
        !input ||
        !preview ||
        !prompt
    ) {
        return;
    }

    /**
     * Make one panel mode authoritative so every request path restores controls.
     */
    function setPanelMode(mode) {
        const isProcessing = mode === "processing";
        const isSuccess = mode === "success";

        panel.classList.toggle("is-processing", isProcessing);
        panel.classList.toggle("is-success", isSuccess);
        panel.classList.toggle("is-recovery", mode === "recovery");
        panel.setAttribute("aria-busy", String(isProcessing));
        processing.hidden = !isProcessing;
        uploader.hidden = isProcessing || isSuccess;
        form.hidden = isProcessing || isSuccess;
    }

    /** Disable every form control while one single-use nonce is in flight. */
    function setControlsDisabled(disabled) {
        form.querySelectorAll("input, button").forEach((control) => {
            control.disabled = disabled;
        });
    }

    /** Release local blob memory and optionally clear the selected file. */
    function clearPreview({ clearFile = true } = {}) {
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            previewUrl = null;
        }
        preview.removeAttribute("src");
        preview.hidden = true;
        prompt.hidden = false;
        if (reviewCopy) {
            reviewCopy.hidden = true;
        }
        if (submitButton) {
            submitButton.hidden = true;
        }
        if (changeButton) {
            changeButton.hidden = true;
        }
        chooseButtons.forEach((button) => {
            button.hidden = false;
        });
        if (clearFile) {
            input.value = "";
        }
    }

    /** Reveal the selected-state controls only after a valid local preview exists. */
    function showSelectedControls() {
        if (reviewCopy) {
            reviewCopy.hidden = false;
        }
        if (submitButton) {
            submitButton.hidden = false;
        }
        chooseButtons.forEach((button) => {
            button.hidden = true;
        });
        if (changeButton) {
            changeButton.hidden = false;
        }
    }

    /**
     * Render a safe client-owned recovery card when HTMX cannot swap server HTML.
     * Response bodies are deliberately ignored because the kiosk is anonymous.
     */
    function showRequestFailure(status = 0) {
        const rateLimited = status === 429;
        const heading = rateLimited
            ? "Too many scan attempts"
            : "We could not process the scan";
        const detail = rateLimited
            ? "Please wait a moment before trying again."
            : "Try once more. If this continues, contact the gate attendant.";

        result.innerHTML = `
            <section class="kiosk-out kiosk-out--error"
                     data-kiosk-result-mode="recovery"
                     role="alert"
                     aria-labelledby="kiosk-result-heading">
                <p class="kiosk-out__status"
                   id="kiosk-result-heading"
                   data-kiosk-result-heading
                   tabindex="-1">${heading}</p>
                <p class="kiosk-out__detail">${detail}</p>
                <div class="kiosk-out__actions">
                    <button class="btn btn--primary" type="button" data-kiosk-retry>
                        Try again
                    </button>
                    <button class="btn btn--secondary" type="button" data-kiosk-attendant>
                        Call attendant
                    </button>
                </div>
            </section>
        `;
        setControlsDisabled(false);
        setPanelMode("recovery");
        result.querySelector("[data-kiosk-result-heading]")?.focus();
    }

    /** Apply the server-selected success or recovery presentation after a swap. */
    function applyResultMode() {
        const resultCard = result.querySelector("[data-kiosk-result-mode]");
        const mode = resultCard?.dataset.kioskResultMode || "recovery";
        setControlsDisabled(false);
        setPanelMode(mode);
        result.querySelector("[data-kiosk-result-heading]")?.focus();
    }

    /** Display only locally validated JPEG/PNG files; the server still revalidates. */
    function showFile(file) {
        if (!file || !["image/jpeg", "image/png"].includes(file.type)) {
            clearPreview();
            result.innerHTML = `
                <section class="kiosk-out kiosk-out--error"
                         data-kiosk-result-mode="recovery"
                         role="alert">
                    <p class="kiosk-out__status">This image cannot be used</p>
                    <p class="kiosk-out__detail">
                        The file must be a valid JPEG or PNG smaller than 10 MB.
                    </p>
                </section>
            `;
            setPanelMode("recovery");
            return;
        }

        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
        }
        previewUrl = URL.createObjectURL(file);
        preview.onload = () => {
            // Releasing after decode prevents memory growth across repeated scans.
            URL.revokeObjectURL(previewUrl);
            previewUrl = null;
        };
        preview.src = previewUrl;
        preview.hidden = false;
        prompt.hidden = true;
        result.innerHTML = "";
        showSelectedControls();
        setPanelMode("ready");
    }

    /** Restrict HTMX lifecycle handling to this kiosk form. */
    function isKioskRequest(event) {
        return event.detail?.requestConfig?.elt === form || event.target === form;
    }

    input.addEventListener("change", () => showFile(input.files[0]));

    chooseButtons.forEach((button) => {
        button.addEventListener("click", () => input.click());
    });

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

    document.body.addEventListener("htmx:beforeRequest", (event) => {
        if (!isKioskRequest(event)) {
            return;
        }
        setControlsDisabled(true);
        setPanelMode("processing");
    });

    document.body.addEventListener("htmx:afterSwap", (event) => {
        if (event.detail.target !== result) {
            return;
        }
        applyResultMode();
    });

    document.body.addEventListener("htmx:afterRequest", (event) => {
        if (!isKioskRequest(event)) {
            return;
        }
        const nonce = event.detail.xhr.getResponseHeader("X-Kiosk-Nonce");
        const nonceInput = document.getElementById("kiosk-nonce");
        if (nonce && nonceInput) {
            nonceInput.value = nonce;
            // form.reset() restores defaultValue, so rotate both properties.
            nonceInput.defaultValue = nonce;
        }
        setControlsDisabled(false);
        if (panel.classList.contains("is-processing")) {
            showRequestFailure(event.detail.xhr.status);
        }
    });

    ["htmx:responseError", "htmx:sendError", "htmx:timeout"].forEach((name) => {
        document.body.addEventListener(name, (event) => {
            if (!isKioskRequest(event)) {
                return;
            }
            showRequestFailure(event.detail.xhr?.status || 0);
        });
    });

    // Result controls arrive through HTMX, so delegation keeps one stable listener.
    document.addEventListener("click", (event) => {
        const resetButton = event.target.closest?.("[data-kiosk-reset]");
        if (resetButton) {
            result.innerHTML = "";
            form.reset();
            clearPreview();
            setPanelMode("ready");
            input.focus();
            return;
        }

        const retakeButton = event.target.closest?.("[data-kiosk-retake]");
        if (retakeButton) {
            clearPreview();
            setPanelMode("recovery");
            input.focus();
            input.click();
            return;
        }

        const retryButton = event.target.closest?.("[data-kiosk-retry]");
        if (retryButton) {
            if (input.files.length > 0) {
                form.requestSubmit();
            } else {
                input.focus();
                input.click();
            }
            return;
        }

        const attendantButton = event.target.closest?.("[data-kiosk-attendant]");
        if (attendantButton) {
            const detail = result.querySelector(".kiosk-out__detail");
            if (detail) {
                detail.textContent =
                    "Please use the gate intercom or wait for an attendant.";
                detail.focus?.();
            }
        }
    });
})();
