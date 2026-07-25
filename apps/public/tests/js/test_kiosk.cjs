"use strict";

const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const SCRIPT_PATH = path.resolve(__dirname, "../../../../static/js/kiosk.js");

function classList() {
    const values = new Set();
    return {
        contains(value) {
            return values.has(value);
        },
        add(value) {
            values.add(value);
        },
        remove(value) {
            values.delete(value);
        },
        toggle(value, force) {
            if (force) {
                values.add(value);
            } else {
                values.delete(value);
            }
        },
    };
}

function element(overrides = {}) {
    const attributes = new Map();
    return {
        hidden: false,
        disabled: false,
        value: "",
        defaultValue: "",
        files: [],
        innerHTML: "",
        classList: classList(),
        dataset: {},
        addEventListener() {},
        removeAttribute(name) {
            attributes.delete(name);
        },
        setAttribute(name, value) {
            attributes.set(name, String(value));
        },
        getAttribute(name) {
            return attributes.get(name);
        },
        querySelector() {
            return null;
        },
        querySelectorAll() {
            return [];
        },
        focus() {},
        reset() {},
        ...overrides,
    };
}

function loadKioskScript({ resultMode = null, activation = false } = {}) {
    const bodyListeners = new Map();
    const documentListeners = new Map();
    const heading = element({
        focus() {
            heading.focused = true;
        },
    });
    const resultCard = resultMode
        ? element({ dataset: { kioskResultMode: resultMode } })
        : null;
    const result = element({
        querySelector(selector) {
            if (selector === "[data-kiosk-result-mode]") {
                return resultCard;
            }
            if (selector === "[data-kiosk-result-heading]") {
                return heading;
            }
            return null;
        },
    });
    const nonce = element({
        value: "server-rendered-nonce",
        defaultValue: "server-rendered-nonce",
    });
    const controls = [element(), element()];
    const form = element({
        querySelectorAll() {
            return controls;
        },
        reset() {
            nonce.value = nonce.defaultValue;
        },
    });
    const panel = element();
    const processing = element({ hidden: true });
    const input = element();
    const preview = element({ hidden: true });
    const prompt = element();
    const reviewCopy = element({ hidden: true });
    const dropzone = element();
    const chooseButtons = [element()];
    const submitButton = element({ hidden: true });
    const changeButton = element({ hidden: true });
    const uploader = element();
    const activationHeading = element({
        focus() {
            activationHeading.focused = true;
        },
    });
    const activationResult = element({
        querySelector(selector) {
            return selector === "[data-kiosk-activation-heading]"
                ? activationHeading
                : null;
        },
    });
    const activationToken = element({ value: "shared-secret-token" });
    const activationForm = activation
        ? element({
              querySelector(selector) {
                  return selector === 'input[name="token"]'
                      ? activationToken
                      : null;
              },
          })
        : null;

    const selectors = new Map([
        ["[data-dropzone]", dropzone],
        ["[data-dropzone-preview]", preview],
        ["[data-dropzone-prompt]", prompt],
        ["[data-kiosk-review-copy]", reviewCopy],
        ["[data-kiosk-submit]", submitButton],
        ["[data-kiosk-uploader]", uploader],
        ["[data-kiosk-activation-form]", activationForm],
        ["[data-kiosk-activation-result]", activation ? activationResult : null],
    ]);
    const ids = new Map([
        ["kiosk-panel", panel],
        ["kiosk-form", form],
        ["kiosk-processing", processing],
        ["kiosk-result", result],
        ["kiosk-image", input],
        ["kiosk-nonce", nonce],
    ]);

    global.document = {
        body: {
            addEventListener(name, listener) {
                bodyListeners.set(name, listener);
            },
        },
        addEventListener(name, listener) {
            documentListeners.set(name, listener);
        },
        querySelector(selector) {
            if (
                activation &&
                ![
                    "[data-kiosk-activation-form]",
                    "[data-kiosk-activation-result]",
                ].includes(selector)
            ) {
                return null;
            }
            return selectors.get(selector) || null;
        },
        querySelectorAll(selector) {
            if (activation) {
                return [];
            }
            return selector === "[data-kiosk-choose-file]" ? chooseButtons : [];
        },
        getElementById(id) {
            if (activation) {
                return null;
            }
            return ids.get(id) || null;
        },
    };
    global.URL = {
        createObjectURL() {
            return "blob:preview";
        },
        revokeObjectURL() {},
    };
    global.DataTransfer = class {
        constructor() {
            this.files = [];
            this.items = {
                add: (file) => {
                    this.files = [file];
                },
            };
        }
    };

    delete require.cache[require.resolve(SCRIPT_PATH)];
    require(SCRIPT_PATH);

    return {
        bodyListeners,
        documentListeners,
        panel,
        form,
        controls,
        processing,
        result,
        heading,
        nonce,
        activationForm,
        activationToken,
        activationResult,
        activationHeading,
    };
}

test("beforeRequest replaces the uploader with the processing state", () => {
    const fixture = loadKioskScript();
    const listener = fixture.bodyListeners.get("htmx:beforeRequest");

    assert.equal(typeof listener, "function");
    listener({ detail: { requestConfig: { elt: fixture.form } } });

    assert.equal(fixture.panel.getAttribute("aria-busy"), "true");
    assert.equal(fixture.form.hidden, true);
    assert.equal(fixture.processing.hidden, false);
    assert.ok(fixture.controls.every((control) => control.disabled));
});

test("afterSwap gives a successful result focused replacement", () => {
    const fixture = loadKioskScript({ resultMode: "success" });
    const listener = fixture.bodyListeners.get("htmx:afterSwap");

    assert.equal(typeof listener, "function");
    listener({ detail: { target: fixture.result } });

    assert.equal(fixture.form.hidden, true);
    assert.equal(fixture.processing.hidden, true);
    assert.equal(fixture.panel.classList.contains("is-success"), true);
    assert.equal(fixture.heading.focused, true);
});

test("responseError restores a safe recovery state without echoing server HTML", () => {
    const fixture = loadKioskScript();
    const listener = fixture.bodyListeners.get("htmx:responseError");

    assert.equal(typeof listener, "function");
    listener({
        detail: {
            requestConfig: { elt: fixture.form },
            xhr: {
                status: 500,
                responseText: "<script>unsafe server response</script>",
            },
        },
    });

    assert.equal(fixture.form.hidden, false);
    assert.equal(fixture.panel.classList.contains("is-recovery"), true);
    assert.match(fixture.result.innerHTML, /could not process the scan/i);
    assert.doesNotMatch(fixture.result.innerHTML, /unsafe server response/i);
});

test("rotated nonce survives resetting for the next vehicle", () => {
    const fixture = loadKioskScript();
    const afterRequest = fixture.bodyListeners.get("htmx:afterRequest");
    const click = fixture.documentListeners.get("click");

    afterRequest({
        detail: {
            requestConfig: { elt: fixture.form },
            xhr: {
                status: 200,
                getResponseHeader(name) {
                    return name === "X-Kiosk-Nonce" ? "rotated-nonce" : null;
                },
            },
        },
    });
    click({
        target: {
            closest(selector) {
                return selector === "[data-kiosk-reset]" ? {} : null;
            },
        },
    });

    assert.equal(fixture.nonce.value, "rotated-nonce");
    assert.equal(fixture.nonce.defaultValue, "rotated-nonce");
});

test("activation errors are visible without echoing the server response", () => {
    const fixture = loadKioskScript({ activation: true });
    const listener = fixture.bodyListeners.get("htmx:responseError");

    assert.equal(typeof listener, "function");
    listener({
        detail: {
            requestConfig: { elt: fixture.activationForm },
            xhr: {
                status: 403,
                responseText: "<script>unsafe activation response</script>",
            },
        },
    });

    assert.match(fixture.activationResult.innerHTML, /activation failed/i);
    assert.doesNotMatch(
        fixture.activationResult.innerHTML,
        /unsafe activation response/i,
    );
    assert.equal(fixture.activationToken.value, "");
    assert.equal(fixture.activationHeading.focused, true);
});
