"use strict";

const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const SCRIPT_PATH = path.resolve(__dirname, "../../../../static/js/plates.js");

test("plate deletion is cancelled when the resident rejects confirmation", () => {
    let submitListener;
    let prevented = false;
    const plateText = { textContent: "ABC123" };
    const row = {
        querySelector(selector) {
            return selector === ".plate-list__text" ? plateText : null;
        },
    };
    const form = {
        addEventListener(name, listener) {
            if (name === "submit") {
                submitListener = listener;
            }
        },
        closest(selector) {
            return selector === ".plate-list__item" ? row : null;
        },
    };

    global.document = {
        querySelectorAll(selector) {
            return selector === "[data-plate-delete]" ? [form] : [];
        },
    };
    global.window = {
        confirm(message) {
            assert.equal(message, "Remove ABC123?");
            return false;
        },
    };

    delete require.cache[require.resolve(SCRIPT_PATH)];
    require(SCRIPT_PATH);
    submitListener({
        preventDefault() {
            prevented = true;
        },
    });

    assert.equal(prevented, true);
});
