// Exercise a failed request followed by a successful retry without a DOM library.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

test('successful revenue retry clears the previous error', async () => {
    const error = { hidden: true };
    let submit;
    let requests = 0;
    const filter = {
        dataset: {},
        querySelector: () => null,
        querySelectorAll: () => [],
        addEventListener: (_, callback) => { submit = callback; },
    };
    const total = {};
    const document = {
        querySelector: (selector) => ({
            '[data-revenue-root]': { dataset: { endpoint: '/revenue/' } },
            '[data-revenue-filter]': filter,
            '[data-chart-error]': error,
            '[data-revenue-total]': total,
        })[selector] || {},
        getElementById: () => ({}),
        querySelectorAll: () => [],
    };
    const data = {
        summary: { total_revenue: '12.50', session_count: 1, average_duration_seconds: 60 },
        daily: [], by_lot: [], hourly: [],
    };
    vm.runInNewContext(fs.readFileSync(path.resolve(__dirname, '../../../../static/js/revenue.js'), 'utf8'), {
        document, URLSearchParams, Intl,
        FormData: class { *[Symbol.iterator]() {} },
        Chart: class { destroy() {} },
        console: { error() {} },
        fetch: async () => ({ ok: ++requests > 1, json: async () => data }),
    });
    await new Promise(setImmediate);
    assert.equal(error.hidden, false);
    submit({ preventDefault() {} });
    await new Promise(setImmediate);
    assert.equal(total.textContent, '$12.50');
    assert.equal(error.hidden, true);
});
