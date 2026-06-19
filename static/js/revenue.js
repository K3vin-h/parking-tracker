/**
 * Render exact-money analytics returned by the staff-only revenue endpoint.
 */
(() => {
    "use strict";

    const root = document.querySelector("[data-revenue-root]");
    if (!root || typeof Chart === "undefined") {
        return;
    }

    const filter = document.querySelector("[data-revenue-filter]");
    const error = document.querySelector("[data-chart-error]");
    const context = document.getElementById("revenue-chart");
    const breakdownContext = document.getElementById("breakdown-chart");
    let revenueChart;
    let breakdownChart;
    let payload;
    let breakdown = "lot";

    const gridColor = "#2e3039";
    const textColor = "#a1a1aa";
    const money = (value) =>
        new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(
            Number(value)
        );

    function commonOptions() {
        // WHY disable animation: filtering feels immediate and screenshots stay deterministic.
        return {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: textColor }, grid: { display: false } },
                y: {
                    beginAtZero: true,
                    ticks: { color: textColor, callback: money },
                    grid: { color: gridColor },
                },
            },
        };
    }

    function renderBreakdown() {
        breakdownChart?.destroy();
        const isLot = breakdown === "lot";
        const rows = isLot ? payload.by_lot : payload.hourly;
        breakdownChart = new Chart(breakdownContext, {
            type: "bar",
            data: {
                labels: rows.map((row) =>
                    isLot ? row.lot_name : `${String(row.hour).padStart(2, "0")}:00`
                ),
                datasets: [{
                    data: rows.map((row) => Number(row.revenue)),
                    backgroundColor: isLot
                        ? ["#3b82f6", "#22c55e", "#eab308", "#a855f7"]
                        : "#3b82f6",
                    borderRadius: 5,
                }],
            },
            options: commonOptions(),
        });
        document.querySelector("[data-breakdown-title]").textContent =
            isLot ? "By lot" : "By hour";
    }

    function render(data) {
        payload = data;
        document.querySelector("[data-revenue-total]").textContent =
            money(data.summary.total_revenue);
        document.querySelector("[data-session-total]").textContent =
            data.summary.session_count.toLocaleString();
        document.querySelector("[data-average-charge]").textContent =
            money(data.summary.average_charge);

        revenueChart?.destroy();
        revenueChart = new Chart(context, {
            type: "line",
            data: {
                labels: data.daily.map((row) => row.date),
                datasets: [{
                    data: data.daily.map((row) => Number(row.revenue)),
                    borderColor: "#3b82f6",
                    backgroundColor: "rgba(59,130,246,.18)",
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.16,
                }],
            },
            options: commonOptions(),
        });
        renderBreakdown();
    }

    async function load() {
        const query = new URLSearchParams(new FormData(filter));
        const selectedPreset = filter.querySelector('button[name="range"].is-selected');
        const hasCustomDates = query.get("start_date") && query.get("end_date");
        if (hasCustomDates && filter.dataset.customRequested === "true") {
            query.set("range", "custom");
            query.set("start", query.get("start_date"));
            query.set("end", query.get("end_date"));
        } else if (selectedPreset) {
            query.set("range", selectedPreset.value);
        }
        query.delete("start_date");
        query.delete("end_date");
        const response = await fetch(`${root.dataset.endpoint}?${query}`, {
            headers: { Accept: "application/json" },
            credentials: "same-origin",
        });
        if (!response.ok) {
            throw new Error("Revenue endpoint failed");
        }
        render(await response.json());
    }

    filter.addEventListener("submit", (event) => {
        event.preventDefault();
        const submitter = event.submitter;
        filter.dataset.customRequested = String(!submitter?.name);
        filter.querySelectorAll('button[name="range"]').forEach((button) => {
            button.classList.toggle("is-selected", button === submitter);
        });
        load().catch(() => { error.hidden = false; });
    });

    document.querySelectorAll("[data-breakdown]").forEach((button) => {
        button.addEventListener("click", () => {
            breakdown = button.dataset.breakdown;
            document.querySelectorAll("[data-breakdown]").forEach((item) => {
                item.classList.toggle("is-active", item === button);
            });
            if (payload) {
                renderBreakdown();
            }
        });
    });

    load().catch(() => { error.hidden = false; });
})();
