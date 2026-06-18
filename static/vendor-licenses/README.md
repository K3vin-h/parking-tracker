# Vendored frontend assets

These files are committed so authenticated workflows do not depend on public
CDNs at runtime.

| Asset | Version | Project license | Local path |
|---|---:|---|---|
| HTMX | 2.0.10 | BSD 2-Clause | `static/js/vendor/htmx-2.0.10.min.js` |
| Chart.js | 4.5.1 | MIT | `static/js/vendor/chart-4.5.1.umd.min.js` |
| JetBrains Mono | 2.304 | SIL Open Font License 1.1 | `static/fonts/jetbrains-mono/` |

The adjacent license files are copied from each upstream release. Versioned
filenames make upgrades explicit and prevent browser caches from mixing builds.
