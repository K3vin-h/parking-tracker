# Plate Font

`synthetic_data.py` looks for `plate_font.ttf` in this directory.

If missing, it falls back to PIL's built-in bitmap font (functional but low-fidelity).

## Recommended font

**Liberation Mono Bold** — Apache 2.0 license, close to highway signage typefaces.

Download from the [Red Hat liberation-fonts releases](https://github.com/liberationfonts/liberation-fonts/releases),
extract, and copy `LiberationMono-Bold.ttf` here as `plate_font.ttf`:

```bash
cp LiberationMono-Bold.ttf apps/cv/training/assets/plate_font.ttf
```

The `.ttf` file is gitignored (binary asset, no license issues in the repo itself).
