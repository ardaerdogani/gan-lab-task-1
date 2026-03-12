# Overleaf Report Package

This folder contains a self-contained LaTeX project for the report in `reports/report.md`.

## Files

- `main.tex` is the Overleaf entrypoint.
- `sections/` contains the report body split by section.
- `figures/` contains the copied image assets used by the report.
- `data/gan_fid_by_epoch.csv` is used by `pgfplots` to render the FID figure directly in LaTeX.

## Overleaf Setup

1. Create a new Blank Project in Overleaf.
2. Upload the contents of this `reports/overleaf/` folder.
3. Ensure `main.tex` is the main file.
4. Compile with the default Overleaf compiler or set it to `XeLaTeX`.
5. Compile once to generate the table of contents and cross-references.
6. Compile a second time to stabilize page numbers and references.

## Compiler Notes

- The project is intended to compile on Overleaf without requiring a compiler switch.
- `XeLaTeX` is still a good choice if you want native `fontspec` handling.

## Editable Metadata

Update these macros near the top of `main.tex`:

- `\reporttitle`
- `\reportauthor`
- `\reportcourse`
- `\reportinstructor`
- `\reportdate`

The rest of the project is organized section-by-section so later edits can stay localized.
