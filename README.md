---
title: Handwriting Studio
emoji: ✍️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Handwriting Studio

Turn your handwriting into a font, then paste any text, including ChatGPT answers full of
LaTeX math, and get pages that look hand-written on ruled paper.

## Run

```
git clone https://github.com/LostRunes/python-scripts.git
cd python-scripts/handwriting_covertor
pip install -r requirements.txt
python app.py            # open http://127.0.0.1:5000
```

The repo ships with two ready handwritings: **y1** (real handwriting, selected by default)
and **demo**. The y1 handwriting is also available as an installable font:
[`fonts/y1-hand.ttf`](fonts/y1-hand.ttf). Double-click it on Windows/macOS and press Install,
and it shows up as "y1 Hand" in Word, Docs and similar apps.

1. **Download the template** (2 pages, 158 characters: A–Z, a–z, 0–9, punctuation, math
   symbols `× ÷ ± ≠ ≤ ≥ ∞ √ ∫ ∑ ∏ ∂ ∇ → ⇒ ∈ ⊂ ∪ ∀ ∃ …` and Greek letters).
   The "2 copies" template gives two variants per character, so repeated letters don't look stamped.
2. **Fill it in** with a dark pen, sitting letters on the solid baseline.
3. **Photograph or scan** each page. Angle and uneven lighting are fine as long as the four
   black corner squares are visible. Pages can be uploaded in any order; the squares at the
   top encode the page number.
4. **Paste text** → *Write it by hand* → download the PDF. You can also download your
   handwriting as an installable **.ttf font**.

## CLI

```
python cli.py template -o template.pdf [--copies 2]
python cli.py extract myhand page1.jpg page2.jpg
python cli.py render myhand samples/gpt_solution.md -o out.pdf --ink black --paper grid
python cli.py font myhand -o myhand.ttf
python cli.py demo
```

## What the renderer understands

- Markdown: `#` headings (bold + underlined), `**bold**`, bullet and numbered lists,
  tables, code blocks, blank lines, `---`
- Math: `$…$`, `\(…\)`, `$$…$$`, `\[…\]`, bare `\begin{align}`
  - `\frac` `\sqrt[n]{}` `^` `_` `\sum` `\int` `\lim` (limits placed under), `\left( \right)`
    (delimiters stretch to fit), `pmatrix/bmatrix/vmatrix`, `cases`, `aligned` (each row
    goes on its own ruled line with the `=` signs lined up), `\boxed`, `\vec \hat \bar \dot`,
    `\cancel`, `\binom`, `\text`, Greek letters and about 150 symbols
  - unicode `x²`, `H₂O`
  - over-long equations wrap before `=` / `+` like hand-written derivations
- Realism: per-letter rotation, size and baseline jitter, random choice between variants,
  line slope and drift, hand-drawn fraction bars, radicals and boxes, ink grain

**Cursive:** tick *Join letters* (or `--cursive`). Lowercase letters are placed close together
and linked with a connecting stroke from each letter's exit point to the next letter's entry point.
It looks best when letters on the template are written with their natural lead-in / lead-out strokes.

Placement is self-correcting: your real baseline, cap height and x-height are measured from
your letters, and punctuation is positioned by type. So it doesn't matter if you wrote on the
dashed line instead of the solid one. A box that clearly holds the wrong thing (e.g. `10` in the
`.` box) is ignored.

Characters you didn't write fall back to a similar character you did write, and otherwise
to a system font.

## Graphs, diagrams, tables, colours

- **Graphs**: fenced ```` ```graph ```` block. One item per line: `y = x^2 - 2x`, `y = sin(x) [red]`,
  implicit curves `x^2 + y^2 = 4`, ranges `x: -2..4` / `x from 0 to 2pi` (π ticks), `y: -1..5`,
  `points: (1,2), (3,0)`, `data: (0,1), (1,3)`, `title:`, `xlabel:`, `ylabel:`, `grid: on`.
  Expressions accept `2x`, `sin x`, `|x|`, `e^x`, `rac{1}{x}`; they run in a sandbox (no Python).
- **Bar / pie charts**: ```` ```bar ```` with `bar: Maths=82, Physics=74`; `pie: A=30, B=70`; mermaid `pie` blocks.
- **Flowcharts**: ```` ```mermaid ```` `graph TD` / `flowchart LR` (boxes `[]`, rounded `()`, decisions `{}`,
  circles `(())`, edge labels `-->|Yes|`, dashed `-.->`), or ```` ```diagram ```` with `Step one -> Step two : label`.
- **ASCII diagrams** in plain code blocks (`+--+`, `|`, `─│┌┐└┘`, `-->`, `v`) are redrawn as hand-drawn lines.
- **Tables**: markdown tables get hand-ruled borders; long cells wrap inside their column.
- **Pen colours**: `[red]..[/red]`, `[color=#aa0033]..[/color]`, `<span style="color:green">..</span>`,
  `	extcolor{red}{x}` / `\color{red}` in maths. They can wrap whole blocks, including graphs and tables.
  The UI has colour buttons that wrap the selected text, plus separate heading and maths inks.
- **Consistency** (0–1, default 0.7) evens out letter heights per group (x-height, ascenders,
  descenders, capitals, digits), slant outliers, ink darkness and pen weight.

Try `samples/figures_demo.md`.

## Layout

```
hw/charset.py     template character set + sheet geometry
hw/template.py    printable template
hw/figures.py     graphs, charts, flowcharts (hand-drawn)
hw/extract.py     photo → marker detection → perspective fix → glyphs
hw/profile.py     glyph library, fallbacks, preview sheet
hw/fontexport.py  glyphs → TrueType font
hw/mathparse.py   LaTeX → node tree
hw/layout.py      TeX-like boxes drawn with your glyphs
hw/render.py      markdown parsing, line breaking, pages, paper, ink
hw/synth.py       fake "photographed" template (demo + tests)
```
