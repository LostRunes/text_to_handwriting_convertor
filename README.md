# Handwriting Studio

Turn your handwriting into a font, then paste any text, including ChatGPT answers full of
LaTeX math, and get pages that look hand-written on ruled paper.

## Run

```
pip install -r requirements.txt
python app.py            # http://127.0.0.1:5000  (builds a 'demo' handwriting on first run)
```

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

Characters you didn't write fall back to a similar character you did write, and otherwise
to a system font.

## Layout

```
hw/charset.py     template character set + sheet geometry
hw/template.py    printable template
hw/extract.py     photo → marker detection → perspective fix → glyphs
hw/profile.py     glyph library, fallbacks, preview sheet
hw/fontexport.py  glyphs → TrueType font
hw/mathparse.py   LaTeX → node tree
hw/layout.py      TeX-like boxes drawn with your glyphs
hw/render.py      markdown parsing, line breaking, pages, paper, ink
hw/synth.py       fake "photographed" template (demo + tests)
```
