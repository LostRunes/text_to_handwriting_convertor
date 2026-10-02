"""Command-line access to the whole pipeline.

  python cli.py template  [--copies 2] [-o template.pdf]
  python cli.py extract   NAME photo1.jpg photo2.jpg [--append]
  python cli.py demo                      # build the 'demo' profile from a synthetic sheet
  python cli.py render    NAME input.txt [-o out.pdf] [--ink blue] [--paper ruled] ...
  python cli.py font      NAME [-o name.ttf]
"""

import argparse
import os
import sys

from hw import charset as C
from hw.extract import extract_profile
from hw.fontexport import export_ttf
from hw.profile import Profile
from hw.render import Options, render_text, save_pdf
from hw.template import save_template_pdf

ROOT = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.join(ROOT, "profiles")


def build_demo(name="demo", log=print):
    from hw.synth import synth_sources
    return extract_profile(synth_sources(), os.path.join(PROFILES, name), name, log=log)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Handwriting -> font -> handwritten pages")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template")
    t.add_argument("--copies", type=int, default=1)
    t.add_argument("-o", "--out", default="template.pdf")

    e = sub.add_parser("extract")
    e.add_argument("name")
    e.add_argument("images", nargs="+")
    e.add_argument("--append", action="store_true")

    sub.add_parser("demo")

    r = sub.add_parser("render")
    r.add_argument("name")
    r.add_argument("input")
    r.add_argument("-o", "--out", default="handwritten.pdf")
    for k, v in vars(Options()).items():
        if k in ("page_w", "page_h"):
            continue
        r.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)

    f = sub.add_parser("font")
    f.add_argument("name")
    f.add_argument("-o", "--out")

    a = p.parse_args(argv)
    if a.cmd == "template":
        save_template_pdf(a.out, a.copies)
        print(f"wrote {a.out} ({len(C.pages_for(a.copies))} pages)")
    elif a.cmd == "extract":
        srcs = [(open(path, "rb").read(), os.path.basename(path)) for path in a.images]
        rep = extract_profile(srcs, os.path.join(PROFILES, a.name), a.name, append=a.append)
        print(f"{rep['glyph_count']} characters; missing: {rep['missing'] or 'none'}")
        for err in rep["errors"]:
            print("ERROR", err)
    elif a.cmd == "demo":
        rep = build_demo()
        print(f"demo profile: {rep['glyph_count']} characters; errors: {rep['errors']}")
    elif a.cmd == "render":
        prof = Profile(os.path.join(PROFILES, a.name))
        opts = Options.from_dict({k: getattr(a, k) for k in vars(Options()) if hasattr(a, k)})
        with open(a.input, encoding="utf-8") as fh:
            pages = render_text(fh.read(), prof, opts)
        if a.out.lower().endswith(".pdf"):
            save_pdf(pages, a.out)
        else:
            stem, ext = os.path.splitext(a.out)
            for i, pg in enumerate(pages):
                pg.save(f"{stem}_{i + 1}{ext}" if len(pages) > 1 else a.out)
        print(f"wrote {a.out} ({len(pages)} page(s))")
    elif a.cmd == "font":
        prof = Profile(os.path.join(PROFILES, a.name))
        out = a.out or f"{a.name}.ttf"
        export_ttf(prof, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
