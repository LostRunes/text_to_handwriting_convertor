"""Web UI:  python app.py  ->  http://127.0.0.1:5000

Hosting settings (environment variables):
  PROTECTED_PROFILES  comma-separated profiles nobody can delete or overwrite
                      through the website (default: "y1,demo")
  ALLOW_UPLOADS       "0" to stop visitors from creating new handwriting profiles
  MAX_TEXT_CHARS      longest text accepted for one render (default 40000)
"""

import io
import os
import re
import shutil
import time
import uuid

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from cli import PROFILES, ROOT, build_demo
from hw.extract import extract_profile
from hw.fontexport import export_ttf
from hw.profile import Profile, list_profiles
from hw.render import Options, render_text, save_pdf
from hw.template import save_template_pdf

OUTPUT = os.path.join(ROOT, "output")
app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024

PROTECTED = {p.strip().lower() for p in os.environ.get("PROTECTED_PROFILES", "y1,demo").split(",") if p.strip()}
ALLOW_UPLOADS = os.environ.get("ALLOW_UPLOADS", "1") != "0"
MAX_TEXT = int(os.environ.get("MAX_TEXT_CHARS", 40000))
OUTPUT_TTL = 3600          # rendered pages are kept for an hour


def cleanup_output():
    if not os.path.isdir(OUTPUT):
        return
    cutoff = time.time() - OUTPUT_TTL
    for job in os.listdir(OUTPUT):
        d = os.path.join(OUTPUT, job)
        try:
            if os.path.getmtime(d) < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def slug(name):
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", (name or "").strip()).strip("-").lower()
    return s[:40]


def profile_dir(name):
    s = slug(name)
    if not s or not os.path.exists(os.path.join(PROFILES, s, "profile.json")):
        abort(404, "unknown profile")
    return os.path.join(PROFILES, s)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/template.pdf")
def template():
    copies = max(1, min(int(request.args.get("copies", 1)), 4))
    buf = io.BytesIO()
    save_template_pdf(buf, copies)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True,
                     download_name=f"handwriting_template_x{copies}.pdf")


@app.get("/api/profiles")
def profiles():
    return jsonify([Profile(os.path.join(PROFILES, n)).summary() | {"id": n, "locked": n in PROTECTED}
                    for n in list_profiles(PROFILES)])


@app.get("/api/config")
def config():
    return jsonify(allow_uploads=ALLOW_UPLOADS)


@app.post("/api/profiles")
def create_profile():
    if not ALLOW_UPLOADS:
        return jsonify(error="Uploading new handwriting is turned off on this site"), 403
    name = slug(request.form.get("name"))
    if not name:
        return jsonify(error="Give the handwriting a name"), 400
    if name in PROTECTED:
        return jsonify(error=f'"{name}" is protected - pick another name'), 403
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="Upload at least one photo or scan"), 400
    append = request.form.get("append") == "1"
    out = os.path.join(PROFILES, name)
    if not append and os.path.exists(out):
        shutil.rmtree(out)
    rep = extract_profile([(f.read(), f.filename) for f in files], out, name, append=append)
    if rep["glyph_count"] == 0:
        shutil.rmtree(out, ignore_errors=True)
        return jsonify(error="No characters found. " + " ".join(rep["errors"])), 400
    return jsonify(rep | {"id": name})


@app.delete("/api/profiles/<name>")
def delete_profile(name):
    d = profile_dir(name)
    if slug(name) in PROTECTED:
        return jsonify(error=f'"{slug(name)}" is protected and can\'t be deleted from the website'), 403
    shutil.rmtree(d)
    return jsonify(ok=True)


@app.get("/api/profiles/<name>/sheet.png")
def sheet(name):
    buf = io.BytesIO()
    Profile(profile_dir(name)).contact_sheet().save(buf, "PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png", max_age=0)


@app.get("/api/profiles/<name>/font.ttf")
def font(name):
    prof = Profile(profile_dir(name))
    buf = io.BytesIO()
    export_ttf(prof, buf)
    buf.seek(0)
    return send_file(buf, mimetype="font/ttf", as_attachment=True, download_name=f"{slug(name)}-hand.ttf")


@app.post("/api/render")
def render():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text.strip():
        return jsonify(error="Paste some text first"), 400
    if len(text) > MAX_TEXT:
        return jsonify(error=f"Text is too long (limit {MAX_TEXT} characters) - split it into parts"), 400
    cleanup_output()
    prof = Profile(profile_dir(data.get("profile")))
    notes = []
    pages = render_text(text, prof, Options.from_dict(data.get("options")), notes)
    job = uuid.uuid4().hex[:12]
    d = os.path.join(OUTPUT, job)
    os.makedirs(d)
    names = []
    for i, p in enumerate(pages):
        fn = f"page_{i + 1}.png"
        p.save(os.path.join(d, fn), optimize=True)
        names.append(f"/output/{job}/{fn}")
    save_pdf(pages, os.path.join(d, "handwritten.pdf"))
    return jsonify(pages=names, pdf=f"/output/{job}/handwritten.pdf", notes=notes)


@app.get("/output/<job>/<fn>")
def output(job, fn):
    if not re.fullmatch(r"[0-9a-f]{12}", job):
        abort(404)
    return send_from_directory(os.path.join(OUTPUT, job), fn, max_age=0)


if __name__ == "__main__":
    if not list_profiles(PROFILES):
        print("building demo handwriting profile...")
        build_demo()
    app.run(debug=False, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)))
