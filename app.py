"""Web UI:  python app.py  ->  http://127.0.0.1:5000"""

import io
import os
import re
import shutil
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
    return jsonify([Profile(os.path.join(PROFILES, n)).summary() | {"id": n} for n in list_profiles(PROFILES)])


@app.post("/api/profiles")
def create_profile():
    name = slug(request.form.get("name"))
    if not name:
        return jsonify(error="Give the handwriting a name"), 400
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
    shutil.rmtree(profile_dir(name))
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
    prof = Profile(profile_dir(data.get("profile")))
    pages = render_text(text, prof, Options.from_dict(data.get("options")))
    job = uuid.uuid4().hex[:12]
    d = os.path.join(OUTPUT, job)
    os.makedirs(d)
    names = []
    for i, p in enumerate(pages):
        fn = f"page_{i + 1}.png"
        p.save(os.path.join(d, fn), optimize=True)
        names.append(f"/output/{job}/{fn}")
    save_pdf(pages, os.path.join(d, "handwritten.pdf"))
    return jsonify(pages=names, pdf=f"/output/{job}/handwritten.pdf")


@app.get("/output/<job>/<fn>")
def output(job, fn):
    if not re.fullmatch(r"[0-9a-f]{12}", job):
        abort(404)
    return send_from_directory(os.path.join(OUTPUT, job), fn, max_age=0)


if __name__ == "__main__":
    if not list_profiles(PROFILES):
        print("building demo handwriting profile...")
        build_demo()
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)))
