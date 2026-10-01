#!/usr/bin/env python3
"""
Make a clear slide PDF from an NPTEL transcript PDF + the lecture videos on YouTube.

What it does:
  1. Reads every "(Refer Slide Time: MM:SS)" marker in the transcript, grouped by
     its "Lecture - NN" heading.
  2. Works out which YouTube link is which lecture (lecture number in the video
     title, otherwise title similarity, otherwise the order you gave the links).
  3. Downloads only the picture of each video (no audio) into a cache folder,
     so running it again doesn't download again.
  4. Grabs the frame at every slide timestamp and builds the PDF in transcript order.
     Lectures without a link keep the PDF's own image (cleaned up if low-res).

Install once:
    pip install pypdf img2pdf pillow yt-dlp imageio-ffmpeg opencv-python numpy

Usage (Windows, quote paths that contain spaces):
    python nptel_hd_slides.py "C:\\path\\week1 (3).pdf" https://youtu.be/AAA https://youtu.be/BBB ...

Force a lecture number for a link if the automatic matching gets it wrong:
    python nptel_hd_slides.py "week1 (3).pdf" 3=https://youtu.be/AAA 5=https://youtu.be/BBB

Check the matching without downloading anything:
    python nptel_hd_slides.py "week1 (3).pdf" <links...> --dry-run
"""

import argparse
import difflib
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import img2pdf
from PIL import Image, ImageFilter, ImageOps
from pypdf import PdfReader

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None

LECTURE_RE = re.compile(r"Lecture\s*[-\u2013]\s*(\d+)")
TIME_RE = re.compile(r"Refer Slide Time:\s*((?:\d+:)?\d+:\d+)")
TITLE_NUM_RE = re.compile(r"lec(?:ture)?\s*[-_:#.]?\s*0*(\d{1,3})", re.IGNORECASE)
PAGE_SIZE = (640.5, 360)   # 16:9 page in points
LOWRES_BELOW = 854         # PDF images narrower than this get cleaned up


# ---------------------------------------------------------------- transcript

def to_seconds(stamp):
    secs = 0
    for part in stamp.split(":"):
        secs = secs * 60 + int(part)
    return secs


def read_transcript(pdf_path, min_width=300):
    """Return (slides, lecture_titles). Each slide: lecture, stamp, seconds, pdf_image."""
    reader = PdfReader(pdf_path)
    slides, pdf_images, titles = [], [], {}
    lecture = 0

    for page in reader.pages:
        text = page.extract_text() or ""
        events = [(m.start(), "L", m) for m in LECTURE_RE.finditer(text)]
        events += [(m.start(), "T", m) for m in TIME_RE.finditer(text)]
        for _, kind, m in sorted(events, key=lambda e: e[0]):
            if kind == "L":
                lecture = int(m.group(1))
                rest = text[m.end():].strip().splitlines()
                titles[lecture] = rest[0].strip() if rest else ""
            else:
                slides.append({"lecture": lecture, "stamp": m.group(1),
                               "seconds": to_seconds(m.group(1)), "pdf_image": None})

        for img in page.images:
            try:
                w, h = Image.open(io.BytesIO(img.data)).size
                data = img.data
            except Exception:
                pil = img.image
                w, h = pil.size
                buf = io.BytesIO()
                pil.convert("RGB").save(buf, format="PNG")
                data = buf.getvalue()
            if w >= min_width and w > h:   # landscape screenshot, not the cover or a logo
                pdf_images.append(data)

    if len(pdf_images) == len(slides):
        for s, data in zip(slides, pdf_images):
            s["pdf_image"] = data
    else:
        print(f"[warn] {len(slides)} slide timestamps but {len(pdf_images)} screenshots in the PDF; "
              "lectures without a video may have missing pages")
    return slides, titles


def enhance_lowres(data, target_width=1280):
    """Clean up and enlarge a low-resolution PDF screenshot (used only without a video)."""
    p = Image.open(io.BytesIO(data)).convert("RGB")
    if cv2 is not None:
        bgr = cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 5, 5, 7, 21)
        p = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    p = p.resize((p.width * 2, p.height * 2), Image.BICUBIC)
    p = p.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))
    p = p.resize((target_width, round(p.height * target_width / p.width)), Image.LANCZOS)
    p = p.filter(ImageFilter.UnsharpMask(radius=2, percent=130, threshold=2))
    p = ImageOps.autocontrast(p, cutoff=0.5)
    buf = io.BytesIO()
    p.save(buf, format="JPEG", quality=95, subsampling=0)
    return buf.getvalue()


# ---------------------------------------------------------------- videos

def find_ffmpeg():
    try:
        import imageio_ffmpeg          # bundles an ffmpeg.exe, handy on Windows
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def parse_sources(items):
    """'URL' or 'N=URL' (a local video file also works instead of a URL)."""
    sources = []
    for item in items:
        m = re.match(r"^(\d+)=(.+)$", item)
        forced, target = (int(m.group(1)), m.group(2)) if m else (None, item)
        sources.append({"target": target, "forced": forced, "title": "", "duration": None})
    return sources


def fetch_info(src):
    if os.path.isfile(src["target"]):
        src["title"] = os.path.splitext(os.path.basename(src["target"]))[0]
        return
    import yt_dlp
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as ydl:
        info = ydl.extract_info(src["target"], download=False)
    src["title"] = info.get("title") or ""
    src["duration"] = info.get("duration")


def match_lectures(sources, titles):
    """Fill src['lecture'] for every source. Returns a list of problems (strings)."""
    lectures = sorted(titles)
    taken, problems = set(), []

    def take(src, n, how):
        src["lecture"], src["how"] = n, how
        taken.add(n)

    for src in sources:                                  # 1. forced with N=URL
        if src["forced"] is not None:
            take(src, src["forced"], "you chose it")
    for src in sources:                                  # 2. lecture number in the title
        if "lecture" in src:
            continue
        for m in TITLE_NUM_RE.finditer(src["title"]):
            n = int(m.group(1))
            if n in titles and n not in taken:
                take(src, n, "number in video title")
                break
    for src in sources:                                  # 3. title looks like the transcript's
        if "lecture" in src:
            continue
        best, score = None, 0.0
        for n in lectures:
            if n in taken or not titles[n]:
                continue
            r = difflib.SequenceMatcher(None, titles[n].lower(), src["title"].lower()).ratio()
            if r > score:
                best, score = n, r
        if best is not None and score >= 0.5:
            take(src, best, f"title match {score:.0%}")
    free = [n for n in lectures if n not in taken]       # 4. remaining links in order
    for src in sources:
        if "lecture" not in src:
            if free:
                take(src, free.pop(0), "order of links (please check!)")
            else:
                src["lecture"], src["how"] = None, "no lecture left to match"
                problems.append(f"Could not place: {src['target']}")

    seen = {}
    for src in sources:
        n = src.get("lecture")
        if n is not None:
            if n in seen:
                problems.append(f"Two links were matched to lecture {n}")
            seen[n] = src
    return problems


def download_video(src, cache_dir, ffmpeg):
    if os.path.isfile(src["target"]):
        return src["target"]
    n = src["lecture"]
    done = [p for p in glob.glob(os.path.join(cache_dir, f"lecture_{n:02d}.*"))
            if not p.endswith((".part", ".ytdl"))]
    if done:
        print(f"  lecture {n}: using already downloaded {os.path.basename(done[0])}")
        return done[0]

    import yt_dlp
    opts = {
        # picture only, best up to 1080p (slides don't need more)
        "format": "bv*[height<=1080][ext=mp4]/bv*[height<=1080]/b[height<=1080]/b",
        "outtmpl": os.path.join(cache_dir, f"lecture_{n:02d}.%(ext)s"),
        "noplaylist": True, "quiet": True, "no_warnings": True, "noprogress": True,
    }
    if ffmpeg:
        opts["ffmpeg_location"] = ffmpeg
    print(f"  lecture {n}: downloading ...")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(src["target"], download=True)
        reqs = info.get("requested_downloads") or [{}]
        path = reqs[0].get("filepath") or ydl.prepare_filename(info)
    print(f"  lecture {n}: saved {os.path.basename(path)} ({info.get('height', '?')}p)")
    return path


def grab_frame(ffmpeg, video, seconds, out_path):
    cmd = [ffmpeg, "-loglevel", "error", "-y", "-ss", f"{seconds:.2f}", "-i", video,
           "-frames:v", "1", "-q:v", "1", out_path]
    subprocess.run(cmd, check=False)
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="NPTEL transcript PDF + YouTube links -> clear slide PDF")
    ap.add_argument("transcript", help="NPTEL transcript PDF")
    ap.add_argument("videos", nargs="+", help="YouTube links (or N=link to force lecture N)")
    ap.add_argument("-o", "--output", help="output PDF (default: next to the transcript)")
    ap.add_argument("--offset", type=float, default=1.0,
                    help="seconds after each timestamp to grab the frame (default 1)")
    ap.add_argument("-j", "--workers", type=int, default=10, help="parallel frame grabs (default 10)")
    ap.add_argument("--cache", help="folder for downloaded videos (default: next to the transcript)")
    ap.add_argument("--save-images", metavar="DIR", help="also save every slide as a JPG")
    ap.add_argument("--dry-run", action="store_true", help="only show which link is which lecture")
    args = ap.parse_args()

    if not os.path.isfile(args.transcript):
        sys.exit(f"Transcript not found: {args.transcript}")
    base = os.path.splitext(args.transcript)[0]

    print("Reading transcript ...")
    slides, titles = read_transcript(args.transcript)
    if not slides:
        sys.exit("No '(Refer Slide Time: ...)' markers found in this PDF.")
    counts = {}
    for s in slides:
        counts[s["lecture"]] = counts.get(s["lecture"], 0) + 1
    for n in sorted(counts):
        print(f"  Lecture {n:>2}: {counts[n]:>3} slides  {titles.get(n, '')}")

    print("Looking up videos ...")
    sources = parse_sources(args.videos)
    for src in sources:
        fetch_info(src)
    problems = match_lectures(sources, titles)

    last_stamp = {}
    for s in slides:
        last_stamp[s["lecture"]] = max(last_stamp.get(s["lecture"], 0), s["seconds"])
    for src in sources:
        n = src.get("lecture")
        dur = src["duration"]
        mins = f"{dur // 60}:{dur % 60:02d}" if dur else "?"
        print(f"  Lecture {n}  <-  \"{src['title']}\"  ({mins} long, {src['how']})")
        if n is not None and dur and last_stamp.get(n, 0) > dur + 5:
            problems.append(f"Lecture {n}'s last slide is at {last_stamp[n] // 60}:{last_stamp[n] % 60:02d} "
                            f"but the video is only {mins} long; is it the right video?")
    missing = [n for n in sorted(titles) if n not in {s.get('lecture') for s in sources}]
    if missing:
        print(f"  No link for lecture(s) {', '.join(map(str, missing))}: their PDF images are kept.")
    for p in problems:
        print(f"[check] {p}")
    if args.dry_run:
        return

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        sys.exit("ffmpeg not found. Run: pip install imageio-ffmpeg")

    cache_dir = args.cache or base + "_videos"
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Getting videos (saved in {cache_dir}) ...")
    videos = {}
    for src in sources:
        if src.get("lecture") is not None:
            videos[src["lecture"]] = download_video(src, cache_dir, ffmpeg)

    print("Grabbing slide frames ...")
    tmp = tempfile.mkdtemp(prefix="slides_")

    def job(item):
        i, s = item
        video = videos.get(s["lecture"])
        if video:
            out = os.path.join(tmp, f"{i:04d}.jpg")
            if grab_frame(ffmpeg, video, s["seconds"] + args.offset, out):
                with open(out, "rb") as f:
                    return f.read(), "video"
            print(f"  [warn] lecture {s['lecture']} {s['stamp']}: frame grab failed, using PDF image")
        data = s["pdf_image"]
        if data is None:
            return None, "missing"
        if Image.open(io.BytesIO(data)).width < LOWRES_BELOW:
            return enhance_lowres(data), "pdf (enhanced)"
        return data, "pdf"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(job, enumerate(slides)))

    pages, summary = [], {}
    for s, (data, source) in zip(slides, results):
        summary[source] = summary.get(source, 0) + 1
        if data is not None:
            pages.append(data)
    shutil.rmtree(tmp, ignore_errors=True)
    if not pages:
        sys.exit("No slides could be produced.")

    if args.save_images:
        os.makedirs(args.save_images, exist_ok=True)
        for i, data in enumerate(pages, start=1):
            with open(os.path.join(args.save_images, f"slide_{i:03d}.jpg"), "wb") as f:
                f.write(data)

    out = args.output or base + "_clear_slides.pdf"
    layout = img2pdf.get_layout_fun(PAGE_SIZE, fit=img2pdf.FitMode.into)
    with open(out, "wb") as f:
        f.write(img2pdf.convert(pages, layout_fun=layout))
    print("Slides by source: " + ", ".join(f"{k}: {v}" for k, v in summary.items()))
    print(f"Done: {len(pages)} slides -> {out}")


if __name__ == "__main__":
    main()