#!/usr/bin/env python3
"""
Extract only the slide screenshots from a lecture-transcript PDF
and put them, one per page, into a new PDF.

How it decides what counts as a "screenshot":
  * the image must be landscape (wider than tall)  -> skips the portrait cover image
  * the image must be at least --min-width pixels wide -> skips tiny logos/icons
  * identical images are kept only once

The original JPEG data is embedded as-is (no re-compression), so quality is preserved.

Requirements:
    pip install pypdf img2pdf pillow

Usage:
    python extract_slides.py test.pdf
    python extract_slides.py test.pdf -o slides.pdf --min-width 400
    python extract_slides.py test.pdf --save-images slides_folder
"""

import argparse
import hashlib
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from pypdf import PdfReader
from PIL import Image, ImageFilter, ImageOps

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None

LOWRES_BELOW = 854   # screenshots narrower than this get cleaned up and enlarged

try:
    import img2pdf
except ImportError:
    img2pdf = None



def enhance_lowres(data, target_width):
    """Clean up and enlarge a low-resolution screenshot so the text is readable."""
    p = Image.open(io.BytesIO(data)).convert("RGB")

    # 1. Remove JPEG block noise first, so it doesn't get enlarged too
    if cv2 is not None:
        bgr = cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 5, 5, 7, 21)
        p = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # 2. Enlarge in two steps, sharpening after each (smoother letter edges)
    p = p.resize((p.width * 2, p.height * 2), Image.BICUBIC)
    p = p.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))
    p = p.resize((target_width, round(p.height * target_width / p.width)), Image.LANCZOS)
    p = p.filter(ImageFilter.UnsharpMask(radius=2, percent=130, threshold=2))

    # 3. Slightly darker text and cleaner background
    p = ImageOps.autocontrast(p, cutoff=0.5)

    buf = io.BytesIO()
    p.save(buf, format="JPEG", quality=95, subsampling=0)
    return buf.getvalue(), p.size


def _scan_pages(pdf_path, page_numbers, min_width, allow_portrait, upscale_to=1280):
    """Worker: runs in its own process and opens its own copy of the PDF."""
    reader = PdfReader(pdf_path)
    found = []
    for page_no in page_numbers:
        page = reader.pages[page_no - 1]
        try:
            images = page.images
        except Exception as e:
            print(f"  [warn] page {page_no}: could not read images ({e})")
            continue

        for img in images:
            data = img.data
            try:
                pil = Image.open(io.BytesIO(data))
                w, h = pil.size
                fmt = pil.format or "PNG"
            except Exception:
                pil = img.image
                w, h = pil.size
                buf = io.BytesIO()
                pil.convert("RGB").save(buf, format="PNG")
                data = buf.getvalue()
                fmt = "PNG"

            if w < min_width:
                continue
            if not allow_portrait and w <= h:
                continue
            # Low-resolution screenshot: clean it up and enlarge it
            if upscale_to and w < LOWRES_BELOW:
                data, (w, h) = enhance_lowres(data, upscale_to)
                fmt = "JPEG"

            ext = "jpg" if fmt.lower() in ("jpeg", "jpg") else fmt.lower()
            found.append((page_no, data, ext, w, h))
    return found


def extract_screenshots(pdf_path, min_width=300, allow_portrait=False, workers=10, upscale_to=1280):
    num_pages = len(PdfReader(pdf_path).pages)
    workers = max(1, min(workers, num_pages))

    # One contiguous block of pages per worker, so page order is kept
    size = -(-num_pages // workers)  # ceiling division
    chunks = [list(range(s, min(s + size, num_pages + 1)))
              for s in range(1, num_pages + 1, size)]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_scan_pages, pdf_path, c, min_width, allow_portrait, upscale_to)
                   for c in chunks]
        batches = [f.result() for f in futures]

    # Remove duplicates here in the main process (keeps the first occurrence)
    seen = set()
    results = []
    for batch in batches:
        for page_no, data, ext, w, h in batch:
            digest = hashlib.md5(data).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            results.append((page_no, data, ext))
            print(f"  page {page_no:>3}: {w}x{h} {ext}")
    return results

def build_pdf(images, out_path):
    if img2pdf is not None:
        # Lossless: JPEGs are embedded directly; each page sized to its image
        with open(out_path, "wb") as f:
            # Every page gets the same size; bigger images just get more detail per inch
            layout = img2pdf.get_layout_fun((640.5, 360), fit=img2pdf.FitMode.into)
            f.write(img2pdf.convert([data for _, data, _ in images], layout_fun=layout))
    else:
        # Fallback with Pillow only (re-encodes images)
        pil_images = [Image.open(io.BytesIO(d)).convert("RGB") for _, d, _ in images]
        pil_images[0].save(out_path, save_all=True, append_images=pil_images[1:],
                           resolution=150)


def main():
    ap = argparse.ArgumentParser(description="Extract slide screenshots from a PDF into a new PDF.")
    ap.add_argument("input", help="input PDF")
    ap.add_argument("-o", "--output", help="output PDF (default: <input>_slides.pdf)")
    ap.add_argument("--min-width", type=int, default=300,
                    help="ignore images narrower than this many pixels (default 300)")
    ap.add_argument("--allow-portrait", action="store_true",
                    help="also keep portrait images (by default only landscape ones are kept)")
    ap.add_argument("--save-images", metavar="DIR",
                    help="also save each screenshot as an image file in DIR")
    ap.add_argument("-j", "--workers", type=int, default=10,
                    help="number of parallel processes (default 10)")
    ap.add_argument("--upscale-to", type=int, default=1280,
                    help="width to enlarge low-res screenshots to (default 1280, 0 = off)")
    args = ap.parse_args()

    out = args.output or os.path.splitext(args.input)[0] + "_slides.pdf"

    print(f"Scanning {args.input} ...")
    images = extract_screenshots(args.input, args.min_width, args.allow_portrait, args.workers, args.upscale_to)

    if not images:
        sys.exit("No screenshots found. Try lowering --min-width or using --allow-portrait.")

    if args.save_images:
        os.makedirs(args.save_images, exist_ok=True)
        for i, (page_no, data, ext) in enumerate(images, start=1):
            path = os.path.join(args.save_images, f"slide_{i:03d}_p{page_no}.{ext}")
            with open(path, "wb") as f:
                f.write(data)
        print(f"Saved {len(images)} image files to {args.save_images}/")

    build_pdf(images, out)
    print(f"Done: {len(images)} screenshots -> {out}")


if __name__ == "__main__":
    main()
