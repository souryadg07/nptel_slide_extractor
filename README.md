# NPTEL Transcript Slide Extractor

Turns an NPTEL transcript PDF into a PDF with just the slides.

## Install

```
pip install pypdf img2pdf pillow opencv-python numpy yt-dlp imageio-ffmpeg
```

## Usage

Slides from the PDF only:

```
python extract_slides.py week1.pdf
```

Output: `week1_slides.pdf`

HD slides using the YouTube lecture videos:

```
python nptel_hd_slides.py week1.pdf https://youtu.be/AAA https://youtu.be/BBB
```
Pass the list of yt video links that contains inside week1.pdf

Output: `week1_clear_slides.pdf`

Put the PDF path in quotes if it has spaces, for example `"week1 (3).pdf"`.
