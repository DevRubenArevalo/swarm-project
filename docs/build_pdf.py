"""Render docs/REPORT.md to a long-form reading PDF.

This is NOT the submission artifact. The submission is the Apart-template
document at docs/swarm_fidelity_report.pdf (editable source: .docx).
"""
import base64
import mimetypes
import pathlib
import re
import subprocess
import sys

from markdown_it import MarkdownIt

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv)>1 else ".").resolve()
SRC  = ROOT / "docs" / "REPORT.md"
OUT  = pathlib.Path(sys.argv[2] if len(sys.argv)>2 else "docs/REPORT_full.pdf").resolve()

md = MarkdownIt("commonmark").enable("table").enable("strikethrough")
html = md.render(SRC.read_text())

# Inline images so Chrome needs no file access beyond the HTML itself.


def embed(m):
    rel = m.group(1)
    f = (SRC.parent / rel).resolve()
    if not f.exists():
        print("  MISSING IMAGE:", rel)
        return m.group(0)
    mime = mimetypes.guess_type(f.name)[0] or "image/png"
    b64 = base64.b64encode(f.read_bytes()).decode()
    return f'src="data:{mime};base64,{b64}"'
html = re.sub(r'src="([^"]+)"', embed, html)

CSS = """
@page { size: A4; margin: 16mm 17mm 15mm 17mm; }
html { font-size: 9.6pt; }
body { font-family: "Source Serif 4","DejaVu Serif",Georgia,serif; line-height: 1.34;
       color: #111; margin: 0; text-align: justify; hyphens: auto; }
h1 { font-size: 17pt; line-height: 1.15; margin: 0 0 2mm; text-align: left; letter-spacing: -.01em; }
h2 { font-size: 11.5pt; margin: 5mm 0 1.6mm; text-align: left;
     border-bottom: .5pt solid #bbb; padding-bottom: .8mm; page-break-after: avoid; }
h3 { font-size: 10pt; margin: 3.4mm 0 1.2mm; text-align: left; page-break-after: avoid; }
h1 + h3 { font-size: 11pt; font-style: italic; font-weight: 600; color: #333;
          margin: 0 0 2.5mm; border: 0; }
p { margin: 0 0 1.7mm; orphans: 2; widows: 2; }
ul, ol { margin: 0 0 1.9mm; padding-left: 4.6mm; }
li { margin-bottom: .8mm; }
code { font-family: "DejaVu Sans Mono",monospace; font-size: .855em; background: #f2f2f0;
       padding: .3mm .7mm; border-radius: 1.5pt; }
pre { background: #f6f6f4; border: .4pt solid #ddd; border-radius: 2pt; padding: 1.8mm 2.2mm;
      font-size: 7.5pt; line-height: 1.26; overflow: hidden; white-space: pre-wrap;
      word-break: break-word; margin: 0 0 2mm; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 1em; }
table { border-collapse: collapse; width: 100%; font-size: 8.1pt; margin: 1.2mm 0 2.4mm;
        page-break-inside: avoid; }
th, td { border: .4pt solid #c8c8c8; padding: .9mm 1.3mm; text-align: left; vertical-align: top; }
th { background: #efefec; font-weight: 600; }
img { max-width: 92%; display: block; margin: 2mm auto 1mm; page-break-inside: avoid; }
blockquote { margin: 0 0 2mm; padding-left: 2.6mm; border-left: 1.2pt solid #ccc; color: #444; }
hr { border: 0; border-top: .4pt solid #ccc; margin: 3.4mm 0; }
em { color: #222; }
a { color: #14506b; text-decoration: none; word-break: break-all; }
h2[id^="appendix"], h2 { page-break-before: auto; }
"""
doc = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Swarm Fidelity</title><style>{CSS}</style>{html}</html>"""

tmp = OUT.with_suffix(".html")
tmp.write_text(doc)
subprocess.run(["google-chrome", "--headless", "--disable-gpu", "--no-pdf-header-footer",
                f"--user-data-dir={OUT.parent/'chrome'}",
                f"--print-to-pdf={OUT}", tmp.as_uri()],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("wrote", OUT)
