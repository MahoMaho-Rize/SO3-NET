"""Convert a Markdown file to PDF via Python-Markdown + headless Chrome.

Usage:
    python3 scripts/md_to_pdf.py <input.md> [output.pdf]
"""

import sys
import subprocess
import tempfile
from pathlib import Path

import markdown


CSS = """
<style>
@page { size: A4; margin: 18mm 16mm; }
body {
  font-family: -apple-system, "Segoe UI", "Helvetica Neue", Helvetica,
               "Source Han Sans SC", "Noto Sans CJK SC", "PingFang SC",
               "Microsoft YaHei", sans-serif;
  font-size: 10.5pt; line-height: 1.45; color: #1b1b1b;
  max-width: 800px; margin: 0 auto;
}
h1 { font-size: 19pt; border-bottom: 2px solid #333; padding-bottom: 4px;
     margin-top: 24px; }
h2 { font-size: 15pt; border-bottom: 1px solid #999; padding-bottom: 3px;
     margin-top: 20px; }
h3 { font-size: 12.5pt; margin-top: 16px; }
h4 { font-size: 11pt; margin-top: 12px; color: #333; }
p  { margin: 6px 0; }
code {
  font-family: "SFMono-Regular", "Monaco", "Menlo", "DejaVu Sans Mono",
               Consolas, monospace;
  font-size: 9.5pt;
  background: #f3f3f3; padding: 1px 4px; border-radius: 3px;
}
pre {
  background: #f6f8fa; padding: 10px 12px; border-radius: 4px;
  border: 1px solid #e1e4e8; overflow-x: auto;
  font-size: 8.8pt; line-height: 1.35;
  page-break-inside: avoid;
}
pre code { background: transparent; padding: 0; font-size: 8.8pt; }
table {
  border-collapse: collapse; margin: 10px 0; font-size: 9.5pt;
  page-break-inside: avoid;
}
th, td { border: 1px solid #bbb; padding: 4px 8px; text-align: left; }
th { background: #eaeaea; }
blockquote {
  margin: 8px 0; padding: 4px 12px;
  border-left: 3px solid #888; color: #555; background: #fafafa;
}
hr { border: 0; border-top: 1px solid #ccc; margin: 18px 0; }
ul, ol { margin: 6px 0; padding-left: 24px; }
li { margin: 2px 0; }
</style>
"""


def main():
    if len(sys.argv) < 2:
        print("usage: md_to_pdf.py <input.md> [output.pdf]")
        sys.exit(1)
    md_path = Path(sys.argv[1]).resolve()
    if len(sys.argv) >= 3:
        pdf_path = Path(sys.argv[2]).resolve()
    else:
        pdf_path = md_path.with_suffix(".pdf")

    text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        text,
        extensions=[
            "fenced_code", "tables", "codehilite",
            "toc", "sane_lists", "nl2br",
        ],
        extension_configs={"codehilite": {"guess_lang": False, "noclasses": True}},
    )
    html_full = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{md_path.stem}</title>{CSS}</head>
<body>{html_body}</body></html>"""

    with tempfile.NamedTemporaryFile("w", suffix=".html",
                                     delete=False, encoding="utf-8") as f:
        f.write(html_full)
        html_file = f.name

    try:
        cmd = [
            "google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
            f"--print-to-pdf={pdf_path}",
            f"--print-to-pdf-no-header",
            f"file://{html_file}",
        ]
        print("Running:", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print("STDERR:", r.stderr)
            sys.exit(r.returncode)
        print(f"Wrote {pdf_path}")
    finally:
        Path(html_file).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
