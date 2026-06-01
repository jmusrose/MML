#!/usr/bin/env python3
"""Clean Food-101 HTML pages into compact text files.

Default input:
    data/texts_html/{class_name}/{sample_id}.html

Default output:
    data/texts_clean/{class_name}/{sample_id}.txt
"""

import argparse
import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path


SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "form",
    "button",
    "input",
    "select",
    "textarea",
    "nav",
    "footer",
    "header",
    "aside",
}

TEXT_BREAK_TAGS = {
    "title",
    "h1",
    "h2",
    "h3",
    "h4",
    "p",
    "li",
    "td",
    "th",
    "figcaption",
    "blockquote",
    "br",
}

NOISE_ATTR_WORDS = (
    "ads",
    "advert",
    "archive",
    "breadcrumb",
    "comment",
    "footer",
    "header",
    "menu",
    "nav",
    "newsletter",
    "pagination",
    "promo",
    "related",
    "reply",
    "search",
    "share",
    "sidebar",
    "social",
    "subscribe",
    "toc",
    "widget",
)

BOILERPLATE_RE = re.compile(
    r"\b("
    r"advertisement|archive|categories|click here|comment|copyright|"
    r"email address|follow us|jump to|leave a reply|login|navigation|"
    r"newsletter|posted in|privacy policy|related posts|rss|search|"
    r"share this|sign up|subscribe|terms of use|trackback"
    r")\b",
    re.IGNORECASE,
)


class FoodHTMLTextExtractor(HTMLParser):
    """Small HTML text extractor tuned for recipe/blog pages."""

    def __init__(self, skip_noisy_attrs=True):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip_depth = 0
        self.skip_noisy_attrs = skip_noisy_attrs

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if self.skip_depth:
            self.skip_depth += 1
            return
        if tag in SKIP_TAGS or (
            self.skip_noisy_attrs and self._attrs_look_noisy(attrs)
        ):
            self.skip_depth = 1
            return
        if tag in TEXT_BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.skip_depth:
            self.skip_depth -= 1
            return
        if tag.lower() in TEXT_BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)

    @staticmethod
    def _attrs_look_noisy(attrs):
        attr_text = " ".join(str(value).lower() for _, value in attrs if value)
        attr_words = set(re.findall(r"[a-z0-9]+", attr_text))
        return any(word in attr_words for word in NOISE_ATTR_WORDS)

    def text(self):
        return "".join(self.parts)


def clean_line(line):
    """Normalize one extracted line and drop obvious webpage boilerplate."""
    line = unescape(line)
    line = re.sub(r"https?://\S+|www\.\S+", " ", line)
    line = re.sub(r"\S+@\S+", " ", line)
    line = re.sub(r"\[[0-9]+\]", " ", line)
    line = re.sub(r"\s+", " ", line).strip(" -_|")

    if len(line) < 3:
        return ""
    if BOILERPLATE_RE.search(line) and len(line.split()) <= 30:
        return ""
    if len(set(line)) <= 2:
        return ""
    return line


def limit_words(text, max_words):
    """Return text capped to max_words while preserving word order."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def extract_clean_lines(html_text, skip_noisy_attrs=True):
    """Extract readable food text from one HTML document."""
    parser = FoodHTMLTextExtractor(skip_noisy_attrs=skip_noisy_attrs)
    parser.feed(html_text)

    seen = set()
    cleaned_lines = []
    for raw_line in parser.text().splitlines():
        line = clean_line(raw_line)
        key = line.lower()
        if line and key not in seen:
            seen.add(key)
            cleaned_lines.append(line)

    return cleaned_lines


def clean_html_text(html_text, max_words=300, min_words_before_fallback=30):
    """Extract readable food text from one HTML document."""
    cleaned_lines = extract_clean_lines(html_text, skip_noisy_attrs=True)
    text = " ".join(cleaned_lines)

    if len(text.split()) < min_words_before_fallback:
        fallback_lines = extract_clean_lines(html_text, skip_noisy_attrs=False)
        fallback_text = " ".join(fallback_lines)
        if len(fallback_text.split()) > len(text.split()):
            text = fallback_text

    return limit_words(text, max_words)


def output_path_for_html(html_path, html_root, out_root):
    """Map an HTML input path to the matching cleaned .txt output path."""
    rel_path = html_path.relative_to(html_root)
    return (out_root / rel_path).with_suffix(".txt")


def iter_html_files(html_root):
    """Yield HTML files under class directories."""
    yield from sorted(html_root.rglob("*.html"))


def clean_html_tree(html_root, out_root, max_words=300, limit=None, overwrite=False):
    """Clean all HTML files and return a compact processing report."""
    html_root = Path(html_root)
    out_root = Path(out_root)
    report = {
        "html_root": str(html_root),
        "out_root": str(out_root),
        "max_words": max_words,
        "files_seen": 0,
        "files_skipped": 0,
        "files_written": 0,
        "empty_outputs": 0,
        "examples": [],
    }

    for html_path in iter_html_files(html_root):
        if limit is not None and report["files_seen"] >= limit:
            break
        report["files_seen"] += 1
        out_path = output_path_for_html(html_path, html_root, out_root)

        if out_path.exists() and not overwrite:
            report["files_skipped"] += 1
            continue

        html_text = html_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_html_text(html_text, max_words=max_words)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(cleaned + "\n", encoding="utf-8")

        report["files_written"] += 1
        if not cleaned:
            report["empty_outputs"] += 1
        if len(report["examples"]) < 10:
            report["examples"].append(
                {
                    "input": str(html_path),
                    "output": str(out_path),
                    "words": len(cleaned.split()),
                }
            )
        if report["files_seen"] % 5000 == 0:
            print(
                f"Seen {report['files_seen']} files; "
                f"written {report['files_written']}; "
                f"skipped {report['files_skipped']}."
            )

    return report


def parse_args():
    data_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Clean Food-101 HTML pages into compact text files."
    )
    parser.add_argument("--html-root", type=Path, default=data_dir / "texts_html")
    parser.add_argument("--out-root", type=Path, default=data_dir / "texts_clean")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=data_dir / "text_cleaning_report.json",
    )
    parser.add_argument("--max-words", type=int, default=300)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional file limit for quick smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing cleaned files instead of resuming.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = clean_html_tree(
        args.html_root,
        args.out_root,
        max_words=args.max_words,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    args.report_path.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"Cleaned {report['files_written']} files to {args.out_root} "
        f"({report['empty_outputs']} empty outputs)."
    )
    print(f"Wrote report to {args.report_path}")


if __name__ == "__main__":
    main()
