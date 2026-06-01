#!/usr/bin/env python3
"""Tests for Food-101 HTML text cleaning utilities."""

from pathlib import Path

from data.clean_food_texts import clean_html_text, clean_html_tree, output_path_for_html


def test_clean_html_text_removes_page_boilerplate_and_limits_words():
    html = """
    <html>
      <body>
        <header>Subscribe to our newsletter Search Navigation</header>
        <nav>Home About Contact Archive</nav>
        <article>
          <h1>Apple pie</h1>
          <p>Apple pie is a fruit pie in which the principal filling ingredient is apple.</p>
          <p>It is often served with whipped cream, ice cream, or cheddar cheese.</p>
        </article>
        <aside>Related posts Comments Leave a reply</aside>
        <footer>Privacy policy Copyright 2026</footer>
      </body>
    </html>
    """

    cleaned = clean_html_text(html, max_words=18)

    assert "Apple pie" in cleaned
    assert "principal filling ingredient" in cleaned
    assert "Subscribe" not in cleaned
    assert "Comments" not in cleaned
    assert len(cleaned.split()) <= 18


def test_output_path_for_html_preserves_class_directory_and_uses_txt_suffix():
    html_root = Path("data/texts_html")
    out_root = Path("data/texts_clean")
    html_path = html_root / "pizza" / "pizza_123.html"

    assert output_path_for_html(html_path, html_root, out_root) == (
        out_root / "pizza" / "pizza_123.txt"
    )


def test_clean_html_tree_skips_existing_outputs_by_default(tmp_path):
    html_root = tmp_path / "texts_html"
    out_root = tmp_path / "texts_clean"
    html_path = html_root / "pizza" / "pizza_123.html"
    out_path = out_root / "pizza" / "pizza_123.txt"
    html_path.parent.mkdir(parents=True)
    out_path.parent.mkdir(parents=True)
    html_path.write_text("<html><body><p>fresh pizza text</p></body></html>")
    out_path.write_text("existing cleaned text\n")

    report = clean_html_tree(html_root, out_root)

    assert report["files_seen"] == 1
    assert report["files_skipped"] == 1
    assert report["files_written"] == 0
    assert out_path.read_text() == "existing cleaned text\n"
