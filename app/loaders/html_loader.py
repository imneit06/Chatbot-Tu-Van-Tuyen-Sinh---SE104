from pathlib import Path
from bs4 import BeautifulSoup

from .html_helpers import (
    html_table_to_rows,
    select_main_content,
    clean_html_lines,
    extract_html_images,
)


def extract_html_text_tables_images(html_path: Path):
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    main_tag = select_main_content(soup, html_path)
    work_soup = BeautifulSoup(str(main_tag), "lxml")

    tables = []
    for table_tag in work_soup.find_all("table"):
        table_rows = html_table_to_rows(table_tag)
        if table_rows:
            tables.append(table_rows)
        table_tag.decompose()

    images = extract_html_images(html_path, work_soup)

    text = work_soup.get_text("\n", strip=True)
    text = clean_html_lines(text)

    # inline clean_text
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    text = "\n".join(lines)

    return text, tables, images