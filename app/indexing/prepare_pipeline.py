    from pathlib import Path
    import json
    import uuid
    from urllib.parse import urlparse, unquote

    import requests
    import pdfplumber
    import pytesseract

    from PIL import Image, ImageOps
    from bs4 import BeautifulSoup
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from app.core.config import (
        PROCESS_PDF,
        PROCESS_HTML,
        PDF_DIR,
        HTML_DIR,
        PROCESSED_DIR,
        DOWNLOADED_IMAGES_DIR,
        PARENTS_PATH,
        CHILDREN_PATH,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )


    # =========================
    # Config
    # =========================

    ID_KEY = "doc_id"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOADED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


    # =========================
    # Common helpers
    # =========================

    def clean_text(text: str) -> str:
        if not text:
            return ""

        lines = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                lines.append(line)

        return "\n".join(lines)


    def clean_cell(cell) -> str:
        if cell is None:
            return ""

        return str(cell).replace("\n", " ").strip()


    def doc_to_json(doc: Document) -> dict:
        return {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }


    def write_jsonl(path: Path, records: list[dict]):
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


    def make_child_content_prefix(
        source: str,
        location,
        child_type: str,
        file_type: str,
        extra_note: str = "",
    ) -> str:
        prefix = [
            f"[Nguồn file] {source}",
            f"[File type] {file_type}",
            f"[Vị trí/trang] {location}",
            f"[Loại child] {child_type}",
        ]

        if extra_note:
            prefix.append(f"[Ghi chú] {extra_note}")

        return "\n".join(prefix) + "\n\n"


    # =========================
    # Table helpers
    # =========================

    def table_to_markdown(table) -> str:
        lines = []

        for row in table:
            cleaned_row = [clean_cell(cell) for cell in row]
            lines.append("| " + " | ".join(cleaned_row) + " |")

        return "\n".join(lines)


    def make_table_summary(table, file_name: str, location, table_idx: int) -> str:
        if not table:
            return ""

        header = [clean_cell(cell) for cell in table[0]]
        header_text = ", ".join([h for h in header if h])

        if not header_text:
            header_text = "không xác định rõ header"

        return (
            f"Bảng số {table_idx} trong file {file_name}, vị trí/trang {location}. "
            f"Các cột gồm: {header_text}. "
            f"Bảng này có thể chứa thông tin tuyển sinh hoặc đào tạo như mã ngành, "
            f"tên ngành, chỉ tiêu, điểm chuẩn, học phí, phương thức xét tuyển, "
            f"tổ hợp môn, môn học, số tín chỉ hoặc ghi chú."
        )


    def make_table_row_docs(
        table,
        parent_id: str,
        source: str,
        location,
        table_idx: int,
        file_type: str,
    ):
        child_docs = []

        if not table or len(table) < 2:
            return child_docs

        header = [clean_cell(cell) for cell in table[0]]

        for row_idx, row in enumerate(table[1:], start=1):
            cells = [clean_cell(cell) for cell in row]

            pairs = []

            for h, c in zip(header, cells):
                if h or c:
                    label = h if h else "Cột không tên"
                    pairs.append(f"{label}: {c}")

            content = " | ".join(pairs)

            if not content.strip():
                continue

            page_content = (
                make_child_content_prefix(
                    source=source,
                    location=location,
                    child_type="table_row",
                    file_type=file_type,
                    extra_note=(
                        "Dòng bảng có thể chứa môn học, số tín chỉ, khối kiến thức, "
                        "điểm chuẩn, chỉ tiêu hoặc học phí."
                    ),
                )
                + f"Dòng bảng tuyển sinh/đào tạo: {content}"
            )

            child_docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        ID_KEY: parent_id,
                        "source": source,
                        "location": location,
                        "child_type": "table_row",
                        "table_index": table_idx,
                        "row_index": row_idx,
                        "file_type": file_type,
                    },
                )
            )

        return child_docs


    def add_table_children(
        child_records,
        tables,
        parent_id: str,
        source: str,
        file_name: str,
        location,
        file_type: str,
    ):
        table_blocks = []

        for table_idx, table in enumerate(tables, start=1):
            if not table:
                continue

            markdown_table = table_to_markdown(table)

            if markdown_table.strip():
                table_blocks.append(
                    f"[BẢNG {table_idx} - {file_name} - vị trí/trang {location}]\n"
                    f"{markdown_table}"
                )

            summary = make_table_summary(
                table=table,
                file_name=file_name,
                location=location,
                table_idx=table_idx,
            )

            if summary.strip():
                page_content = (
                    make_child_content_prefix(
                        source=source,
                        location=location,
                        child_type="table_summary",
                        file_type=file_type,
                        extra_note="Tóm tắt bảng để hỗ trợ truy vấn RAG.",
                    )
                    + summary
                )

                summary_doc = Document(
                    page_content=page_content,
                    metadata={
                        ID_KEY: parent_id,
                        "source": source,
                        "location": location,
                        "child_type": "table_summary",
                        "table_index": table_idx,
                        "file_type": file_type,
                    },
                )

                child_records.append(doc_to_json(summary_doc))

            row_docs = make_table_row_docs(
                table=table,
                parent_id=parent_id,
                source=source,
                location=location,
                table_idx=table_idx,
                file_type=file_type,
            )

            for row_doc in row_docs:
                child_records.append(doc_to_json(row_doc))

        return table_blocks


    # =========================
    # HTML table helpers
    # =========================

    def html_table_to_rows(table_tag):
        rows = []

        for tr in table_tag.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            row = [cell.get_text(" ", strip=True) for cell in cells]

            if row:
                rows.append(row)

        return rows


    # =========================
    # HTML main content selector
    # =========================

    def tag_desc(tag) -> str:
        if tag is None:
            return "None"

        tag_id = tag.get("id", "")
        tag_class = " ".join(tag.get("class", [])) if tag.get("class") else ""

        return f"<{tag.name} id='{tag_id}' class='{tag_class}'>"


    def score_main_candidate(tag) -> int:
        text = tag.get_text("\n", strip=True)
        lower = text.lower()

        useful_keywords = [
            "cử nhân",
            "kỹ sư",
            "ngành",
            "giới thiệu chung",
            "mục tiêu đào tạo",
            "đối tượng tuyển sinh",
            "quy chế đào tạo",
            "chuẩn đầu ra",
            "chương trình đào tạo",
            "khung chương trình",
            "khối kiến thức",
            "tỷ lệ các khối kiến thức",
            "tín chỉ",
            "cơ sở ngành",
            "chuyên ngành",
            "tốt nghiệp",
            "môn học",
            "thực tập",
            "đồ án",
        ]

        noise_keywords = [
            "đăng nhập",
            "tên truy cập",
            "mật khẩu",
            "tìm kiếm",
            "liên kết website",
            "webmail",
            "website trường",
            "website môn học",
            "ctdt khóa",
            "ctdt khoá",
            "lịch phòng",
            "what is the",
        ]

        useful_score = sum(1 for kw in useful_keywords if kw in lower)
        noise_score = sum(1 for kw in noise_keywords if kw in lower)

        table_score = len(tag.find_all("table"))
        image_score = len(tag.find_all("img"))

        score = (
            useful_score * 3000
            + table_score * 1500
            + image_score * 300
            + min(len(text), 20000)
            - noise_score * 2000
        )

        return score


    def select_main_content(soup: BeautifulSoup, html_path: Path):
        """
        Chọn vùng nội dung chính theo nhiều fallback.
        Không phụ thuộc cứng vào một format HTML.
        """

        selectors = [
            "main",
            "article",
            "div[role='main']",

            ".main-content",
            ".region-content",
            ".content-main",
            ".page-content",
            ".node",
            ".node-content",
            ".field-name-body",
            ".field-items",
            ".field-item",

            ".col-md-9",
            ".col-sm-9",
            ".col-lg-9",
            ".col-md-8",
            ".col-sm-8",
            ".col-lg-8",

            "#content",
            "#main-content",
            "#content-area",
            "#main",
        ]

        candidates = []

        # Tầng 1: selector phổ biến
        for selector in selectors:
            for tag in soup.select(selector):
                text = tag.get_text("\n", strip=True)
                if len(text) >= 200:
                    candidates.append(tag)

        # Tầng 2: tìm các div/section/article có keyword nội dung chính
        important_keywords = [
            "chương trình đào tạo",
            "mục tiêu đào tạo",
            "chuẩn đầu ra",
            "tỷ lệ các khối kiến thức",
            "khung chương trình",
            "cơ sở ngành",
            "chuyên ngành",
            "tín chỉ",
        ]

        for tag in soup.find_all(["div", "section", "article"]):
            text = tag.get_text("\n", strip=True).lower()

            if len(text) >= 300 and any(kw in text for kw in important_keywords):
                candidates.append(tag)

        # Tầng 3: fallback body
        if not candidates:
            body = soup.find("body")
            if body:
                print(f"[WARNING] Không tìm thấy main content rõ ràng, dùng <body>: {html_path}")
                return body

            print(f"[WARNING] Không có body, dùng toàn bộ soup: {html_path}")
            return soup

        unique_candidates = []
        seen = set()

        for tag in candidates:
            obj_id = id(tag)
            if obj_id not in seen:
                seen.add(obj_id)
                unique_candidates.append(tag)

        best = max(unique_candidates, key=score_main_candidate)
        best_score = score_main_candidate(best)
        best_text_len = len(best.get_text("\n", strip=True))

        print(f"[HTML] File: {html_path}")
        print(f"[HTML] Chọn main content: {tag_desc(best)}")
        print(f"[HTML] Score: {best_score}")
        print(f"[HTML] Text length: {best_text_len}")
        print(f"[HTML] Tables: {len(best.find_all('table'))}")
        print(f"[HTML] Images: {len(best.find_all('img'))}")

        if best_score < 3000:
            print(f"[WARNING] Main content score thấp, nên inspect lại file này: {html_path}")

        return best


    def clean_html_lines(text: str) -> str:
        noise_exact_lines = {
            "Lịch",
            "TKB",
            "Lịch phòng",
            "Tìm kiếm",
            "Đăng Nhập",
            "Đăng nhập",
            "Tên truy cập",
            "Mật khẩu",
            "Mật khẩu chứng thực",
            "Dùng tài khoản chứng thực",
            "Liên kết",
            "Website trường",
            "Webmail",
            "Website môn học",
            "Tài khoản chứng thực",
            "Diễn đàn sinh viên",
            "Microsoft Azure",
            "Hệ chính quy",
            "Danh mục môn học",
            "Tóm tắt môn học",
            "Đề án mở ngành",
            "*",
        }

        noise_prefixes = [
            "CTDT Khóa",
            "CTDT Khoá",
            "CTDT Khoa",
            "What is the",
            "-----",
            "--",
        ]

        cleaned = []
        seen = set()

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            if line in noise_exact_lines:
                continue

            if any(line.startswith(prefix) for prefix in noise_prefixes):
                continue

            if len(line) <= 2 and not line.isalnum():
                continue

            if line in seen:
                continue

            seen.add(line)
            cleaned.append(line)

        return "\n".join(cleaned)


    # =========================
    # Image helpers
    # =========================

    def get_img_src(img_tag) -> str:
        candidates = [
            img_tag.get("src"),
            img_tag.get("data-src"),
            img_tag.get("data-original"),
            img_tag.get("data-lazy-src"),
        ]

        for item in candidates:
            if item and str(item).strip():
                return str(item).strip()

        srcset = img_tag.get("srcset")
        if srcset:
            first_src = srcset.split(",")[0].strip().split(" ")[0]
            return first_src

        return ""


    def get_nearby_caption(img_tag) -> str:
        figure = img_tag.find_parent("figure")
        if figure:
            caption = figure.find("figcaption")
            if caption:
                return caption.get_text(" ", strip=True)

        parent = img_tag.parent
        if parent:
            return parent.get_text(" ", strip=True)[:500]

        return ""


    def is_remote_url(src: str) -> bool:
        parsed = urlparse(src)
        return parsed.scheme in ["http", "https"]


    def is_data_uri(src: str) -> bool:
        return src.startswith("data:")


    def resolve_local_image_path(html_path: Path, src: str) -> Path | None:
        if not src:
            return None

        if is_data_uri(src) or is_remote_url(src):
            return None

        clean_src = src.split("?")[0].split("#")[0]
        clean_src = unquote(clean_src)

        img_path = (html_path.parent / clean_src).resolve()

        if img_path.exists():
            return img_path

        return None


    def download_remote_image(src: str, file_name_prefix: str) -> Path | None:
        try:
            response = requests.get(src, timeout=10)
            response.raise_for_status()

            parsed = urlparse(src)
            suffix = Path(parsed.path).suffix.lower()

            if suffix not in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
                suffix = ".png"

            out_path = DOWNLOADED_IMAGES_DIR / f"{file_name_prefix}_{uuid.uuid4().hex}{suffix}"

            with open(out_path, "wb") as f:
                f.write(response.content)

            return out_path

        except Exception as e:
            print(f"Không tải được ảnh remote: {src}. Lỗi: {e}")
            return None


    def preprocess_image_for_ocr(image_path: Path) -> Image.Image:
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert("L")

        width, height = image.size

        if width < 1600:
            scale = 1600 / max(width, 1)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size)

        return image


    def ocr_image(image_path: Path) -> str:
        try:
            image = preprocess_image_for_ocr(image_path)
            text = pytesseract.image_to_string(image, lang="vie+eng")
            return clean_text(text)

        except Exception as e:
            print(f"OCR lỗi với ảnh {image_path}: {e}")
            return ""


    def image_metadata_has_relevant_keyword(image_info: dict) -> bool:
        src = image_info.get("src", "").lower()
        alt = image_info.get("alt", "").lower()
        title = image_info.get("title", "").lower()
        caption = image_info.get("caption", "").lower()

        text = " ".join([src, alt, title, caption])

        skip_keywords = [
            "logo",
            "icon",
            "facebook",
            "youtube",
            "zalo",
            "banner",
            "background",
            "avatar",
            "sprite",
            "menu",
            "search",
            "footer",
            "header",
        ]

        for kw in skip_keywords:
            if kw in text:
                return False

        keep_keywords = [
            "tuyển sinh",
            "tuyen sinh",
            "điểm chuẩn",
            "diem chuan",
            "chỉ tiêu",
            "chi tieu",
            "học phí",
            "hoc phi",
            "chương trình",
            "chuong trinh",
            "đào tạo",
            "dao tao",
            "khung",
            "môn học",
            "mon hoc",
            "tín chỉ",
            "tin chi",
            "ngành",
            "nganh",
            "xét tuyển",
            "xet tuyen",
            "ctdt",
            "curriculum",
        ]

        return any(kw in text for kw in keep_keywords)


    def is_large_image_candidate(image_info: dict) -> bool:
        local_path = image_info.get("local_path", "")

        if not local_path:
            return False

        try:
            img = Image.open(local_path)
            w, h = img.size

            if w < 500 or h < 250:
                return False

            return True

        except Exception:
            return False


    def is_useful_ocr_text(ocr_text: str) -> bool:
        text = ocr_text.lower()

        if len(text.strip()) < 80:
            return False

        useful_keywords = [
            "tuyển sinh",
            "tuyen sinh",
            "điểm chuẩn",
            "diem chuan",
            "chỉ tiêu",
            "chi tieu",
            "học phí",
            "hoc phi",
            "chương trình",
            "chuong trinh",
            "đào tạo",
            "dao tao",
            "khung chương trình",
            "môn học",
            "mon hoc",
            "tín chỉ",
            "tin chi",
            "cơ sở ngành",
            "co so nganh",
            "đại cương",
            "dai cuong",
            "chuyên ngành",
            "chuyen nganh",
            "tốt nghiệp",
            "tot nghiep",
            "chuẩn đầu ra",
            "chuan dau ra",
            "mục tiêu đào tạo",
            "muc tieu dao tao",
        ]

        keyword_count = sum(1 for kw in useful_keywords if kw in text)

        return keyword_count >= 2


    def extract_html_images(html_path: Path, soup: BeautifulSoup):
        images = []

        for idx, img_tag in enumerate(soup.find_all("img"), start=1):
            src = get_img_src(img_tag)
            alt = img_tag.get("alt", "").strip()
            title = img_tag.get("title", "").strip()
            caption = get_nearby_caption(img_tag)

            image_path = resolve_local_image_path(html_path, src)

            if image_path is None and is_remote_url(src):
                image_path = download_remote_image(
                    src=src,
                    file_name_prefix=html_path.stem,
                )

            image_info = {
                "image_index": idx,
                "src": src,
                "alt": alt,
                "title": title,
                "caption": caption,
                "local_path": str(image_path) if image_path else "",
            }

            if src or alt or title or caption or image_path:
                images.append(image_info)

            img_tag.decompose()

        return images


    def add_image_children(
        child_records,
        images,
        parent_id: str,
        source: str,
        file_name: str,
        location,
        file_type: str,
    ):
        image_blocks = []

        for image in images:
            image_idx = image.get("image_index")
            src = image.get("src", "")
            alt = image.get("alt", "")
            title = image.get("title", "")
            caption = image.get("caption", "")
            local_path = image.get("local_path", "")

            metadata_relevant = image_metadata_has_relevant_keyword(image)
            large_candidate = is_large_image_candidate(image)

            if not metadata_relevant and not large_candidate:
                continue

            ocr_text = ""

            if local_path:
                ocr_text = ocr_image(Path(local_path))

            ocr_relevant = is_useful_ocr_text(ocr_text)

            if not metadata_relevant and not ocr_relevant:
                continue

            summary_content = (
                make_child_content_prefix(
                    source=source,
                    location=location,
                    child_type="image_summary",
                    file_type=file_type,
                    extra_note="Tóm tắt metadata ảnh trong HTML/PDF.",
                )
                + f"Hình ảnh số {image_idx} trong file {file_name}. "
                + f"Alt: {alt}. "
                + f"Title: {title}. "
                + f"Caption/ngữ cảnh gần ảnh: {caption}. "
                + f"Đường dẫn ảnh: {src}."
            )

            summary_doc = Document(
                page_content=summary_content,
                metadata={
                    ID_KEY: parent_id,
                    "source": source,
                    "location": location,
                    "child_type": "image_summary",
                    "image_index": image_idx,
                    "image_src": src,
                    "image_local_path": local_path,
                    "file_type": file_type,
                },
            )

            child_records.append(doc_to_json(summary_doc))

            image_block = f"[HÌNH {image_idx} - {file_name}]\n{summary_content}"

            if ocr_relevant:
                ocr_content = (
                    make_child_content_prefix(
                        source=source,
                        location=location,
                        child_type="image_ocr_text",
                        file_type=file_type,
                        extra_note=(
                            "Nội dung được OCR từ ảnh, có thể chứa khung chương trình, "
                            "môn học hoặc số tín chỉ."
                        ),
                    )
                    + f"Nội dung OCR từ hình ảnh số {image_idx} "
                    + f"trong file {file_name}:\n{ocr_text}"
                )

                ocr_doc = Document(
                    page_content=ocr_content,
                    metadata={
                        ID_KEY: parent_id,
                        "source": source,
                        "location": location,
                        "child_type": "image_ocr_text",
                        "image_index": image_idx,
                        "image_src": src,
                        "image_local_path": local_path,
                        "file_type": file_type,
                    },
                )

                child_records.append(doc_to_json(ocr_doc))
                image_block += "\n\n[OCR TEXT]\n" + ocr_text

            image_blocks.append(image_block)

        return image_blocks


    # =========================
    # HTML extraction
    # =========================

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

        return clean_text(text), tables, images


    # =========================
    # Parent + text chunk helpers
    # =========================

    def add_parent(
        parent_records,
        parent_id: str,
        content: str,
        source: str,
        location,
        parent_type: str,
        file_type: str,
    ):
        if not content.strip():
            return

        parent_records.append(
            {
                "doc_id": parent_id,
                "page_content": content,
                "metadata": {
                    "source": source,
                    "location": location,
                    "parent_type": parent_type,
                    "file_type": file_type,
                },
            }
        )


    def add_text_chunks(
        child_records,
        splitter,
        text: str,
        parent_id: str,
        source: str,
        location,
        file_type: str,
        extra_metadata: dict | None = None,
    ):
        if not text.strip():
            return

        metadata = {
            ID_KEY: parent_id,
            "source": source,
            "location": location,
            "child_type": "text_chunk",
            "file_type": file_type,
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        text_docs = splitter.split_documents(
            [
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            ]
        )

        for doc in text_docs:
            prefix = make_child_content_prefix(
                source=source,
                location=location,
                child_type="text_chunk",
                file_type=file_type,
                extra_note="Đoạn văn bản được tách từ một section của PDF/HTML.",
            )

            doc.page_content = prefix + doc.page_content
            child_records.append(doc_to_json(doc))


    # =========================
    # Process PDF
    # =========================

    def process_pdfs(parent_records, child_records, splitter):
        pdf_files = list(PDF_DIR.rglob("*.pdf"))

        for pdf_path in pdf_files:
            print(f"Đang xử lý PDF: {pdf_path}")

            with pdfplumber.open(pdf_path) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    parent_id = str(uuid.uuid4())

                    text = clean_text(page.extract_text() or "")
                    tables = page.extract_tables()

                    table_blocks = add_table_children(
                        child_records=child_records,
                        tables=tables,
                        parent_id=parent_id,
                        source=str(pdf_path),
                        file_name=pdf_path.name,
                        location=page_idx,
                        file_type="pdf",
                    )

                    parent_content = (
                        text
                        + "\n\n"
                        + "\n\n".join(table_blocks)
                    )

                    add_parent(
                        parent_records=parent_records,
                        parent_id=parent_id,
                        content=parent_content,
                        source=str(pdf_path),
                        location=page_idx,
                        parent_type="pdf_page",
                        file_type="pdf",
                    )

                    add_text_chunks(
                        child_records=child_records,
                        splitter=splitter,
                        text=text,
                        parent_id=parent_id,
                        source=str(pdf_path),
                        location=page_idx,
                        file_type="pdf",
                    )


    # =========================
    # Process HTML
    # =========================


    def is_section_heading(line: str) -> bool:
        line = line.strip()
        lower = line.lower()

        if not line:
            return False

        heading_keywords = [
            "giới thiệu chung",
            "mục tiêu đào tạo",
            "đối tượng tuyển sinh",
            "quy chế đào tạo",
            "chuẩn đầu ra",
            "chương trình đào tạo",
            "tỷ lệ các khối kiến thức",
            "khung chương trình",
            "khối kiến thức",
            "cơ sở ngành",
            "chuyên ngành",
            "tốt nghiệp",
            "thực tập",
            "đồ án",
        ]

        if any(keyword in lower for keyword in heading_keywords):
            return True

        # Dạng: 1. GIỚI THIỆU CHUNG, 5.1 Tỷ lệ...
        import re
        if re.match(r"^\d+(\.\d+)*\.?\s+.{3,}$", line):
            return True

        # Dạng heading viết hoa tương đối dài
        letters = [c for c in line if c.isalpha()]
        if len(line) >= 8 and letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio >= 0.7:
                return True

        return False


    def guess_major_name(text: str, source: str = "") -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        for line in lines[:30]:
            lower = line.lower()

            if "cử nhân ngành" in lower or "kỹ sư ngành" in lower:
                return line

            if line.lower().startswith("ngành "):
                return line

        # fallback theo tên file
        name = Path(source).stem
        return name


    def split_text_into_sections(text: str, min_section_chars: int = 300):
        """
        Chia text HTML thành các section dựa trên heading.
        Nếu không tìm được heading tốt thì trả về 1 section toàn bộ.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        sections = []
        current_title = "Nội dung chính"
        current_lines = []

        for line in lines:
            if is_section_heading(line) and current_lines:
                content = "\n".join(current_lines).strip()

                if content:
                    sections.append(
                        {
                            "title": current_title,
                            "content": content,
                        }
                    )

                current_title = line
                current_lines = [line]
            else:
                if not current_lines and is_section_heading(line):
                    current_title = line

                current_lines.append(line)

        if current_lines:
            content = "\n".join(current_lines).strip()

            if content:
                sections.append(
                    {
                        "title": current_title,
                        "content": content,
                    }
                )

        # Nếu split ra quá vụn, gộp các section quá ngắn vào section trước
        merged = []

        for sec in sections:
            if merged and len(sec["content"]) < min_section_chars:
                merged[-1]["content"] += "\n" + sec["content"]
            else:
                merged.append(sec)

        if not merged:
            return [
                {
                    "title": "Nội dung chính",
                    "content": text,
                }
            ]

        return merged


    def process_htmls(parent_records, child_records, splitter):
        html_files = list(HTML_DIR.rglob("*.html")) + list(HTML_DIR.rglob("*.htm"))

        for html_path in html_files:
            print(f"Đang xử lý HTML: {html_path}")

            text, tables, images = extract_html_text_tables_images(html_path)

            major_name = guess_major_name(text, str(html_path))
            sections = split_text_into_sections(text)

            print(f"[HTML] Major guess: {major_name}")
            print(f"[HTML] Số section text: {len(sections)}")
            print(f"[HTML] Số bảng: {len(tables)}")
            print(f"[HTML] Số ảnh: {len(images)}")

            # =========================
            # 1. Mỗi section text = 1 parent
            # =========================

            for section_idx, section in enumerate(sections, start=1):
                parent_id = str(uuid.uuid4())

                section_title = section["title"]
                section_content = section["content"]

                parent_content = (
                    f"[SECTION {section_idx}] {section_title}\n\n"
                    f"{section_content}"
                )

                add_parent(
                    parent_records=parent_records,
                    parent_id=parent_id,
                    content=parent_content,
                    source=str(html_path),
                    location=f"html_section_{section_idx}",
                    parent_type="html_section",
                    file_type="html",
                )

                add_text_chunks(
                    child_records=child_records,
                    splitter=splitter,
                    text=parent_content,
                    parent_id=parent_id,
                    source=str(html_path),
                    location=f"html_section_{section_idx}",
                    file_type="html",
                    extra_metadata={
                        "doc_type": "curriculum",
                        "major_name": major_name,
                        "section_title": section_title,
                        "section_index": section_idx,
                    },
                )

            # =========================
            # 2. Mỗi bảng HTML = 1 parent riêng
            # =========================

            for table_idx, table in enumerate(tables, start=1):
                parent_id = str(uuid.uuid4())

                table_blocks = add_table_children(
                    child_records=child_records,
                    tables=[table],
                    parent_id=parent_id,
                    source=str(html_path),
                    file_name=html_path.name,
                    location=f"html_table_{table_idx}",
                    file_type="html",
                )

                if not table_blocks:
                    continue

                parent_content = (
                    f"[TABLE SECTION] Bảng {table_idx} trong {html_path.name}\n"
                    f"[Ngành] {major_name}\n\n"
                    + "\n\n".join(table_blocks)
                )

                add_parent(
                    parent_records=parent_records,
                    parent_id=parent_id,
                    content=parent_content,
                    source=str(html_path),
                    location=f"html_table_{table_idx}",
                    parent_type="html_table",
                    file_type="html",
                )

            # =========================
            # 3. Mỗi ảnh OCR liên quan = 1 parent riêng
            # =========================

            for image_idx, image in enumerate(images, start=1):
                parent_id = str(uuid.uuid4())

                image_blocks = add_image_children(
                    child_records=child_records,
                    images=[image],
                    parent_id=parent_id,
                    source=str(html_path),
                    file_name=html_path.name,
                    location=f"html_image_{image_idx}",
                    file_type="html",
                )

                if not image_blocks:
                    continue

                parent_content = (
                    f"[IMAGE SECTION] Hình {image_idx} trong {html_path.name}\n"
                    f"[Ngành] {major_name}\n\n"
                    + "\n\n".join(image_blocks)
                )

                add_parent(
                    parent_records=parent_records,
                    parent_id=parent_id,
                    content=parent_content,
                    source=str(html_path),
                    location=f"html_image_{image_idx}",
                    parent_type="html_image",
                    file_type="html",
                )


    # =========================
    # Main
    # =========================

    def run_prepare_multivector():
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        parent_records = []
        child_records = []

        if PROCESS_PDF:
            process_pdfs(parent_records, child_records, splitter)

        if PROCESS_HTML:
            process_htmls(parent_records, child_records, splitter)

        write_jsonl(PARENTS_PATH, parent_records)
        write_jsonl(CHILDREN_PATH, child_records)

        print("Xong.")
        print(f"Số parent docs: {len(parent_records)}")
        print(f"Số child docs: {len(child_records)}")
        print(f"Parent lưu tại: {PARENTS_PATH}")
        print(f"Child lưu tại: {CHILDREN_PATH}")


    if __name__ == "__main__":
        run_prepare_multivector()