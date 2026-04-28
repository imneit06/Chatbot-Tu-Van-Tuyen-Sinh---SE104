from pathlib import Path
from urllib.parse import urlparse, unquote
import uuid

import requests
import pytesseract # pyright: ignore[reportMissingImports]
from PIL import Image, ImageOps
from bs4 import BeautifulSoup

from app.core.config import DOWNLOADED_IMAGES_DIR


# ---- Image helpers ----

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
        # inline clean_text
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                lines.append(line)
        return "\n".join(lines)
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
        "logo", "icon", "facebook", "youtube", "zalo", "banner",
        "background", "avatar", "sprite", "menu", "search",
        "footer", "header",
    ]
    for kw in skip_keywords:
        if kw in text:
            return False

    keep_keywords = [
        "tuyển sinh", "tuyen sinh", "điểm chuẩn", "diem chuan",
        "chỉ tiêu", "chi tieu", "học phí", "hoc phi",
        "chương trình", "chuong trinh", "đào tạo", "dao tao",
        "khung", "môn học", "mon hoc", "tín chỉ", "tin chi",
        "ngành", "nganh", "xét tuyển", "xet tuyen",
        "ctdt", "curriculum",
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
        "tuyển sinh", "tuyen sinh", "điểm chuẩn", "diem chuan",
        "chỉ tiêu", "chi tieu", "học phí", "hoc phi",
        "chương trình", "chuong trinh", "đào tạo", "dao tao",
        "khung chương trình", "môn học", "mon hoc",
        "tín chỉ", "tin chi", "cơ sở ngành", "co so nganh",
        "đại cương", "dai cuong", "chuyên ngành", "chuyen nganh",
        "tốt nghiệp", "tot nghiep", "chuẩn đầu ra", "chuan dau ra",
        "mục tiêu đào tạo", "muc tieu dao tao",
    ]
    keyword_count = sum(1 for kw in useful_keywords if kw in text)
    return keyword_count >= 2

    # ---- HTML table helpers ----

def html_table_to_rows(table_tag):
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        row = [cell.get_text(" ", strip=True) for cell in cells]
        if row:
            rows.append(row)
    return rows


# ---- HTML main content selector ----

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
        "cử nhân", "kỹ sư", "ngành", "giới thiệu chung",
        "mục tiêu đào tạo", "đối tượng tuyển sinh", "quy chế đào tạo",
        "chuẩn đầu ra", "chương trình đào tạo", "khung chương trình",
        "khối kiến thức", "tỷ lệ các khối kiến thức", "tín chỉ",
        "cơ sở ngành", "chuyên ngành", "tốt nghiệp", "môn học",
        "thực tập", "đồ án",
    ]
    noise_keywords = [
        "đăng nhập", "tên truy cập", "mật khẩu", "tìm kiếm",
        "liên kết website", "webmail", "website trường",
        "website môn học", "ctdt khóa", "ctdt khoá",
        "lịch phòng", "what is the",
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
    selectors = [
        "main", "article", "div[role='main']",
        ".main-content", ".region-content", ".content-main",
        ".page-content", ".node", ".node-content",
        ".field-name-body", ".field-items", ".field-item",
        ".col-md-9", ".col-sm-9", ".col-lg-9",
        ".col-md-8", ".col-sm-8", ".col-lg-8",
        "#content", "#main-content", "#content-area", "#main",
    ]

    candidates = []
    for selector in selectors:
        for tag in soup.select(selector):
            text = tag.get_text("\n", strip=True)
            if len(text) >= 200:
                candidates.append(tag)

    important_keywords = [
        "chương trình đào tạo", "mục tiêu đào tạo",
        "chuẩn đầu ra", "tỷ lệ các khối kiến thức",
        "khung chương trình", "cơ sở ngành",
        "chuyên ngành", "tín chỉ",
    ]

    for tag in soup.find_all(["div", "section", "article"]):
        text = tag.get_text("\n", strip=True).lower()
        if len(text) >= 300 and any(kw in text for kw in important_keywords):
            candidates.append(tag)

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
        "Lịch", "TKB", "Lịch phòng", "Tìm kiếm",
        "Đăng Nhập", "Đăng nhập", "Tên truy cập", "Mật khẩu",
        "Mật khẩu chứng thực", "Dùng tài khoản chứng thực",
        "Liên kết", "Website trường", "Webmail", "Website môn học",
        "Tài khoản chứng thực", "Diễn đàn sinh viên",
        "Microsoft Azure", "Hệ chính quy", "Danh mục môn học",
        "Tóm tắt môn học", "Đề án mở ngành", "*",
    }
    noise_prefixes = [
        "CTDT Khóa", "CTDT Khoá", "CTDT Khoa",
        "What is the", "-----", "--",
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

    # ---- Extract images from HTML ----

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