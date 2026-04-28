from pathlib import Path
from langchain_core.documents import Document  # pyright: ignore[reportMissingImports]

from ..utils.jsonl_utils import doc_to_json
from ..loaders.html_helpers import (
    make_child_content_prefix,
    ocr_image,
    image_metadata_has_relevant_keyword,
    is_large_image_candidate,
    is_useful_ocr_text,
)


ID_KEY = "doc_id"


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