from pathlib import Path
import pdfplumber
import pandas as pd


PDF_DIR = Path("data/raw/pdf")
OUT_DIR = Path("data/processed/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_cell(x):
    if x is None:
        return ""
    return str(x).replace("\n", " ").strip()


def main():
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("Chưa có file PDF trong data/raw/pdf/")
        return

    for pdf_path in pdf_files:
        print("=" * 80)
        print(f"Đang kiểm tra file: {pdf_path}")

        with pdfplumber.open(pdf_path) as pdf:
            print(f"Số trang: {len(pdf.pages)}")

            for page_idx, page in enumerate(pdf.pages, start=1):
                print("\n" + "-" * 80)
                print(f"Trang {page_idx}")

                text = page.extract_text() or ""
                print("Text preview:")
                print(text[:500])

                tables = page.extract_tables()
                print(f"Số bảng tìm thấy: {len(tables)}")

                for table_idx, table in enumerate(tables, start=1):
                    if not table:
                        continue

                    cleaned_table = [
                        [clean_cell(cell) for cell in row]
                        for row in table
                    ]

                    df = pd.DataFrame(cleaned_table)

                    out_path = OUT_DIR / f"{pdf_path.stem}_page_{page_idx}_table_{table_idx}.csv"
                    df.to_csv(out_path, index=False, header=False, encoding="utf-8-sig")

                    print(f"Đã lưu bảng: {out_path}")


if __name__ == "__main__":
    main()