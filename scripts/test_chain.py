from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.rag.chain import answer_question, clear_memory


def main():
    session_id = input("Nhập session_id, Enter để dùng default: ").strip()
    if not session_id:
        session_id = "default"

    print(f"Đang dùng session_id: {session_id}")
    print("Gõ /clear để xóa memory session này.")
    print("Gõ exit để thoát.")

    while True:
        question = input("\nNhập câu hỏi: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            break

        if question == "/clear":
            clear_memory(session_id)
            print("Đã xóa memory của session này.")
            continue

        result = answer_question(
            question=question,
            session_id=session_id,
        )

        print("\n" + "=" * 100)
        print("SESSION:", result["session_id"])
        print("FILTER:", result["filter"])
        print("STANDALONE QUESTION:", result["standalone_question"])

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCES:")
        for source in result["sources"]:
            print(source)


if __name__ == "__main__":
    main()