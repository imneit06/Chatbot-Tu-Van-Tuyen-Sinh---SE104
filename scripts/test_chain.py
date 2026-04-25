from app.rag.chain import answer_question


def main():
    while True:
        question = input("\nNhập câu hỏi, hoặc gõ exit để thoát: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            break

        result = answer_question(question)

        print("\n" + "=" * 100)
        print("FILTER:", result["filter"])

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCES:")
        for source in result["sources"]:
            print(source)


if __name__ == "__main__":
    main()