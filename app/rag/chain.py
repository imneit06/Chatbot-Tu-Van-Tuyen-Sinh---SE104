from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.prompts import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE
from app.rag.retriever import retrieve_docs, format_context, format_sources


load_dotenv(dotenv_path=Path(".env"))


def build_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0.2,
    )


def answer_question(question: str):
    docs, metadata_filter = retrieve_docs(question, k=5)

    context = format_context(docs)

    prompt = (
        SYSTEM_PROMPT
        + "\n\n"
        + RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )
    )

    llm = build_llm()
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "filter": metadata_filter,
        "sources": format_sources(docs),
    }