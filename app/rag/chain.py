from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import (
    GEMINI_MODEL,
    LLM_TEMPERATURE,
    MEMORY_MAX_RECENT_TURNS,
    MEMORY_SUMMARY_TRIGGER_TURNS,
    QUESTION_REWRITE_ENABLED,
    RETRIEVAL_TOP_K,
)
from app.core.prompts import (
    SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
    REWRITE_PROMPT_TEMPLATE,
    MEMORY_SUMMARY_PROMPT_TEMPLATE,
)
from app.rag.retriever import retrieve_docs, format_context, format_sources
from app.rag.memory import FileChatMemoryStore, format_messages


memory_store = FileChatMemoryStore()


def build_llm():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=LLM_TEMPERATURE,
    )


def should_rewrite_question(question: str, session: dict) -> bool:
    if not QUESTION_REWRITE_ENABLED:
        return False

    messages = session.get("messages", [])

    if not messages:
        return False

    q = question.lower().strip()

    followup_markers = [
        "ngành đó",
        "cái đó",
        "cái này",
        "vậy",
        "vậy còn",
        "nó",
        "trường này",
        "phần đó",
        "mấy môn đó",
        "bao nhiêu tín chỉ",
        "còn",
        "thì sao",
        "như thế nào",
    ]

    if any(marker in q for marker in followup_markers):
        return True

    clear_keywords = [
        "khoa học máy tính",
        "công nghệ thông tin",
        "kỹ thuật phần mềm",
        "hệ thống thông tin",
        "điểm chuẩn",
        "phương thức xét tuyển",
        "chỉ tiêu",
        "chương trình đào tạo",
    ]

    # Nếu câu hỏi đã có keyword rất rõ thì không cần rewrite
    if any(keyword in q for keyword in clear_keywords):
        return False

    # Nếu có history nhưng câu hỏi không rõ ràng, vẫn rewrite
    return True


def rewrite_question(question: str, session_id: str) -> str:
    session = memory_store.load(session_id)

    if not should_rewrite_question(question, session):
        return question

    memory_summary, recent_chat_history = memory_store.get_memory_parts(session_id)

    prompt = REWRITE_PROMPT_TEMPLATE.format(
        memory_summary=memory_summary,
        recent_chat_history=recent_chat_history,
        question=question,
    )

    llm = build_llm()
    response = llm.invoke(prompt)

    rewritten = response.content.strip()

    if not rewritten:
        return question

    # Tránh trường hợp model trả lời dài hoặc lỡ giải thích.
    first_line = rewritten.splitlines()[0].strip()

    return first_line if first_line else question


def maybe_summarize_memory(session_id: str):
    session = memory_store.load(session_id)
    messages = session.get("messages", [])

    trigger_messages = MEMORY_SUMMARY_TRIGGER_TURNS * 2
    keep_messages = MEMORY_MAX_RECENT_TURNS * 2

    if len(messages) <= trigger_messages:
        return

    old_messages = messages[:-keep_messages]
    recent_messages = messages[-keep_messages:]

    if not old_messages:
        return

    old_summary = session.get("summary", "").strip()
    old_chat_history = format_messages(old_messages)

    prompt = MEMORY_SUMMARY_PROMPT_TEMPLATE.format(
        old_summary=old_summary or "Chưa có tóm tắt cũ.",
        old_chat_history=old_chat_history,
    )

    llm = build_llm()
    response = llm.invoke(prompt)

    new_summary = response.content.strip()

    if not new_summary:
        return

    session["summary"] = new_summary
    session["messages"] = recent_messages

    memory_store.save(session_id, session)


def answer_question(
    question: str,
    session_id: str = "default",
):
    """
    RAG chain có memory.

    Bước xử lý:
    1. Load memory theo session_id
    2. Rewrite question nếu là follow-up
    3. Retrieve bằng standalone_question
    4. Generate answer bằng context + memory
    5. Save user/assistant message
    6. Summarize memory nếu history dài
    """

    standalone_question = rewrite_question(
        question=question,
        session_id=session_id,
    )

    docs, metadata_filter = retrieve_docs(
        standalone_question,
        k=RETRIEVAL_TOP_K,
    )

    context = format_context(docs)
    memory_summary, recent_chat_history = memory_store.get_memory_parts(session_id)

    prompt = (
        SYSTEM_PROMPT
        + "\n\n"
        + RAG_PROMPT_TEMPLATE.format(
            memory_summary=memory_summary,
            recent_chat_history=recent_chat_history,
            context=context,
            question=question,
            standalone_question=standalone_question,
        )
    )

    llm = build_llm()
    response = llm.invoke(prompt)

    answer = response.content

    memory_store.append_message(
        session_id=session_id,
        role="user",
        content=question,
    )

    memory_store.append_message(
        session_id=session_id,
        role="assistant",
        content=answer,
    )

    maybe_summarize_memory(session_id)

    return {
        "answer": answer,
        "question": question,
        "standalone_question": standalone_question,
        "filter": metadata_filter,
        "sources": format_sources(docs),
        "session_id": session_id,
    }


def clear_memory(session_id: str = "default"):
    memory_store.clear(session_id)