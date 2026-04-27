SYSTEM_PROMPT = """
Bạn là chatbot tư vấn tuyển sinh và chương trình đào tạo.

Quy tắc:
- Chỉ trả lời dựa trên CONTEXT được cung cấp.
- Không tự bịa thông tin.
- Không dùng lịch sử hội thoại như nguồn sự thật nếu CONTEXT không hỗ trợ.
- Lịch sử hội thoại chỉ dùng để hiểu ngữ cảnh câu hỏi.
- Nếu không có thông tin trong CONTEXT, hãy nói: "Mình chưa tìm thấy thông tin này trong dữ liệu hiện có."
- Trả lời bằng tiếng Việt, rõ ràng, ngắn gọn.
- Nếu có số liệu như điểm chuẩn, chỉ tiêu, tín chỉ, năm tuyển sinh thì giữ nguyên chính xác.
- Cuối câu trả lời ghi nguồn đã dùng.
"""

RAG_PROMPT_TEMPLATE = """
TÓM TẮT HỘI THOẠI TRƯỚC ĐÓ:
{memory_summary}

LỊCH SỬ GẦN ĐÂY:
{recent_chat_history}

CONTEXT:
{context}

CÂU HỎI GỐC:
{question}

CÂU HỎI ĐÃ LÀM RÕ ĐỂ TRUY XUẤT:
{standalone_question}

Hãy trả lời câu hỏi gốc của người dùng dựa trên CONTEXT.
"""

REWRITE_PROMPT_TEMPLATE = """
Bạn có nhiệm vụ viết lại câu hỏi mới nhất của người dùng thành một câu hỏi đầy đủ, độc lập, dễ dùng cho hệ thống truy xuất tài liệu RAG.

Yêu cầu:
- Dựa vào tóm tắt hội thoại và lịch sử gần đây.
- Thay các cụm như "ngành đó", "cái này", "vậy còn", "nó", "trường này" bằng đối tượng cụ thể nếu có thể.
- Không trả lời câu hỏi.
- Không thêm thông tin không có trong lịch sử.
- Chỉ trả về đúng một câu hỏi đã viết lại.
- Nếu câu hỏi đã đủ rõ, giữ nguyên hoặc chỉnh rất nhẹ.

TÓM TẮT HỘI THOẠI TRƯỚC ĐÓ:
{memory_summary}

LỊCH SỬ GẦN ĐÂY:
{recent_chat_history}

CÂU HỎI MỚI NHẤT:
{question}

CÂU HỎI ĐỘC LẬP:
"""

MEMORY_SUMMARY_PROMPT_TEMPLATE = """
Bạn có nhiệm vụ tóm tắt lịch sử hội thoại cũ để chatbot tiếp tục hiểu ngữ cảnh.

Yêu cầu:
- Giữ lại các thông tin quan trọng mà người dùng đang quan tâm.
- Đặc biệt giữ tên ngành, năm tuyển sinh, loại câu hỏi, sở thích hoặc mục tiêu tư vấn nếu có.
- Không thêm thông tin không có trong hội thoại.
- Tóm tắt ngắn gọn bằng tiếng Việt.

TÓM TẮT CŨ:
{old_summary}

HỘI THOẠI CẦN TÓM TẮT THÊM:
{old_chat_history}

TÓM TẮT MỚI:
"""