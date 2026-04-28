from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
import time

from app.db.session import get_db
from app.models.chat_history import ChatHistory

router = APIRouter()

# Schema (Pydantic) để validate dữ liệu từ React gửi lên
class ChatRequest(BaseModel):
    message: str
    user_id: str = "guest"

class ChatResponse(BaseModel):
    reply: str
    status: str

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    # 1. Giả lập delay suy nghĩ của Chatbot
    time.sleep(1)
    
    # 2. Logic trả lời tạm thời (Sau này sẽ thay bằng gọi hàm Pipeline RAG ở đây)
    user_msg = request.message.lower()
    if "học phí" in user_msg:
        reply_text = "Học phí dự kiến của UIT năm 2026 dao động từ 30 - 50 triệu VNĐ/năm tùy chương trình đào tạo."
    elif "điểm chuẩn" in user_msg:
        reply_text = "Điểm chuẩn các ngành của UIT năm ngoái từ 25.5 đến 28.1 điểm. Bạn muốn hỏi cụ thể ngành nào?"
    else:
        reply_text = f"Hệ thống đang được nâng cấp để trả lời câu hỏi: '{request.message}'. Chức năng tra cứu thông minh bằng RAG sẽ sớm ra mắt!"

    # 3. Lưu vào Database (Lịch sử hỏi đáp)
    new_history = ChatHistory(
        user_id=request.user_id,
        question=request.message,
        answer=reply_text,
        status="Đã trả lời"
    )
    db.add(new_history)
    db.commit()
    db.refresh(new_history)

    return ChatResponse(reply=reply_text, status="success")