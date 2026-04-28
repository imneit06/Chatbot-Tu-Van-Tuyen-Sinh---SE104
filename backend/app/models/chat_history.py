from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from app.db.session import Base

class ChatHistory(Base):
    __tablename__ = "chat_histories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, default="guest") # Mặc định là guest nếu chưa đăng nhập
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    status = Column(String, default="Đã trả lời")
    created_at = Column(DateTime, default=datetime.utcnow)