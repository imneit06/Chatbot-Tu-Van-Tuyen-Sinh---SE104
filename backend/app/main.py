from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat
from app.api import auth  # Import router auth
from app.db.session import engine, Base

# Tự động tạo các bảng trong database (nếu chưa có)
Base.metadata.create_all(bind=engine)

# 1. KHỞI TẠO APP TRƯỚC (Rất quan trọng, phải nằm trên cùng)
app = FastAPI(title="UIT Admission Chatbot API")

# 2. CẤU HÌNH CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. ĐĂNG KÝ ROUTER (Dùng biến 'app' sau khi nó đã được tạo ở trên)
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])

@app.get("/")
def read_root():
    return {"message": "Welcome to UIT Admission Backend API"}