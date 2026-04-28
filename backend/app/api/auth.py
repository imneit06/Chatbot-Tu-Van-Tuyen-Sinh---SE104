from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    # Mock User 1: Admin
    if request.email == "admin@uit.edu.vn" and request.password == "123456":
        return LoginResponse(id=1, name="Admin Trí", email=request.email, role="admin")
    
    # Mock User 2: Sinh viên bình thường
    elif request.email == "student@gmail.com" and request.password == "123456":
        return LoginResponse(id=2, name="Học sinh 12", email=request.email, role="user")
    
    # Báo lỗi nếu sai thông tin
    raise HTTPException(status_code=401, detail="Email hoặc mật khẩu không đúng")