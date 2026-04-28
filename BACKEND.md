Cài giùm bố **Node.js** và **Python 3.10+**.

## 1. Chạy Backend (FastAPI)
Mở Terminal mới và chạy các lệnh naỳ:
```bash
cd backend
python -m venv venv         # Tạo môi trường ảo
source venv/bin/activate   
pip install -r requirements.txt
uvicorn app.main:app --reload