import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar/Navbar';
import ChatPage from './pages/ChatPage';
import LookupPage from './pages/LookupPage';
import AdminPage from './pages/AdminPage';
import HistoryPage from './pages/HistoryPage';
import LoginPage from './pages/LoginPage';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-[#f8fafc] flex flex-col font-sans">
        <Navbar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Navigate to="/chat" replace />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/lookup" element={<LookupPage />} />
            <Route path="/admin" element={<AdminPage />} />
            
            {/* Các trang chưa code xong tạm thời để chữ */}
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/login" element={<LoginPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;