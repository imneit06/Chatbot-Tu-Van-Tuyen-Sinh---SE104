import React, { useState, useContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mail, Lock, User, ArrowRight, Home } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { AuthContext } from '../context/AuthContext';

const LoginPage = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  
  const { login } = useContext(AuthContext);
  const navigate = useNavigate();

  const handleAuthSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (isLogin) {
      // Logic Đăng nhập
      try {
        const response = await axios.post('http://localhost:8000/api/v1/auth/login', {
          email,
          password
        });
        login(response.data);
        if(response.data.role === 'admin') navigate('/admin');
        else navigate('/chat');
      } catch (err) {
        setError('Email hoặc mật khẩu không chính xác!');
      }
    } else {
      // Logic Đăng ký (Tạm thời giả lập báo lỗi để bạn nối API sau)
      if (password !== confirmPassword) {
        setError('Mật khẩu xác nhận không khớp!');
        return;
      }
      setError('Chức năng đăng ký đang được bảo trì!');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden bg-[#002244] pt-24 pb-12 px-4">
      
      <div className="absolute inset-0 z-0">
        <div className="absolute top-[0%] left-[-10%] w-[40%] h-[40%] bg-[#0ea5e9] rounded-full mix-blend-screen filter blur-[120px] opacity-40 animate-blob"></div>
        <div className="absolute bottom-[0%] right-[-10%] w-[50%] h-[50%] bg-[#8b5cf6] rounded-full mix-blend-screen filter blur-[150px] opacity-30 animate-blob animation-delay-2000"></div>
      </div>

      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="w-full max-w-md relative z-10"
      >
        <div className="bg-white/15 backdrop-blur-2xl border border-white/30 p-8 rounded-[2.5rem] shadow-[0_8px_32px_0_rgba(0,0,0,0.3)]">
          
          <div className="text-center mb-8">
            <Link to="/" className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-white/20 mb-4 hover:bg-white/30 transition-colors">
              <Home className="w-5 h-5 text-white" />
            </Link>
            <h2 className="text-3xl font-black text-white tracking-tight mb-2 drop-shadow-md">
              {isLogin ? 'Mừng trở lại!' : 'Tạo tài khoản'}
            </h2>
            <p className="text-blue-100 text-sm">
              {isLogin ? 'Đăng nhập để xem lịch sử tư vấn' : 'Trải nghiệm Chatbot tư vấn tuyển sinh UIT'}
            </p>
          </div>

          <AnimatePresence mode="wait">
            <motion.form 
              key={isLogin ? 'login' : 'register'}
              initial={{ opacity: 0, x: isLogin ? -20 : 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: isLogin ? 20 : -20 }}
              transition={{ duration: 0.3 }}
              onSubmit={handleAuthSubmit}
              className="space-y-4"
            >
              {error && (
                <div className="text-red-300 text-sm font-bold text-center bg-red-500/20 py-2.5 rounded-2xl mb-4 border border-red-500/30">
                  {error}
                </div>
              )}

              {!isLogin && (
                <div className="relative">
                  <User className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/80" />
                  <input 
                    type="text" 
                    required
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="Họ và tên" 
                    className="w-full bg-white/20 border border-white/40 rounded-2xl py-3.5 pl-12 pr-4 text-white placeholder-white/70 focus:outline-none focus:ring-2 focus:ring-white focus:bg-white/30 transition-all"
                  />
                </div>
              )}
              
              <div className="relative">
                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/80" />
                <input 
                  type="email" 
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Email" 
                  className="w-full bg-white/20 border border-white/40 rounded-2xl py-3.5 pl-12 pr-4 text-white placeholder-white/70 focus:outline-none focus:ring-2 focus:ring-white focus:bg-white/30 transition-all"
                />
              </div>

              <div className="relative">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/80" />
                <input 
                  type="password" 
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Mật khẩu" 
                  className="w-full bg-white/20 border border-white/40 rounded-2xl py-3.5 pl-12 pr-4 text-white placeholder-white/70 focus:outline-none focus:ring-2 focus:ring-white focus:bg-white/30 transition-all"
                />
              </div>

              {!isLogin && (
                <div className="relative">
                  <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/80" />
                  <input 
                    type="password" 
                    required
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="Xác nhận mật khẩu" 
                    className="w-full bg-white/20 border border-white/40 rounded-2xl py-3.5 pl-12 pr-4 text-white placeholder-white/70 focus:outline-none focus:ring-2 focus:ring-white focus:bg-white/30 transition-all"
                  />
                </div>
              )}

              {isLogin && (
                <div className="flex justify-end">
                  <a href="#" className="text-sm text-blue-100 hover:text-white transition-colors">Quên mật khẩu?</a>
                </div>
              )}

              <button 
                type="submit" 
                className="w-full bg-[#0ea5e9] text-white font-bold py-4 rounded-2xl flex items-center justify-center gap-2 hover:bg-blue-400 hover:shadow-[0_0_20px_rgba(14,165,233,0.5)] transition-all hover:-translate-y-0.5 mt-6"
              >
                {isLogin ? 'Đăng nhập' : 'Đăng ký ngay'}
                <ArrowRight className="w-5 h-5" />
              </button>
            </motion.form>
          </AnimatePresence>

          <div className="mt-8 text-center text-sm text-blue-100">
            {isLogin ? 'Chưa có tài khoản? ' : 'Đã có tài khoản? '}
            <button 
              onClick={() => {
                setIsLogin(!isLogin);
                setError('');
              }}
              className="text-white font-bold hover:underline"
            >
              {isLogin ? 'Đăng ký' : 'Đăng nhập'}
            </button>
          </div>

        </div>
      </motion.div>
    </div>
  );
};

export default LoginPage;