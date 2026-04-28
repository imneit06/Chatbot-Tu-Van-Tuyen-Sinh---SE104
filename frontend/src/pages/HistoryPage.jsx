import React from 'react';
import { motion } from 'framer-motion';
import { MessageSquare, Clock, Calendar, ArrowRight, Bot, User } from 'lucide-react';

const HistoryPage = () => {
  // Mock data mô phỏng lịch sử hỏi đáp của sinh viên
  const chatHistory = [
    {
      id: 1,
      date: '28/04/2026',
      time: '14:30',
      question: 'Ngành Khoa học Máy tính học phí bao nhiêu?',
      answer: 'Chào bạn, học phí dự kiến của ngành Khoa học Máy tính (chương trình đại trà) năm 2026 là khoảng 35.000.000 VNĐ/năm nhé.',
      status: 'Đã trả lời'
    },
    {
      id: 2,
      date: '25/04/2026',
      time: '09:15',
      question: 'Mình muốn hỏi về phương thức xét tuyển học bạ',
      answer: 'UIT hiện tại có phương thức ưu tiên xét tuyển theo quy định của ĐHQG-HCM, bạn có thể xem chi tiết tại trang Tra cứu nhé.',
      status: 'Đã trả lời'
    },
    {
      id: 3,
      date: '20/04/2026',
      time: '20:45',
      question: 'Trường có ký túc xá không bot?',
      answer: 'Sinh viên UIT sẽ được bố trí ở tại Ký túc xá ĐHQG-HCM (Khu A hoặc Khu B) rất hiện đại và tiện nghi. Từ KTX có xe bus đi thẳng đến trường.',
      status: 'Đã trả lời'
    }
  ];

  return (
    <div className="min-h-screen pt-32 pb-12 px-4 bg-[#f8fafc] relative overflow-hidden">
      {/* Background Blobs cho đồng bộ với trang Chat */}
      <div className="absolute top-0 right-0 w-96 h-96 bg-blue-200 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob"></div>
      <div className="absolute bottom-0 left-0 w-96 h-96 bg-purple-200 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob animation-delay-2000"></div>

      <div className="max-w-4xl mx-auto relative z-10">
        <header className="mb-10 text-center">
          <h1 className="text-3xl font-black text-[#003366] uppercase tracking-tight mb-3">Lịch sử tương tác</h1>
          <p className="text-gray-500">Xem lại các nội dung bạn đã trao đổi với Chatbot tuyển sinh</p>
        </header>

        <div className="space-y-6">
          {chatHistory.map((item, index) => (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              key={item.id}
              className="bg-white/80 backdrop-blur-md p-6 rounded-3xl border border-white/60 shadow-[0_4px_20px_rgb(0,0,0,0.05)] hover:shadow-lg transition-all group cursor-pointer"
            >
              {/* Header của thẻ lịch sử */}
              <div className="flex justify-between items-center mb-4 pb-4 border-b border-gray-100">
                <div className="flex items-center gap-4 text-sm text-gray-500">
                  <span className="flex items-center gap-1.5 bg-blue-50 text-[#0ea5e9] px-3 py-1 rounded-full font-medium">
                    <Calendar className="w-4 h-4" /> {item.date}
                  </span>
                  <span className="flex items-center gap-1.5">
                    <Clock className="w-4 h-4" /> {item.time}
                  </span>
                </div>
                <span className="text-[10px] font-black uppercase tracking-widest bg-green-50 text-green-600 px-3 py-1 rounded-full">
                  {item.status}
                </span>
              </div>

              {/* Nội dung hỏi đáp */}
              <div className="space-y-4">
                <div className="flex gap-3">
                  <div className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center flex-shrink-0">
                    <User className="w-4 h-4 text-gray-600" />
                  </div>
                  <p className="text-gray-800 font-medium pt-1">{item.question}</p>
                </div>
                <div className="flex gap-3 pl-11">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#003366] to-[#0ea5e9] flex items-center justify-center flex-shrink-0">
                    <Bot className="w-4 h-4 text-white" />
                  </div>
                  <div className="bg-gray-50 p-4 rounded-2xl rounded-tl-sm border border-gray-100 text-gray-600 text-sm leading-relaxed w-full">
                    {item.answer}
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default HistoryPage;