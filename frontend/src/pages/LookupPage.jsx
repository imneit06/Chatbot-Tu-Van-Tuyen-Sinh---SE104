import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Filter, BookOpen, Banknote, GraduationCap, X } from 'lucide-react';

const LookupPage = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [showFilter, setShowFilter] = useState(false);
  const [filterBlock, setFilterBlock] = useState('All');

  // Dữ liệu giả lập các ngành của UIT (Mock Data)
  const majors = [
    { id: 1, name: 'Khoa học Máy tính', fee: '35,000,000 VNĐ', code: '7480101', admission: 'A00, A01, D01', description: 'Đào tạo chuyên sâu về thuật toán, Trí tuệ nhân tạo (AI) và phát triển phần mềm lõi.' },
    { id: 2, name: 'An toàn Thông tin', fee: '33,000,000 VNĐ', code: '7480202', admission: 'A00, A01', description: 'Nghiên cứu về bảo mật hệ thống, mật mã học, và phòng chống tấn công mạng.' },
    { id: 3, name: 'Kỹ thuật Phần mềm', fee: '40,000,000 VNĐ', code: '7480103', admission: 'A00, A01, D01', description: 'Tìm hiểu quy trình phát triển, kiểm thử và bảo trì phần mềm chuyên nghiệp.' },
    { id: 4, name: 'Hệ thống Thông tin', fee: '33,000,000 VNĐ', code: '7480104', admission: 'A00, A01, D01', description: 'Tập trung vào phân tích, thiết kế và quản trị cơ sở dữ liệu doanh nghiệp.' },
    { id: 5, name: 'Thương mại Điện tử', fee: '35,000,000 VNĐ', code: '7480109', admission: 'A00, A01, D01', description: 'Kết hợp công nghệ thông tin và kinh tế trong các mô hình kinh doanh số.' }
  ];

  // Thuật toán lọc kép: Tìm kiếm theo tên/mã ngành + Lọc theo tổ hợp môn
  const filteredMajors = majors.filter(m => {
    const matchSearch = m.name.toLowerCase().includes(searchTerm.toLowerCase()) || m.code.includes(searchTerm);
    const matchBlock = filterBlock === 'All' || m.admission.includes(filterBlock);
    return matchSearch && matchBlock;
  });

  return (
    <div className="min-h-screen pt-32 pb-12 px-4 bg-gray-50 relative">
      <div className="max-w-6xl mx-auto">
        <header className="mb-10">
          <h1 className="text-3xl font-black text-[#003366] mb-2 uppercase tracking-tight">Tra cứu thông tin ngành học</h1>
          <p className="text-gray-500 italic">Dữ liệu tuyển sinh mới nhất năm 2026</p>
        </header>

        {/* Thanh tìm kiếm & Nút Lọc nâng cao */}
        <div className="flex flex-col md:flex-row gap-4 mb-8">
          <div className="relative flex-1 group">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400 group-focus-within:text-[#0ea5e9] transition-colors" />
            <input 
              type="text" 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Tìm kiếm tên ngành, mã ngành..." 
              className="w-full pl-12 pr-4 py-4 rounded-2xl border-none shadow-sm focus:ring-2 focus:ring-[#0ea5e9] transition-all outline-none text-gray-700"
            />
          </div>
          
          <div className="relative">
            <button 
              onClick={() => setShowFilter(!showFilter)}
              className={`flex items-center justify-center gap-2 px-6 py-4 rounded-2xl font-bold transition-all border ${showFilter ? 'bg-blue-50 border-blue-200 text-[#0ea5e9]' : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-100 shadow-sm'}`}
            >
              <Filter className="w-4 h-4" />
              Lọc nâng cao
            </button>

            {/* Popup Bộ lọc (Dropdown Modal) */}
            <AnimatePresence>
              {showFilter && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 10 }}
                  className="absolute right-0 top-full mt-3 w-72 bg-white rounded-2xl shadow-2xl border border-gray-100 p-5 z-50"
                >
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="font-bold text-gray-800">Chọn Tổ hợp môn</h3>
                    <button onClick={() => setShowFilter(false)} className="p-1 rounded-full hover:bg-gray-100 transition-colors">
                      <X className="w-4 h-4 text-gray-400 hover:text-red-500" />
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {['All', 'A00', 'A01', 'D01'].map(block => (
                      <button 
                        key={block}
                        onClick={() => setFilterBlock(block)}
                        className={`px-4 py-2 rounded-xl text-sm font-bold transition-all ${
                          filterBlock === block 
                            ? 'bg-[#003366] text-white shadow-md' 
                            : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                        }`}
                      >
                        {block === 'All' ? 'Tất cả' : block}
                      </button>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Danh sách Card Ngành Học */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <AnimatePresence>
            {filteredMajors.map((major) => (
              <motion.div 
                layout
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                key={major.id}
                className="bg-white p-6 rounded-3xl border border-gray-100 shadow-sm hover:shadow-xl hover:-translate-y-1 transition-all duration-300 group flex flex-col"
              >
                <div className="flex justify-between items-start mb-4">
                  <div className="p-3 bg-blue-50 rounded-2xl text-[#0ea5e9] group-hover:bg-[#0ea5e9] group-hover:text-white transition-colors duration-300">
                    <BookOpen className="w-6 h-6" />
                  </div>
                  <span className="text-[10px] font-black bg-gray-100 px-2 py-1 rounded text-gray-400 uppercase tracking-widest">{major.code}</span>
                </div>
                
                <h3 className="text-lg font-bold text-gray-800 mb-2">{major.name}</h3>
                <p className="text-sm text-gray-500 mb-6 line-clamp-3 italic flex-1">{major.description}</p>
                
                <div className="space-y-2 border-t border-gray-50 pt-4 mt-auto">
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <Banknote className="w-4 h-4 text-green-500" />
                    <span>Học phí: <strong className="text-gray-800">{major.fee}</strong></span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <GraduationCap className="w-4 h-4 text-purple-500" />
                    <span>Tổ hợp: <strong className="text-gray-800">{major.admission}</strong></span>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {/* Hiển thị khi không tìm thấy kết quả nào */}
          {filteredMajors.length === 0 && (
            <div className="col-span-full py-12 text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gray-100 mb-4">
                <Search className="w-8 h-8 text-gray-300" />
              </div>
              <h3 className="text-lg font-bold text-gray-600">Không tìm thấy kết quả</h3>
              <p className="text-gray-400 mt-1">Thử thay đổi từ khóa hoặc bộ lọc xem sao nhé.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LookupPage;