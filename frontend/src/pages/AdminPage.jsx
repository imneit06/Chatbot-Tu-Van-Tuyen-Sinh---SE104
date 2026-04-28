import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { BarChart3, Database, FileText, RefreshCw, Upload, Plus, Trash2 } from 'lucide-react';

const AdminPage = () => {
  const [activeTab, setActiveTab] = useState('stats');

  const tabs = [
    { id: 'stats', name: 'Thống kê báo cáo', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'knowledge', name: 'Tri thức Chatbot', icon: <FileText className="w-4 h-4" /> },
    { id: 'data', name: 'Dữ liệu hệ thống', icon: <Database className="w-4 h-4" /> },
  ];

  return (
    <div className="min-h-screen pt-32 pb-12 px-4 bg-[#f8fafc]">
      <div className="max-w-7xl mx-auto">
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
          <div>
            <h1 className="text-3xl font-black text-[#003366] uppercase">Admin Dashboard</h1>
            <p className="text-gray-500">Quản lý tri thức và dữ liệu tuyển sinh UIT</p>
          </div>
          <button className="flex items-center gap-2 px-6 py-3 bg-[#0ea5e9] text-white rounded-2xl font-bold shadow-lg shadow-blue-500/20 hover:bg-blue-600 transition-all active:scale-95">
            <Plus className="w-5 h-5" />
            Thêm mới
          </button>
        </header>

        {/* Tab Navigation */}
        <div className="flex gap-2 p-1.5 bg-gray-200/50 rounded-2xl w-fit mb-8 backdrop-blur-sm">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all ${
                activeTab === tab.id 
                  ? 'bg-white text-[#003366] shadow-sm' 
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab.icon}
              {tab.name}
            </button>
          ))}
        </div>

        {/* Tab Content: Knowledge Base Management (UC-07 -> UC-10) */}
        {activeTab === 'knowledge' && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
            <div className="bg-white p-6 rounded-3xl border border-gray-100 shadow-sm">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-bold text-gray-800">Tài liệu tri thức hiện tại (BM 4.1)</h2>
                <button className="flex items-center gap-2 px-4 py-2 bg-blue-50 text-[#0ea5e9] rounded-xl font-bold text-xs hover:bg-blue-100 transition-colors">
                  <RefreshCw className="w-4 h-4" />
                  Ingest/Re-index (UC-10)
                </button>
              </div>
              <table className="w-full text-left">
                <thead>
                  <tr className="text-gray-400 text-xs uppercase tracking-widest border-b border-gray-50">
                    <th className="py-4 font-black">Tên tài liệu</th>
                    <th className="py-4 font-black">Loại</th>
                    <th className="py-4 font-black">Ngày cập nhật</th>
                    <th className="py-4 font-black">Trạng thái</th>
                    <th className="py-4 font-black text-right">Thao tác</th>
                  </tr>
                </thead>
                <tbody className="text-sm">
                  {[1, 2, 3].map((i) => (
                    <tr key={i} className="border-b border-gray-50 hover:bg-gray-50 transition-colors">
                      <td className="py-4 font-bold text-gray-700">Quy-che-tuyen-sinh-2026.pdf</td>
                      <td className="py-4 text-gray-500">PDF</td>
                      <td className="py-4 text-gray-500">20/04/2026</td>
                      <td className="py-4">
                        <span className="px-3 py-1 bg-green-50 text-green-600 rounded-full text-[10px] font-black uppercase">Đang sử dụng</span>
                      </td>
                      <td className="py-4 text-right">
                        <button className="p-2 text-red-400 hover:bg-red-50 rounded-lg transition-colors">
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        )}

        {/* Tab Content: Stats (UC-14 & UC-15) */}
        {activeTab === 'stats' && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-white p-8 rounded-3xl shadow-sm border border-gray-100">
              <p className="text-gray-400 text-xs font-black uppercase tracking-widest mb-2">Số lượt hỏi đáp</p>
              <h3 className="text-4xl font-black text-[#003366]">1,284</h3>
              <div className="mt-4 text-green-500 text-xs font-bold">+12% so với tuần trước</div>
            </div>
            {/* Các card thống kê khác tương tự... */}
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default AdminPage;