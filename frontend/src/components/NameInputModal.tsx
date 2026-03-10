import { useState } from 'react';

interface NameInputModalProps {
  isOpen: boolean;
  currentName: string; // 현재 구역 이름 (예: 거실)
  onClose: () => void;
  onSave: (newName: string) => void;
}

const NameInputModal = ({ isOpen, currentName, onClose, onSave }: NameInputModalProps) => {
  const [newName, setNewName] = useState('');

  if (!isOpen) return null;

  const handleSave = () => {
    if (newName.trim() === '') {
      alert('이름을 입력해주세요!');
      return;
    }
    onSave(newName);
    setNewName(''); // 입력값 초기화
  };

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/40">
      <div className="relative w-[400px] rounded-[32px] bg-white p-8 shadow-2xl">
        
        {/* X 닫기 버튼 */}
        <button 
          onClick={onClose}
          className="absolute right-6 top-6 text-main-blue hover:scale-110 transition-transform p-2"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>

        {/* 현재 이름 표시 섹션 */}
        <div className="mb-8 text-center">
          <h2 className="text-[32px] font-[900] text-gray-900 leading-tight">
            현재 이름: <span className="text-main-blue">{currentName}</span>
          </h2>
        </div>

        {/* 입력 폼 섹션 */}
        <div className="mb-10 flex flex-col items-center">
          <div className="relative w-full h-[80px] rounded-[15px] bg-main-sky p-2 shadow-inner border-2 border-transparent focus-within:border-main-blue transition-all flex items-center">
            <input
              type="text"
              placeholder="방 이름을 입력하세요"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              className="w-full bg-transparent text-center text-[24px] font-bold text-gray-700 outline-none placeholder:text-gray-400"
              autoFocus
            />
          </div>
        </div>

        {/* 저장 버튼 */}
        <div className="flex justify-center">
          <button
            onClick={handleSave}
            className="h-[65px] w-[140px] rounded-[20px] bg-main-blue text-[28px] font-[900] text-white shadow-lg active:scale-95 transition-all"
          >
            저장
          </button>
        </div>
      </div>
    </div>
  );
};

export default NameInputModal;