import { useEffect, useState } from 'react';

interface NameInputModalProps {
  isOpen: boolean;
  currentName: string;
  onClose: () => void;
  onSave: (newName: string) => void;
}

const NameInputModal = ({ isOpen, currentName, onClose, onSave }: NameInputModalProps) => {
  const [newName, setNewName] = useState(currentName);

  useEffect(() => {
    if (isOpen) setNewName(currentName);
  }, [currentName, isOpen]);

  if (!isOpen) return null;

  const handleSave = () => {
    const trimmedName = newName.trim();

    if (!trimmedName) {
      alert('이름을 입력해주세요.');
      return;
    }

    onSave(trimmedName);
  };

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/40 px-6">
      <div className="relative w-full max-w-[400px] rounded-[32px] bg-white p-8 shadow-2xl">
        <button
          onClick={onClose}
          className="absolute right-6 top-6 p-2 text-main-blue transition-transform hover:scale-110"
          aria-label="닫기"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>

        <div className="mb-8 text-center">
          <h2 className="text-[30px] font-[900] leading-tight text-gray-900">
            구역 이름 설정
          </h2>
          <p className="mt-2 text-[18px] font-bold text-gray-500">
            현재 이름: <span className="text-main-blue">{currentName}</span>
          </p>
        </div>

        <div className="mb-10 flex h-[78px] items-center rounded-[15px] border-2 border-transparent bg-main-sky p-2 shadow-inner transition-all focus-within:border-main-blue">
          <input
            type="text"
            placeholder="방 이름을 입력하세요"
            value={newName}
            onChange={(event) => setNewName(event.target.value)}
            className="w-full bg-transparent text-center text-[24px] font-bold text-gray-700 outline-none placeholder:text-gray-400"
            autoFocus
          />
        </div>

        <div className="flex justify-center">
          <button
            onClick={handleSave}
            className="h-[65px] w-[140px] rounded-[20px] bg-main-blue text-[28px] font-[900] text-white shadow-lg transition-all active:scale-95"
          >
            저장
          </button>
        </div>
      </div>
    </div>
  );
};

export default NameInputModal;
