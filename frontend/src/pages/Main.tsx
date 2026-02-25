import { useState } from 'react';
import CommonModal from '../components/CommonModal';
// 'type' 키워드를 붙여서 가져와야 에러가 발생하지 않습니다.
import type { ModalType } from '../components/CommonModal';
import '../index.css';

const Main = () => {
  const [modalState, setModalState] = useState<{
    isOpen: boolean;
    type: ModalType;
  }>({
    isOpen: false,
    type: 'RESET',
  });

  const openModal = (type: ModalType) => setModalState({ isOpen: true, type });
  const closeModal = () => setModalState((prev) => ({ ...prev, isOpen: false }));

  const handleConfirm = () => {
    closeModal();
  };

  return (
    <div className="flex h-screen flex-col items-center justify-center gap-4 bg-gray-100">
      <h1 className="text-xl font-bold text-gray-800">로봇 제어 시스템</h1>
      <div className="flex gap-2">
        <button onClick={() => openModal('RESET')} className="rounded bg-gray-700 px-4 py-2 text-white">초기화 테스트</button>
        <button onClick={() => openModal('AI_RESET')} className="rounded bg-gray-700 px-4 py-2 text-white">초기화 테스트</button>
        <button onClick={() => openModal('ERROR')} className="rounded bg-gray-700 px-4 py-2 text-white">오류 테스트</button>
      </div>

      <CommonModal
        isOpen={modalState.isOpen}
        type={modalState.type}
        onConfirm={handleConfirm}
        onClose={closeModal}
      />
    </div>
  );
};

export default Main;