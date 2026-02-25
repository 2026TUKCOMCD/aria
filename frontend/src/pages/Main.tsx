import { useState } from 'react';
import CommonModal from '../components/CommonModal';
import SleepTimeModal from '../components/SleepTimeModal'; // 새로 만든 모달 임포트
import type { ModalType } from '../components/CommonModal';

const Main = () => {
  // 공통 모달 상태
  const [commonModal, setCommonModal] = useState<{
    isOpen: boolean;
    type: ModalType;
  }>({
    isOpen: false,
    type: 'RESET',
  });

  // 수면 시간 모달 상태
  const [isSleepModalOpen, setIsSleepModalOpen] = useState(false);

  // 공통 모달 제어 함수
  const openCommonModal = (type: ModalType) => setCommonModal({ isOpen: true, type });
  const closeCommonModal = () => setCommonModal((prev) => ({ ...prev, isOpen: false }));

  // 수면 시간 저장 핸들러
  const handleSleepSave = (sleep: string, wake: string) => {
    console.log('설정된 수면 시간:', sleep);
    console.log('설정된 기상 시간:', wake);
    alert(`취침: ${sleep}\n기상: ${wake}\n시간이 저장되었습니다.`);
    setIsSleepModalOpen(false);
  };

  return (
    <div className="flex h-screen flex-col items-center justify-center gap-6 bg-gray-100">
      <h1 className="text-2xl font-black text-gray-800">로봇 제어 시스템 UI 테스트</h1>
      
      {/* 테스트 버튼 그룹 */}
      <div className="grid grid-cols-2 gap-4">
        <button 
          onClick={() => openCommonModal('RESET')} 
          className="rounded-xl bg-gray-700 px-6 py-3 text-white font-bold hover:bg-gray-800 transition-all"
        >
          초기화 모달
        </button>
        <button 
          onClick={() => openCommonModal('ERROR')} 
          className="rounded-xl bg-red-600 px-6 py-3 text-white font-bold hover:bg-red-700 transition-all"
        >
          에러 모달
        </button>
        <button 
          onClick={() => setIsSleepModalOpen(true)} 
          className="col-span-2 rounded-xl bg-main-blue px-6 py-4 text-white text-xl font-black shadow-lg hover:scale-105 transition-all"
        >
          수면 시간 설정 모달 열기
        </button>
      </div>

      {/* 1. 공통 안내/에러 모달 */}
      <CommonModal
        isOpen={commonModal.isOpen}
        type={commonModal.type}
        onConfirm={closeCommonModal}
        onClose={closeCommonModal}
      />

      {/* 2. E2-2 수면 시간 설정 모달 */}
      <SleepTimeModal
        isOpen={isSleepModalOpen}
        onClose={() => setIsSleepModalOpen(false)}
        onSave={handleSleepSave}
      />
    </div>
  );
};

export default Main;