import { useState } from 'react';
import CommonModal from '../components/CommonModal';
import SleepTimeModal from '../components/SleepTimeModal';
import NameInputModal from '../components/NameInputModal'; 
import type { ModalType } from '../components/CommonModal';

const Main = () => {
  // 1. 공통 모달 상태 (초기화/에러)
  const [commonModal, setCommonModal] = useState<{
    isOpen: boolean;
    type: ModalType;
  }>({
    isOpen: false,
    type: 'RESET',
  });

  // 2. 구역 이름 모달 상태
  const [isNameModalOpen, setIsNameModalOpen] = useState(false);
  const [roomName, setRoomName] = useState('거실');

  // 3. 수면 시간 모달 상태
  const [isSleepModalOpen, setIsSleepModalOpen] = useState(false);

  // 제어 함수들
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
    <div className="flex h-screen flex-col items-center justify-center gap-6 bg-gray-100 p-4">
      <div className="text-center mb-4">
        <h1 className="text-2xl font-black text-gray-800">로봇 제어 시스템 UI 테스트</h1>
        {/* 현재 설정된 방 이름을 화면에서 바로 확인 가능하도록 추가 */}
        <p className="mt-2 text-lg font-bold text-main-blue">현재 설정 구역: {roomName}</p>
      </div>
      
      {/* 테스트 버튼 그룹: 2열 그리드 구성 */}
      <div className="grid grid-cols-2 gap-4 w-full max-w-md">
        <button 
          onClick={() => openCommonModal('RESET')} 
          className="rounded-xl bg-gray-700 px-6 py-4 text-white font-bold hover:bg-gray-800 shadow-md active:scale-95 transition-all"
        >
          초기화 모달
        </button>
        <button 
          onClick={() => openCommonModal('ERROR')} 
          className="rounded-xl bg-red-600 px-6 py-4 text-white font-bold hover:bg-red-700 shadow-md active:scale-95 transition-all"
        >
          에러 모달
        </button>
        
        <button 
          onClick={() => setIsNameModalOpen(true)} 
          className="col-span-1 rounded-xl bg-main-sky border-2 border-main-blue px-6 py-4 text-main-blue font-black shadow-md hover:bg-blue-50 active:scale-95 transition-all"
        >
          구역 이름 변경
        </button>

        <button 
          onClick={() => setIsSleepModalOpen(true)} 
          className="col-span-1 rounded-xl bg-main-blue px-6 py-4 text-white font-black shadow-md hover:bg-blue-500 active:scale-95 transition-all"
        >
          수면 시간 설정
        </button>
      </div>

      {/* --- 모달 컴포넌트 모음 --- */}

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

      {/* 3. E2-3 이름 입력 모달 */}
      <NameInputModal
        isOpen={isNameModalOpen}
        currentName={roomName}
        onClose={() => setIsNameModalOpen(false)}
        onSave={(newName) => {
          setRoomName(newName);
          setIsNameModalOpen(false);
        }}
      />
    </div>
  );
};

export default Main;