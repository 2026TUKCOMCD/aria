import { useState } from 'react';
import CommonModal from '../components/CommonModal';
import SleepTimeModal from '../components/SleepTimeModal';
import NameInputModal from '../components/NameInputModal'; 
import Navigation from '../components/Navigation'; // 하단 탭 컴포넌트
import type { ModalType } from '../components/CommonModal';

const Main = () => {
  // 1. 공통 안내/에러 모달 상태
  const [commonModal, setCommonModal] = useState<{
    isOpen: boolean;
    type: ModalType;
  }>({
    isOpen: false,
    type: 'RESET',
  });

  // 2. 구역 이름 입력 모달 상태
  const [isNameModalOpen, setIsNameModalOpen] = useState(false);
  const [roomName, setRoomName] = useState('거실');

  // 3. 수면 시간 설정 모달 상태
  const [isSleepModalOpen, setIsSleepModalOpen] = useState(false);

  // --- 제어 함수 로직 ---

  // 공통 모달 열기/닫기
  const openCommonModal = (type: ModalType) => setCommonModal({ isOpen: true, type });
  const closeCommonModal = () => setCommonModal((prev) => ({ ...prev, isOpen: false }));

  // 수면 시간 저장 핸들러
  const handleSleepSave = (sleep: string, wake: string) => {
    console.log('설정된 수면 시간:', sleep);
    console.log('설정된 기상 시간:', wake);
    alert(`취침: ${sleep}\n기상: ${wake}\n시간이 저장되었습니다.`);
    setIsSleepModalOpen(false);
  };

  // 구역 이름 저장 핸들러
  const handleNameSave = (newName: string) => {
    setRoomName(newName);
    setIsNameModalOpen(false);
  };

  return (
    // 하단 네비게이션 높이(80px)를 고려해 pb-[100px] 여백 확보
    <div className="flex min-h-screen flex-col items-center justify-center gap-6 bg-gray-100 p-4 pb-[100px]">
      
      {/* 상단 텍스트 영역 */}
      <div className="text-center mb-8">
        <h1 className="text-2xl font-black text-gray-800 tracking-tight">
          ARIA 로봇 제어 시스템
        </h1>
        <div className="mt-4 inline-block px-4 py-2 rounded-full bg-white shadow-sm border border-gray-200">
          <p className="text-lg font-bold">
            현재 구역: <span className="text-main-blue">{roomName}</span>
          </p>
        </div>
      </div>
      
      {/* 테스트용 버튼 그리드 레이아웃 */}
      <div className="grid grid-cols-2 gap-4 w-full max-w-sm">
        <button 
          onClick={() => openCommonModal('RESET')} 
          className="rounded-2xl bg-gray-700 px-6 py-4 text-white font-bold hover:bg-gray-800 shadow-md active:scale-95 transition-all"
        >
          초기화 모달
        </button>
        <button 
          onClick={() => openCommonModal('ERROR')} 
          className="rounded-2xl bg-red-600 px-6 py-4 text-white font-bold hover:bg-red-700 shadow-md active:scale-95 transition-all"
        >
          에러 모달
        </button>
        
        <button 
          onClick={() => setIsNameModalOpen(true)} 
          className="col-span-1 rounded-2xl bg-main-sky border-2 border-main-blue px-4 py-4 text-main-blue font-black shadow-md hover:bg-blue-50 active:scale-95 transition-all"
        >
          구역명 변경
        </button>

        <button 
          onClick={() => setIsSleepModalOpen(true)} 
          className="col-span-1 rounded-2xl bg-main-blue px-4 py-4 text-white font-black shadow-md hover:bg-blue-500 active:scale-95 transition-all"
        >
          수면 설정
        </button>
      </div>

      {/* --- 모달 컴포넌트 렌더링 영역 --- */}

      {/* 1. 공통 안내/에러 모달 */}
      <CommonModal
        isOpen={commonModal.isOpen}
        type={commonModal.type}
        onConfirm={closeCommonModal}
        onClose={closeCommonModal}
      />

      {/* 2. 수면 시간 설정 모달 */}
      <SleepTimeModal
        isOpen={isSleepModalOpen}
        onClose={() => setIsSleepModalOpen(false)}
        onSave={handleSleepSave}
      />

      {/* 3. 구역 이름 입력 모달 */}
      <NameInputModal
        isOpen={isNameModalOpen}
        currentName={roomName}
        onClose={() => setIsNameModalOpen(false)}
        onSave={handleNameSave}
      />

      {/* 하단 네비게이션 탭 */}
      <Navigation />
    </div>
  );
};

export default Main;