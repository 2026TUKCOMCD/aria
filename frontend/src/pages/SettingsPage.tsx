import { useState } from 'react';
import Navigation from '../components/Navigation';
import CommonModal, { type ModalType } from '../components/CommonModal';
import SleepTimeModal from '../components/SleepTimeModal';

const SettingsPage = () => {
  // 1. 공통 모달(초기화 등) 상태
  const [isCommonOpen, setIsCommonOpen] = useState(false);
  const [modalType, setModalType] = useState<ModalType>('RESET');

  // 2. 수면 시간 설정 모달 상태
  const [isSleepOpen, setIsSleepOpen] = useState(false);

  // 초기화 버튼 클릭 핸들러
  const handleOpenReset = (type: ModalType) => {
    setModalType(type);
    setIsCommonOpen(true);
  };

  // 모달 확인(예) 버튼 클릭 시 실행될 로직
  const handleConfirmReset = () => {
    console.log(`${modalType} 처리됨`);
    setIsCommonOpen(false);
    // 여기에 실제 데이터 초기화 로직 추가
  };

  // 수면 시간 저장 핸들러
  const handleSaveSleepTime = (sleep: string, wake: string) => {
    console.log(`취침: ${sleep}, 기상: ${wake}`);
    setIsSleepOpen(false);
    // 여기에 API 저장 로직 추가
  };

  return (
    <div className="flex min-h-screen flex-col pb-[100px] font-sans">
      <header className="flex items-center justify-center pt-42 pb-16">
        <h1 className="text-[40px] font-black text-black">설정</h1>
      </header>

      <section className="flex flex-col gap-8 px-10">
        <button 
          onClick={() => setIsSleepOpen(true)}
          className="flex h-[80px] w-full items-center justify-center rounded-[20px] bg-main-blue text-[24px] font-black text-white shadow-xl active:scale-95 transition-all"
        >
          수면 시간 설정
        </button>

        <button 
          onClick={() => handleOpenReset('RESET')}
          className="flex h-[80px] w-full items-center justify-center rounded-[20px] bg-main-blue text-[24px] font-black text-white shadow-xl active:scale-95 transition-all"
        >
          맵 초기화
        </button>

        <button 
          onClick={() => handleOpenReset('AI_RESET')}
          className="flex h-[80px] w-full items-center justify-center rounded-[20px] bg-main-blue text-[24px] font-black text-white shadow-xl active:scale-95 transition-all"
        >
          AI 초기화
        </button>
      </section>

      {/* --- 모달 레이어 --- */}
      
      {/* 1. 공통 모달 (초기화 확인용) */}
      <CommonModal 
        isOpen={isCommonOpen}
        type={modalType}
        onConfirm={handleConfirmReset}
        onClose={() => setIsCommonOpen(false)}
      />

      {/* 2. 수면 시간 설정 모달 */}
      <SleepTimeModal 
        isOpen={isSleepOpen}
        onClose={() => setIsSleepOpen(false)}
        onSave={handleSaveSleepTime}
      />

      <Navigation />
    </div>
  );
};

export default SettingsPage;