import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Navigation from '../components/Navigation';
import CommonModal, { type ModalType } from '../components/CommonModal';
import SleepTimeModal from '../components/SleepTimeModal';
import useAuthStore from '../store/useAuthStore';
import { resetRobotData } from '../api/ARIARobotController';

const SettingsPage = () => {
  const navigate = useNavigate();
  const logout = useAuthStore((state) => state.logout);
  const authRobotId = useAuthStore((state) => state.robotId);
  // 1. 공통 모달(초기화 등) 상태
  const [isCommonOpen, setIsCommonOpen] = useState(false);
  const [modalType, setModalType] = useState<ModalType>('RESET');

  // 2. 수면 시간 설정 모달 상태
  const [isSleepOpen, setIsSleepOpen] = useState(false);

  // 환경 변수 불러오기
  const ROBOT_ID = import.meta.env.VITE_ROBOT_ID || authRobotId || "1";

  // --- [추가] 초기화 버튼 클릭 시 모달을 여는 함수 ---
  const handleOpenReset = (type: ModalType) => {
    setModalType(type);
    setIsCommonOpen(true);
  };

  // --- [통합] 초기화 API 호출 핸들러 ---
  const handleConfirmReset = async () => {
    // RESET -> MAP 데이터 초기화, AI_RESET -> AI 데이터 초기화
    const target = modalType === 'RESET' ? 'MAP' : 'AI';

    try {
      const data = await resetRobotData(ROBOT_ID, target);
      alert(data.message || "초기화 명령이 성공적으로 전송되었습니다.");
    } catch (error) {
      console.error("Reset Error:", error);
      alert("초기화 중 오류가 발생했습니다.");
    } finally {
      setIsCommonOpen(false); // 작업 완료 후 모달 닫기
    }
  };

  // 수면 시간 저장 핸들러 (UI 업데이트용)
  const handleSaveSleepTime = (sleep: string, wake: string) => {
    console.log(`설정된 시간 - 취침: ${sleep}, 기상: ${wake}`);
    // SleepTimeModal 내부에서 이미 API 호출을 하므로 여기서는 UI 처리만 합니다.
    setIsSleepOpen(false);
  };

  const handleLogout = () => {
    logout();
    navigate('/auth', { replace: true });
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

        <button
          onClick={handleLogout}
          className="flex h-[70px] w-full items-center justify-center rounded-[20px] bg-white text-[22px] font-black text-main-blue shadow-xl active:scale-95 transition-all"
        >
          로그아웃
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
        robotId={ROBOT_ID}
      />

      <Navigation />
    </div>
  );
};

export default SettingsPage;
