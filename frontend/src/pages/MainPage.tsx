import { useState } from 'react';
import CheckIcon from '../assets/check.svg?react';
import MapIcon from '../assets/map.svg?react';
import AIIcon from '../assets/ai.svg?react';
import PinIcon from '../assets/pin.svg?react';
import Navigation from '../components/Navigation';

// --- 로봇 상태별 커스텀 아이콘 ---
import Activity from '../assets/activity.svg?react';      // 활동 감지
import Non_Activity from '../assets/non_activity.svg?react'; // 비활동 감지
import Cooking from '../assets/cooking.svg?react';       // 요리 오염감지
import Non_Cooking from '../assets/non_cooking.svg?react';   // 비요리 오염감지
import Sleep from '../assets/sleep.svg?react';           // 취침
import Morning from '../assets/morning.svg?react';       // 기상
import Return from '../assets/return.svg?react';         // 복귀중
import Home from '../assets/home.svg?react';             // 복귀 완료
import Patrol from '../assets/patrol.svg?react';         // 순찰중

import EventLogModal from '../components/EventLogModal';
import { sendRobotCommand } from '../api/ARIARobotController'; 
import useRobotStore from '../store/useRobotStore'; 

const MainPage = () => {
  const { logs, isRunning, setIsRunning, battery, isAiMode, setAiMode, fetchLogs } = useRobotStore();
  const [hasMapData, setHasMapData] = useState(false); 
  const [isLogOpen, setIsLogOpen] = useState(false);
  
  const robotId = import.meta.env.VITE_ROBOT_ID || "1";
  
  // --- 로그 보기 버튼 클릭 핸들러 ---
  const handleOpenLogs = async () => {
    setIsLogOpen(true);
    await fetchLogs(robotId); // 버튼 누르는 순간 서버 C파트(GET /api/events) 실행!
  };

  /**
   * 메시지 텍스트에 따라 적절한 SVG 아이콘을 반환하는 함수
   */
  const renderStatusIcon = (message: string) => {
    const iconClass = "w-7 h-7 fill-current"; // 공통 스타일

    if (message.includes('요리 오염')) return <Cooking className={`${iconClass} text-main-blue animate-pulse`} />;
    if (message.includes('비요리 오염')) return <Non_Cooking className={`${iconClass} text-main-blue animate-pulse`} />;
    if (message.includes('비활동')) return <Non_Activity className={`${iconClass} text-main-blue animate-pulse`} />;
    if (message.includes('활동')) return <Activity className={`${iconClass} text-main-blue animate-pulse`} />;
    if (message.includes('취침')) return <Sleep className={`${iconClass} text-main-blue animate-pulse`} />;
    if (message.includes('기상')) return <Morning className={`${iconClass} text-main-blue animate-pulse`} />;
    if (message.includes('복귀중')) return <Return className={`${iconClass} text-main-blue animate-pulse`} />;
    if (message.includes('복귀 완료')) return <Home className={`${iconClass} text-main-blue animate-pulse`} />;
    if (message.includes('순찰')) return <Patrol className={`${iconClass} text-main-blue animate-pulse`} />;

    // 기본값 (매칭되는게 없을 때 기존 빨간 점)
    return <div className="w-3 h-3 bg-main-red rounded-full animate-pulse" />;
  };

  const displayStatusMessage = () => {
    if (logs.length > 0) return logs[0].content;
    return isAiMode ? 'AI 자동 청정 시작' : '사용자 지정 청정 시작';
  };

  const handleModeSelect = (targetMode: 'BASIC' | 'AI') => {
    if (isRunning) {
      alert("로봇이 작동 중일 때는 모드를 변경할 수 없습니다. 먼저 중지해주세요.");
      return;
    }
    setAiMode(targetMode === 'AI');
  };

  const handleActionClick = async () => {
    // 1. 맵 데이터가 없는 경우: 맵 생성(SLAM ON) 명령 전송
    if (!hasMapData) {
      try {
        // 로봇에게 SLAM ON 명령 전송
        await sendRobotCommand(robotId, 'SLAM', 'ON');
        
        // 상태 업데이트: 생성 중임을 알림 (필요 시 별도 로딩 상태 관리 가능)
        alert("맵 데이터 생성을 시작합니다. 로봇이 주변을 스캔합니다. 잠시 기다려주세요...");
        
        // 테스트용으로 true 설정 (실제로는 로봇이 맵 업로드 후 DB를 통해 확인하는 것이 좋음)
        setHasMapData(true); 
      } catch (error) {
        console.error("맵 생성 시작 실패:", error);
        alert("맵 생성 명령을 전달하지 못했습니다.");
      }
      return;
    }

    // 2. 맵 데이터가 있는 경우: 청정 시작/중지 제어
    try {
      if (!isRunning) {
        const apiModeValue = isAiMode ? 'AUTO' : 'MANUAL';
        // 모드 설정 후 전원 ON
        await sendRobotCommand(robotId, 'MODE', apiModeValue);
        await sendRobotCommand(robotId, 'POWER', 'ON');
        setIsRunning(true);
      } else {
        // 전원 OFF
        await sendRobotCommand(robotId, 'POWER', 'OFF');
        setIsRunning(false);
      }
    } catch (error) {
      console.error("로봇 제어 실패:", error);
      alert("로봇에게 명령을 전달하지 못했습니다.");
    }
  };

  return (
    <div className="flex min-h-screen flex-col pb-[100px] font-sans">
      <header className="flex items-center justify-between px-9 pt-10">
        <h1 className="text-[20px] font-[900] text-black tracking-tight">ARIA</h1>
        <span className="text-[18px] font-bold text-main-blue">배터리: {battery}%</span>
      </header>

      <section className="mt-3 px-6">
        <div className={`w-full rounded-[30px] p-3 shadow-xl border transition-all ${
          isRunning ? 'bg-gray-200 border-gray-300' : 'bg-main-sky border-main-sky'
        }`}>
          <div className="flex h-[65px] w-full items-center rounded-[25px] bg-white p-1.5">
            <button
              onClick={() => handleModeSelect('BASIC')}
              className={`flex flex-1 h-full items-center justify-center gap-2 rounded-[20px] text-[18px] font-black transition-all ${
                !isAiMode ? 'bg-main-blue text-white shadow-md' : 'text-gray-400'
              } ${isRunning && isAiMode ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              기본 모드 
              {!isAiMode && <CheckIcon className="w-5 h-5 fill-current" />}
            </button>
            <button
              onClick={() => handleModeSelect('AI')}
              className={`flex flex-1 h-full items-center justify-center gap-2 rounded-[20px] text-[18px] font-black transition-all ${
                isAiMode ? 'bg-main-blue text-white shadow-md' : 'text-gray-400'
              } ${isRunning && !isAiMode ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              AI 모드 
              {isAiMode && <CheckIcon className="w-5 h-5 fill-current" />}
            </button>
          </div>
        </div>
      </section>

      <section className="mt-3 px-6">
        <div className="flex flex-col gap-2 rounded-[25px] bg-main-sky p-3 shadow-xl">
          <div className="flex items-center gap-3 px-2 min-h-[40px]">
            {isRunning ? (
              <>
                {/* [변경] 빨간 점 대신 텍스트에 맞는 아이콘 출력 */}
                {renderStatusIcon(displayStatusMessage())}
                <span className="text-[20px] font-black text-main-blue">
                  {displayStatusMessage()}
                </span>
              </>
            ) : hasMapData ? (
              <>
                {!isAiMode ? (
                  <>
                    <PinIcon className="w-7 h-7 text-main-blue" />
                    <span className="text-[20px] font-black text-main-blue">핀 위치로 이동합니다.</span>
                  </>
                ) : (
                  <>
                    <AIIcon className="w-7 h-7 text-main-blue" />
                    <span className="text-[20px] font-black text-main-blue">AI가 공기질을 감지합니다.</span>
                  </>
                )}
              </>
            ) : (
              <>
                <MapIcon className="w-7 h-7 text-main-blue" />
                <span className="text-[20px] font-black text-main-blue">맵 데이터를 생성해주세요</span>
              </>
            )}
          </div>
          <button 
            onClick={handleOpenLogs}
            className="w-full rounded-[40px] bg-white px-4 py-1 text-[16px] font-extrabold text-black shadow-inner mt-1 text-left active:scale-[0.98] transition-transform"
          >
            이벤트 로그 보기 &gt;
          </button>
        </div>
      </section>

      <section className="mt-3 flex flex-1 px-6">
        <div className="relative flex w-full flex-col items-center justify-center overflow-hidden rounded-[30px] border-4 border-gray-200 bg-white shadow-lg">
          {hasMapData ? (
            <div className="h-full w-full flex items-center justify-center">
              {isRunning ? (
                <div className="flex flex-col items-center gap-2">
                   <div className="w-20 h-20 border-8 border-main-blue border-t-transparent rounded-full animate-spin" />
                   <span className="text-main-blue font-black mt-4 uppercase tracking-widest">
                     {isAiMode ? 'AI Analyzing...' : 'Moving to Pin...'}
                   </span>
                </div>
              ) : (
                <span className="text-gray-400 font-bold">맵 렌더링 준비됨</span>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center gap-4">
              <span className="text-[20px] font-black text-black opacity-30">No Map Data</span>
            </div>
          )}
        </div>
      </section>

      <section className="mt-3 mb-1 px-6">
        <button
          onClick={handleActionClick}
          className={`flex h-[60px] w-full items-center justify-center gap-3 rounded-[20px] text-[20px] font-black shadow-lg active:scale-95 transition-all ${
            isRunning 
              ? 'bg-white text-main-blue border-2 border-main-blue' 
              : 'bg-main-blue text-white'
          }`}
        >
          {!hasMapData ? '맵 데이터 생성' : isRunning ? '청정 중지 ■' : '청정 시작 ▶'}
        </button>
      </section>

      <Navigation />

      <EventLogModal 
        isOpen={isLogOpen} 
        onClose={() => setIsLogOpen(false)} 
        logs={logs} 
      />
    </div>
  );
};

export default MainPage;