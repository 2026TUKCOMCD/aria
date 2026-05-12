import { useState } from 'react';
import { saveRobotSchedule } from '../api/ARIARobotController';

interface SleepTimeModalProps {
  isOpen: boolean;
  onClose: () => void;
  // onSave는 성공 후 부모의 UI 상태를 업데이트하는 용도로 유지합니다.
  onSave: (sleepTime: string, wakeTime: string) => void;
  robotId?: string; // 로봇 ID를 프롭으로 받으면 더 유연합니다.
}

const SleepTimeModal = ({ isOpen, onClose, onSave, robotId = "1" }: SleepTimeModalProps) => {
  const ROBOT_ID = import.meta.env.VITE_ROBOT_ID || "1";
  
  const [sleepTime, setSleepTime] = useState({ period: '오후', hour: '11', minute: '00' });
  const [wakeTime, setWakeTime] = useState({ period: '오전', hour: '07', minute: '00' });
  const [enabled, setEnabled] = useState(true);
  const [isLoading, setIsLoading] = useState(false); // 로딩 상태 관리

  if (!isOpen) return null;

  // --- 유틸리티: 시간 변경 핸들러 ---
  const handleHourChange = (type: 'sleep' | 'wake', value: string) => {
    // 숫자만 추출
    let num = value.replace(/[^0-9]/g, '');
    
    if (num !== '') {
      let intNum = parseInt(num);
      if (intNum > 12) num = '12';
      // 입력 중일 때는 01로 강제하지 않고 빈 값이나 숫자 그대로 둠
    }
    
    const setter = type === 'sleep' ? setSleepTime : setWakeTime;
    const state = type === 'sleep' ? sleepTime : wakeTime;
    // padStart를 제거하여 입력하는 대로 보이게 함
    setter({ ...state, hour: num });
  };

  // --- 분(Minute) 핸들러 ---
  const handleMinuteChange = (type: 'sleep' | 'wake', value: string) => {
    let num = value.replace(/[^0-9]/g, '');
    
    if (num !== '') {
      let intNum = parseInt(num);
      if (intNum > 59) num = '59';
    }
    
    const setter = type === 'sleep' ? setSleepTime : setWakeTime;
    const state = type === 'sleep' ? sleepTime : wakeTime;
    // padStart를 제거
    setter({ ...state, minute: num });
  };

  const togglePeriod = (type: 'sleep' | 'wake') => {
    const setter = type === 'sleep' ? setSleepTime : setWakeTime;
    const state = type === 'sleep' ? sleepTime : wakeTime;
    setter({ ...state, period: state.period === '오전' ? '오후' : '오전' });
  };

  const formatTo24H = (period: string, hour: string) => {
    let h = parseInt(hour);
    if (period === '오후' && h < 12) h += 12;
    if (period === '오전' && h === 12) h = 0;
    return h.toString().padStart(2, '0');
  };

  // --- 핵심: API 호출 함수 ---
  const handleSaveRequest = async () => {
    // 저장할 때 한 자릿수면 앞에 0을 붙여줌
    const formatStr = (val: string) => val.padStart(2, '0');
    
    const sHour = formatStr(sleepTime.hour || "12");
    const sMin = formatStr(sleepTime.minute || "00");
    const wHour = formatStr(wakeTime.hour || "07");
    const wMin = formatStr(wakeTime.minute || "00");

    const sTime = `${formatTo24H(sleepTime.period, sHour)}:${sMin}`;
    const wTime = `${formatTo24H(wakeTime.period, wHour)}:${wMin}`;
    
    
    setIsLoading(true);

    try {
      await saveRobotSchedule(robotId || ROBOT_ID, {
        wake_time: wTime,
        sleep_time: sTime,
        enabled,
      });

      onSave(sTime, wTime); 
      alert("수면 시간이 저장되었습니다.");
      onClose();
    } catch (error) {
      console.error("저장 실패:", error);
      alert("저장 중 오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/40">
      <div className="relative w-[400px] rounded-[32px] bg-white p-8 shadow-2xl">
        
        {/* 우측 상단 X 닫기 버튼 */}
        <button 
          onClick={onClose}
          className="absolute right-6 top-6 text-main-blue hover:scale-110 transition-transform p-2"
          aria-label="닫기"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>

        <h2 className="mb-10 text-center text-[32px] font-[900] text-gray-900">수면 시간 설정</h2>

        <div className="space-y-6">
          {[
            { label: '취침', state: sleepTime, type: 'sleep' },
            { label: '기상', state: wakeTime, type: 'wake' },
          ].map((item) => (
            <div key={item.label} className="flex items-center justify-between">
              <span className="text-[28px] font-[900]">{item.label}</span>
              <div className="flex h-[80px] w-[260px] items-center justify-center gap-2 rounded-[15px] bg-main-sky p-2 shadow-inner border-2 border-transparent focus-within:border-main-blue transition-all">
                <button 
                  onClick={() => togglePeriod(item.type as 'sleep' | 'wake')}
                  className="text-[24px] font-bold text-main-blue px-2 hover:bg-white/50 rounded-lg transition-colors"
                >
                  {item.state.period}
                </button>

                <input
                  type="text"
                  inputMode="numeric"
                  value={item.state.hour}
                  onChange={(e) => handleHourChange(item.type as 'sleep' | 'wake', e.target.value)}
                  className="w-[60px] bg-transparent text-center text-[28px] font-bold text-gray-700 outline-none focus:text-main-blue"
                />
                <span className="text-[28px] font-bold text-gray-400">:</span>
                <input
                  type="text"
                  inputMode="numeric"
                  value={item.state.minute}
                  onChange={(e) => handleMinuteChange(item.type as 'sleep' | 'wake', e.target.value)}
                  className="w-[60px] bg-transparent text-center text-[28px] font-bold text-gray-700 outline-none focus:text-main-blue"
                />
              </div>
            </div>
          ))}
        </div>

        <button
          type="button"
          onClick={() => setEnabled((value) => !value)}
          className="mt-8 flex w-full items-center justify-between rounded-[18px] bg-main-sky px-5 py-4 shadow-inner transition-all active:scale-[0.98]"
        >
          <span className="text-[18px] font-black text-gray-800">스케줄 자동 적용</span>
          <span
            className={`flex h-8 w-14 items-center rounded-full p-1 transition-colors ${
              enabled ? 'bg-main-blue' : 'bg-gray-300'
            }`}
          >
            <span
              className={`h-6 w-6 rounded-full bg-white shadow-md transition-transform ${
                enabled ? 'translate-x-6' : 'translate-x-0'
              }`}
            />
          </span>
        </button>

        <div className="mt-10 flex justify-center">
          <button
            onClick={handleSaveRequest}
            disabled={isLoading}
            className={`h-[65px] w-[140px] rounded-[20px] text-[28px] font-[900] text-white shadow-lg active:scale-95 transition-all ${
              isLoading ? 'bg-gray-400' : 'bg-main-blue'
            }`}
          >
            {isLoading ? "..." : "저장"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default SleepTimeModal;
