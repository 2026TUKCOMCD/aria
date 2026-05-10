import { useState } from 'react';

interface SleepTimeModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (sleepTime: string, wakeTime: string) => void;
}

const SleepTimeModal = ({ isOpen, onClose, onSave }: SleepTimeModalProps) => {
  const [sleepTime, setSleepTime] = useState({ period: '오후', hour: '11', minute: '00' });
  const [wakeTime, setWakeTime] = useState({ period: '오전', hour: '07', minute: '00' });

  if (!isOpen) return null;

  const handleHourChange = (type: 'sleep' | 'wake', value: string) => {
    let num = value.replace(/[^0-9]/g, '');
    if (num !== '') {
      let intNum = parseInt(num);
      if (intNum > 12) num = '12';
      if (intNum === 0) num = '01';
    }
    const setter = type === 'sleep' ? setSleepTime : setWakeTime;
    const state = type === 'sleep' ? sleepTime : wakeTime;
    setter({ ...state, hour: num.padStart(2, '0').slice(-2) });
  };

  const handleMinuteChange = (type: 'sleep' | 'wake', value: string) => {
    let num = value.replace(/[^0-9]/g, '');
    if (num !== '') {
      let intNum = parseInt(num);
      if (intNum > 59) num = '59';
    }
    const setter = type === 'sleep' ? setSleepTime : setWakeTime;
    const state = type === 'sleep' ? sleepTime : wakeTime;
    setter({ ...state, minute: num.padStart(2, '0').slice(-2) });
  };

  const togglePeriod = (type: 'sleep' | 'wake') => {
    const setter = type === 'sleep' ? setSleepTime : setWakeTime;
    const state = type === 'sleep' ? sleepTime : wakeTime;
    setter({ ...state, period: state.period === '오전' ? '오후' : '오전' });
  };

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/40">
      <div className="relative w-[400px] rounded-[32px] bg-white p-8 shadow-2xl">
        
        {/* 우측 상단 X 닫기 버튼 추가 */}
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

        <div className="mt-10 flex justify-center">
          <button
            onClick={() => onSave(`${sleepTime.period} ${sleepTime.hour}:${sleepTime.minute}`, `${wakeTime.period} ${wakeTime.hour}:${wakeTime.minute}`)}
            className="h-[65px] w-[140px] rounded-[20px] bg-main-blue text-[28px] font-[900] text-white shadow-lg active:scale-95 transition-all"
          >
            저장
          </button>
        </div>
      </div>
    </div>
  );
};

export default SleepTimeModal;