import { useState } from 'react';
import AiIcon from '../assets/ai.svg?react';
import CheckIcon from '../assets/check.svg?react';
import CookingIcon from '../assets/cooking.svg?react';
import HomeIcon from '../assets/home.svg?react';
import MapIcon from '../assets/map.svg?react';
import MorningIcon from '../assets/morning.svg?react';
import MovingIcon from '../assets/moving.svg?react';
import PatrolIcon from '../assets/patrol.svg?react';
import PinIcon from '../assets/pin.svg?react';
import SleepIcon from '../assets/sleep.svg?react';
import Navigation from '../components/Navigation';

const MainPage = () => {
  const [mode, setMode] = useState<'BASIC' | 'AI'>('BASIC');
  const [hasMapData, setHasMapData] = useState(false);
  const [battery] = useState(80);

  return (
    <div className="flex min-h-screen flex-col pb-[100px] font-sans">
      <header className="flex items-center justify-between px-9 pt-10">
        <h1 className="text-[20px] font-[900] text-black tracking-tight">ARIA</h1>
        <span className="text-[18px] font-bold text-main-blue">배터리: {battery}%</span>
      </header>

{/* 2. 모드 변경 스위치 영역 */}
<section className="mt-3 px-6">
  {/* 바깥쪽 흰색 배경 컨테이너 */}
  <div className="w-full rounded-[30px] bg-main-sky p-3 shadow-xl border border-main-sky">
    
    <div className="flex h-[65px] w-full items-center rounded-[25px] bg-white p-1.5">
      <button
        onClick={() => setMode('BASIC')}
        className={`flex flex-1 h-full items-center justify-center gap-2 rounded-[20px] text-[18px] font-black transition-all ${
          mode === 'BASIC' 
            ? 'bg-main-blue text-white shadow-md' 
            : 'text-gray-400'
        }`}
      >
        기본 모드 
        {mode === 'BASIC' && <CheckIcon className="w-5 h-5 fill-current" />}
      </button>

      <button
        onClick={() => setMode('AI')}
        className={`flex flex-1 h-full items-center justify-center gap-2 rounded-[20px] text-[18px] font-black transition-all ${
          mode === 'AI' 
            ? 'bg-main-blue text-white shadow-md' 
            : 'text-gray-400'
        }`}
      >
        AI 모드 
        {mode === 'AI' && <CheckIcon className="w-5 h-5 fill-current" />}
      </button>
    </div>

  </div>
</section>

      {/* 3. 이벤트 상태 및 로그 영역 */}
<section className="mt-3 px-6">
  {/* 부모 컨테이너 */}
  <div className="flex flex-col gap-2 rounded-[25px] bg-main-sky p-3 shadow-xl">
    
    {/* 상단 상태 메시지 영역 */}
    <div className="flex items-center gap-3 px-2">
      {hasMapData ? (
        <>
          <PinIcon className="w-7 h-7 text-main-blue" />
          <span className="text-[20px] font-black text-main-blue">핀 위치로 이동합니다.</span>
        </>
      ) : (
        <>
          <MapIcon className="w-7 h-7 text-main-blue" />
          <span className="text-[20px] font-black text-main-blue">맵 데이터를 생성해주세요</span>
        </>
      )}
    </div>

    <button className="w-full rounded-[40px] bg-white px-4 py-1 text-[16px] font-extrabold text-black shadow-inner mt-1 text-left active:scale-[0.98] transition-transform">
      이벤트 로그 보기 &gt;
    </button>
  </div>
</section>

      <section className="mt-3 flex flex-1 px-6">
        <div className="relative flex w-full flex-col items-center justify-center overflow-hidden rounded-[30px] border-4 border-gray-200 bg-white shadow-xl">
          {hasMapData ? (
            <div className="h-full w-full ">
              <div className="relative h-full w-full rounded-2xl border border-blue-100 flex items-center justify-center">
                <span className="text-gray-400 font-bold">맵 렌더링 영역</span>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-4">
              <span className="text-[20px] font-black text-black">비어있음</span>
            </div>
          )}
        </div>
      </section>

      <section className="mt-3 mb-1 px-6">
        <button
          onClick={() => !hasMapData && setHasMapData(true)}
          className="flex h-[60px] w-full items-center justify-center gap-3 rounded-[20px] bg-main-blue text-[20px] font-black text-white shadow-lg active:scale-95 transition-all"
        >
          {hasMapData ? '청정 시작 ▶' : '맵 데이터 생성'}
        </button>
      </section>

      <Navigation />
    </div>
  );
};

export default MainPage;