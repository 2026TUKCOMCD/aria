import { useState } from 'react';
import CheckIcon from '../assets/check.svg?react';
import MapIcon from '../assets/map.svg?react';
import Navigation from '../components/Navigation';

const MapPage = () => {
  // 상태 관리: 구역이름(AREA) vs 위치설정(LOCATION)
  const [tab, setTab] = useState<'AREA' | 'LOCATION'>('AREA');

  return (
    <div className="flex min-h-screen flex-col pb-[100px] font-sans">
      
      {/* 1. 상단 안내 타이틀 - 탭 상태에 따라 문구 변경 */}
      <header className="flex flex-col items-center justify-center pt-14">
        <h2 className="text-[20px] font-black text-main-blue tracking-tight">
          {tab === 'AREA' 
            ? '구역을 눌러서 이름을 설정해주세요' 
            : '핀을 이동하여 위치를 설정해주세요'}
        </h2>
      </header>

      {/* 2. 상단 탭 토글 영역 - MainPage와 규격 동일 */}
      <section className="mt-3 px-6">
        <div className="w-full rounded-[30px] bg-main-sky p-3 shadow-xl border border-main-sky">
          <div className="flex h-[65px] w-full items-center rounded-[25px] bg-white p-1.5">
            <button
              onClick={() => setTab('AREA')}
              className={`flex flex-1 h-full items-center justify-center gap-2 rounded-[20px] text-[18px] font-black transition-all ${
                tab === 'AREA' 
                  ? 'bg-main-blue text-white shadow-md' 
                  : 'text-gray-400'
              }`}
            >
              구역이름 
              {tab === 'AREA' && <CheckIcon className="w-5 h-5 fill-current" />}
            </button>

            <button
              onClick={() => setTab('LOCATION')}
              className={`flex flex-1 h-full items-center justify-center gap-2 rounded-[20px] text-[18px] font-black transition-all ${
                tab === 'LOCATION' 
                  ? 'bg-main-blue text-white shadow-md' 
                  : 'text-gray-400'
              }`}
            >
              위치설정 
              {tab === 'LOCATION' && <CheckIcon className="w-5 h-5 fill-current" />}
            </button>
          </div>
        </div>
      </section>

      {/* 3. 중앙 맵 영역 */}
      <section className="mt-3 flex flex-1 px-6">
        <div className="relative flex w-full flex-col items-center justify-center overflow-hidden rounded-[30px] border-4 border-gray-200 bg-white shadow-xl">
          <div className="h-full w-full">
            <div className="relative h-full w-full rounded-2xl flex items-center justify-center">
              <div className="flex flex-col items-center gap-4">
                <MapIcon className="w-20 h-20 text-gray-200 opacity-50" />
                <span className="text-gray-400 font-bold">지도 데이터를 로드 중입니다</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4. 하단 저장 버튼 */}
      <section className="mt-3 mb-1 px-6">
        <button
          className="flex h-[60px] w-full items-center justify-center rounded-[20px] bg-main-blue text-[20px] font-black text-white shadow-lg active:scale-95 transition-all"
        >
          현재 상태 저장
        </button>
      </section>

      <Navigation />
    </div>
  );
};

export default MapPage;