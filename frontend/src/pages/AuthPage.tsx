import AriaSymbol from '../assets/aria_symbol.svg?react';

const AuthPage = () => {
  return (
    <div className="flex h-screen w-full flex-col items-center justify-between bg-aria-gradient p-10 py-20">
      <div className="flex flex-col items-center gap-6 mt-20">
        {/* 중앙 로고 아이콘: aria_symbol.svg 적용 */}
        <div className="flex h-40 w-40 items-center justify-center rounded-[40px] bg-main-blue shadow-lg transition-transform ">
          <AriaSymbol className="w-24 h-24 text-white" />
        </div>
        <h1 className="text-[28px] font-black text-black">ARIA 로봇 공기청정기</h1>
      </div>

      {/* 하단 안내 문구 */}
      <div className="mb-10 text-center">
        <button className="text-[34px] font-black text-black underline decoration-gray-400 underline-offset-8 hover:scale-103">
          QR 코드를 스캔해주세요
        </button>
      </div>
    </div>
  );
};

export default AuthPage;