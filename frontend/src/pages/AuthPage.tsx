import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import AriaSymbol from '../assets/aria_symbol.svg?react';
import useAuthStore from '../store/useAuthStore';

const AuthPage = () => {
  const navigate = useNavigate();
  const login = useAuthStore((state) => state.login);
  const [qrToken, setQrToken] = useState('');
  const [isVerifying, setIsVerifying] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleVerify = async () => {
    const trimmedToken = qrToken.trim();

    if (!trimmedToken) {
      setErrorMessage('QR 토큰을 입력해주세요.');
      return;
    }

    setIsVerifying(true);
    setErrorMessage(null);

    try {
      login({
        robotId: import.meta.env.VITE_ROBOT_ID || '1',
        userName: '민재',
        robotName: 'ARIA_01',
        qrToken: trimmedToken,
      });
      navigate('/', { replace: true });
    } finally {
      setIsVerifying(false);
    }
  };

  return (
    <div className="flex min-h-screen w-full flex-col items-center justify-between bg-aria-gradient px-8 py-14 font-sans">
      <div className="mt-12 flex flex-col items-center gap-6">
        <div className="flex h-40 w-40 items-center justify-center rounded-[40px] bg-main-blue shadow-lg">
          <AriaSymbol className="h-24 w-24 text-white" />
        </div>
        <div className="text-center">
          <h1 className="text-[28px] font-black text-black">ARIA 로봇 공기청정기</h1>
          <p className="mt-2 text-[15px] font-bold text-gray-500">QR 코드로 기기를 등록합니다.</p>
        </div>
      </div>

      <div className="w-full space-y-4">
        <div className="rounded-[28px] bg-white p-5 shadow-xl">
          <div className="mx-auto mb-5 grid h-40 w-40 grid-cols-5 grid-rows-5 gap-2 rounded-[20px] bg-gray-900 p-4">
            {Array.from({ length: 25 }).map((_, index) => (
              <span
                key={index}
                className={`rounded-sm ${
                  [0, 1, 2, 5, 10, 12, 14, 19, 20, 21, 22, 24].includes(index)
                    ? 'bg-white'
                    : 'bg-gray-900'
                }`}
              />
            ))}
          </div>

          <label className="block text-[13px] font-black text-gray-500">QR 토큰</label>
          <input
            value={qrToken}
            onChange={(event) => setQrToken(event.target.value)}
            className="mt-2 h-[56px] w-full rounded-[18px] bg-main-sky px-4 text-center text-[18px] font-black text-main-blue outline-none"
            placeholder="QR 토큰 입력"
          />

          {errorMessage && (
            <p className="mt-3 rounded-[14px] bg-main-red/10 px-4 py-3 text-center text-[13px] font-bold text-main-red">
              {errorMessage}
            </p>
          )}
        </div>

        <button
          onClick={handleVerify}
          disabled={isVerifying}
          className="h-[64px] w-full rounded-[22px] bg-main-blue text-[22px] font-black text-white shadow-lg transition-all active:scale-95 disabled:bg-gray-400"
        >
          {isVerifying ? '확인 중' : 'QR 코드 확인'}
        </button>
      </div>
    </div>
  );
};

export default AuthPage;
