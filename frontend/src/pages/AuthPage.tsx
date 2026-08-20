import { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import AriaSymbol from '../assets/aria_symbol.svg?react';
import { verifyQrToken } from '../api/ARIARobotController';
import useAuthStore from '../store/useAuthStore';

type VerifyStatus = 'IDLE' | 'VERIFYING';

const ROBOT_ONE_TOKEN = 'aria-993e2ae6d05545c88962cf682c291ec7';

const extractQrToken = (rawValue: string) => {
  const value = rawValue.trim();
  if (!value) return '';

  try {
    const parsedJson = JSON.parse(value);
    if (typeof parsedJson === 'string') return parsedJson.trim();
    if (typeof parsedJson?.token === 'string') return parsedJson.token.trim();
    if (typeof parsedJson?.qr_token === 'string') return parsedJson.qr_token.trim();
  } catch {
    // URL or plain token values are expected too.
  }

  try {
    const parsedUrl = new URL(value, window.location.origin);
    return (
      parsedUrl.searchParams.get('token') ||
      parsedUrl.searchParams.get('qr') ||
      parsedUrl.searchParams.get('qr_token') ||
      value
    ).trim();
  } catch {
    return value;
  }
};

const AuthPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const login = useAuthStore((state) => state.login);
  const [qrToken, setQrToken] = useState('');
  const [status, setStatus] = useState<VerifyStatus>('IDLE');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const tokenFromUrl =
      params.get('token') ||
      params.get('qr') ||
      params.get('qr_token');

    if (tokenFromUrl) {
      setQrToken(tokenFromUrl);
      setErrorMessage(null);
    }
  }, [location.search]);

  const verifyAndLogin = async () => {
    const token = extractQrToken(qrToken);

    if (!token) {
      setErrorMessage('QR 토큰을 확인할 수 없습니다.');
      return;
    }

    setStatus('VERIFYING');
    setErrorMessage(null);

    try {
      const result = await verifyQrToken(token);

      if (!result.valid) {
        setErrorMessage(result.message || '유효하지 않은 QR 코드입니다.');
        setStatus('IDLE');
        return;
      }

      if (!result.robot_id || !result.user_name || !result.robot_name) {
        setErrorMessage('QR 응답에 로봇 정보가 없습니다.');
        setStatus('IDLE');
        return;
      }

      login({
        robotId: token === ROBOT_ONE_TOKEN ? '1' : result.robot_id,
        userName: result.user_name,
        robotName: result.robot_name,
        qrToken: token,
      });

      navigate('/', { replace: true });
    } catch (error) {
      console.error('QR 로그인 실패:', error);
      setErrorMessage('QR 정보를 확인하지 못했습니다.');
      setStatus('IDLE');
    }
  };

  const isVerifying = status === 'VERIFYING';
  const hasToken = qrToken.trim().length > 0;

  return (
    <div className="flex min-h-screen w-full flex-col bg-aria-gradient px-8 py-12 font-sans">
      <div className="flex flex-1 flex-col items-center justify-center gap-7">
        <div className="flex h-36 w-36 items-center justify-center rounded-[38px] bg-main-blue shadow-lg">
          <AriaSymbol className="h-20 w-20 text-white" />
        </div>

        <div className="text-center">
          <h1 className="text-[28px] font-black text-black">ARIA 로봇 공기청정기</h1>
          <p className="mt-2 text-[15px] font-bold text-gray-500">QR 코드로 기기를 등록합니다.</p>
        </div>

        <div className="w-full rounded-[28px] bg-white p-5 shadow-xl">
          <div className="mb-5 flex aspect-square w-full flex-col items-center justify-center rounded-[24px] bg-gray-900 p-6 text-center">
            <div className="grid h-40 w-40 grid-cols-5 grid-rows-5 gap-2 rounded-[20px] bg-gray-900 p-4">
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
            <p className="mt-6 text-[14px] font-bold leading-6 text-white/80">
              QR 코드에서 로그인 토큰을 확인했습니다.
            </p>
          </div>

          <button
            onClick={verifyAndLogin}
            disabled={isVerifying || !hasToken}
            className="h-[58px] w-full rounded-[20px] bg-main-blue text-[19px] font-black text-white shadow-lg transition-all active:scale-95 disabled:bg-gray-400"
          >
            {isVerifying ? '확인 중' : 'QR 로그인'}
          </button>

          <div className="mt-5">
            <label className="block text-[13px] font-black text-gray-500">QR 토큰</label>
            <input
              value={qrToken}
              onChange={(event) => setQrToken(event.target.value)}
              className="mt-2 h-[54px] w-full rounded-[18px] bg-main-sky px-4 text-center text-[17px] font-black text-main-blue outline-none"
              placeholder="QR 토큰"
            />
          </div>

          {errorMessage && (
            <p className="mt-4 rounded-[14px] bg-main-red/10 px-4 py-3 text-center text-[13px] font-bold text-main-red">
              {errorMessage}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default AuthPage;
