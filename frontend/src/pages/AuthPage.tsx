import { useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import AriaSymbol from '../assets/aria_symbol.svg?react';
import { verifyQrToken } from '../api/ARIARobotController';
import useAuthStore from '../store/useAuthStore';

type ScanStatus = 'IDLE' | 'SCANNING' | 'VERIFYING';

const extractQrToken = (rawValue: string) => {
  const value = rawValue.trim();
  if (!value) return '';

  try {
    const parsedJson = JSON.parse(value);
    if (typeof parsedJson === 'string') return parsedJson.trim();
    if (typeof parsedJson?.token === 'string') return parsedJson.token.trim();
    if (typeof parsedJson?.qr_token === 'string') return parsedJson.qr_token.trim();
  } catch {
    // Plain token or URL QR values are expected too.
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
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const scanTimerRef = useRef<number | null>(null);
  const autoVerifyTokenRef = useRef<string | null>(null);
  const [qrToken, setQrToken] = useState('');
  const [status, setStatus] = useState<ScanStatus>('IDLE');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isCameraReady, setIsCameraReady] = useState(false);

  const stopCamera = () => {
    if (scanTimerRef.current) {
      window.clearTimeout(scanTimerRef.current);
      scanTimerRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setIsCameraReady(false);
    setStatus((current) => (current === 'SCANNING' ? 'IDLE' : current));
  };

  useEffect(() => stopCamera, []);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const tokenFromUrl =
      params.get('token') ||
      params.get('qr') ||
      params.get('qr_token');

    if (!tokenFromUrl || autoVerifyTokenRef.current === tokenFromUrl) return;

    autoVerifyTokenRef.current = tokenFromUrl;
    setQrToken(tokenFromUrl);
    verifyAndLogin(tokenFromUrl);
  }, [location.search]);

  const verifyAndLogin = async (rawToken: string) => {
    const token = extractQrToken(rawToken);

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
        robotId: result.robot_id,
        userName: result.user_name,
        robotName: result.robot_name,
        qrToken: token,
      });
      stopCamera();
      navigate('/', { replace: true });
    } catch (error) {
      console.error('QR 로그인 실패:', error);
      setErrorMessage('QR 정보를 확인하지 못했습니다.');
      setStatus('IDLE');
    }
  };

  const startQrScan = async () => {
    if (!('BarcodeDetector' in window)) {
      setErrorMessage('이 브라우저는 QR 카메라 스캔을 지원하지 않습니다. 토큰을 직접 입력해주세요.');
      return;
    }

    try {
      setErrorMessage(null);
      setStatus('SCANNING');

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsCameraReady(true);
      }

      const detector = new window.BarcodeDetector({ formats: ['qr_code'] });

      const scanFrame = async () => {
        if (!videoRef.current || !canvasRef.current) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;

        if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const context = canvas.getContext('2d');

          if (context) {
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const codes = await detector.detect(canvas);
            const firstCode = codes[0]?.rawValue;

            if (firstCode) {
              setQrToken(firstCode);
              await verifyAndLogin(firstCode);
              return;
            }
          }
        }

        scanTimerRef.current = window.setTimeout(scanFrame, 250);
      };

      scanFrame();
    } catch (error) {
      console.error('카메라 실행 실패:', error);
      setErrorMessage('카메라를 실행하지 못했습니다. 권한을 확인해주세요.');
      stopCamera();
    }
  };

  const handleManualVerify = () => {
    verifyAndLogin(qrToken);
  };

  const isBusy = status === 'SCANNING' || status === 'VERIFYING';

  return (
    <div className="flex min-h-screen w-full flex-col bg-aria-gradient px-8 py-12 font-sans">
      <div className="flex flex-1 flex-col items-center justify-center gap-7">
        <div className="flex h-36 w-36 items-center justify-center rounded-[38px] bg-main-blue shadow-lg">
          <AriaSymbol className="h-20 w-20 text-white" />
        </div>

        <div className="text-center">
          <h1 className="text-[28px] font-black text-black">ARIA 로봇 공기청정기ㅎㅎ</h1>
          <p className="mt-2 text-[15px] font-bold text-gray-500">QR 코드로 기기를 등록합니다.</p>
        </div>

        <div className="w-full rounded-[28px] bg-white p-5 shadow-xl">
          <div className="relative mb-5 flex aspect-square w-full items-center justify-center overflow-hidden rounded-[24px] bg-gray-900">
            <video
              ref={videoRef}
              className={`h-full w-full object-cover ${isCameraReady ? 'block' : 'hidden'}`}
              muted
              playsInline
            />
            <canvas ref={canvasRef} className="hidden" />

            {!isCameraReady && (
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
            )}

            {status === 'SCANNING' && (
              <div className="pointer-events-none absolute inset-8 rounded-[22px] border-4 border-white/80" />
            )}
          </div>

          <button
            onClick={status === 'SCANNING' ? stopCamera : startQrScan}
            disabled={status === 'VERIFYING'}
            className="h-[58px] w-full rounded-[20px] bg-main-blue text-[19px] font-black text-white shadow-lg transition-all active:scale-95 disabled:bg-gray-400"
          >
            {status === 'SCANNING' ? '스캔 중지' : status === 'VERIFYING' ? '확인 중' : 'QR 로그인'}
          </button>

          <div className="mt-5">
            <label className="block text-[13px] font-black text-gray-500">QR 토큰 직접 입력</label>
            <input
              value={qrToken}
              onChange={(event) => setQrToken(event.target.value)}
              className="mt-2 h-[54px] w-full rounded-[18px] bg-main-sky px-4 text-center text-[17px] font-black text-main-blue outline-none"
              placeholder="QR 토큰"
            />
            <button
              onClick={handleManualVerify}
              disabled={isBusy}
              className="mt-3 h-[50px] w-full rounded-[18px] border-2 border-main-blue bg-white text-[16px] font-black text-main-blue transition-all active:scale-95 disabled:border-gray-300 disabled:text-gray-400"
            >
              토큰으로 확인
            </button>
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
