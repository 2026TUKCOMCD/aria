import { useMemo } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import MapIcon from '../assets/map.svg?react';
import ReturnIcon from '../assets/return.svg?react';

type ErrorKind = 'network' | 'system' | 'map' | 'unknown';

const errorCopy: Record<ErrorKind, { title: string; description: string; action: string }> = {
  network: {
    title: '통신 오류가 발생했습니다.',
    description: '중앙 서버 또는 클라우드 연결 상태를 확인한 뒤 다시 시도해주세요.',
    action: '연결 다시 확인',
  },
  system: {
    title: '시스템 오류가 발생했습니다.',
    description: '로봇 상태가 정상적으로 동기화되지 않았습니다. 초기화 후 다시 확인할 수 있습니다.',
    action: '상태 초기화',
  },
  map: {
    title: '맵 데이터를 불러오지 못했습니다.',
    description: '기존 맵은 유지되며, 서버에 최신 맵이 올라온 뒤 다시 갱신할 수 있습니다.',
    action: '맵 다시 확인',
  },
  unknown: {
    title: '오류가 발생했습니다.',
    description: '잠시 후 다시 시도하거나 메인 화면에서 상태를 확인해주세요.',
    action: '다시 시도',
  },
};

const normalizeErrorKind = (value: string | null): ErrorKind => {
  if (value === 'network' || value === 'system' || value === 'map') return value;
  return 'unknown';
};

const ErrorPage = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const errorInfo = useMemo(() => {
    const params = new URLSearchParams(location.search);
    return errorCopy[normalizeErrorKind(params.get('type'))];
  }, [location.search]);

  return (
    <div className="flex min-h-screen w-full flex-col bg-aria-gradient px-8 pb-[110px] pt-16 font-sans">
      <header className="text-center">
        <div className="mx-auto flex h-24 w-24 items-center justify-center rounded-[28px] bg-white text-main-red shadow-xl">
          <MapIcon className="h-14 w-14" />
        </div>
        <h1 className="mt-8 text-[34px] font-black leading-tight text-black">
          {errorInfo.title}
        </h1>
        <p className="mt-4 text-[17px] font-bold leading-7 text-gray-500">
          {errorInfo.description}
        </p>
      </header>

      <section className="mt-10 space-y-3">
        <button
          onClick={() => navigate(-1)}
          className="flex h-[62px] w-full items-center justify-center rounded-[20px] bg-main-blue text-[20px] font-black text-white shadow-lg transition-all active:scale-95"
        >
          {errorInfo.action}
        </button>
        <button
          onClick={() => navigate('/settings')}
          className="flex h-[62px] w-full items-center justify-center rounded-[20px] bg-white text-[20px] font-black text-main-blue shadow-lg transition-all active:scale-95"
        >
          초기화 설정으로 이동
        </button>
        <button
          onClick={() => navigate('/')}
          className="flex h-[62px] w-full items-center justify-center gap-2 rounded-[20px] bg-white text-[20px] font-black text-gray-500 shadow-lg transition-all active:scale-95"
        >
          <ReturnIcon className="h-6 w-6" />
          메인페이지로
        </button>
      </section>
    </div>
  );
};

export default ErrorPage;
