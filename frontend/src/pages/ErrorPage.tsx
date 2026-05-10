import { useNavigate } from 'react-router-dom';

const ErrorPage = () => {
  const navigate = useNavigate();

  return (
    <div className="flex h-screen w-full flex-col items-center justify-center bg-aria-gradient p-10">
      <div className="text-center space-y-6">
        <h1 className="text-[40px] font-black text-black leading-tight">
          에러가 발생하였습니다.
        </h1>
        
        {/* 메인페이지로 이동 링크 */}
        <button 
          onClick={() => navigate('/')}
          className="text-[24px] font-bold text-gray-500 underline decoration-gray-400 underline-offset-4 hover:text-main-blue transition-colors"
        >
          메인페이지로
        </button>
      </div>
    </div>
  );
};

export default ErrorPage;