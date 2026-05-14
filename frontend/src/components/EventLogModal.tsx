import { createPortal } from 'react-dom';

interface LogItem {
  time: string;
  content: string;
}

interface EventLogModalProps {
  isOpen: boolean;
  onClose: () => void;
  logs: LogItem[]; // DB나 부모로부터 전달받을 로그 리스트
}

const EventLogModal = ({ isOpen, onClose, logs }: EventLogModalProps) => {
  if (!isOpen) return null;

  // 모달을 root 외부의 별도 포탈에 띄워 겹침 문제를 원천 차단합니다.
  return createPortal(
    <div className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/40 px-6">
      {/* 모달 본체: 이미지의 파란색 굵은 테두리 반영 */}
      <div className="relative w-full max-w-[380px] rounded-[30px] border-[6px] border-main-blue bg-white p-8 shadow-2xl animate-in fade-in zoom-in duration-200">
        
        {/* 우측 상단 닫기(X) 버튼 */}
        <button 
          onClick={onClose}
          className="absolute right-6 top-6 text-main-blue hover:scale-110 transition-transform"
        >
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>

        {/* 로그 리스트 레이아웃 */}
        <div className="mt-8 flex flex-col gap-6 max-h-[500px] overflow-y-auto pr-2">
          {logs.map((log, index) => (
            <div key={index} className="flex items-center justify-between px-4">
              {/* 시간 영역: 이미지처럼 굵고 정렬된 느낌 */}
              <span className="text-[20px] font-black text-black w-24">
                {log.time}
              </span>
              {/* 내용 영역: 오른쪽 정렬 */}
              <span className="text-[18px] font-black text-black text-center flex-1">
                {log.content}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>,
    document.body
  );
};

export default EventLogModal;