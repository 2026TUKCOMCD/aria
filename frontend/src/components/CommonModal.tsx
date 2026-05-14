type ModalType = 'RESET' | 'ERROR' | 'AI_RESET';

interface CommonModalProps {
  isOpen: boolean;
  type: ModalType;
  onConfirm: () => void;
  onClose: () => void;
}

const CommonModal = ({ isOpen, type, onConfirm, onClose }: CommonModalProps) => {
  if (!isOpen) return null;

  const modalConfig = {
    RESET: {
      title: "현재 맵 데이터를",
      highlight: "초기화",
      suffix: "하시겠습니까?",
      confirmBtn: "예",
      cancelBtn: "아니오",
      confirmClass: "bg-main-gray text-black",
      cancelClass: "bg-main-blue text-white",
    },
    AI_RESET: {
      title: "AI를",
      highlight: "초기화",
      suffix: "하시겠습니까?",
      confirmBtn: "예",
      cancelBtn: "아니오",
      confirmClass: "bg-main-gray text-black",
      cancelClass: "bg-main-blue text-white",
    },
    ERROR: {
      title: "오류가 발생하였습니다.",
      highlight: "",
      suffix: "다시 시도하시겠습니까?",
      confirmBtn: "예",
      cancelBtn: "아니오",
      confirmClass: "bg-main-blue text-white",
      cancelClass: "bg-main-gray text-black",
    },
  };

  const config = modalConfig[type];

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/40">
      {/* 팝업창 본체: border-main-sky 적용 */}
      <div className="w-[370px] h-[290px] rounded-[15px] border-[4px] border-main-sky bg-white shadow-2xl flex flex-col items-center justify-center p-6">
        
        {/* 본문 문구: text-main-red 적용 */}
        <h2 className="mb-10 text-center text-[28px] font-[900] leading-tight text-gray-900">
          {config.title} <br />
          {config.highlight && <span className="text-main-red">{config.highlight} </span>}
          {config.suffix}
        </h2>

        {/* 버튼 영역: flex gap과 h- 문법 사용 */}
        <div className="flex gap-[20px]">
          <button
            onClick={onConfirm}
            className={`w-[108px] h-[60px] rounded-[15px] text-[28px] font-[900] shadow-md transition-all active:scale-95 ${config.confirmClass}`}
          >
            {config.confirmBtn}
          </button>
          <button
            onClick={onClose}
            className={`w-[108px] h-[60px] rounded-[15px] text-[28px] font-[900] shadow-md transition-all active:scale-95 ${config.cancelClass}`}
          >
            {config.cancelBtn}
          </button>
        </div>
      </div>
    </div>
  );
};

export type { ModalType }; // 타입을 내보낼 때는 'type' 키워드를 붙여주면 더 안전합니다.
export default CommonModal;