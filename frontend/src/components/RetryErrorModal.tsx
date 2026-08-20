interface RetryErrorModalProps {
  isOpen: boolean;
  onRetry: () => void;
  onClose: () => void;
}

const RetryErrorModal = ({ isOpen, onRetry, onClose }: RetryErrorModalProps) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/35 px-8">
      <div className="w-full max-w-[260px] rounded-[4px] border-4 border-main-sky bg-white px-6 py-8 text-center shadow-2xl">
        <h2 className="text-[18px] font-black leading-7 text-black">
          오류가 발생하였습니다.
          <br />
          다시 시도하시겠습니까?
        </h2>

        <div className="mt-8 flex justify-center gap-5">
          <button
            type="button"
            onClick={onRetry}
            className="h-12 w-16 rounded-[8px] bg-main-blue text-[16px] font-black text-white shadow-md active:scale-95"
          >
            예
          </button>
          <button
            type="button"
            onClick={onClose}
            className="h-12 w-16 rounded-[8px] bg-gray-200 text-[16px] font-black text-black shadow-md active:scale-95"
          >
            아니오
          </button>
        </div>
      </div>
    </div>
  );
};

export default RetryErrorModal;
