interface SetupFlowModalProps {
  isOpen: boolean;
  message: string;
  description?: string;
  confirmText?: string;
  cancelText?: string;
  onConfirm: () => void;
  onCancel?: () => void;
}

const SetupFlowModal = ({
  isOpen,
  message,
  description,
  confirmText = '확인',
  cancelText,
  onConfirm,
  onCancel,
}: SetupFlowModalProps) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/35 px-8">
      <div className="w-full max-w-[280px] rounded-[4px] border-4 border-main-sky bg-white px-6 py-8 text-center shadow-2xl">
        <h2 className="whitespace-pre-line text-[18px] font-black leading-7 text-black">
          {message}
        </h2>

        {description && (
          <p className="mt-4 whitespace-pre-line text-[14px] font-bold leading-6 text-gray-600">
            {description}
          </p>
        )}

        <div className="mt-8 flex justify-center gap-5">
          <button
            type="button"
            onClick={onConfirm}
            className="h-12 min-w-16 rounded-[8px] bg-main-blue px-4 text-[16px] font-black text-white shadow-md active:scale-95"
          >
            {confirmText}
          </button>

          {cancelText && onCancel && (
            <button
              type="button"
              onClick={onCancel}
              className="h-12 min-w-16 rounded-[8px] bg-gray-200 px-4 text-[16px] font-black text-black shadow-md active:scale-95"
            >
              {cancelText}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default SetupFlowModal;
