import { useRef } from "react";

import { useT } from "../i18n.js";

function UploadIcon() {
  return (
    <svg
      className="dropzone-icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 15V4" />
      <path d="m7.5 8.5 4.5-4.5 4.5 4.5" />
      <path d="M4 15v3.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V15" />
    </svg>
  );
}

export default function Composer({ onPick, disabled, dragging, busy }) {
  const inputRef = useRef(null);
  const t = useT();

  function handleChange(event) {
    const [file] = event.target.files;
    if (file) onPick(file);
    // Сброс, иначе повторный выбор того же файла не вызовет change.
    event.target.value = "";
  }

  return (
    <div className="composer">
      <input ref={inputRef} type="file" accept="image/*" onChange={handleChange} hidden />

      {/* Кнопка, а не div: клавиатура, фокус и Enter работают сами собой. */}
      <button
        type="button"
        className={`dropzone${dragging ? " dropzone--active" : ""}`}
        onClick={() => inputRef.current?.click()}
        disabled={disabled}
      >
        <UploadIcon />
        <span className="dropzone-title">
          {busy ? t.dropBusy : dragging ? t.dropActive : t.dropIdle}
        </span>
        <span className="dropzone-hint">
          {t.dropClick} · {t.dropPaste} <kbd>Ctrl</kbd>+<kbd>V</kbd>
        </span>
      </button>
    </div>
  );
}
