import { useEffect } from "react";

import Prediction from "./Prediction.jsx";
import { useT } from "../i18n.js";

export default function Message({ message, modelName, onRelease }) {
  const { role, textKey, system, imageUrl, result, errorCode, errorText, pending } = message;
  const t = useT();

  // Превью живёт в blob-URL; освобождаем его, когда сообщение уходит.
  useEffect(() => {
    if (!imageUrl) return undefined;
    return () => onRelease(imageUrl);
  }, [imageUrl, onRelease]);

  if (role === "system") {
    return <p className="system-note">{t[system.key](system.arg)}</p>;
  }

  return (
    <article className={`message message--${role}`}>
      <div className="bubble">
        {imageUrl && <img className="thumb" src={imageUrl} alt={t.thumbAlt} />}
        {textKey && <p className="text">{t[textKey]}</p>}
        {pending && (
          <p className="text pending">
            {t.pending}
            <span className="dots" aria-hidden="true" />
          </p>
        )}
        {(errorCode || errorText) && (
          <p className="text error">{t.errors[errorCode] ?? errorText}</p>
        )}
        {result && <Prediction result={result} modelName={modelName} />}
      </div>
    </article>
  );
}
