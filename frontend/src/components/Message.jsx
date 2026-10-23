import { useEffect } from "react";

import Prediction from "./Prediction.jsx";

export default function Message({ message, modelName, onRelease }) {
  const { role, text, imageUrl, result, error, pending } = message;

  // Превью живёт в blob-URL; освобождаем его, когда сообщение уходит.
  useEffect(() => {
    if (!imageUrl) return undefined;
    return () => onRelease(imageUrl);
  }, [imageUrl, onRelease]);

  if (role === "system") {
    return <p className="system-note">{text}</p>;
  }

  return (
    <article className={`message message--${role}`}>
      <div className="bubble">
        {imageUrl && <img className="thumb" src={imageUrl} alt="Отправленное изображение" />}
        {text && <p className="text">{text}</p>}
        {pending && (
          <p className="text pending">
            Смотрю<span className="dots" aria-hidden="true" />
          </p>
        )}
        {error && <p className="text error">{error}</p>}
        {result && <Prediction result={result} modelName={modelName} />}
      </div>
    </article>
  );
}
