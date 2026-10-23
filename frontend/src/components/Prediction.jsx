const percent = (probability) => `${(probability * 100).toFixed(1)}%`;

/**
 * Top-3 как одна серия горизонтальных полос: сравнение величин.
 * Цвет у всех полос одинаковый — он обозначает «вероятность», а не ранг;
 * лидера выделяет типографика, а не отдельный оттенок.
 */
export default function Prediction({ result, modelName }) {
  const [best, ...rest] = result.top;

  return (
    <div className="prediction">
      {result.uncertain && (
        <p className="uncertain" role="status">
          <span aria-hidden="true">⚠</span>
          Не уверен, что это один из цветов, которые я знаю. Вот самые близкие варианты:
        </p>
      )}

      {!result.uncertain && (
        <p className="lead">
          Похоже на <strong>{best.label}</strong>
        </p>
      )}

      <ol className="bars">
        {[best, ...rest].map((item) => (
          <li key={item.label} className="bar-row">
            <span className="bar-label">{item.label}</span>
            <span className="bar-track" title={`${item.label}: ${percent(item.probability)}`}>
              <span
                className="bar-fill"
                style={{ inlineSize: `${Math.max(item.probability * 100, 0.8)}%` }}
              />
            </span>
            <span className="bar-value">{percent(item.probability)}</span>
          </li>
        ))}
      </ol>

      <p className="meta">
        {modelName && <>{modelName} · </>}
        {result.latency_ms} мс на CPU
      </p>
    </div>
  );
}
