import { LANGS, useT } from "../i18n.js";

const percent = (value) => `${(value * 100).toFixed(1)}%`;

function Mark() {
  return (
    <svg className="mark" viewBox="0 0 32 32" aria-hidden="true">
      {[0, 60, 120, 180, 240, 300].map((angle) => (
        <ellipse
          key={angle}
          cx="16"
          cy="9"
          rx="4.2"
          ry="7"
          transform={`rotate(${angle} 16 16)`}
        />
      ))}
      <circle className="mark-core" cx="16" cy="16" r="3.6" />
    </svg>
  );
}

function LangSwitch({ lang, onChange }) {
  return (
    <div className="lang-switch" role="group" aria-label="Language">
      {LANGS.map((code) => (
        <button
          key={code}
          type="button"
          className={code === lang ? "is-active" : ""}
          aria-pressed={code === lang}
          onClick={() => onChange(code)}
        >
          {code.toUpperCase()}
        </button>
      ))}
    </div>
  );
}

export default function Header({ models, selected, onSelect, lang, onLang, error, busy }) {
  const t = useT();
  const current = models.find((model) => model.id === selected);

  return (
    <header className="header">
      <div className="brand">
        <Mark />
        <h1>{t.title}</h1>
        <LangSwitch lang={lang} onChange={onLang} />
      </div>

      {error && (
        <p className="header-meta header-meta--warn">
          <span aria-hidden="true">⚠</span>
          {t.errors[error] ?? t.errors.network}
        </p>
      )}

      {current && (
        <div className="header-row">
          <ul className="chips">
            <li className="chip">{current.architecture}</li>
            <li className="chip">{t.species(current.num_classes)}</li>
            {current.accuracy != null && (
              <li className="chip chip--accent">top-1 {percent(current.accuracy)}</li>
            )}
          </ul>

          {models.length > 1 ? (
            <label className="model-picker">
              <span>{t.modelLabel}</span>
              <select
                value={selected}
                disabled={busy}
                onChange={(event) => onSelect(event.target.value)}
              >
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <span className="model-single">{current.name}</span>
          )}
        </div>
      )}
    </header>
  );
}
