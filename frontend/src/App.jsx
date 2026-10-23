import { useCallback, useEffect, useRef, useState } from "react";

import Composer from "./components/Composer.jsx";
import Header from "./components/Header.jsx";
import Message from "./components/Message.jsx";
import { fetchModels, predict } from "./api.js";

const GREETING = {
  id: "greeting",
  role: "assistant",
  text:
    "Привет. Пришли фотографию цветка, и я назову его вид — " +
    "из тех, что знаю по выбранной модели.",
};

let nextId = 1;
const makeId = () => `m${nextId++}`;

export default function App() {
  const [messages, setMessages] = useState([GREETING]);
  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    fetchModels()
      .then(({ models: list, default: fallback }) => {
        setModels(list);
        setSelected(fallback);
      })
      .catch((exc) => setError(exc.message));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const releaseUrl = useCallback((url) => URL.revokeObjectURL(url), []);

  const nameOf = useCallback(
    (id) => models.find((model) => model.id === id)?.name ?? id,
    [models],
  );

  const send = useCallback(
    async (file) => {
      if (!file.type.startsWith("image/")) return;

      const pendingId = makeId();
      setBusy(true);
      setMessages((previous) => [
        ...previous,
        { id: makeId(), role: "user", imageUrl: URL.createObjectURL(file) },
        { id: pendingId, role: "assistant", pending: true },
      ]);

      try {
        const result = await predict(file, selected);
        setMessages((previous) =>
          previous.map((message) =>
            message.id === pendingId ? { ...message, pending: false, result } : message,
          ),
        );
      } catch (exc) {
        setMessages((previous) =>
          previous.map((message) =>
            message.id === pendingId
              ? { ...message, pending: false, error: exc.message }
              : message,
          ),
        );
      } finally {
        setBusy(false);
      }
    },
    [selected],
  );

  function switchModel(id) {
    setSelected(id);
    setMessages((previous) => [
      ...previous,
      { id: makeId(), role: "system", text: `Переключился на «${nameOf(id)}»` },
    ]);
  }

  useEffect(() => {
    function handlePaste(event) {
      const file = [...event.clipboardData.files][0];
      if (file) send(file);
    }
    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, [send]);

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    const [file] = event.dataTransfer.files;
    if (file) send(file);
  }

  return (
    <div
      className={`app${dragging ? " app--dragging" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <Header
        models={models}
        selected={selected}
        onSelect={switchModel}
        error={error}
        busy={busy}
      />

      <main className="chat">
        {messages.map((message) => (
          <Message
            key={message.id}
            message={message}
            modelName={models.length > 1 ? nameOf(message.result?.model) : null}
            onRelease={releaseUrl}
          />
        ))}
        <div ref={endRef} />
      </main>

      <footer className="footer">
        <Composer
          onPick={send}
          disabled={busy || !selected}
          dragging={dragging}
          busy={busy}
        />
      </footer>
    </div>
  );
}
