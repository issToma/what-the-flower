/** Ошибка от сервера. Код машиночитаемый, текст — запасной вариант. */
export class ApiError extends Error {
  constructor(code, detail) {
    super(detail);
    this.code = code;
  }
}

async function toError(response) {
  try {
    const body = await response.json();
    return new ApiError(body.code, body.detail ?? `ошибка ${response.status}`);
  } catch {
    return new ApiError(undefined, `ошибка ${response.status}`);
  }
}

export async function fetchModels() {
  let response;
  try {
    response = await fetch("/api/models");
  } catch {
    throw new ApiError("network", "network error");
  }
  if (!response.ok) throw await toError(response);
  return response.json();
}

export async function predict(file, modelId) {
  const form = new FormData();
  form.append("image", file);
  if (modelId) form.append("model", modelId);

  let response;
  try {
    response = await fetch("/api/predict", { method: "POST", body: form });
  } catch {
    throw new ApiError("network", "network error");
  }
  if (!response.ok) throw await toError(response);
  return response.json();
}
