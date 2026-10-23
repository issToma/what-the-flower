async function detail(response, fallback) {
  try {
    const body = await response.json();
    return body.detail ?? fallback;
  } catch {
    return fallback;
  }
}

export async function fetchModels() {
  const response = await fetch("/api/models");
  if (!response.ok) throw new Error(await detail(response, "модели недоступны"));
  return response.json();
}

export async function predict(file, modelId) {
  const form = new FormData();
  form.append("image", file);
  if (modelId) form.append("model", modelId);

  const response = await fetch("/api/predict", { method: "POST", body: form });
  if (!response.ok) {
    throw new Error(await detail(response, `ошибка ${response.status}`));
  }
  return response.json();
}
