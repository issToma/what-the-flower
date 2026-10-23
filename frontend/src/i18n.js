import { createContext, useContext } from "react";

export const LANGS = ["ru", "en"];

export const STRINGS = {
  ru: {
    title: "What the flower?",
    greeting:
      "Привет. Пришли фотографию цветка, и я назову его вид — " +
      "из тех, что знаю по выбранной модели.",
    modelLabel: "Модель",
    species: (n) => `${n} видов`,
    checking: "проверяю модель…",
    lead: "Похоже на",
    uncertain: "Не уверен, что это один из видов, которые я знаю. Вот самые близкие варианты:",
    latency: (ms) => `${ms} мс на CPU`,
    pending: "Смотрю",
    thumbAlt: "Отправленное изображение",
    switched: (name) => `Переключился на «${name}»`,
    dropIdle: "Перетащи фотографию цветка сюда",
    dropActive: "Отпусти — сейчас посмотрю",
    dropBusy: "Смотрю на фотографию…",
    dropClick: "или нажми, чтобы выбрать файл",
    dropPaste: "вставь из буфера",
    errors: {
      not_an_image: "Это не похоже на изображение.",
      too_large: "Файл больше 10 МБ.",
      empty_file: "Файл пустой.",
      bad_image: "Не удалось прочитать изображение: оно повреждено или слишком большое.",
      unknown_model: "Такой модели нет.",
      no_models:
        "Модель не загружена. Обучи её по ноутбуку из notebooks/ " +
        "и положи файлы в папку model/.",
      network: "Сервер не отвечает.",
    },
  },
  en: {
    title: "What the flower?",
    greeting:
      "Hi. Send a photo of a flower and I will name its species — " +
      "from the list the selected model knows.",
    modelLabel: "Model",
    species: (n) => `${n} species`,
    checking: "checking the model…",
    lead: "Looks like",
    uncertain: "Not sure this is one of the species I know. Here are the closest guesses:",
    latency: (ms) => `${ms} ms on CPU`,
    pending: "Looking",
    thumbAlt: "Uploaded image",
    switched: (name) => `Switched to “${name}”`,
    dropIdle: "Drop a photo of a flower here",
    dropActive: "Let go — I will take a look",
    dropBusy: "Looking at the photo…",
    dropClick: "or click to pick a file",
    dropPaste: "paste from the clipboard",
    errors: {
      not_an_image: "That does not look like an image.",
      too_large: "The file is larger than 10 MB.",
      empty_file: "The file is empty.",
      bad_image: "Could not read the image: it is damaged or too large.",
      unknown_model: "No such model.",
      no_models:
        "No model is loaded. Train one with a notebook from notebooks/ " +
        "and put the files into the model/ folder.",
      network: "The server is not responding.",
    },
  },
};

export const LangContext = createContext("ru");

export function useT() {
  return STRINGS[useContext(LangContext)];
}

export function detectLang() {
  try {
    const saved = localStorage.getItem("lang");
    if (LANGS.includes(saved)) return saved;
  } catch {
    // приватный режим браузера запрещает доступ к localStorage
  }
  return navigator.language?.toLowerCase().startsWith("ru") ? "ru" : "en";
}

export function rememberLang(lang) {
  try {
    localStorage.setItem("lang", lang);
  } catch {
    // выбор не сохранится, но интерфейс работает
  }
}
