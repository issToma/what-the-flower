import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // В разработке фронтенд живёт на 5173, API — на 8000.
    // Прокси избавляет от CORS и делает пути в коде одинаковыми
    // для dev и для прода, где FastAPI раздаёт собранный dist.
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
