# 🎬 Видео → Текст (Whisper Web App)

Веб-приложение на Streamlit для преобразования видео и аудио в текст с помощью [OpenAI Whisper](https://github.com/openai/whisper).

## ✨ Возможности

- 📤 Загрузка видео/аудио прямо в браузере
- 🌍 Поддержка множества языков (автоопределение)
- 🎚️ Выбор размера модели (от tiny до large)
- 📝 Экспорт в **TXT**, **TXT с тайм-кодами** и **SRT-субтитры**
- 🎞️ Встроенный предпросмотр видео/аудио

## 🚀 Локальный запуск

### 1. Установите ffmpeg

- **Windows:** [ffmpeg.org/download](https://ffmpeg.org/download.html) (добавьте в PATH)
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt-get install ffmpeg`

### 2. Установите зависимости

```bash
git clone https://github.com/USERNAME/REPO.git
cd REPO
pip install -r requirements.txt
