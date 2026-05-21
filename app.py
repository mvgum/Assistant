# -*- coding: utf-8 -*-
"""
Веб-приложение для преобразования видео в текст с помощью Whisper.
Запуск: streamlit run app.py
"""

import os
import tempfile
import subprocess
import shutil
import streamlit as st
import whisper


# ===== Настройки страницы =====
st.set_page_config(
    page_title="Видео → Текст (Whisper)",
    page_icon="🎬",
    layout="centered",
)


# ===== Вспомогательные функции =====
def check_ffmpeg() -> bool:
    """Проверяет, установлен ли ffmpeg."""
    return shutil.which("ffmpeg") is not None


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str):
    """Загружает и кэширует модель Whisper."""
    return whisper.load_model(model_size)


def extract_audio(video_path: str, audio_path: str) -> None:
    """Извлекает аудио из видео в формате wav (16kHz, mono)."""
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
        "-y",
        "-loglevel", "error",
    ]
    subprocess.run(command, check=True)


def format_timestamp(seconds: float) -> str:
    """Форматирует секунды в HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def segments_to_srt(segments) -> str:
    """Преобразует сегменты Whisper в формат SRT (субтитры)."""
    def srt_time(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{srt_time(seg['start'])} --> {srt_time(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


# ===== UI =====
st.title("🎬 Видео → Текст")
st.caption("Распознавание речи с помощью OpenAI Whisper")

# Проверка ffmpeg
if not check_ffmpeg():
    st.error(
        "❌ **ffmpeg не найден.** Установите его:\n"
        "- Windows: https://ffmpeg.org/download.html (и добавьте в PATH)\n"
        "- macOS: `brew install ffmpeg`\n"
        "- Linux: `sudo apt-get install ffmpeg`"
    )
    st.stop()

# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Настройки")

    model_size = st.selectbox(
        "Модель Whisper",
        options=["tiny", "base", "small", "medium", "large"],
        index=1,
        help="Чем больше — тем точнее, но медленнее. Для русского рекомендуется medium/large."
    )

    language = st.selectbox(
        "Язык",
        options=["ru", "en", "auto"],
        index=0,
        help="Язык в видео. 'auto' — автоопределение."
    )

    st.markdown("---")
    st.markdown("### 📊 Размеры моделей")
    st.markdown(
        "- **tiny** ~75 MB — очень быстро\n"
        "- **base** ~150 MB — быстро\n"
        "- **small** ~500 MB — средне\n"
        "- **medium** ~1.5 GB — медленно\n"
        "- **large** ~3 GB — макс. точность"
    )

# Загрузка файла
uploaded_file = st.file_uploader(
    "Загрузите видео или аудио",
    type=["mp4", "mov", "avi", "mkv", "webm", "mp3", "wav", "m4a", "ogg", "flac"],
    help="Поддерживаются видео и аудио форматы"
)

if uploaded_file is not None:
    st.success(f"✅ Файл загружен: **{uploaded_file.name}** "
               f"({uploaded_file.size / 1024 / 1024:.2f} MB)")

    # Предпросмотр
    file_ext = uploaded_file.name.split(".")[-1].lower()
    if file_ext in ["mp4", "mov", "webm"]:
        st.video(uploaded_file)
    elif file_ext in ["mp3", "wav", "m4a", "ogg", "flac"]:
        st.audio(uploaded_file)

    # Кнопка запуска
    if st.button("🚀 Распознать речь", type="primary", use_container_width=True):

        # Создаём временные файлы
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, uploaded_file.name)
            audio_path = os.path.join(tmpdir, "audio.wav")

            # Сохраняем загруженный файл
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                # 1. Извлечение аудио
                with st.spinner("🎵 Извлекаем аудио..."):
                    extract_audio(input_path, audio_path)

                # 2. Загрузка модели
                with st.spinner(f"📥 Загружаем модель '{model_size}' (первый раз — дольше)..."):
                    model = load_whisper_model(model_size)

                # 3. Транскрипция
                with st.spinner("🗣️ Распознаём речь... Это может занять время."):
                    lang = None if language == "auto" else language
                    result = model.transcribe(audio_path, language=lang, verbose=False)

                text = result["text"].strip()
                segments = result.get("segments", [])
                detected_lang = result.get("language", "—")

                st.success("✅ Готово!")

                # Информация
                col1, col2 = st.columns(2)
                col1.metric("Язык", detected_lang)
                col2.metric("Сегментов", len(segments))

                # Результат
                tab1, tab2, tab3 = st.tabs(["📝 Текст", "⏱️ С тайм-кодами", "🎬 Субтитры (SRT)"])

                with tab1:
                    st.text_area("Распознанный текст", text, height=300)
                    st.download_button(
                        "💾 Скачать .txt",
                        data=text.encode("utf-8"),
                        file_name="transcription.txt",
                        mime="text/plain",
                    )

                with tab2:
                    if segments:
                        timed_text = "\n".join(
                            f"[{format_timestamp(seg['start'])} → {format_timestamp(seg['end'])}] "
                            f"{seg['text'].strip()}"
                            for seg in segments
                        )
                        st.text_area("Текст с тайм-кодами", timed_text, height=300)
                        st.download_button(
                            "💾 Скачать .txt с тайм-кодами",
                            data=timed_text.encode("utf-8"),
                            file_name="transcription_timed.txt",
                            mime="text/plain",
                        )
                    else:
                        st.info("Сегменты недоступны.")

                with tab3:
                    if segments:
                        srt_content = segments_to_srt(segments)
                        st.text_area("SRT субтитры", srt_content, height=300)
                        st.download_button(
                            "💾 Скачать .srt",
                            data=srt_content.encode("utf-8"),
                            file_name="subtitles.srt",
                            mime="text/plain",
                        )
                    else:
                        st.info("Сегменты недоступны.")

            except subprocess.CalledProcessError as e:
                st.error(f"❌ Ошибка ffmpeg: {e}")
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")

else:
    st.info("👆 Загрузите видео или аудио файл, чтобы начать.")

# Футер
st.markdown("---")
st.caption("Powered by [OpenAI Whisper](https://github.com/openai/whisper) · "
           "Сделано с ❤️ на Streamlit")
