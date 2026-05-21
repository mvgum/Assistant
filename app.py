# -*- coding: utf-8 -*-
"""
Преобразование видео в текст (аудио -> транскрипция) с помощью Whisper.
Локальная версия (адаптировано из Google Colab).

Требования:
- Python 3.8+
- Установленный ffmpeg в системе (https://ffmpeg.org/download.html)
- Зависимости из requirements.txt
"""

import os
import sys
import argparse
import subprocess
import shutil
import whisper


def check_ffmpeg():
    """Проверяет, установлен ли ffmpeg в системе."""
    if shutil.which("ffmpeg") is None:
        print("❌ Ошибка: ffmpeg не найден в системе.")
        print("Установите его:")
        print("  - Windows: https://ffmpeg.org/download.html (и добавьте в PATH)")
        print("  - macOS:   brew install ffmpeg")
        print("  - Linux:   sudo apt-get install ffmpeg")
        sys.exit(1)


def extract_audio(video_path: str, audio_path: str) -> None:
    """Извлекает аудио из видео в формате wav (16kHz, mono)."""
    print("🎵 Извлекаем аудио из видео...")
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


def transcribe(audio_path: str, model_size: str = "base", language: str = "ru") -> str:
    """Транскрибирует аудио с помощью Whisper."""
    print(f"📥 Загружаем модель Whisper '{model_size}'...")
    model = whisper.load_model(model_size)

    print("🗣️  Выполняется распознавание речи...")
    # language=None — автоопределение
    lang = None if language.lower() in ("auto", "none", "") else language
    result = model.transcribe(audio_path, language=lang)
    return result["text"]


def main():
    parser = argparse.ArgumentParser(
        description="Преобразование видео в текст с помощью Whisper"
    )
    parser.add_argument(
        "video",
        help="Путь к видеофайлу (например, video.mp4)"
    )
    parser.add_argument(
        "-m", "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Размер модели Whisper (по умолчанию: base)"
    )
    parser.add_argument(
        "-l", "--language",
        default="ru",
        help="Язык распознавания: ru, en, auto и т.д. (по умолчанию: ru)"
    )
    parser.add_argument(
        "-o", "--output",
        default="transcription.txt",
        help="Путь для сохранения транскрипции (по умолчанию: transcription.txt)"
    )
    args = parser.parse_args()

    # Проверки
    check_ffmpeg()

    if not os.path.exists(args.video):
        print(f"❌ Файл не найден: {args.video}")
        sys.exit(1)

    print(f"📹 Работаем с файлом: {args.video}")

    audio_path = "temp_audio.wav"

    try:
        # 1. Извлекаем аудио
        extract_audio(args.video, audio_path)

        # 2. Распознаём речь
        text = transcribe(audio_path, model_size=args.model, language=args.language)

        # 3. Выводим и сохраняем результат
        print("\n=== РАСШИФРОВКА ===\n")
        print(text)

        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"\n✅ Транскрипция сохранена в: {args.output}")

    finally:
        # 4. Удаляем временный аудиофайл
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print("🧹 Временный аудиофайл удалён.")


if __name__ == "__main__":
    main()
