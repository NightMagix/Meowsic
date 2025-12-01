import os
import threading
import asyncio
import time
import tempfile
import subprocess
import uuid
from typing import Dict, Any

import numpy as np
import librosa
import pyloudnorm as pyln

from flask import Flask, request

from openai import OpenAI

from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import CommandStart

# ============== КОНФИГ ==============

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY не найден в переменных окружения")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN не найден в переменных окружения")

client = OpenAI(api_key=OPENAI_API_KEY)
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# ============== НАСТРОЙКИ АНАЛИЗА ==============

TARGET_SR = 22050                 # рабочий sample rate
MAX_ANALYSIS_DURATION = 45.0      # макс. длительность для громкости, сек
MAX_SPECTRUM_DURATION = 15.0      # макс. длительность для спектра, сек

_METERS: Dict[int, pyln.Meter] = {}

# event loop бота для рассылки из Flask-потока
BOT_LOOP: asyncio.AbstractEventLoop | None = None

# список пользователей, которым можно слать рассылки
subscribers: set[int] = set()


def register_subscriber(chat_id: int):
    subscribers.add(chat_id)


def get_meter(sr: int) -> pyln.Meter:
    meter = _METERS.get(sr)
    if meter is None:
        meter = pyln.Meter(sr)
        _METERS[sr] = meter
    return meter


# ============== ЛИЧНОСТЬ МЯУЗИКА ==============

SYSTEM_PROMPT = """
Ты — Мяузик (Meowsic), цифровой кот-саундпродюсер.
Ты эксперт по звуку, миксу и мастерингу и даёшь рекомендации по цифрам: LUFS, пиковый уровень, динамический диапазон, спектр по полосам.
Всегда опирайся только на переданные параметры анализа, не придумывай, что ты "слышишь" трек.
Объясняй простым языком, но технически точно. Иногда можно мяукать: "мяу", "мур", "фрр".
"""

# ============== ИСТОРИИ ЧАТА ==============

user_histories: Dict[int, list] = {}


def update_history(uid: int, role: str, content: str):
    if uid not in user_histories:
        user_histories[uid] = [{"role": "system", "content": SYSTEM_PROMPT}]
    user_histories[uid].append({"role": role, "content": content})
    if len(user_histories[uid]) > 12:
        user_histories[uid] = [user_histories[uid][0]] + user_histories[uid][-10:]


# ============== РАССЫЛКА ==============

async def broadcast_message(text: str) -> int:
    count = 0
    for chat_id in list(subscribers):
        try:
            await bot.send_message(
                chat_id,
                f"📢 Сообщение от Meowsic:\n\n{text}"
            )
            count += 1
            await asyncio.sleep(0.05)
        except Exception as e:
            print("broadcast error:", chat_id, repr(e))
    return count


# ============== КЛАВИАТУРА ==============

main_keyboard = ReplyKeyboardMarkup(
    resize_keyboard=True,
    keyboard=[
        [KeyboardButton(text="Анализ трека")],
    ],
)

# ============== АУДИО-АНАЛИТИКА ==============


def prepare_audio_with_ffmpeg(src_path: str) -> str:
    """
    Через ffmpeg обрезаем до MAX_ANALYSIS_DURATION, приводим к mono 22050.
    Если ffmpeg недоступен, возвращаем исходный путь.
    """
    tmp_dir = tempfile.gettempdir()
    out_path = os.path.join(tmp_dir, f"meowsic_pre_{uuid.uuid4().hex}.wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", src_path,
        "-vn",
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-t", str(MAX_ANALYSIS_DURATION),
        out_path,
    ]
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        if os.path.exists(out_path):
            return out_path
    except Exception as e:
        print("ffmpeg error, fallback to original:", repr(e))
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
    return src_path


def load_audio_mono_fast(
    path: str,
    target_sr: int = TARGET_SR,
    max_duration: float = MAX_ANALYSIS_DURATION,
) -> tuple[np.ndarray, int, float]:
    """
    Быстрая загрузка уже подготовленного ffmpeg файла: моно, target_sr.
    """
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if y.size == 0:
        raise RuntimeError("Пустой аудиофайл")
    duration = len(y) / sr
    if duration > max_duration:
        samples = int(max_duration * sr)
        y = y[:samples]
        duration = max_duration
    return y.astype(np.float32), sr, float(duration)


def analyze_audio(y: np.ndarray, sr: int, duration_sec: float) -> Dict[str, Any]:
    meter = get_meter(sr)
    loudness = float(meter.integrated_loudness(y))

    peak_lin = float(np.max(np.abs(y)) + 1e-12)
    true_peak_db = 20.0 * np.log10(peak_lin)

    rms_lin = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    rms_db = 20.0 * np.log10(rms_lin)
    dr = float(true_peak_db - loudness)

    max_spec_samples = int(sr * MAX_SPECTRUM_DURATION)
    y_spec = y[:max_spec_samples] if len(y) > max_spec_samples else y

    spec = np.fft.rfft(y_spec)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(len(y_spec), 1.0 / sr)

    def band_energy_db(f_lo: float, f_hi: float) -> float:
        idx = np.where((freqs >= f_lo) & (freqs < f_hi))[0]
        if idx.size == 0:
            return -120.0
        e = float(np.mean(mag[idx] ** 2) + 1e-20)
        return 10.0 * np.log10(e)

    bands = {
        "sub": (20, 60),
        "bass": (60, 120),
        "low_mid": (120, 500),
        "mid": (500, 3000),
        "high_mid": (3000, 8000),
        "air": (8000, 20000),
    }

    band_db = {name: band_energy_db(*rng) for name, rng in bands.items()}
    tilt = band_db["air"] - band_db["bass"]

    return {
        "loudness_lufs": loudness,
        "true_peak_db": true_peak_db,
        "rms_db": rms_db,
        "dr": dr,
        "bands_db": band_db,
        "tilt_db": tilt,
        "duration_sec": duration_sec,
        "sr": sr,
    }


def format_analysis_for_llm(analysis: Dict[str, Any]) -> str:
    b = analysis["bands_db"]
    return f"""
Технический анализ (по усечённому фрагменту трека):
- Проанализированная длительность: {analysis['duration_sec']:.1f} сек
- Loudness (integrated LUFS): {analysis['loudness_lufs']:.2f} LUFS
- True Peak: {analysis['true_peak_db']:.2f} dBFS
- RMS: {analysis['rms_db']:.2f} dBFS
- Оценочный динамический диапазон (DR ≈ TP - LUFS): {analysis['dr']:.2f} dB

Спектральный баланс (примерные средние уровни по полосам, dB)
(рассчитан по первым ~{min(analysis['duration_sec'], MAX_SPECTRUM_DURATION):.0f} сек трека):
- Sub (20–60 Hz): {b['sub']:.2f} dB
- Bass (60–120 Hz): {b['bass']:.2f} dB
- Low-mid (120–500 Hz): {b['low_mid']:.2f} dB
- Mid (500–3000 Hz): {b['mid']:.2f} dB
- High-mid (3–8 kHz): {b['high_mid']:.2f} dB
- Air (8–20 kHz): {b['air']:.2f} dB

Общий спектральный наклон (Air - Bass): {analysis['tilt_db']:.2f} dB
"""


def analyze_file_sync(path: str) -> Dict[str, Any]:
    """
    Синхронный пайплайн: ffmpeg-подготовка -> загрузка -> анализ.
    Вызывается из отдельного потока.
    """
    prep_path = prepare_audio_with_ffmpeg(path)
    try:
        y, sr, dur = load_audio_mono_fast(prep_path)
        return analyze_audio(y, sr, dur)
    finally:
        if prep_path != path and os.path.exists(prep_path):
            try:
                os.remove(prep_path)
            except OSError:
                pass


# ============== КОМАНДЫ / КНОПКИ ==============

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    register_subscriber(message.chat.id)
    text = (
        "Мяу! Я Мяузик — кот-саундпродюсер.\n\n"
        "💿 Что я умею сейчас:\n"
        "• Пришлёшь трек — я по цифрам оценю громкость (LUFS), пики, динамику и спектр,\n"
        "  и дам рекомендации, что подкрутить в миксе/мастеринге.\n\n"
        "Я смотрю только первые ~45 секунд трека, чтобы отвечать быстрее.\n\n"
        "Просто скинь мне аудиофайл (как аудио или документ), или нажми кнопку «Анализ трека»."
    )
    await message.answer(text, reply_markup=main_keyboard)


@dp.message(F.text == "Анализ трека")
async def on_analysis_button(message: types.Message):
    register_subscriber(message.chat.id)
    await message.answer(
        "Мур! Отправь мне трек (как аудио или документ).\n"
        "Я быстро пробегусь по первым ~45 сек и дам отчёт по:\n"
        "• Loudness (LUFS)\n"
        "• True Peak\n"
        "• условному DR\n"
        "• балансу по частотным полосам\n\n"
        "И выдам понятный отчёт и рекомендации 😺",
        reply_markup=main_keyboard,
    )


# ============== ЗАГРУЗКА АУДИО И АНАЛИЗ ==============

async def download_audio_to_temp(message: types.Message) -> str:
    if message.audio:
        file_obj = message.audio
    elif message.document and message.document.mime_type and "audio" in message.document.mime_type:
        file_obj = message.document
    else:
        raise RuntimeError("Нет аудио в сообщении")

    tmp_dir = tempfile.gettempdir()
    ext = ".ogg"
    if file_obj.file_name and "." in file_obj.file_name:
        ext = "." + file_obj.file_name.split(".")[-1]

    tmp_path = os.path.join(tmp_dir, f"meowsic_{file_obj.file_id}{ext}")
    await bot.download(file_obj, destination=tmp_path)
    return tmp_path


@dp.message(F.audio | (F.document & F.document.mime_type.contains("audio")))
async def on_audio_message(message: types.Message):
    register_subscriber(message.chat.id)

    await message.answer(
        "Мяу, качаю и анализирую твой трек.\n"
        "Смотрю первые ~45 секунд, чтобы ответить побыстрее 🔍🎧"
    )

    tmp_path = None
    try:
        tmp_path = await download_audio_to_temp(message)
        analysis = await asyncio.to_thread(analyze_file_sync, tmp_path)
    except Exception as e:
        print("Audio processing error:", repr(e))
        await message.answer("Что-то пошло не так при чтении файла. Попробуй другой формат или файл, мяу.")
        return
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    analysis_text = format_analysis_for_llm(analysis)
    prompt = f"""
Пользователь прислал трек на анализ. Вот технические параметры (громкость, пики, динамика и спектр):

{analysis_text}

Сделай краткий, но полезный разбор:
1) Оцени громкость (LUFS, true peak, DR): тихо/норм/очень громко. Подходит ли под стриминги? под клуб?
2) Оцени спектр: низ, низ-середина, середина, верхняя середина, воздух. Где перебор, где нехватка.
3) Дай 5–10 конкретных рекомендаций по эквализации, компрессии и лимитеру.
4) Пиши в образе Meowsic — кот-саундпродюсер, немного с юмором, но без воды.
Ответ сделай компактным, чтобы его можно было прочитать с телефона.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=600,
        )
        answer = response.choices[0].message.content
        await message.answer(answer, reply_markup=main_keyboard)
    except Exception as e:
        print("OpenAI error (analysis):", repr(e))
        await message.answer(
            "Мур... не смог договориться с OpenAI. Попробуй ещё раз чуть позже.",
            reply_markup=main_keyboard,
        )


# ============== ОБЫЧНЫЙ ЧАТ ==============

@dp.message()
async def generic_chat(message: types.Message):
    chat_id = message.chat.id
    uid = message.from_user.id
    text = message.text or ""

    register_subscriber(chat_id)

    await bot.send_chat_action(chat_id, "typing")
    update_history(uid, "user", text)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=user_histories[uid],
            temperature=0.8,
            max_tokens=500,
        )
        answer = response.choices[0].message.content
        update_history(uid, "assistant", answer)
        await message.answer(answer, reply_markup=main_keyboard)
    except Exception as e:
        print("OpenAI error (chat):", repr(e))
        await message.answer(
            "Мяу... у меня лапки, что-то пошло не так с OpenAI. Попробуй ещё раз.",
            reply_markup=main_keyboard,
        )


# ============== FLASK ДЛЯ RENDER ==============

app = Flask(__name__)


HTML_FORM = """
<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8">
    <title>Meowsic Broadcast</title>
  </head>
  <body>
    <h2>Meowsic: рассылка сообщений</h2>
    <form method="post">
      <div>
        <label>Пароль:</label><br>
        <input type="password" name="password">
      </div>
      <div style="margin-top:10px;">
        <label>Сообщение для рассылки:</label><br>
        <textarea name="message" rows="6" cols="60"></textarea>
      </div>
      <div style="margin-top:10px;">
        <button type="submit">Отправить</button>
      </div>
    </form>
    <p style="color: green;">{status}</p>
  </body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return HTML_FORM.format(status="")
    password = request.form.get("password", "")
    text = request.form.get("message", "").strip()

    if password != "12345678":
        return HTML_FORM.format(status="Неверный пароль.")

    if not text:
        return HTML_FORM.format(status="Пустое сообщение.")

    global BOT_LOOP
    if BOT_LOOP is None:
        return HTML_FORM.format(status="Бот ещё не запущен.")

    try:
        fut = asyncio.run_coroutine_threadsafe(broadcast_message(text), BOT_LOOP)
        count = fut.result(timeout=60)
        return HTML_FORM.format(status=f"Отправлено {count} сообщений.")
    except Exception as e:
        print("broadcast exception:", repr(e))
        return HTML_FORM.format(status="Ошибка при рассылке.")


@app.route("/health")
def health():
    return "ok"


def start_web():
    port = int(os.environ.get("PORT", 10000))
    print(f"🌐 Meowsic: поднимаю веб-сервер на порту {port}...")
    app.run(host="0.0.0.0", port=port, threaded=True)


# ============== MAIN ==============

async def main():
    global BOT_LOOP
    BOT_LOOP = asyncio.get_running_loop()
    print("🎧 Meowsic: запускаю aiogram polling...")
    while True:
        try:
            await bot.delete_webhook(drop_pending_updates=True)
            await dp.start_polling(
                bot,
                allowed_updates=dp.resolve_used_update_types(),
            )
        except Exception as e:
            print("❌ Ошибка в polling:", repr(e))
            print("⏳ Перезапуск polling через 5 секунд...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    web_thread = threading.Thread(target=start_web, daemon=True)
    web_thread.start()
    time.sleep(1)
    asyncio.run(main())

