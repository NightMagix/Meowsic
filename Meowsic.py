import os
import threading
import asyncio
import time
import tempfile
from typing import Dict, Any

import numpy as np
import librosa
import pyloudnorm as pyln

from flask import Flask
from openai import OpenAI

from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import CommandStart, Command

# ===================== КОНФИГ =====================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY не найден в переменных окружения")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN не найден в переменных окружения")

client = OpenAI(api_key=OPENAI_API_KEY)

# --------- Gemini SDK ---------
try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = GEMINI_API_KEY is not None
except Exception:
    genai = None
    GEMINI_AVAILABLE = False

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# ===================== ЛИЧНОСТЬ МЯУЗИКА =====================

SYSTEM_PROMPT = """
Ты — Meowsic, цифровой кот-саундпродюсер.
Говоришь компактно, дружелюбно, местами мяукаешь: "мяу", "мур".
Делаешь разбор по:
- громкости: LUFS, true peak, DR;
- спектру: sub, bass, low-mid, mid, high-mid, air;
- даёшь рекомендации по EQ/компрессии/лимитеру.
Не придумывай, что "слышишь трек" — опирайся только на предоставленные цифры.
"""

# ===================== СОСТОЯНИЕ ПОЛЬЗОВАТЕЛЕЙ =====================

user_histories: Dict[int, list] = {}
user_llm: Dict[int, str] = {}  # gpt / gemini


def set_user_model(uid: int, model: str):
    user_llm[uid] = model


def get_user_model(uid: int) -> str:
    return user_llm.get(uid, "gpt")  # по умолчанию GPT


def update_history(uid: int, role: str, content: str):
    """Храним только последние 4 сообщения + системное."""
    if uid not in user_histories:
        user_histories[uid] = [{"role": "system", "content": SYSTEM_PROMPT}]
    user_histories[uid].append({"role": role, "content": content})

    if len(user_histories[uid]) > 6:
        user_histories[uid] = [user_histories[uid][0]] + user_histories[uid][-5:]


# ===================== ВЫЗОВ МОДЕЛИ =====================

async def call_llm(uid: int, messages: list[dict], max_tokens: int, temperature: float = 0.7) -> str:
    model_choice = get_user_model(uid)

    # ---------- GEMINI ----------
    if model_choice == "gemini" and GEMINI_AVAILABLE and genai:
        try:
            prompt_text = ""
            for m in messages:
                role = m.get("role", "user")
                prefix = {"system": "[SYSTEM]", "assistant": "[AI]", "user": "[USER]"}[role]
                prompt_text += f"{prefix} {m['content']}\n\n"

            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            resp = model.generate_content(
                prompt_text,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                },
            )
            return (resp.text or "").strip()
        except Exception as e:
            print("Gemini error:", repr(e))
            # fallback → GPT
            model_choice = "gpt"

    # ---------- GPT ----------
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


# ===================== КЛАВИАТУРА =====================

main_keyboard = ReplyKeyboardMarkup(
    resize_keyboard=True,
    keyboard=[
        [KeyboardButton(text="Анализ трека")],
    ],
)

# ===================== АУДИО-АНАЛИТИКА =====================

def load_audio_mono_fast(path: str, target_sr: int = 22050, max_duration: float = 120.0):
    y, sr = librosa.load(path, sr=target_sr, mono=True, duration=max_duration)
    duration = len(y) / sr
    return y.astype(np.float32), sr, duration


def analyze_audio(y: np.ndarray, sr: int, duration_sec: float):
    meter = pyln.Meter(sr)
    loudness = float(meter.integrated_loudness(y))

    peak_lin = float(np.max(np.abs(y)) + 1e-12)
    true_peak_db = 20 * np.log10(peak_lin)

    rms_lin = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    rms_db = 20 * np.log10(rms_lin)
    dr = float(true_peak_db - loudness)

    spec = np.fft.rfft(y)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)

    def band(f_lo, f_hi):
        idx = np.where((freqs >= f_lo) & (freqs < f_hi))[0]
        if len(idx) == 0:
            return -120
        e = float(np.mean(mag[idx] ** 2) + 1e-20)
        return 10 * np.log10(e)

    bands = {
        "sub": band(20, 60),
        "bass": band(60, 120),
        "low_mid": band(120, 500),
        "mid": band(500, 3000),
        "high_mid": band(3000, 8000),
        "air": band(8000, 20000),
    }

    tilt = bands["air"] - bands["bass"]

    return {
        "loudness_lufs": loudness,
        "true_peak_db": true_peak_db,
        "rms_db": rms_db,
        "dr": dr,
        "bands": bands,
        "tilt": tilt,
        "duration_sec": duration_sec,
    }


def format_analysis(a):
    b = a["bands"]
    return (
        f"dur={a['duration_sec']:.1f}s; "
        f"LUFS={a['loudness_lufs']:.1f}; "
        f"TP={a['true_peak_db']:.1f}dB; "
        f"DR≈{a['dr']:.1f}dB; "
        f"sub={b['sub']:.1f}, bass={b['bass']:.1f}, lowmid={b['low_mid']:.1f}, "
        f"mid={b['mid']:.1f}, highmid={b['high_mid']:.1f}, air={b['air']:.1f}; "
        f"tilt={a['tilt']:.1f}"
    )


# ===================== КОМАНДЫ =====================

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    uid = message.from_user.id
    set_user_model(uid, "gpt")

    await message.answer(
        "Мяу! Я Meowsic — твой кот-саундпродюсер.\n\n"
        "Пришли мне трек — я быстро разберу его по цифрам: громкость, динамика, спектр.\n"
        "Используй кнопки внизу.\n\n"
        "Команды:\n"
        "• /gpt — использовать GPT\n"
        "• /gemini — использовать Gemini",
        reply_markup=main_keyboard,
    )


@dp.message(Command("gpt"))
async def cmd_gpt(message: types.Message):
    set_user_model(message.from_user.id, "gpt")
    await message.answer("Мяу! Работаю в своём обычном режиме.", reply_markup=main_keyboard)


@dp.message(Command("gemini"))
async def cmd_gemini(message: types.Message):
    if not GEMINI_AVAILABLE:
        await message.answer("Мур… Gemini недоступен. Остаюсь в обычном режиме.", reply_markup=main_keyboard)
        return

    set_user_model(message.from_user.id, "gemini")
    await message.answer("Мур! Переключился. Готов работать!", reply_markup=main_keyboard)


@dp.message(F.text == "Анализ трека")
async def on_press_analysis(message: types.Message):
    await message.answer(
        "Пришли мне аудиофайл (как аудио или документ). "
        "Я проанализирую первые ~2 минуты и дам рекомендации.",
        reply_markup=main_keyboard,
    )


# ===================== АНАЛИЗ АУДИО =====================

async def download_audio(message: types.Message):
    file_obj = message.audio or message.document
    tmp = os.path.join(tempfile.gettempdir(), f"meowsic_{file_obj.file_id}.tmp")
    await bot.download(file_obj, destination=tmp)
    return tmp


@dp.message(F.audio | (F.document & F.document.mime_type.contains("audio")))
async def handle_audio(message: types.Message):
    uid = message.from_user.id
    await message.answer("Мяу… анализирую твой трек…")

    try:
        path = await download_audio(message)
        y, sr, dur = load_audio_mono_fast(path)
        a = analyze_audio(y, sr, dur)
    except Exception as e:
        print("Audio error:", repr(e))
        await message.answer("Не смог прочитать файл. Попробуй другой формат.")
        return

    compact = format_analysis(a)

    prompt = (
        "Вот результаты анализа трека:\n"
        f"{compact}\n\n"
        "Сделай короткий разбор:\n"
        "1) громкость: тихо/норм/громко, подходит ли под стриминг (-14) или громкий мастерин (-9..-7)\n"
        "2) динамика: DR\n"
        "3) спектр: переборы/провалы\n"
        "4) 5–7 советов по EQ/компрессии/лимитеру\n"
        "Пиши компактно, дружелюбно, как кот."
    )

    response = await call_llm(
        uid,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=350,
        temperature=0.6,
    )

    await message.answer(response, reply_markup=main_keyboard)


# ===================== ОБЩЕНИЕ =====================

@dp.message()
async def general_chat(message: types.Message):
    uid = message.from_user.id
    update_history(uid, "user", message.text)

    response = await call_llm(
        uid,
        messages=user_histories[uid],
        max_tokens=250,
        temperature=0.8,
    )

    update_history(uid, "assistant", response)
    await message.answer(response, reply_markup=main_keyboard)


# ===================== FLASK (для Render) =====================

app = Flask(__name__)

@app.route("/")
def index():
    return "Meowsic is alive 🐾"

@app.route("/health")
def health():
    return "ok"


def start_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=True)


# ===================== MAIN =====================

async def main():
    print("🐾 Meowsic running…")
    while True:
        try:
            await bot.delete_webhook(drop_pending_updates=True)
            await dp.start_polling(bot)
        except Exception as e:
            print("Polling error:", repr(e))
            await asyncio.sleep(5)


if __name__ == "__main__":
    threading.Thread(target=start_web, daemon=True).start()
    time.sleep(1)
    asyncio.run(main())
