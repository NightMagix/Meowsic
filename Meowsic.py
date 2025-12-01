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

# ==== КОНФИГ ====

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY не найден в переменных окружения")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN не найден в переменных окружения")

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Gemini (опционально) ---
try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None

if GEMINI_API_KEY and genai is not None:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    genai = None  # чтобы ниже было понятно, что Gemini недоступен

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# ==== ЛИЧНОСТЬ МЯУЗИКА (КОРОТКАЯ) ====

SYSTEM_PROMPT = """
Ты — Meowsic, цифровой кот-саундпродюсер.
Кратко и по делу объясняешь результаты анализа трека:
- громкость: LUFS, true peak, DR;
- спектр: низ, низ-средина, средина, верхняя середина, воздух;
- даёшь практичные советы по эквализации, компрессии и лимитеру.
Всегда опираешься только на переданные численные параметры, не придумывая, что ты "слышишь" трек.
Пиши компактно, максимум около 1200–1500 символов, используй списки.
Иногда можно вставлять кошачьи вставки "мяу", "мур", но без перегруза.
"""

# ==== ХРАНЕНИЕ ИСТОРИИ И ВЫБОР МОДЕЛИ ====

user_histories: Dict[int, list] = {}
user_llm: Dict[int, str] = {}  # "gpt" или "gemini"


def get_user_model(uid: int) -> str:
    # по умолчанию GPT
    return user_llm.get(uid, "gpt")


def set_user_model(uid: int, model: str):
    user_llm[uid] = model


def update_history(uid: int, role: str, content: str):
    if uid not in user_histories:
        user_histories[uid] = [{"role": "system", "content": SYSTEM_PROMPT}]
    user_histories[uid].append({"role": role, "content": content})
    # системное + последние 4 сообщения
    if len(user_histories[uid]) > 6:
        user_histories[uid] = [user_histories[uid][0]] + user_histories[uid][-5:]


async def call_llm(
    uid: int,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.7,
) -> str:
    """
    Универсальный вызов LLM:
    - если пользователь выбрал /gemini и есть ключ + библиотека — используем Gemini;
    - иначе — GPT-4.1-mini.
    """
    model_choice = get_user_model(uid)

    # ----- Gemini -----
    if model_choice == "gemini" and genai is not None and GEMINI_API_KEY:
        # Для простоты превращаем chat-историю в одну строку.
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not content:
                continue
            if role == "system":
                prefix = "[SYSTEM]"
            elif role == "assistant":
                prefix = "[ASSISTANT]"
            else:
                prefix = "[USER]"
            parts.append(f"{prefix} {content}")
        prompt_text = "\n\n".join(parts)

        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(
            prompt_text,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        return (resp.text or "").strip()

    # ----- GPT по умолчанию -----
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


# ==== КЛАВИАТУРА ====

main_keyboard = ReplyKeyboardMarkup(
    resize_keyboard=True,
    keyboard=[
        [KeyboardButton(text="Анализ трека")],
    ],
)

# ==== АУДИО-АНАЛИТИКА (БЫСТРАЯ) ====

def load_audio_mono_fast(
    path: str,
    target_sr: int = 22050,
    max_duration: float = 120.0,
) -> tuple[np.ndarray, int, float]:
    """Быстрая загрузка: моно, пониженный SR, ограничение по длительности."""
    y, sr = librosa.load(path, sr=target_sr, mono=True, duration=max_duration)
    if y.size == 0:
        raise RuntimeError("Пустой аудиофайл")
    duration = len(y) / sr
    return y.astype(np.float32), sr, float(duration)


def analyze_audio(y: np.ndarray, sr: int, duration_sec: float) -> Dict[str, Any]:
    meter = pyln.Meter(sr)
    loudness = float(meter.integrated_loudness(y))

    peak_lin = float(np.max(np.abs(y)) + 1e-12)
    true_peak_db = 20.0 * np.log10(peak_lin)

    rms_lin = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    rms_db = 20.0 * np.log10(rms_lin)
    dr = float(true_peak_db - loudness)

    spec = np.fft.rfft(y)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)

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


def format_analysis_compact(analysis: Dict[str, Any]) -> str:
    """Компактное представление анализа для LLM (минимум токенов)."""
    b = analysis["bands_db"]
    return (
        f"dur_sec={analysis['duration_sec']:.1f}; "
        f"LUFS={analysis['loudness_lufs']:.1f}; "
        f"TP={analysis['true_peak_db']:.1f} dBFS; "
        f"RMS={analysis['rms_db']:.1f} dBFS; "
        f"DR≈{analysis['dr']:.1f} dB; "
        f"bands(dB): sub={b['sub']:.1f}, bass={b['bass']:.1f}, "
        f"low_mid={b['low_mid']:.1f}, mid={b['mid']:.1f}, "
        f"high_mid={b['high_mid']:.1f}, air={b['air']:.1f}; "
        f"tilt(Air-Bass)={analysis['tilt_db']:.1f} dB."
    )


# ==== КОМАНДЫ / ПЕРЕКЛЮЧЕНИЕ МОДЕЛИ ====

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    uid = message.from_user.id
    set_user_model(uid, "gpt")  # по умолчанию GPT
    text = (
        "Мяу! Я Meowsic — кот-саундпродюсер.\n\n"
        "Сейчас я умею быстро разбирать твой трек по цифрам:\n"
        "• громкость (LUFS, true peak, DR)\n"
        "• спектр по полосам\n\n"
        "Просто пришли мне аудиофайл (как аудио или документ) — я проанализирую первые ~2 минуты "
        "и дам короткие рекомендации.\n\n"
        "Команды моделей:\n"
        "• /gpt — использовать GPT-4.1-mini\n"
        "• /gemini — использовать Gemini (если настроен GEMINI_API_KEY)"
    )
    await message.answer(text, reply_markup=main_keyboard)


@dp.message(Command("gpt"))
async def cmd_gpt(message: types.Message):
    uid = message.from_user.id
    set_user_model(uid, "gpt")
    await message.answer(
        "Мяу! Теперь я отвечаю через GPT-4.1-mini. Это основной режим.",
        reply_markup=main_keyboard,
    )


@dp.message(Command("gemini"))
async def cmd_gemini(message: types.Message):
    uid = message.from_user.id
    if genai is None or not GEMINI_API_KEY:
        await message.answer(
            "Мур… Gemini сейчас недоступен (нет библиотеки или GEMINI_API_KEY). "
            "Остаюсь на GPT-4.1-mini.",
            reply_markup=main_keyboard,
        )
        return
    set_user_model(uid, "gemini")
    await message.answer(
        "Мур! Теперь я буду отвечать через Gemini (gemini-1.5-flash). "
        "Если что, вернуться к GPT можно командой /gpt.",
        reply_markup=main_keyboard,
    )


@dp.message(F.text == "Анализ трека")
async def on_analysis_button(message: types.Message):
    await message.answer(
        "Отправь трек как аудио или документ. Я быстро прогоню первые ~2 минуты и дам советы по "
        "громкости, динамике и спектру. Мур!",
        reply_markup=main_keyboard,
    )


# ==== ЗАГРУЗКА АУДИО И АНАЛИЗ ====

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
    uid = message.from_user.id
    await message.answer("Мяу, скачиваю и жму твой трек в анализ...")

    try:
        tmp_path = await download_audio_to_temp(message)
        y, sr, dur = load_audio_mono_fast(tmp_path)
        analysis = analyze_audio(y, sr, dur)
    except Exception as e:
        print("Audio processing error:", repr(e))
        await message.answer("Не получилось прочитать файл. Попробуй другой формат или перезакинь, мяу.")
        return

    compact = format_analysis_compact(analysis)

    prompt = (
        "Вот численные результаты анализа фрагмента трека "
        "(громкость, динамика, спектр):\n\n"
        f"{compact}\n\n"
        "Сделай КРАТКИЙ разбор для звукорежиссёра:\n"
        "1) Оцени громкость: тихо/норм/громко, подходит ли под стриминг (≈ -14 LUFS) "
        "и под современный громкий мастерин (≈ -9…-7 LUFS).\n"
        "2) Оцени динамику по DR: зажатый / средний / живой.\n"
        "3) Оцени спектр: где перебор или провал (sub, bass, low-mid, mid, high-mid, air).\n"
        "4) Дай 5–7 конкретных советов: где примерно поднять/срезать EQ (диапазоны и ±дБ), "
        "нужна ли компрессия/лимитер.\n"
        "Пиши очень компактно, без воды, максимум 8 пунктов. Используй списки."
    )

    try:
        answer = await call_llm(
            uid,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=350,
        )
        await message.answer(answer, reply_markup=main_keyboard)
    except Exception as e:
        print("LLM error (analysis):", repr(e))
        await message.answer(
            "Мур... с моделью что-то не срослось (лимит или сеть). Попробуй ещё раз чуть позже.",
            reply_markup=main_keyboard,
        )


# ==== ОБЫЧНЫЙ ЧАТ ====

@dp.message()
async def generic_chat(message: types.Message):
    chat_id = message.chat.id
    uid = message.from_user.id
    text = message.text or ""

    await bot.send_chat_action(chat_id, "typing")
    update_history(uid, "user", text)

    try:
        answer = await call_llm(
            uid,
            messages=user_histories[uid],
            temperature=0.8,
            max_tokens=220,
        )
        update_history(uid, "assistant", answer)
        await message.answer(answer, reply_markup=main_keyboard)
    except Exception as e:
        print("LLM error (chat):", repr(e))
        await message.answer(
            "Мяу... у меня лапки, модель сейчас не отвечает. Попробуй ещё раз.",
            reply_markup=main_keyboard,
        )


# ==== FLASK ДЛЯ RENDER ====

app = Flask(__name__)

@app.route("/")
def index():
    return "Meowsic bot is alive 🐾 (GPT/Gemini switch)"

@app.route("/health")
def health():
    return "ok"


def start_web():
    port = int(os.environ.get("PORT", 10000))
    print(f"🌐 Meowsic: поднимаю веб-сервер на порту {port}...")
    app.run(host="0.0.0.0", port=port, threaded=True)


# ==== MAIN ====

async def main():
    print("🎧 Meowsic: запускаю aiogram polling (with GPT/Gemini switch)...")
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
