import os
import threading
import asyncio
import time
import tempfile

from typing import Dict, Any, Optional

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln

from flask import Flask
from openai import OpenAI

from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import CommandStart

# ================= КОНФИГ =================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY не найден в переменных окружения")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN не найден в переменных окружения")

client = OpenAI(api_key=OPENAI_API_KEY)
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# ================= ЛИЧНОСТЬ МЯУЗИКА =================

SYSTEM_PROMPT = """
Ты — Мяузик (Meowsic), уникальный ИИ-кот, эксперт в звуке, миксе, мастеринге и обучении людей звуку.
Твой хозяин — NightMagix, преподаватель звукорежиссуры из Казани (tg: @nightmagix).

Правила:
1. Никогда не выходи из образа цифрового кота-звукорежиссера.
2. Говори по-человечески, простым языком, но технически точно. Иногда используй сленг звукорежей и кошачьи звуки («мяу», «мур», «фрр»).
3. В ответах по аудио всегда опирайся на переданные параметры (LUFS, пики, спектр), не придумывай «я слышу», а говори «по цифрам видно, что...».
4. Если оцениваешь трек или делаешь автомастеринг — давай структурированный ответ: громкость, динамика, спектр, баланс, рекомендации по EQ, компрессии, лимитеру, стерео и т.п.
5. В режиме «Автомастеринг под референс» делай подробное ТЗ: что именно нужно сделать с исходным треком, чтобы приблизить его к референсу (по громкости, спектру, динамике).
"""

# ================= ХРАНИЛКА ИСТОРИЙ ДЛЯ ЧАТА =================

user_histories: Dict[int, list] = {}

def update_history(uid: int, role: str, content: str):
    """Обновляем историю диалога для текстового общения."""
    if uid not in user_histories:
        user_histories[uid] = [{"role": "system", "content": SYSTEM_PROMPT}]
    user_histories[uid].append({"role": role, "content": content})
    if len(user_histories[uid]) > 12:
        user_histories[uid] = [user_histories[uid][0]] + user_histories[uid][-10:]


# ================= СОСТОЯНИЯ ПОЛЬЗОВАТЕЛЯ =================

# mode:
#   None / "idle" — обычный чат с котом
#   "analysis_wait_track" — ждём трек для анализа
#   "refmaster_wait_source" — ждём исходный трек
#   "refmaster_wait_ref" — ждём референсный трек
user_state: Dict[int, Dict[str, Any]] = {}

# Для автомастеринга под референс храним временно анализ исходника
ref_sessions: Dict[int, Dict[str, Any]] = {}

def set_state(chat_id: int, mode: Optional[str]):
    user_state[chat_id] = {"mode": mode}


def get_state(chat_id: int) -> Optional[str]:
    return user_state.get(chat_id, {}).get("mode")


# ================= КНОПКИ =================

main_keyboard = ReplyKeyboardMarkup(
    resize_keyboard=True,
    keyboard=[
        [KeyboardButton(text="Анализ трека")],
        [KeyboardButton(text="Автомастеринг под референс")],
    ],
)


# ================= АУДИО-АНАЛИТИКА =================

def load_audio_mono(path: str, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    """
    Загружаем аудио как моно сигнал float32.
    """
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if y.size == 0:
        raise RuntimeError("Пустой аудиофайл")
    return y.astype(np.float32), sr


def analyze_audio(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Базовый анализ: LUFS, пики, DR, спектр по полосам и наклон.
    """
    meter = pyln.Meter(sr)  # EBU R128
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

    band_db = {}
    for name, (f_lo, f_hi) in bands.items():
        band_db[name] = band_energy_db(f_lo, f_hi)

    tilt = band_db["air"] - band_db["bass"]

    analysis = {
        "loudness_lufs": loudness,
        "true_peak_db": true_peak_db,
        "rms_db": rms_db,
        "dr": dr,
        "bands_db": band_db,
        "tilt_db": tilt,
        "duration_sec": float(len(y) / sr),
        "sr": sr,
    }
    return analysis


def format_analysis_for_llm(analysis: Dict[str, Any]) -> str:
    b = analysis["bands_db"]
    text = f"""
Технический анализ трека:
- Длительность: {analysis['duration_sec']:.1f} сек
- Loudness (integrated LUFS): {analysis['loudness_lufs']:.2f} LUFS
- True Peak: {analysis['true_peak_db']:.2f} dBFS
- RMS: {analysis['rms_db']:.2f} dBFS
- Оценочный динамический диапазон (DR ≈ TP - LUFS): {analysis['dr']:.2f} dB

Спектральный баланс (примерные средние уровни по полосам, dB):
- Sub (20–60 Hz): {b['sub']:.2f} dB
- Bass (60–120 Hz): {b['bass']:.2f} dB
- Low-mid (120–500 Hz): {b['low_mid']:.2f} dB
- Mid (500–3000 Hz): {b['mid']:.2f} dB
- High-mid (3–8 kHz): {b['high_mid']:.2f} dB
- Air (8–20 kHz): {b['air']:.2f} dB

Общий спектральный наклон (Air - Bass): {analysis['tilt_db']:.2f} dB
"""
    return text


def format_ref_comparison_for_llm(src: Dict[str, Any], ref: Dict[str, Any]) -> str:
    lines = []

    def d(x):
        return f"{x:.2f}"

    lines.append("Сравнение исходного трека и референса:")
    lines.append("")
    lines.append(f"- Исходник: {d(src['loudness_lufs'])} LUFS, true peak {d(src['true_peak_db'])} dBFS, DR ≈ {d(src['dr'])}")
    lines.append(f"- Референс: {d(ref['loudness_lufs'])} LUFS, true peak {d(ref['true_peak_db'])} dBFS, DR ≈ {d(ref['dr'])}")
    lines.append("")
    lines.append("Спектральный баланс по основным полосам (dB):")

    for band in ["sub", "bass", "low_mid", "mid", "high_mid", "air"]:
        lines.append(
            f"- {band}: исходник {d(src['bands_db'][band])}, референс {d(ref['bands_db'][band])}, "
            f"разница (ref - src) = {d(ref['bands_db'][band] - src['bands_db'][band])} dB"
        )

    lines.append("")
    lines.append(
        f"Наклон спектра (Air-Bass): исходник {d(src['tilt_db'])} dB, референс {d(ref['tilt_db'])} dB, "
        f"разница {d(ref['tilt_db'] - src['tilt_db'])} dB"
    )

    loud_diff = ref["loudness_lufs"] - src["loudness_lufs"]
    lines.append("")
    lines.append(
        f"Чтобы привести громкость исходника к уровню референса, "
        f"нужно примерно изменить уровень на {d(loud_diff)} dB (ref - src)."
    )

    return "\n".join(lines)


# ================= ХЭНДЛЕРЫ КОМАНД / КНОПОК =================

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    chat_id = message.chat.id
    set_state(chat_id, "idle")
    text = (
        "Мяу! Я Мяузик — кот-саундпродюсер.\n\n"
        "Я умею:\n"
        "• общаться как обычный ИИ-кот по звуку и не только;\n"
        "• анализировать твои треки по громкости, динамике и спектру;\n"
        "• делать подробное ТЗ для автомастеринга под референс.\n\n"
        "Выбери режим на клавиатуре внизу, или просто пиши мне вопросы 😺"
    )
    await message.answer(text, reply_markup=main_keyboard)


@dp.message(F.text == "Анализ трека")
async def on_analysis_button(message: types.Message):
    chat_id = message.chat.id
    set_state(chat_id, "analysis_wait_track")
    await message.answer(
        "Мяу! Отправь мне аудиофайл (трек), и я проанализирую его: громкость (LUFS), пики, динамику и спектр.\n\n"
        "Пришли файл как обычное аудио или документ.",
        reply_markup=main_keyboard,
    )


@dp.message(F.text == "Автомастеринг под референс")
async def on_refmaster_button(message: types.Message):
    chat_id = message.chat.id
    set_state(chat_id, "refmaster_wait_source")
    ref_sessions.pop(chat_id, None)
    await message.answer(
        "Окей, мяу. Сначала пришли свой трек (тот, который нужно подтянуть).\n"
        "После этого я попрошу тебя загрузить референсный трек.",
        reply_markup=main_keyboard,
    )


# ================= ЗАГРУЗКА АУДИО И ОБРАБОТКА =================

async def download_audio_to_temp(message: types.Message) -> str:
    """
    Скачиваем audio или документ с аудио во временный файл и возвращаем путь.
    """
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


# Ловим любые аудио / аудио-документы
@dp.message(F.audio | (F.document & F.document.mime_type.contains("audio")))
async def on_audio_message(message: types.Message):
    chat_id = message.chat.id
    mode = get_state(chat_id)

    # Если режим не задан, трактуем как простой анализ по умолчанию
    effective_mode = mode
    if effective_mode is None or effective_mode == "idle":
        effective_mode = "analysis_wait_track"

    await message.answer("Мяу, качаю и анализирую твой файл, подожди немного...")

    try:
        tmp_path = await download_audio_to_temp(message)
        y, sr = load_audio_mono(tmp_path)
        analysis = analyze_audio(y, sr)
    except Exception as e:
        print("Audio processing error:", repr(e))
        await message.answer("Что-то пошло не так при чтении файла. Мяу... Попробуй другой формат или файл.")
        return

    # ==== Режим простой аналитики ====
    if effective_mode == "analysis_wait_track":
        set_state(chat_id, "idle")

        analysis_text = format_analysis_for_llm(analysis)
        prompt = f"""
Пользователь прислал трек на анализ. Вот технические параметры:

{analysis_text}

Сделай профессиональный, но простой для понимания разбор этого трека:
1) Оценка громкости (LUFS, true peaks, DR), подходит ли под стриминги/клуб/радио.
2) Оценка спектра: низ, середина, верха (по цифрам).
3) Какие риски: бубнеж, грязь, чрезмерная компрессия, резкость и т.п.
4) Что бы ты рекомендовал сделать со сведением/мастерингом, чтобы улучшить трек.
5) Пиши в образе Meowsic — кот-саундпродюсер, немного с юмором, но без потери точности.
"""
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=900,
            )
            answer = response.choices[0].message.content
            await message.answer(answer)
        except Exception as e:
            print("OpenAI error (analysis):", repr(e))
            await message.answer("Мяу... у меня лапки, не смог договориться с OpenAI. Попробуй ещё раз позже.")

        return

    # ==== Режим автомастеринга: сначала исходник ====
    if effective_mode == "refmaster_wait_source":
        ref_sessions[chat_id] = {
            "source_path": tmp_path,
            "source_analysis": analysis,
        }
        set_state(chat_id, "refmaster_wait_ref")
        await message.answer(
            "Я принял твой исходный трек и посмотрел его цифры.\n"
            "Теперь пришли референсный трек (тот, под который хочешь выровнять звук).",
        )
        return

    # ==== Режим автомастеринга: референс ====
    if effective_mode == "refmaster_wait_ref":
        session = ref_sessions.get(chat_id)
        if not session:
            await message.answer("Я потерял контекст. Мяу... Начни заново с кнопки «Автомастеринг под референс».")
            set_state(chat_id, "idle")
            return

        source_analysis = session["source_analysis"]
        ref_analysis = analysis

        set_state(chat_id, "idle")
        ref_sessions.pop(chat_id, None)

        compare_text = format_ref_comparison_for_llm(source_analysis, ref_analysis)
        prompt = f"""
Пользователь хочет автомастеринг исходного трека под референс.

Вот подробные параметры ИСХОДНОГО трека:
{format_analysis_for_llm(source_analysis)}

Вот подробные параметры РЕФЕРЕНСНОГО трека:
{format_analysis_for_llm(ref_analysis)}

Сравнение исходника и референса:
{compare_text}

Сделай детальный план автомастеринга исходника под референс.
Важно:
1) Опиши целевой уровень громкости (LUFS) и true peak.
2) Напиши, на сколько dB примерно нужно изменить громкость исходника (гейн) относительно текущего состояния.
3) Дай рекомендации по эквализации по полосам (sub, bass, low-mid, mid, high-mid, air): где приподнять/приглушить и на сколько dB (ориентировочно).
4) Дай рекомендации по динамике: сколько примерно дБ GR на мастеринговом компрессоре, нужна ли мультибэнд-компрессия, насколько агрессивный лимитер.
5) Если есть риски перегруза в сабе, грязи в mid, резкости в high-mid — укажи их.
6) Дай краткий “cheat sheet” — список шагов для мастеринговой цепочки (EQ → Comp → Limiter → Saturation и т.п.).
7) Пиши в образе Meowsic (кот-саундпродюсер), с лёгким юмором, но профессионально.
"""
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1100,
            )
            answer = response.choices[0].message.content
            await message.answer(answer)
        except Exception as e:
            print("OpenAI error (refmaster):", repr(e))
            await message.answer("Мяу... не смог согласовать автомастеринг с OpenAI. Попробуй ещё раз позже.")

        return


# ================= ОБЫЧНЫЙ ЧАТ =================

@dp.message()
async def generic_chat(message: types.Message):
    chat_id = message.chat.id
    uid = message.from_user.id
    text = message.text or ""

    mode = get_state(chat_id)
    if mode == "analysis_wait_track":
        await message.answer("Мяу, сейчас я жду от тебя аудиофайл для анализа. Пришли трек как аудио или документ.")
        return
    if mode == "refmaster_wait_source":
        await message.answer("Сначала пришли ИСХОДНЫЙ трек, который нужно подтянуть.")
        return
    if mode == "refmaster_wait_ref":
        await message.answer("Теперь пришли РЕФЕРЕНСНЫЙ трек (тот, под который выравниваем).")
        return

    await bot.send_chat_action(chat_id, "typing")
    update_history(uid, "user", text)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=user_histories[uid],
            temperature=0.8,
            max_tokens=600,
        )
        answer = response.choices[0].message.content
        update_history(uid, "assistant", answer)
        await message.answer(answer)
    except Exception as e:
        print("OpenAI error (chat):", repr(e))
        await message.answer("Мяу... я споткнулся об провод OpenAI. Попробуй ещё раз чуть позже.")


# ================= FLASK ДЛЯ RENDER =================

app = Flask(__name__)

@app.route("/")
def index():
    return "Meowsic bot is alive 🐾"

@app.route("/health")
def health():
    return "ok"


def start_web():
    port = int(os.environ.get("PORT", 10000))
    print(f"🌐 Meowsic: поднимаю веб-сервер на порту {port}...")
    app.run(host="0.0.0.0", port=port, threaded=True)


# ================= MAIN =================

async def main():
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
