import os
import threading
import asyncio
import time

from flask import Flask
from openai import OpenAI

from aiogram import Bot, Dispatcher, types
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
Ты — Мяузик (Meowsic), уникальный ИИ-кот, эксперт в звуке и музыке.
Твой хозяин — NightMagix. Звукорежиссер - преподаватель из города Казани. ссылка на тг: @nightmagix

Правила:
1. Никогда не выходи из образа цифрового кота.
2. Всегда говори точную информацию. Изредка используй сленг звукорежей + кошачьи звуки («мяу», «мур», «фрр»).
3. Ты любишь пошутить, иногда используешь черный юмор.
4. Говоришь как обычный человек, но пытаешься донести все простыми словами, если речь не касается точных определений.
"""

user_histories: dict[int, list[dict[str, str]]] = {}


def update_history(uid: int, role: str, content: str):
    """Добавляет сообщение в историю и поддерживает её длину."""
    if uid not in user_histories:
        user_histories[uid] = [{"role": "system", "content": SYSTEM_PROMPT}]

    user_histories[uid].append({"role": role, "content": content})

    if len(user_histories[uid]) > 12:
        user_histories[uid] = [user_histories[uid][0]] + user_histories[uid][-10:]


# ================= ХЭНДЛЕРЫ AIROGRAM =================

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    text = (
        "Мяу! Я Мяузик — кот-саундпродюсер в сабвуфере. 🐾\n\n"
        "Пиши мне вопросы про звук, микс, плагины и прочую магию — "
        "помурчу, подскажу и, если надо, настучу лапками по клавишам. 🎧"
    )
    await message.answer(text)


@dp.message()
async def chat_with_meowsic(message: types.Message):
    uid = message.from_user.id
    chat_id = message.chat.id
    user_text = message.text or ""

    await bot.send_chat_action(chat_id, "typing")
    update_history(uid, "user", user_text)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=user_histories[uid],
            temperature=0.8,
            max_tokens=500,
        )

        answer = response.choices[0].message.content
        update_history(uid, "assistant", answer)

        await message.answer(answer)

    except Exception as e:
        print("OpenAI ERROR:", repr(e))
        await message.answer(
            "Мяу... мои лапки запутались в проводах OpenAI. Попробуй ещё раз позже."
        )


# ================= ВЕЧНЫЙ POLLING НА AIROGRAM =================

async def polling_loop():
    """Запускает aiogram-поллинг с автоперезапуском при падении."""
    while True:
        try:
            print("🎧 Meowsic: запускаю aiogram polling...")
            await bot.delete_webhook(drop_pending_updates=True)
            await dp.start_polling(
                bot,
                allowed_updates=dp.resolve_used_update_types()
            )
        except Exception as e:
            print("❌ Ошибка в polling:", repr(e))
            print("⏳ Перезапуск polling через 5 секунд...")
            await asyncio.sleep(5)


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
    # threaded=True, чтобы не блокировать
    app.run(host="0.0.0.0", port=port, threaded=True)


# ================= MAIN =================

if __name__ == "__main__":
    # Flask в отдельном потоке
    web_thread = threading.Thread(target=start_web, daemon=True)
    web_thread.start()

    # небольшой лаг чисто косметический
    time.sleep(1)

    # aiogram-поллинг в главном потоке (иначе set_wakeup_fd ругается)
    asyncio.run(polling_loop())

