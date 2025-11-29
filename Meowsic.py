import os
import threading
import time

import telebot
from openai import OpenAI
from flask import Flask

# ================= КОНФИГУРАЦИЯ =================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY не найден в переменных окружения")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN не найден в переменных окружения")

client = OpenAI(api_key=OPENAI_API_KEY)
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# ================= ЛИЧНОСТЬ КОТА =================

SYSTEM_PROMPT = """
Ты — Мяузик (Meowsic), уникальный ИИ-кот, эксперт в звуке и музыке.
Твой хозяин — NightMagix.

Правила:
1. Никогда не выходи из образа цифрового кота.
2. Стиль общения — сленг звукорежей + «мяу», «мур».
3. Ты ленивый, саркастичный, но милый.
4. Используй метафоры про звук: частоты, басы, шум.
5. Если пишешь код — говори, что настучал лапками.
"""

# ================= ПАМЯТЬ =================

user_histories = {}


def update_history(user_id, role, content):
    """Добавляет сообщение в историю и поддерживает ее длину."""
    if user_id not in user_histories:
        user_histories[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    user_histories[user_id].append({"role": role, "content": content})

    # Оставляем system prompt + последние 10 сообщений
    if len(user_histories[user_id]) > 12:
        user_histories[user_id] = [user_histories[user_id][0]] + user_histories[user_id][-10:]


# ================= ОБРАБОТЧИК СООБЩЕНИЙ =================

@bot.message_handler(func=lambda message: True)
def chat_with_meowsic(message):
    user_id = message.chat.id
    user_text = message.text or ""

    bot.send_chat_action(user_id, "typing")
    update_history(user_id, "user", user_text)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=user_histories[user_id],
            temperature=0.8,
            max_tokens=500,
        )

        bot_answer = response.choices[0].message.content
        update_history(user_id, "assistant", bot_answer)

        bot.reply_to(message, bot_answer)

    except Exception as e:
        print("Ошибка OpenAI:", e)
        bot.send_message(
            user_id,
            "Мяу... мои усы запутались в проводах. (Ошибка API)"
        )


# ================= МИНИ ВЕБ-СЕРВЕР ДЛЯ RENDER =================

app = Flask(__name__)


@app.route("/")
def index():
    return "Meowsic bot is alive 🐾"


@app.route("/health")
def health():
    return "ok"


def run_bot():
    # Вечный цикл: если polling упал — поднимем заново
    while True:
        try:
            print("🎧 Meowsic: запускаю Telegram-поллинг...")
            bot.remove_webhook()  # на всякий, если где-то остался webhook
            bot.infinity_polling(skip_pending=True, timeout=60)
        except Exception as e:
            print("❌ Ошибка в polling:", repr(e))
            # маленькая пауза, чтобы не крутить перезапуск сотни раз в секунду
            time.sleep(5)


def run_web():
    port = int(os.environ.get("PORT", 10000))
    print(f"🌐 Meowsic: поднимаю веб-сервер на порту {port}...")
    app.run(host="0.0.0.0", port=port)


# ================= ЗАПУСК =================

if __name__ == "__main__":
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    run_web()
