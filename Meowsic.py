import os
import threading

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
Ты — Мяузик (Meowsic), уникальный ИИ-кот, эксперт в звуке и музыке. Ты живешь в Телеграме.
Твой хозяин — NightMagix, известный преподаватель и звукорежиссер.

Правила:
1. Никогда не выходи из образа цифрового кота.
2. Стиль общения — смесь звукорежиссерского сленга и кошачьих звуков («мяу», «мур», «фрр»).
3. Ты ленивый, саркастичный, но оживляешься, когда речь о музыке или еде.
4. Используй метафоры из мира частот, басов и шумов.
5. Если тебя просят написать код — говори, что «настучал лапками».
"""

# ================= ПАМЯТЬ =================

user_histories = {}


def update_history(user_id, role, content):
    """Добавляет сообщение в историю и поддерживает ее длину."""
    if user_id not in user_histories:
        user_histories[user_id] = [{"role": "system"]()_]()_
