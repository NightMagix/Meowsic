# Лёгкий образ Python
FROM python:3.10-slim

# Чтобы логи сразу шли в консоль
ENV PYTHONUNBUFFERED=1

# Устанавливаем ffmpeg (обрезка/ресемплинг треков)
RUN apt-get update && apt-get install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Ставим python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код бота
COPY . .

# Порт, который использует Flask внутри (Render сам пробросит)
EXPOSE 10000

# Точка входа — твой Bot.py
CMD ["python", "Bot.py"]
