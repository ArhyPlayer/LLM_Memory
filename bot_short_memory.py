"""
Telegram-бот с короткой памятью (последние 10 сообщений)
Использует aiogram 3.x и OpenAI API (proxyAPI)
"""

import os
import logging
from typing import Dict, List
from collections import deque

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import Message
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Загрузка настроек из переменных окружения
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
OPENAI_MAX_COMPLETION_TOKENS = int(os.getenv("OPENAI_MAX_COMPLETION_TOKENS", os.getenv("OPENAI_MAX_TOKENS", "2000")))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Проверка обязательных параметров
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не найден в переменных окружения!")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не найден в переменных окружения!")

# Вывод настроек при запуске
logger.info(f"Настройки бота:")
logger.info(f"  - Модель: {OPENAI_MODEL}")
logger.info(f"  - Temperature: {OPENAI_TEMPERATURE}")
logger.info(f"  - Max completion tokens: {OPENAI_MAX_COMPLETION_TOKENS}")
logger.info(f"  - Base URL: {OPENAI_BASE_URL if OPENAI_BASE_URL else 'default (api.openai.com)'}")
logger.info(f"  - Уровень логирования: {LOG_LEVEL}")


# Инициализация бота и диспетчера
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Инициализация OpenAI клиента (работает с proxyAPI)
# Если указан OPENAI_BASE_URL, используем его для ProxyAPI
if OPENAI_BASE_URL:
    openai_client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
else:
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Хранилище истории диалогов: user_id -> deque последних сообщений
# deque ограничена maxlen, автоматически удаляет старые элементы
HISTORY_SIZE = 10
user_histories: Dict[int, deque] = {}


def get_user_history(user_id: int) -> deque:
    """
    Получить историю диалога пользователя.
    Если истории нет - создать новую.
    """
    if user_id not in user_histories:
        user_histories[user_id] = deque(maxlen=HISTORY_SIZE)
    return user_histories[user_id]


async def get_ai_response(user_id: int, user_message: str) -> str:
    """
    Отправить запрос к OpenAI API с историей диалога.
    
    Args:
        user_id: ID пользователя Telegram
        user_message: Текущее сообщение пользователя
    
    Returns:
        Ответ от модели
    """
    try:
        # Получаем историю пользователя
        history = get_user_history(user_id)
        
        # Формируем список сообщений для API
        messages = [
            {
                "role": "system",
                "content": "Ты полезный AI-ассистент. Отвечай кратко и по делу."
            }
        ]
        
        # Добавляем историю диалога
        messages.extend(list(history))
        
        # Добавляем текущее сообщение пользователя
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Запрос к OpenAI API
        logger.info(f"Отправка запроса к API для user_id={user_id}")
        
        # Параметры запроса
        api_params = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "max_completion_tokens": OPENAI_MAX_COMPLETION_TOKENS
        }
        
        # gpt-5-mini не поддерживает temperature (только значение по умолчанию 1)
        if "gpt-5" not in OPENAI_MODEL.lower():
            api_params["temperature"] = OPENAI_TEMPERATURE
            logger.debug(f"Параметры запроса: model={OPENAI_MODEL}, max_completion_tokens={OPENAI_MAX_COMPLETION_TOKENS}, temperature={OPENAI_TEMPERATURE}")
        else:
            logger.debug(f"Параметры запроса: model={OPENAI_MODEL}, max_completion_tokens={OPENAI_MAX_COMPLETION_TOKENS} (temperature=default для gpt-5)")
        
        logger.debug(f"Количество сообщений в истории: {len(messages)}")
        
        response = await openai_client.chat.completions.create(**api_params)
        
        # Извлекаем ответ
        ai_message = response.choices[0].message.content
        logger.info(f"Получен ответ от API, длина: {len(ai_message)} символов")
        
        # Сохраняем сообщения в историю
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": ai_message})
        
        return ai_message
        
    except Exception as e:
        # Детальное логирование ошибки
        logger.error(f"Ошибка при запросе к API: {type(e).__name__}: {str(e)}")
        logger.error(f"Модель: {OPENAI_MODEL}, Base URL: {OPENAI_BASE_URL if OPENAI_BASE_URL else 'default'}")
        
        # Проверка конкретных типов ошибок
        error_message = "❌ Произошла ошибка при обработке запроса.\n\n"
        
        if "500" in str(e) or "Internal Server Error" in str(e):
            error_message += "🔴 Ошибка сервера ProxyAPI (500).\n"
            error_message += "Возможные причины:\n"
            error_message += "• Проверьте правильность API ключа\n"
            error_message += "• Проверьте, что модель поддерживается вашим ProxyAPI\n"
            error_message += f"• Текущая модель: {OPENAI_MODEL}\n"
            error_message += "• Попробуйте изменить модель в .env файле"
        elif "401" in str(e) or "Unauthorized" in str(e):
            error_message += "🔑 Ошибка авторизации.\n"
            error_message += "Проверьте правильность OPENAI_API_KEY в файле .env"
        elif "429" in str(e) or "rate_limit" in str(e).lower():
            error_message += "⏱️ Превышен лимит запросов.\n"
            error_message += "Попробуйте повторить через некоторое время."
        elif "404" in str(e):
            error_message += "🔍 Модель не найдена.\n"
            error_message += f"Модель '{OPENAI_MODEL}' не доступна.\n"
            error_message += "Проверьте название модели в .env файле."
        else:
            error_message += f"Детали: {str(e)}"
        
        return error_message


@dp.message(Command("start"))
async def cmd_start(message: Message):
    """
    Обработчик команды /start
    """
    user_id = message.from_user.id
    
    # Очищаем историю при старте
    if user_id in user_histories:
        user_histories[user_id].clear()
    
    await message.answer(
        "👋 <b>Привет! Я бот с короткой памятью.</b>\n\n"
        "Я запоминаю последние 10 сообщений нашего диалога.\n"
        "Просто напиши мне что-нибудь, и я отвечу!\n\n"
        "📋 <b>Доступные команды:</b>\n"
        "/start - Начать диалог заново\n"
        "/clear - Очистить историю\n"
        "/settings - Показать настройки\n"
        "/help - Справка по командам",
        parse_mode="HTML"
    )


@dp.message(Command("help"))
async def cmd_help(message: Message):
    """
    Обработчик команды /help - справка
    """
    help_text = (
        "📚 <b>Справка по использованию бота</b>\n\n"
        "<b>Команды:</b>\n"
        "/start - Начать диалог заново (очищает историю)\n"
        "/clear - Очистить историю диалога\n"
        "/settings - Показать текущие настройки бота\n"
        "/help - Показать эту справку\n\n"
        "<b>Как работает бот:</b>\n"
        "• Бот запоминает последние 10 сообщений диалога\n"
        "• Каждый пользователь имеет свою историю\n"
        "• История используется для контекста при генерации ответов\n"
        "• При достижении лимита старые сообщения автоматически удаляются\n\n"
        "<b>Советы:</b>\n"
        "• Используйте /clear если хотите начать новую тему\n"
        "• Проверьте /settings если возникают ошибки\n"
        "• История хранится в памяти и сбрасывается при перезапуске бота"
    )
    
    await message.answer(help_text, parse_mode="HTML")


@dp.message(Command("clear"))
async def cmd_clear(message: Message):
    """
    Обработчик команды /clear - очистка истории диалога
    """
    user_id = message.from_user.id
    
    if user_id in user_histories:
        user_histories[user_id].clear()
    
    await message.answer("🧹 История диалога очищена!")


@dp.message(Command("settings"))
async def cmd_settings(message: Message):
    """
    Обработчик команды /settings - показать текущие настройки
    """
    settings_text = (
        "⚙️ <b>Текущие настройки бота:</b>\n\n"
        f"🤖 <b>Модель:</b> <code>{OPENAI_MODEL}</code>\n"
        f"🌡️ <b>Temperature:</b> <code>{OPENAI_TEMPERATURE}</code>\n"
        f"📝 <b>Max completion tokens:</b> <code>{OPENAI_MAX_COMPLETION_TOKENS}</code>\n"
        f"🌐 <b>Base URL:</b> <code>{OPENAI_BASE_URL if OPENAI_BASE_URL else 'default (api.openai.com)'}</code>\n"
        f"📊 <b>Уровень логирования:</b> <code>{LOG_LEVEL}</code>\n"
        f"💾 <b>Размер памяти:</b> <code>{HISTORY_SIZE} сообщений</code>\n\n"
        f"<i>Для изменения настроек отредактируйте файл .env и перезапустите бота</i>"
    )
    
    await message.answer(settings_text, parse_mode="HTML")


@dp.message(F.text)
async def handle_text_message(message: Message):
    """
    Обработчик всех текстовых сообщений.
    Отправляет запрос к AI и возвращает ответ.
    """
    user_id = message.from_user.id
    user_text = message.text
    
    logger.info(f"Получено сообщение от user_id={user_id}: {user_text}")
    
    # Показываем, что бот печатает
    await message.bot.send_chat_action(
        chat_id=message.chat.id,
        action="typing"
    )
    
    # Получаем ответ от AI
    ai_response = await get_ai_response(user_id, user_text)
    
    # Отправляем ответ пользователю
    await message.answer(ai_response)


async def main():
    """
    Главная функция запуска бота
    """
    logger.info("Бот запущен!")
    
    # Удаляем webhook если он был установлен
    await bot.delete_webhook(drop_pending_updates=True)
    
    # Запускаем polling
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")

