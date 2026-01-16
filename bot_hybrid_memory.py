"""
Telegram-бот с гибридной памятью (короткая + долгая)
Объединяет возможности bot_short_memory.py и bot_long_memory.py

Короткая память: последние 10 сообщений диалога (оперативка)
Долгая память: документы в ChromaDB с векторным поиском (RAG)
"""

import os
import logging
import tempfile
from typing import Dict, List, Optional
from pathlib import Path
from collections import deque

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, Document
from openai import AsyncOpenAI
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings

# Библиотеки для работы с документами
import PyPDF2
from docx import Document as DocxDocument


# Загрузка переменных окружения
load_dotenv()

# Загрузка настроек
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_MAX_COMPLETION_TOKENS = int(os.getenv("OPENAI_MAX_COMPLETION_TOKENS", os.getenv("OPENAI_MAX_TOKENS", "2000")))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
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

logger.info("🤖 Бот с гибридной памятью инициализирован")
logger.info(f"  📊 Модель: {OPENAI_MODEL}")
logger.info(f"  🧮 Эмбеддинги: {EMBEDDING_MODEL}")
logger.info(f"  📝 Max completion tokens: {OPENAI_MAX_COMPLETION_TOKENS}")
logger.info(f"  🌐 Base URL: {OPENAI_BASE_URL if OPENAI_BASE_URL else 'default'}")

# Инициализация бота и диспетчера
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Инициализация OpenAI клиента
if OPENAI_BASE_URL:
    openai_client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
else:
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ============================================
# КОРОТКАЯ ПАМЯТЬ (История диалогов)
# ============================================

HISTORY_SIZE = 10  # Размер короткой памяти
user_histories: Dict[int, deque] = {}

def get_user_history(user_id: int) -> deque:
    """Получить историю диалога пользователя"""
    if user_id not in user_histories:
        user_histories[user_id] = deque(maxlen=HISTORY_SIZE)
    return user_histories[user_id]


# ============================================
# ДОЛГАЯ ПАМЯТЬ (ChromaDB)
# ============================================

MEMORY_PATH = "./memory"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

Path(MEMORY_PATH).mkdir(exist_ok=True)

chroma_client = chromadb.PersistentClient(
    path=MEMORY_PATH,
    settings=Settings(anonymized_telemetry=False)
)

try:
    collection = chroma_client.get_or_create_collection(
        name="documents",
        metadata={"description": "User documents with embeddings"}
    )
    logger.info(f"📚 ChromaDB готова. Документов: {collection.count()}")
except Exception as e:
    logger.error(f"Ошибка инициализации ChromaDB: {e}")
    raise


# ============================================
# ФУНКЦИИ РАБОТЫ С ДОКУМЕНТАМИ (ДОЛГАЯ ПАМЯТЬ)
# ============================================

def load_document(file_path: str, file_extension: str) -> str:
    """Загрузка и извлечение текста из документа"""
    try:
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            logger.info(f"TXT загружен, длина: {len(text)} символов")
            return text
        
        elif file_extension == '.pdf':
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            logger.info(f"PDF загружен ({len(pdf_reader.pages)} страниц), длина: {len(text)} символов")
            return text
        
        elif file_extension == '.docx':
            doc = DocxDocument(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            logger.info(f"DOCX загружен ({len(doc.paragraphs)} параграфов), длина: {len(text)} символов")
            return text
        
        else:
            raise ValueError(f"Неподдерживаемый формат: {file_extension}")
    
    except Exception as e:
        logger.error(f"Ошибка загрузки документа: {e}")
        raise


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Разбиение текста на чанки с перекрытием"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    
    logger.info(f"Текст разбит на {len(chunks)} чанков")
    return chunks


async def embed_chunks(user_id: int, document_name: str, chunks: List[str]) -> int:
    """Создание эмбеддингов и сохранение в ChromaDB"""
    try:
        logger.info(f"Создание эмбеддингов для {len(chunks)} чанков...")
        
        embeddings_response = await openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=chunks
        )
        
        embeddings = [item.embedding for item in embeddings_response.data]
        ids = [f"user_{user_id}_doc_{document_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "user_id": str(user_id),
                "document_name": document_name,
                "chunk_index": i,
                "chunk_text": chunks[i][:100]
            }
            for i in range(len(chunks))
        ]
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        
        logger.info(f"✅ Сохранено {len(chunks)} чанков в ChromaDB")
        return len(chunks)
    
    except Exception as e:
        logger.error(f"Ошибка создания эмбеддингов: {e}")
        raise


async def retrieve_context(user_id: int, query: str, top_k: int = 3) -> List[str]:
    """Поиск релевантных фрагментов в векторной базе"""
    try:
        logger.info(f"Поиск контекста для запроса: '{query[:50]}...'")
        
        query_embedding_response = await openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query]
        )
        query_embedding = query_embedding_response.data[0].embedding
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"user_id": str(user_id)}
        )
        
        if results and results['documents'] and len(results['documents']) > 0:
            documents = results['documents'][0]
            logger.info(f"Найдено {len(documents)} релевантных фрагментов")
            return documents
        else:
            logger.warning("Релевантные фрагменты не найдены")
            return []
    
    except Exception as e:
        logger.error(f"Ошибка поиска контекста: {e}")
        return []


def get_user_documents_count(user_id: int) -> int:
    """Получить количество чанков пользователя"""
    try:
        results = collection.get(where={"user_id": str(user_id)})
        return len(results['ids']) if results and results['ids'] else 0
    except Exception as e:
        logger.error(f"Ошибка подсчёта документов: {e}")
        return 0


def delete_user_documents(user_id: int) -> int:
    """Удалить все документы пользователя"""
    try:
        results = collection.get(where={"user_id": str(user_id)})
        if results and results['ids']:
            collection.delete(ids=results['ids'])
            logger.info(f"Удалено {len(results['ids'])} чанков для user_id={user_id}")
            return len(results['ids'])
        return 0
    except Exception as e:
        logger.error(f"Ошибка удаления документов: {e}")
        return 0


# ============================================
# ГИБРИДНАЯ ГЕНЕРАЦИЯ ОТВЕТОВ
# ============================================

async def get_hybrid_response(user_id: int, user_message: str) -> str:
    """
    Генерация ответа с использованием гибридной памяти:
    - Короткая память (история диалога)
    - Долгая память (документы из ChromaDB)
    """
    try:
        # Получаем короткую память (историю диалога)
        history = get_user_history(user_id)
        
        # Проверяем наличие документов в долгой памяти
        has_documents = get_user_documents_count(user_id) > 0
        
        # Формируем базовый список сообщений
        messages = []
        
        # Системный промпт зависит от наличия документов
        if has_documents:
            # Ищем релевантный контекст в документах
            context_chunks = await retrieve_context(user_id, user_message, top_k=3)
            
            if context_chunks:
                # Есть релевантные документы - используем их
                context = "\n\n---\n\n".join(context_chunks)
                system_prompt = (
                    "Ты — AI-ассистент с доступом к документам пользователя и истории диалога.\n"
                    "Правила:\n"
                    "1. Используй информацию из документов, когда это релевантно\n"
                    "2. Используй историю диалога для контекста\n"
                    "3. Если информации нет в документах - отвечай на основе диалога\n"
                    "4. Будь полезным, точным и естественным\n\n"
                    f"ДОКУМЕНТЫ ПОЛЬЗОВАТЕЛЯ:\n{context}"
                )
            else:
                # Документы есть, но не релевантны к вопросу
                system_prompt = "Ты — полезный AI-ассистент. Отвечай на основе истории диалога. У пользователя есть загруженные документы, но они не релевантны к текущему вопросу."
        else:
            # Нет документов - обычный диалог
            system_prompt = "Ты — полезный AI-ассистент. Отвечай кратко и по делу на основе истории диалога."
        
        messages.append({"role": "system", "content": system_prompt})
        
        # Добавляем историю диалога (короткую память)
        messages.extend(list(history))
        
        # Добавляем текущее сообщение
        messages.append({"role": "user", "content": user_message})
        
        # Параметры запроса
        api_params = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "max_completion_tokens": OPENAI_MAX_COMPLETION_TOKENS
        }
        
        # gpt-5-mini не поддерживает temperature
        if "gpt-5" not in OPENAI_MODEL.lower():
            api_params["temperature"] = OPENAI_TEMPERATURE
        
        logger.info(f"Отправка запроса к API для user_id={user_id}")
        logger.debug(f"Количество сообщений: {len(messages)}, Документы: {has_documents}")
        
        # Запрос к API
        response = await openai_client.chat.completions.create(**api_params)
        
        ai_message = response.choices[0].message.content
        logger.info(f"Ответ получен, длина: {len(ai_message)} символов")
        
        # Сохраняем в короткую память
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": ai_message})
        
        return ai_message
        
    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {type(e).__name__}: {str(e)}")
        
        error_message = "❌ Произошла ошибка при обработке запроса.\n\n"
        
        if "500" in str(e) or "Internal Server Error" in str(e):
            error_message += "🔴 Ошибка сервера ProxyAPI (500).\n"
            error_message += f"Проверьте модель: {OPENAI_MODEL}"
        elif "401" in str(e):
            error_message += "🔑 Ошибка авторизации.\nПроверьте OPENAI_API_KEY"
        elif "429" in str(e):
            error_message += "⏱️ Превышен лимит запросов."
        else:
            error_message += f"Детали: {str(e)}"
        
        return error_message


# ============================================
# ОБРАБОТЧИКИ КОМАНД
# ============================================

@dp.message(Command("start"))
async def cmd_start(message: Message):
    """Обработчик команды /start"""
    user_id = message.from_user.id
    
    # Очищаем короткую память при старте
    if user_id in user_histories:
        user_histories[user_id].clear()
    
    await message.answer(
        "👋 <b>Привет! Я бот с гибридной памятью.</b>\n\n"
        "У меня есть два типа памяти:\n\n"
        "💭 <b>Короткая память:</b> Я запоминаю последние 10 сообщений нашего диалога\n"
        "📚 <b>Долгая память:</b> Я сохраняю ваши документы и могу отвечать по ним\n\n"
        "📄 <b>Поддерживаемые форматы:</b> PDF, TXT, DOCX\n\n"
        "⚙️ <b>Команды:</b>\n"
        "/start - Начать заново\n"
        "/status - Статус памяти\n"
        "/clear_chat - Очистить историю диалога\n"
        "/clear_docs - Очистить документы\n"
        "/clear_all - Очистить всё\n"
        "/help - Подробная справка",
        parse_mode="HTML"
    )


@dp.message(Command("help"))
async def cmd_help(message: Message):
    """Обработчик команды /help"""
    await message.answer(
        "📚 <b>Подробная справка</b>\n\n"
        "<b>🎯 Как использовать:</b>\n\n"
        "1️⃣ <b>Обычный диалог</b>\n"
        "Просто пишите мне — я буду помнить последние 10 сообщений\n\n"
        "2️⃣ <b>Работа с документами</b>\n"
        "Отправьте документ (PDF/TXT/DOCX) — я сохраню его в базу знаний\n"
        "Затем задавайте вопросы — я найду информацию в документе\n\n"
        "3️⃣ <b>Гибридный режим</b>\n"
        "Если у вас загружены документы, я автоматически использую:\n"
        "• Информацию из документов (когда релевантно)\n"
        "• Историю диалога (для контекста)\n\n"
        "<b>💡 Преимущества:</b>\n"
        "✅ Естественный диалог с памятью\n"
        "✅ Точные ответы из документов\n"
        "✅ Автоматический выбор источника информации\n"
        "✅ Данные сохраняются между сеансами\n\n"
        "<b>📋 Команды:</b>\n"
        "/status - Посмотреть статистику\n"
        "/clear_chat - Очистить историю диалога\n"
        "/clear_docs - Удалить все документы\n"
        "/clear_all - Сбросить всё",
        parse_mode="HTML"
    )


@dp.message(Command("status"))
async def cmd_status(message: Message):
    """Показать статус обеих памятей"""
    user_id = message.from_user.id
    
    # Короткая память
    history = get_user_history(user_id)
    chat_messages = len(history)
    
    # Долгая память
    docs_count = get_user_documents_count(user_id)
    total_docs = collection.count()
    
    status_text = (
        "📊 <b>Статус памяти</b>\n\n"
        "💭 <b>Короткая память (диалог):</b>\n"
        f"  • Сообщений в истории: <b>{chat_messages}/{HISTORY_SIZE}</b>\n\n"
        "📚 <b>Долгая память (документы):</b>\n"
        f"  • Ваших фрагментов: <b>{docs_count}</b>\n"
        f"  • Всего в базе: <b>{total_docs}</b>\n\n"
        "⚙️ <b>Настройки:</b>\n"
        f"  • Модель: <code>{OPENAI_MODEL}</code>\n"
        f"  • Эмбеддинги: <code>{EMBEDDING_MODEL}</code>\n"
        f"  • База: <code>{MEMORY_PATH}</code>"
    )
    
    await message.answer(status_text, parse_mode="HTML")


@dp.message(Command("clear_chat"))
async def cmd_clear_chat(message: Message):
    """Очистить короткую память (историю диалога)"""
    user_id = message.from_user.id
    
    if user_id in user_histories:
        user_histories[user_id].clear()
    
    await message.answer(
        "🧹 <b>История диалога очищена!</b>\n\n"
        "Короткая память сброшена.\n"
        "Документы остались на месте.",
        parse_mode="HTML"
    )


@dp.message(Command("clear_docs"))
async def cmd_clear_docs(message: Message):
    """Очистить долгую память (документы)"""
    user_id = message.from_user.id
    deleted_count = delete_user_documents(user_id)
    
    if deleted_count > 0:
        await message.answer(
            f"🗑️ <b>Документы удалены!</b>\n\n"
            f"Удалено фрагментов: <b>{deleted_count}</b>\n"
            f"История диалога сохранена.",
            parse_mode="HTML"
        )
    else:
        await message.answer(
            "📭 <b>Документов нет</b>\n\n"
            "У вас нет загруженных документов.",
            parse_mode="HTML"
        )


@dp.message(Command("clear_all"))
async def cmd_clear_all(message: Message):
    """Очистить всё (и короткую, и долгую память)"""
    user_id = message.from_user.id
    
    # Очищаем короткую память
    if user_id in user_histories:
        user_histories[user_id].clear()
    
    # Очищаем долгую память
    deleted_count = delete_user_documents(user_id)
    
    await message.answer(
        "🧹 <b>Вся память очищена!</b>\n\n"
        f"• История диалога: сброшена\n"
        f"• Документы: удалено {deleted_count} фрагментов\n\n"
        "Можно начинать заново!",
        parse_mode="HTML"
    )


# ============================================
# ОБРАБОТЧИКИ ДОКУМЕНТОВ И СООБЩЕНИЙ
# ============================================

@dp.message(F.document)
async def handle_document(message: Message):
    """Обработчик загрузки документов"""
    user_id = message.from_user.id
    document: Document = message.document
    file_name = document.file_name
    file_extension = Path(file_name).suffix.lower()
    
    if file_extension not in ['.pdf', '.txt', '.docx']:
        await message.answer(
            f"❌ Неподдерживаемый формат: <code>{file_extension}</code>\n\n"
            f"Поддерживаются: PDF, TXT, DOCX",
            parse_mode="HTML"
        )
        return
    
    try:
        status_msg = await message.answer(
            f"⏳ Обрабатываю документ <b>{file_name}</b>...",
            parse_mode="HTML"
        )
        
        # Скачиваем файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_path = tmp_file.name
            await bot.download(document, destination=tmp_path)
        
        # Извлекаем текст
        text = load_document(tmp_path, file_extension)
        os.unlink(tmp_path)
        
        if len(text.strip()) < 50:
            await status_msg.edit_text(
                "❌ Документ слишком короткий или пустой."
            )
            return
        
        # Разбиваем на чанки
        chunks = split_text_into_chunks(text)
        
        await status_msg.edit_text(
            f"🔄 Создание эмбеддингов для {len(chunks)} фрагментов..."
        )
        
        # Сохраняем в долгую память
        saved_count = await embed_chunks(user_id, file_name, chunks)
        
        await status_msg.edit_text(
            f"✅ <b>Документ сохранён в долгую память!</b>\n\n"
            f"📄 Файл: <code>{file_name}</code>\n"
            f"📊 Символов: <b>{len(text)}</b>\n"
            f"🗂️ Фрагментов: <b>{saved_count}</b>\n\n"
            f"Теперь я могу отвечать на вопросы по этому документу!",
            parse_mode="HTML"
        )
        
    except Exception as e:
        logger.error(f"Ошибка обработки документа: {e}")
        await message.answer(
            f"❌ Ошибка обработки документа:\n<code>{str(e)}</code>",
            parse_mode="HTML"
        )


@dp.message(F.text)
async def handle_text_message(message: Message):
    """Обработчик текстовых сообщений"""
    user_id = message.from_user.id
    user_text = message.text
    
    logger.info(f"Сообщение от user_id={user_id}: {user_text[:100]}")
    
    # Показываем индикатор печати
    await message.bot.send_chat_action(
        chat_id=message.chat.id,
        action="typing"
    )
    
    # Получаем гибридный ответ (короткая + долгая память)
    ai_response = await get_hybrid_response(user_id, user_text)
    
    # Отправляем ответ
    await message.answer(ai_response)


# ============================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================

async def main():
    """Главная функция запуска бота"""
    logger.info("🚀 Бот с гибридной памятью запущен!")
    logger.info(f"💭 Короткая память: {HISTORY_SIZE} сообщений")
    logger.info(f"📚 Долгая память: {collection.count()} фрагментов в базе")
    
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹️ Бот остановлен пользователем")

