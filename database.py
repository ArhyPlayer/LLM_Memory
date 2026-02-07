"""
Модуль базы данных (SQLite3) для хранения тезисов по пользователям.
Для каждого пользователя при первом обращении создаётся таблица user_<user_id>.
В каждой строке хранятся тезисы одного ответа бота (JSON).
Файл БД: memory/database.db
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


# Путь к файлу БД в memory/
MEMORY_PATH = Path(__file__).resolve().parent / "memory"
DB_PATH = MEMORY_PATH / "database.db"


def _ensure_requests_table(conn: sqlite3.Connection) -> None:
    """Создаёт таблицу user_requests, если её нет."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def add_user_request(user_id: int, text: str) -> None:
    """Сохраняет один запрос пользователя (текст из строки ввода)."""
    conn = _get_connection()
    try:
        _ensure_requests_table(conn)
        conn.execute(
            "INSERT INTO user_requests (user_id, text) VALUES (?, ?)",
            (user_id, text)
        )
        conn.commit()
        logger.debug("Запрос добавлен для user_id=%s", user_id)
    finally:
        conn.close()


def get_all_user_requests(user_id: int) -> List[str]:
    """Возвращает все сохранённые запросы пользователя по порядку."""
    if not DB_PATH.exists():
        return []
    conn = _get_connection()
    try:
        _ensure_requests_table(conn)
        cur = conn.execute(
            "SELECT text FROM user_requests WHERE user_id = ? ORDER BY id ASC",
            (user_id,)
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows]


def get_all_user_ids_with_requests() -> List[int]:
    """Возвращает список user_id, у которых есть сохранённые запросы."""
    if not DB_PATH.exists():
        return []
    conn = _get_connection()
    try:
        _ensure_requests_table(conn)
        cur = conn.execute("SELECT DISTINCT user_id FROM user_requests ORDER BY user_id")
        rows = cur.fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows]


def get_all_user_ids() -> List[int]:
    """
    Возвращает список всех user_id, для которых есть таблица в БД (есть тезисы).
    """
    if not DB_PATH.exists():
        return []
    conn = _get_connection()
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'user_%'"
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    result = []
    for (name,) in rows:
        try:
            uid = int(name.replace("user_", "", 1))
            result.append(uid)
        except ValueError:
            continue
    return result


def _get_connection() -> sqlite3.Connection:
    """Возвращает подключение к SQLite (создаёт файл и каталог при необходимости)."""
    MEMORY_PATH.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def _table_name(user_id: int) -> str:
    """Имя таблицы для пользователя: user_<user_id>. Безопасно для SQL (только цифры в id)."""
    return f"user_{user_id}"


def ensure_user_table(user_id: int) -> None:
    """
    Создаёт таблицу user_<user_id>, если её ещё нет.
    Вызывается при первом сообщении пользователя.
    """
    table = _table_name(user_id)
    conn = _get_connection()
    try:
        conn.execute(
            f'''
            CREATE TABLE IF NOT EXISTS "{table}" (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                theses TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            '''
        )
        conn.commit()
        logger.debug("Таблица %s готова", table)
    finally:
        conn.close()


def add_theses(user_id: int, theses: List[str]) -> None:
    """Добавляет одну запись с тезисами ответа бота в таблицу пользователя."""
    ensure_user_table(user_id)
    table = _table_name(user_id)
    theses_json = json.dumps(theses, ensure_ascii=False)
    conn = _get_connection()
    try:
        conn.execute(
            f'INSERT INTO "{table}" (theses) VALUES (?)',
            (theses_json,)
        )
        conn.commit()
        logger.debug("Тезисы добавлены для user_id=%s", user_id)
    finally:
        conn.close()


def get_all_theses(user_id: int) -> List[List[str]]:
    """
    Достаёт из БД все сохранённые тезисы пользователя (каждая строка — один ответ).
    Возвращает список списков строк: [ [тезис1, тезис2, ...], ... ].
    """
    ensure_user_table(user_id)
    table = _table_name(user_id)
    conn = _get_connection()
    try:
        cur = conn.execute(
            f'SELECT theses FROM "{table}" ORDER BY id ASC',
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    result: List[List[str]] = []
    for (theses_json,) in rows:
        try:
            theses = json.loads(theses_json)
            if isinstance(theses, list):
                result.append([str(t) for t in theses])
            else:
                result.append([str(theses)])
        except (json.JSONDecodeError, TypeError):
            result.append([theses_json])
    return result


def get_theses_for_prompt(user_id: int) -> str:
    """
    Возвращает все тезисы пользователя в виде одного текста для системного промпта.
    Если тезисов нет — пустая строка.
    """
    all_theses = get_all_theses(user_id)
    if not all_theses:
        return ""

    lines: List[str] = []
    for i, block in enumerate(all_theses, 1):
        for t in block:
            if t.strip():
                lines.append(f"  • {t.strip()}")
        if i < len(all_theses):
            lines.append("")  # разделитель между блоками

    return (
        "\nТЕЗИСЫ ИЗ БАЗЫ ДАННЫХ (история прошлых диалогов с пользователем):\n"
        + "\n".join(lines)
    )
