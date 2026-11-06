from __future__ import annotations


def normalize_text(text: str) -> str:
    """
    Нормализация без доменных regex-правил.

    Мы не ищем слова по словарям и не маршрутизируем по ключевым словам.
    Здесь только безопасная техническая нормализация текста.
    """
    text = str(text)
    text = text.replace("ё", "е")
    text = text.replace("\u00a0", " ")

    lines = [line.strip() for line in text.splitlines()]
    text = " ".join(line for line in lines if line)

    return " ".join(text.split()).strip()