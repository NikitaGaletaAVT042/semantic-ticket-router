from __future__ import annotations


def split_text_for_routing(
    text: str,
    *,
    min_chunk_chars: int = 40,
    max_chunk_chars: int = 700,
    max_chunks: int = 8,
) -> list[str]:
    """
    Делит длинное обращение на смысловые фрагменты для scoring.

    Это не доменная логика и не маршрутизация по словам.
    Это техническая нарезка текста, чтобы длинные обращения
    не превращались в один размазанный embedding.
    """
    text = " ".join(str(text).split()).strip()

    if not text:
        return []

    rough_parts = _split_by_soft_separators(text)

    chunks: list[str] = []
    for part in rough_parts:
        part = part.strip()

        if len(part) < min_chunk_chars:
            continue

        if len(part) <= max_chunk_chars:
            chunks.append(part)
            continue

        chunks.extend(
            _split_long_part(
                part,
                min_chunk_chars=min_chunk_chars,
                max_chunk_chars=max_chunk_chars,
            )
        )

    result: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        normalized = " ".join(chunk.split()).strip()

        if len(normalized) < min_chunk_chars:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        result.append(normalized)

        if len(result) >= max_chunks:
            break

    return result


def _split_by_soft_separators(text: str) -> list[str]:
    separators = [
        "\n\n",
        "\r\n\r\n",
        ". ",
        "? ",
        "! ",
        "; ",
    ]

    parts = [text]

    for separator in separators:
        new_parts: list[str] = []

        for part in parts:
            split_parts = part.split(separator)

            for index, split_part in enumerate(split_parts):
                split_part = split_part.strip()

                if not split_part:
                    continue

                if index < len(split_parts) - 1 and separator.strip() in {".", "?", "!", ";"}:
                    split_part = split_part + separator.strip()

                new_parts.append(split_part)

        parts = new_parts

    return parts


def _split_long_part(
    text: str,
    *,
    min_chunk_chars: int,
    max_chunk_chars: int,
) -> list[str]:
    words = text.split()

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        extra = len(word) + 1

        if current and current_len + extra > max_chunk_chars:
            chunk = " ".join(current).strip()

            if len(chunk) >= min_chunk_chars:
                chunks.append(chunk)

            current = [word]
            current_len = len(word)
            continue

        current.append(word)
        current_len += extra

    tail = " ".join(current).strip()
    if len(tail) >= min_chunk_chars:
        chunks.append(tail)

    return chunks