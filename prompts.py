import re

B_INSTRUCTION = "Ответь строго одним словом: «верно» или «неверно»."
D_INSTRUCTION = ("Напиши полную программу на Python со sklearn одним блоком ```python. "
                 "Не задавай уточняющих вопросов.")

E_HEADER = ("Определи по показаниям датчиков, произойдёт ли отказ станка: "
            "класс 1 — отказ, класс 0 — отказа нет.")


def build_prompt(task: dict, lang: str = "ru") -> str:
    """Промпт пользователя для типов A–D, F. Системный промпт добавляет ask()-совместимый формат."""
    key = f"текст_{lang}" if f"текст_{lang}" in task else "текст_ru"
    text = task[key]
    t = task["тип"].upper()
    if t == "B":
        return f"{text}\n{B_INSTRUCTION}"
    if t == "D":
        return f"{text}\n{D_INSTRUCTION}"
    return text  # A, C, F: текст задачи без добавок


def build_type_e_prompt(row_text: str, examples: list[tuple[str, int]]) -> str:
    """Промпт типа E: 0/5/20 примеров + одна тестовая строка."""
    if examples:
        lines = "\n".join(f"{txt} -> класс {lab}" for txt, lab in examples)
        body = f"Примеры с известными классами:\n{lines}\n\nНовый объект: {row_text}"
    else:
        body = f"Новый объект: {row_text}"
    return f"{E_HEADER}\n{body}\nКакой класс у нового объекта?"


def extract_code(answer: str) -> str | None:
    """Для типа D: извлечение кода для запуска (pass@1 проверяется отдельно, вручную/скриптом)."""
    m = re.search(r"```python\n(.*?)```", answer, re.DOTALL)
    return m.group(1) if m else None
