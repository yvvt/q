import json

REQUIRED_FIELDS = ["id", "тип", "тема", "текст_ru", "эталон", "источник"]


def load_bank(path: str) -> list[dict]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            missing = [k for k in REQUIRED_FIELDS if k not in t]
            if missing:
                raise ValueError(f"Строка {n}: не хватает полей {missing}")
            tasks.append(t)
    ids = [t["id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Обнаружены дубликаты id задач")
    return tasks
