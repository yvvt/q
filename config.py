import os
from dotenv import load_dotenv

load_dotenv()

# Системный промпт — одинаковый для всех моделей.
# Отлаживайте на 5 задачах, не входящих в оценку.
SYSTEM_PROMPT = """Ты решаешь задачу по машинному обучению.
Реши задачу пошагово. Пиши ответ на русском языке.
В последней строке напиши ровно в формате (для задач классификации в ответ пиши 0 или 1):
ОТВЕТ: <число или слово(а) (e.g. 100 или верно или 0 или логистическая регрессия)>
УВЕРЕННОСТЬ: <число от 0 до 100>
"""

# Формат ответа, который ожидаем от моделей
ANSWER_PATTERN = r"ОТВЕТ:\s*(.+)"
CONFIDENCE_PATTERN = r"УВЕРЕННОСТЬ:\s*(\d+(?:\.\d+)?)"

# Допуск для числовых ответов (относительный)
RELATIVE_TOLERANCE = 0.01  # 1%

MODELS = {
    "gigachat": {
        "client_id": os.getenv("GIGACHAT_AUTH_KEY"),
        "model_name": "GigaChat-3-Ultra", 
        "scope": "GIGACHAT_API_PERS",
    },
    "yandexgpt": {
        "folder_id":os.getenv("YANDEX_FOLDER_ID"),
        "api_key": os.getenv("YANDEX_API_KEY"),
        "base_url": "https://ai.api.cloud.yandex.net/v1",
        "model_name": "yandexgpt-5-lite",  
    },
    "foreign": {
        "folder_id": os.getenv("YANDEX_FOLDER_ID"),
        "api_key": os.getenv("YANDEX_API_KEY"),
        "base_url": "https://ai.api.cloud.yandex.net/v1",
        "model_name": "gpt-oss-20b/latest",   
    },
    "local_qwen": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",          # Ollama не проверяет ключ
        "model_name": "qwen3:4b-instruct"
    },
}

LOGS_DIR = "logs"
RESULTS_DIR = "results"
BANK_PATH = "data/tasks.jsonl"
