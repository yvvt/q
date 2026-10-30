import json
import time
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import requests
from openai import OpenAI

from config import MODELS, SYSTEM_PROMPT, LOGS_DIR
##from dotenv import load_dotenv

os.makedirs(LOGS_DIR, exist_ok=True)
##load_dotenv()

def _log_entry(model_key: str, entry: dict) -> None:
    """Дописывает одну запись в JSONL-лог."""
    path = os.path.join(LOGS_DIR, f"{model_key}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
API_BASE = "https://api.giga.chat/v1"

AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY")  # те самые 100 символов из кабинета, хранить в .env
SCOPE = "GIGACHAT_API_PERS"

VERIFY = r"combined_ca.pem"

_token_cache = {"token": None,
                "obtained_at": 0.0}


def get_gigachat_token(force: bool = False) -> str:
    """Ступень 1: Basic <ключ авторизации> -> access_token."""
    if not force and _token_cache["token"] and (time.time() - _token_cache["obtained_at"]) < 25 * 60:
        return _token_cache["token"]  # токен живёт 30 минут, берём с запасом

    resp = requests.post(
        OAUTH_URL,
        data={"scope": SCOPE},
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),          # обязательный заголовок, свой uuid4 на каждый запрос
            "Authorization": f"Basic {AUTH_KEY}",  # ключ как есть, без перекодирования
        },
        verify=VERIFY,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    _token_cache.update(token=data["access_token"], obtained_at=time.time())
    return data["access_token"]


def gigachat_list_models() -> list:
    """Точные идентификаторы доступных вам моделей."""
    resp = requests.get(
        f"{API_BASE}/models",
        headers={"Authorization": f"Bearer {get_gigachat_token()}"},
        verify=VERIFY,
        timeout=30,
    )
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]


def ask(
    model_key: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    task_id: str | None = None,
    task_type: str | None = None,
    meta: dict | None = None,
) -> tuple[str, int, float, str]:
    """
    Отправляет запрос к модели и возвращает (ответ, число токенов, время, сырой ответ).
    Логирует запрос и ответ в JSONL.

    Args:
        model_key: ключ модели из config.MODELS
        prompt: текст задачи
        temperature: температура генерации
        max_tokens: максимальное число токенов ответа

    Returns:
        (answer_text, total_tokens, elapsed_seconds, raw_response_json)
    """
    cfg = MODELS[model_key]
    start = time.perf_counter()

    if model_key == "gigachat":
        
        # GigaChat через REST API
        token = get_gigachat_token()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        resp = requests.post(
            f"{API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg["model_name"],
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            verify=VERIFY,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        total_tokens = data.get("usage", {}).get("total_tokens", 0)
        raw = json.dumps(data, ensure_ascii=False)

    elif model_key == "yandexgpt" or model_key == "foreign":
        folder_id = cfg["folder_id"]
        client = OpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],   # https://ai.api.cloud.yandex.net/v1
            project=folder_id,          # уходит в заголовок OpenAI-Project
        )
        completion = client.chat.completions.create(
            model=f"gpt://{folder_id}/{cfg['model_name']}",   # кавычки внутри f-string — одинарные
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,    # диапазон 0..1, ноль допустим
            max_tokens=max_tokens,
        )
        answer = completion.choices[0].message.content
        total_tokens = completion.usage.total_tokens if completion.usage else 0
        raw = completion.model_dump_json()
    elif model_key == "local_qwen":
        client = OpenAI(
            base_url=cfg["base_url"],   # http://localhost:1234/v1
            api_key=cfg["api_key"],     # LM Studio ключ не проверяет; любая непустая строка
        )
        completion = client.chat.completions.create(
            model=cfg["model_name"],    # ТОЧНЫЙ id из GET /v1/models, см. замечание 1
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = completion.choices[0].message.content
        total_tokens = completion.usage.total_tokens if completion.usage else 0
        raw = completion.model_dump_json()
    else:
        raise ValueError(f"Неизвестная модель: {model_key}")

    elapsed = time.perf_counter() - start

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_key": model_key,
        "model_name": cfg["model_name"],
        "temperature": temperature,
        "task_id": task_id,
        "task_type": task_type,
        "prompt": prompt,
        "answer": answer,
        "total_tokens": total_tokens,
        "elapsed_seconds": round(elapsed, 3),
    }
    if meta:
        entry.update(meta)
    _log_entry(model_key, entry)
    return answer, total_tokens, elapsed, raw
