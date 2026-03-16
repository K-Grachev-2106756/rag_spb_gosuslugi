# RAG Document Generation Service

Сервис для генерации официальных документов (инструкций/регламентов) с использованием архитектуры RAG (Retrieval-Augmented Generation).

## Описание

Система анализирует запрос пользователя, находит релевантную информацию в базе знаний (HTML-документы) и генерирует структурированный текст официального документа с помощью LLM (Mistral API).

## Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│  1. Data Preparation                                            │
│     └─ HTML Parsing → Recursive Chunking → Context Enrichment   │
│                                                                 │
│  2. Indexing (Embeddings)                                       │
│     └─ sentence-transformers (local) → ChromaDB                 │
│                                                                 │
│  3. Retrieval                                                   │
│     └─ Semantic Search → Top-K Relevant Chunks                  │
│                                                                 │
│  4. Generation (SGR Pattern)                                    │
│     └─ Mistral API → Structured Document                        │
└─────────────────────────────────────────────────────────────────┘
```

## Технологический стек

- **Backend**: FastAPI + Uvicorn
- **Embeddings**: sentence-transformers (локальная модель)
- **Vector DB**: ChromaDB
- **LLM**: Mistral API
- **Containerization**: Docker + Docker Compose

## Обоснование выбора моделей

### Embedding модель
**deepvk/USER-bge-m3**

- Обучена на русскоязычных данных
- Поддерживает мультиязычный контекст
- Высокое качество семантического поиска
- Размерность: 1024

### LLM модель
**Mistral API (mistral-small-latest)**

- Отличное качество генерации текстов
- Поддержка русского языка
- Оптимальное соотношение цена/качество
- Streaming поддержка

## Структура проекта

```
rag_spb/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Точка входа приложения
│   ├── config.py               # Конфигурация
│   ├── api/
│   │   └── app.py              # FastAPI endpoints
│   ├── core/
│   │   └── logging.py          # Логирование
│   ├── data_processing/
│   │   ├── parse.py            # Парсинг HTML (существующий)
│   │   ├── loader.py           # Загрузка документов
│   │   └── chunking.py         # Рекурсивный чанкинг
│   ├── embeddings/
│   │   └── embedding.py        # Embedding провайдер
│   ├── vector_store/
│   │   └── store.py            # ChromaDB интеграция
│   ├── retrieval/
│   │   └── retriever.py        # Поиск релевантных чанков
│   ├── generation/
│   │   └── generator.py        # Mistral API генератор
│   └── pipeline/
│       └── rag.py              # RAG оркестратор
├── tests/
│   ├── test_api.py
│   ├── test_chunking.py
│   └── test_vector_store.py
├── data/                       # HTML документы (база знаний)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Быстрый старт

### 1. Клонирование и настройка

```bash
# Клонировать репозиторий
cd rag_spb

# Создать .env файл
cp .env.example .env

# Отредактировать .env и добавить MISTRAL_API_KEY
```

### 2. Запуск через Docker

```bash
# Сборка и запуск
docker-compose up --build

# Сервис доступен на http://localhost:8000
```

### 3. Локальный запуск

```bash
# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установить зависимости
pip install -r requirements.txt

# Запустить сервер
python -m src.main
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Index Documents
```bash
POST /index
Content-Type: application/json

{}
```

### Generate Document
```bash
POST /generate
Content-Type: application/json

{
  "query": "Мне необходимо составить инструкцию по получению соц обслуживания для жителя блокадного Ленинграда"
}
```

### Generate Document (Streaming)
```bash
POST /generate/stream
Content-Type: application/json

{
  "query": "Ваш запрос"
}
```

### Get Statistics
```bash
GET /stats
```

## Пример запроса

**Входные данные:**
```json
{
  "query": "Мне необходимо составить инструкцию по получению соц обслуживания для жителя блокадного Ленинграда"
}
```

**Выходные данные:**
```json
{
  "query": "Мне необходимо составить инструкцию по получению соц обслуживания для жителя блокадного Ленинграда",
  "document": "# ИНСТРУКЦИЯ\n## по получению социального обслуживания...\n\n### 1. Общие положения\n...",
  "sources": ["Меры поддержки жителей блокадного Ленинграда"]
}
```

## Принцип работы чанкинга

1. **Парсинг HTML**: Извлечение названия, общей информации и основных блоков
2. **Рекурсивный чанкинг**: Разбиение на фрагменты с приоритетом по разделителям (`\n\n`, `\n`, `. `, ` `)
3. **Контекстуализация**: Каждый чанк содержит:
   - Название документа
   - Общую информацию о документе
   - Заголовок блока
   - Основное содержание

```
[Document: title] 
[Description: description] 
[Info: info_title] 
Content
```

## Тестирование

```bash
# Запустить тесты
pytest

# Запустить с покрытием
pytest --cov=src
```

## Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `MISTRAL_API_KEY` | API ключ Mistral | - |
| `MISTRAL_BASE_URL` | URL Mistral API | `https://api.mistral.ai/v1` |
| `LLM_MODEL` | Модель для генерации | `mistral-small-latest` |
| `EMBEDDING_MODEL` | Модель для эмбеддингов | `deepvk/USER-bge-m3` |
| `CHUNK_SIZE` | Размер чанка | `256` |
| `CHUNK_OVERLAP` | Перекрытие чанков | `64` |
| `TOP_K_RESULTS` | Количество результатов поиска | `5` |
| `DOC_MIN_SCORE` | Минимальный_score для документов | `0.5` |
| `CHROMA_PERSIST_DIR` | Путь к хранилищу ChromaDB | `./data/chroma_db` |
| `DATA_DIR` | Путь к данным | `./data` |
| `DEBUG` | Режим отладки | `false` |
| `HOST` | Хост сервера | `0.0.0.0` |
| `PORT` | Порт сервера | `8000` |
