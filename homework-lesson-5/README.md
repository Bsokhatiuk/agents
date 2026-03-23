# Research Agent з RAG-системою

`homework-lesson-5` розширює попередній Research Agent: тепер агент працює не лише з веб-пошуком, а й з локальною базою знань через RAG.

Проєкт складається з двох окремих частин:

1. `ingest.py` будує або оновлює локальний індекс знань із документів у `data/`
2. `main.py` запускає агента, який комбінує:
   - `knowledge_search` для пошуку в локальній базі знань
   - `web_search` для пошуку в інтернеті
   - `read_url` для читання конкретних сторінок
   - `write_report` для збереження фінального Markdown-звіту

## Що змінилося в lesson 5

Порівняно з попередніми версіями, у проєкті з'явилися:

- `ingest.py` для індексації локальних документів
- `retriever.py` з hybrid retrieval
- новий tool `knowledge_search`
- локальне сховище індексу в `storage/faiss_index/`
- комбінований пошук: semantic + BM25 + reranking

## Швидкий запуск

### 1. Встановіть залежності

```bash
pip install -r requirements.txt
```

### 2. Налаштуйте `.env`

Створіть `.env` на основі `.env.example`:

```env
OPENAI_API_KEY=your_actual_api_key_here
```

Обов'язковий ключ:

- `OPENAI_API_KEY`

### 3. Покладіть документи в `data/`

Згідно із завданням основний сценарій ingestion орієнтований на PDF-документи. Поточна реалізація також підтримує файли:

- `.pdf`
- `.txt`
- `.md`

### 4. Побудуйте індекс

```bash
python ingest.py
```

Скрипт:

- читає документи з `data/`
- розбиває їх на чанки через `RecursiveCharacterTextSplitter`
- створює embeddings через OpenAI
- оновлює FAISS-індекс
- експортує JSON для BM25
- зберігає manifest зі станом файлів

Після успішного запуску будуть оновлені:

- `storage/faiss_index/index.faiss`
- `storage/faiss_index/index.pkl`
- `storage/faiss_index/manifest.json`
- `storage/faiss_index/bm25_chunks.json`

### 5. Запустіть агента

```bash
python main.py
```

Після запуску відкриється інтерактивна консоль. Для виходу використовуйте `exit` або `quit`.

## Важливо

`main.py` імпортує retriever під час старту застосунку, тому перед запуском агента потрібно хоча б один раз виконати `python ingest.py`. Якщо індекс відсутній або застарів після змін у `data/`, агент працюватиме некоректно.

Якщо ви додали, видалили або змінили документи в `data/`, повторно запустіть:

```bash
python ingest.py
```

## Як працює RAG у цьому проєкті

### Ingestion pipeline

`ingest.py`:

- знаходить усі підтримувані документи в `data/`
- обчислює `sha256` для відстеження змін
- перевикористовує існуючий індекс замість повної перебудови
- видаляє з індексу чанки файлів, які були видалені або змінені
- додає нові чанки в FAISS

### Retrieval pipeline

`retriever.py` реалізує триетапний пошук:

1. semantic search через FAISS
2. BM25 lexical search через `BM25Retriever`
3. reranking через cross-encoder `BAAI/bge-reranker-base`

Для об'єднання semantic і BM25 результатів використовується `EnsembleRetriever`.

### Agent workflow

Агент:

- сам вирішує, коли шукати у локальній базі знань
- за потреби доповнює відповідь веб-пошуком
- може комбінувати локальні документи та зовнішні джерела
- зберігає фінальний звіт у `output/`

## Основні файли

- `main.py` — консольний запуск агента та стрімінг подій у термінал
- `agent.py` — створення LangChain/LangGraph-агента з усіма tools
- `tools.py` — `knowledge_search`, `web_search`, `read_url`, `write_report`
- `retriever.py` — hybrid retrieval і reranking
- `ingest.py` — pipeline індексації документів
- `config.py` — конфігурація, моделі та системний prompt
- `data/` — локальні документи для ingestion
- `storage/faiss_index/` — згенерований індекс і службові файли retriever'а
- `output/` — фінальні Markdown-звіти агента
- `TASK.md` — формулювання домашнього завдання

## Структура проєкту

```text
homework-lesson-5/
├── agent.py
├── config.py
├── ingest.py
├── main.py
├── README.md
├── requirements.txt
├── retriever.py
├── TASK.md
├── tools.py
├── data/
├── output/
└── storage/
    └── faiss_index/
```

## Основні залежності

- `langchain`, `langgraph` — агент і orchestration
- `langchain-openai` — chat model та embeddings
- `faiss-cpu` — векторна база
- `rank_bm25` — lexical retrieval
- `sentence-transformers` — reranker
- `ddgs` — веб-пошук
- `trafilatura` — витяг тексту зі сторінок
- `pypdf` — читання PDF
- `pydantic`, `pydantic-settings` — конфігурація

## Поточні значення за замовчуванням

У `config.py` зараз задано:

- LLM: `openai:gpt-5.4`
- embeddings: `text-embedding-3-large`
- reranker: `BAAI/bge-reranker-base`
- `chunk_size=1000`
- `chunk_overlap=200`
- `retrieval_top_k=10`
- `rerank_top_n=3`

## Приклад сценарію використання

1. Запустити `python ingest.py`
2. Запустити `python main.py`
3. Поставити запит на кшталт:

```text
Що таке RAG і які підходи до retrieval описані в локальних документах?
```

4. Агент може:
   - викликати `knowledge_search`
   - за потреби додатково викликати `web_search`
   - прочитати зовнішнє джерело через `read_url`
   - зберегти результат у `output/research_report.md`

## Очікуваний результат

Після виконання домашнього завдання проєкт повинен забезпечувати:

- ingestion локальних документів без повторного embedding незмінених файлів
- hybrid retrieval по локальній базі знань
- reranking результатів перед передачею контексту агенту
- комбінування локального knowledge base і веб-джерел
- збереження фінального Markdown-звіту в `output/`
