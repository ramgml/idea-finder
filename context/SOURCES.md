# Реестр источников болей

> Порядок добавления: волна 1 — лёгкий доступ (RSS/API/простой HTML), волна 2 — тяжелее (каталоги, антибот).
> Новый источник = новый адаптер по контракту `SourceAdapter` (см. AGENTS.md), пайплайн не меняется.

## Волна 1 — MVP и первые расширения

| Источник | Доступ | kind | Статус | Задача |
|---|---|---|---|---|
| FL.ru | RSS лент заказов | demand | MVP-адаптер | T302 |
| Habr | RSS по хабам + полный текст постов | discussion | MVP-адаптер | T303 |
| Google Play (RU-отзывы) | google-play-scraper, без ключа | complaint | MVP-адаптер | T304 |
| Freelance.ru | RSS | demand | бэклог | — |
| Habr Freelance | RSS | demand | бэклог | — |
| Weblancer | RSS/API | demand | бэклог | — |
| Otzovik / iRecommend | HTML, средний антибот | complaint | бэклог | — |
| Otvet Mail.ru | простой HTML | discussion | бэклог | — |
| RuStore отзывы | официальный API | complaint | бэклог | — |
| Форумы (woman.ru, нишевые) | простой HTML | discussion/complaint | бэклог | — |
| Telegram-чаты | Telethon (нужен клиент) | complaint | бэклог, требует настройки | — |

## Волна 2 — тяжёлый доступ (антибот/каталоги/закрытые)

| Источник | Доступ | kind | Примечание |
|---|---|---|---|
| Kwork | каталог/sitemap, средний антибот | demand | ценен: микруслуги = готовый спрос |
| Яндекс.Дзен | RSS по каналам | discussion | антибот на страницах |
| Ozon / Wildberries отзывы | закрыто, тяжёлый антибот | complaint | только через поисковый API |
| YouDo / Profi.ru | закрыто | demand | вторая волна |
| YouTube-комментарии | официальный API (нужен ключ) | complaint | позже |
| vc.ru | антибот; RSS хабов существуют | discussion | через RSS если живы, иначе поиск |

## Контракт адаптера

```
SourceAdapter:
    name: str
    fetch_new(since: datetime) -> list[RawPost]
```

- `RawPost.kind` обязателен: `demand | complaint | discussion`.
- Рейт-лимиты per-domain (default 1 rps), реалистичный UA, robots.txt уважать.
- Идемпотентность: повторный `fetch_new` отдаёт только новое; дедуп по canonical URL в БД.
- Антибот-домены напрямую не парсить (vc.ru/Дзен/Ozon) — RSS/каталоги или поисковый API второго эшелона.

## Обновление реестра

Новый источник: добавить строку в таблицу + задача на адаптер в Orenda (проект #15).
После внедрения: статус → «работает», дата, счётчики health видны на странице «Источники» дашборда.
