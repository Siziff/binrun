# Гибридная агентная система распознавания и нормализации метаданных гербарных этикеток (OCR + VLM + Tool-Using Agent)

**Постановка исследовательской задачи и программа экспериментов**

Рабочий корпус: микологический гербарий Ботанического института им. В. Л. Комарова РАН, подколлекция головнёвых грибов (Ustilaginomycotina, "Smuts"). Документ построен по структуре научной статьи: введение и мотивация, обзор state-of-the-art, дизайн системы, дизайн экспериментов, метрики, риски, дорожная карта, библиография. Ссылки на литературу даны в виде [N] и приведены в разделе 11.

---

## 1. Введение и научная значимость

Биологические коллекции — материальная основа таксономии и одной из крупнейших задач биологических наук, документирования биоразнообразия Земли. По оценкам Index Herbariorum, в мире насчитывается около 3500 действующих гербариев, содержащих ~398 миллионов физических образцов сосудистых растений, мохообразных, водорослей, лишайников и грибов [1, 2]. Эти коллекции лежат в основе систематических, экологических, климатических и природоохранных исследований [3, 4] и одновременно служат верифицируемой летописью присутствия таксонов в конкретной точке пространства и времени [5, 6].

**Ботанический институт им. В. Л. Комарова РАН (LE)** хранит крупнейшее в России и пятое в мире гербарное собрание — более 7.3 млн образцов высушенных растений и грибов, собранных за 300 лет существования института. Микологический фонд LE (~350 тыс. образцов) является крупнейшей коллекцией грибов в России и включает многочисленные типовые сборы Ustilaginales, Uredinales и других групп.

**Проблема — «dark data» в гербариях.** Несмотря на колоссальную научную ценность, метаданные коллекций остаются «тёмными данными»: оцифровано в виде изображений лишь ~5 % сосудистых растений LE, тогда как для микологической части доля изображений составляет менее 1 %, а структурированные метаданные имеются только у ~22 % образцов. На глобальном уровне темпы транскрипции метаданных не растут уже более десяти лет, оставаясь главным «бутылочным горлышком» мобилизации биоразнообразия в открытые базы [7, 8, 9, 10]. Ручная транскрипция дорога, медленна и подвержена шуму от разнобоя кураторских школ и эпох. Эти данные должны соответствовать принципам FAIR (Findable, Accessible, Interoperable, Reusable) [101] и стандартам биологических данных — Darwin Core [27] и ABCD [26].

**Цель исследования.** Разработать и экспериментально валидировать многоступенчатую гибридную систему извлечения структурированных метаданных из изображений этикеток микологического гербария LE. Система объединяет:

1. классические state-of-the-art методы оптического распознавания текста (OCR/HTR) [11, 12, 13, 14, 15];
2. визуально-языковые модели (Vision-Language Models, VLM) как нормализаторы и структуризаторы [16, 17, 18, 19, 20, 21];
3. tool-using агента-валидатора с доступом к авторитативным таксономическим, биографическим и географическим справочникам [22, 23, 24, 25, 26].

Конечной целью является получение машиночитаемой таблицы метаданных, согласованной со стандартом **Darwin Core** [27], с per-field оценкой уверенности и журналом решений, пригодных для аудита куратором.

---

## 2. Характеристика материала и предварительный анализ

Корпус, на котором ведётся прототипирование, содержит 394 JPEG-изображения этикеток образцов с инвентарными префиксами `LE F-*` из подколлекции головнёвых грибов (Ustilaginales). Первичный визуальный анализ выявил высокую гетерогенность материала по нескольким измерениям (Таблица 1).

**Таблица 1. Гетерогенность подколлекции Smuts.**

| Характеристика | Примеры | Сложность для распознавания |
|---|---|---|
| Эпохи | 1898 (Sydow, Ustilagineen); 1946 (Кузнецова); 1988, 1996 (K. Vánky) | Шрифты разных эпох, выцветание, бумажные дефекты |
| Языки | латынь (научные названия), английский, немецкий, русский (включая дореф. орфографию) | Многоязычность, диакритика (Vánky, ä, ö), кириллица |
| Тип текста | типографский, машинопись, рукопись, смешанный | Рукопись требует HTR-моделей, отличных от OCR |
| Форматы дат | `20. 10. 1898`, `13. VI. 1946`, `26.II.1988`, `3.III.1996` | Римские/арабские цифры, разные разделители |
| Геолокация | устаревшие топонимы («USSR: Kazachstaniae, Alatau Transiliense»); современные координаты (42°55' S, 171°33' E) | Геокодинг исторических административных единиц |
| Статусные пометки | Isotypus, Holotypus (цветные стикеры) | Семантика выражена визуально, а не только текстом |
| Композиция | несколько этикеток в кадре, штрих-коды, разворот конверта | Требуется детекция и сегментация компонентов |

Эта гетерогенность является и научной мотивацией работы: ни один из существующих прямых аналогов (см. §3) не валидирован на материале такого временнóго и лингвистического диапазона, а большинство — заточены под коллекции *растений*, использующие справочник WFO [28], а не справочники грибной номенклатуры [22, 23].

### 2.1. Целевая схема метаданных

Финальная схема будет уточнена совместно с микологами LE; предварительно она построена как подмножество Darwin Core [27], расширенное полями, специфичными для микологии (`host_plant`, `type_status`):

- `catalog_number` — инвентарный номер (например, `LE F-198013`);
- `barcode` — штрих-код;
- `collection_series`, `series_number` — серия (Sydow Ustilagineen; K. Vánky: Ustilaginales и пр.) и её номер;
- `scientific_name`, `taxon_authority`, `taxon_rank` — научное название гриба, автор и таксономический ранг;
- `type_status` — типовой статус (holotypus / isotypus / paratypus / —);
- `host_plant` (Matrix) — растение-хозяин (`Oreobolus strictus`, `Lepidosperma gunnii`);
- `country`, `country_modern` — страна по этикетке и современная страна (например, USSR → Kazakhstan);
- `locality_raw`, `locality_normalized` — место сбора как на этикетке и его нормализованная форма;
- `latitude`, `longitude`, `altitude_m` — координаты и высота;
- `collection_date_raw`, `collection_date_iso` — дата в исходной и нормализованной (ISO 8601) форме;
- `collector` (`leg.`), `determiner` (`det.`) — коллекторы и определители;
- `notes` — прочие пометки (например, `later homonyme`, `ex AA`, `KV: 1500`).

---

## 3. Обзор state of the art

### 3.1. Прямые аналоги: digitization гербариев

**Hespi** (Turnbull et al., 2024/2025) [21] — наиболее близкий к нашей постановке открытый пайплайн. В нём последовательно применяются: двухуровневая объектная детекция (YOLOv8 [29] для компонентов гербарного листа, затем для полей внутри primary specimen label); классификатор типа текста (ResNet-34 [30] / Swin Transformer V2 [31]); связка Tesseract [11] + TrOCR-large [13]; кросс-проверка таксонов со списками **World Flora Online** [28] и **Australian National Species List**; финальная коррекция мультимодальной LLM (GPT-4o) [16]. На печатных этикетках Hespi достигает text similarity 92.7 %, на рукописных — 88.8 %, на гетерогенном DILLEN-наборе из десяти европейских гербариев [32] — 67.1 %. Авторы прямо указывают, что LLM-коррекция даёт прирост 6–11 п.п. метрики.

**Wen et al., 2025** [33] — параллельный подход: связка **PaddleOCR** [12] + text-only LLM **DeepSeek-V3** [34] с трёхуровневой системой confidence-тиров и 16 полями Darwin Core. На печатных растительных этикетках достигнуто 95.4 % field-level accuracy при полностью локальном inference (без отправки изображений во внешние API). Эта работа важна как естественный baseline вопроса «нужен ли мультимодальный VLM, или достаточно классического OCR + текстового LLM».

**Дополнительные бенчмарки и работы:** Owen et al. [9] показали, что HTR-инструменты (ABBYY FineReader, Google Cloud Vision) уже пригодны для части материала; Haston et al. [35], Drinkwater et al. [36] и Alzuru et al. [37] независимо подтвердили принцип «OCR на кадрированной этикетке точнее, чем на полном листе»; Milleville et al. [38] и Walker et al. [4] показали, что Google Cloud Vision на латинских названиях даёт лишь ~24 % полных совпадений и ~36 % частичных, что мотивирует кросс-сверку с таксономическими справочниками. Guralnick et al. [10] и Weaver et al. [39] первыми описали применение современных VLM (GPT-4V) к гербарным этикеткам и сформулировали дизайн «instructable VLM» для биоразнообразия.

### 3.2. Document layout и детекция

Семантическая сегментация изображения этикетки — обязательный шаг: текст распознаётся точнее, когда модель работает на кадрированной этикетке, и тем более — построчно [35, 37]. SOTA-инструменты:

- **YOLOv8 / v10 / v11** [29] и **RT-DETR** [40] — универсальные one-stage детекторы.
- **DocLayout-YOLO** (Zhao et al., 2024) [41] — детектор, специально предобученный на синтетическом DocSynth-300K и оптимизированный для документов; превосходит generic YOLO на бенчмарках document layout.
- **Grounding DINO** [42] + **SAM 2** [43] — open-vocabulary детекция и сегментация: позволяет находить «isotypus-стикер», «штрих-код», «штамп» без полной переразметки.

### 3.3. OCR и HTR

**Современные OCR-движки** для печатного текста: PaddleOCR PP-OCRv4 / PP-StructureV3 [12]; Surya OCR [44]; docTR (Mindee) [45]; Tesseract 5 [11, 14]; EasyOCR [15] как альтернативный baseline. Для исторического печатного материала (например, Sydow 1898) релевантны Kraken [46] и Calamari [47], позволяющие тонкий файнтюн под конкретный шрифт.

**HTR (Handwritten Text Recognition).** Базовая открытая модель — TrOCR-large-handwritten [13]; для исторической рукописи на кириллице и немецкой каллиграфии XIX–XX вв. практичны PyLaia [48], Kraken [46] и публичные модели платформы Transkribus [49]. В последний год SOTA сдвинулась к VLM общего назначения (Florence-2 [50], Qwen2.5-VL [17, 51]; см. также PaliGemma [20]), которые в режиме HTR показывают сравнимое или лучшее качество без необходимости отдельного датасета — за счёт огромного предобучения.

### 3.4. End-to-end / OCR-free document understanding

Альтернативная парадигма — «прочитать документ» одной моделью без явного OCR-шага:

- **Donut** (Kim et al., ECCV 2022) [18] — OCR-free Transformer, обучаемый напрямую от изображения к JSON.
- **Nougat** (Blecher et al., Meta, 2023) [52] — для научных страниц с формулами и таблицами.
- **GOT-OCR 2.0** (Wei et al., 2024) [19] — 580 M unified end-to-end модель, читает обычный текст, формулы, ноты, таблицы; выдаёт markdown/LaTeX.
- **Florence-2** (Microsoft, 2024) [50] — мультизадачная VLM с режимами dense OCR + region grounding.

### 3.5. Vision-Language Models общего назначения

- **Open-weight**: Qwen2.5-VL (7B / 32B / 72B) [17, 51]; InternVL 2.5 / 3 [53]; MiniCPM-V 2.6 [54]; Llama-3.2-Vision [55]; PaliGemma 2 [56].
- **Frontier**: GPT-4o / GPT-4.1 [16]; Claude 3.5 / 4 Sonnet [57]; Gemini 2.0 / 2.5 Pro [58].
- **Инфраструктура локального inference**: vLLM [59], SGLang [60], llama.cpp.

Бенчмарки документного понимания, релевантные нашей задаче: DocVQA [61], OCRBench v2 [62], MMDocBench [63].

### 3.6. Структурное извлечение в JSON

- **Constrained decoding**: Outlines [64], Instructor [65], JSON-schema-guided decoding в vLLM/SGLang — гарантируют синтаксически валидный JSON без лишних шагов парсинга.
- **NER для специфических полей**: Quaesitor [66] — распознавание латинских научных названий в тексте; GLiNER [67] — zero-shot NER для произвольных типов сущностей.
- **Парсеры дат и адресов**: dateparser, libpostal; regex-эвристики для координат в DMS-формате.

### 3.7. Авторитативные источники для микологии

Принципиально важно, что Hespi и Wen опираются на ботанические справочники (WFO, APNI и др.). Для грибов они бесполезны. Адекватные источники:

- **Index Fungorum** [22] и **MycoBank** [23] — канонические БД грибной номенклатуры (~500 тыс. имён каждая, с синонимикой и authorities).
- **GBIF Backbone Taxonomy** [24], **Catalogue of Life** [68], **NCBI Taxonomy** [69] — кросс-сверка.
- **Bionomia** [25] — связывает имена коллекторов и определителей с Wikidata QID, годами жизни, ORCID и реестром образцов; критично для исторических лиц (E. W. D. Holway, M. N. Kuznetzova, K. Vánky).
- **Harvard Index of Botanists (HUH)** [70] — стандартные сокращения авторов таксонов (`P. Henn.`, `Schauer` и т. п.).
- **JSTOR Global Plants** [71] и **GBIF Occurrences** [24] — поиск дубликатов одного и того же образца в других гербариях. Для типов это особенно сильный сигнал: Iso/Holotypus, хранящийся в LE, почти всегда имеет дубликаты в S, NY, K, BPI.

### 3.8. Геокодинг и историческая география

Современный геокодинг: **GeoNames** [72], **Nominatim/OSM** [73], **Pelias** [74], **Wikidata SPARQL** [75]. Для исторических топонимов (USSR, Persia, Ceylon, провинции дореволюционной России): **Getty TGN** [76], **World Historical Gazetteer** [77]. Для качества координат: **CoordinateCleaner** [78] и **bdc** [79] — пакеты biodiversity-data cleaning с типовыми проверками соответствия координат и страны.

### 3.9. Калибровка, агентная оркестрация и валидация

- **Калибровка**: temperature/Platt scaling per-field вероятностей [80]; self-consistency для LLM [81]; LLM-as-a-judge для авто-оценки [82].
- **Оркестрация**: LangGraph [83], Pydantic-AI [84], smolagents [85], Llama-Index Agents [86]; MCP-серверы [87] под каждый внешний источник.
- **Архитектура с независимой проверкой**: разделение Extractor- и Verifier-агентов (по логике CriticGPT [88], Reflexion [89]) для снижения галлюцинаций.

### 3.10. Принципиальные новизны нашего исследования

В таблице 2 сведены ключевые отличия предлагаемой системы от ближайших аналогов [21, 33].

**Таблица 2. Отличия от Hespi и Wen et al.**

| Аспект | Hespi [21] | Wen et al. [33] | Настоящая работа |
|---|---|---|---|
| Целевая группа | Сосудистые растения | Сосудистые растения | **Грибы (микология)** |
| Таксономический бэкенд | WFO + AusNSL | (LLM «на знаниях») | **Index Fungorum + MycoBank** |
| Поле `host_plant` (Matrix) | нет | нет | **есть, нормализуется через WFO** |
| Поле `type_status` со стикером | нет | нет | **есть, отдельный классификатор** |
| Кросс-гербарная сверка дубликатов | нет | нет | **GBIF Occurrences + JSTOR** |
| Историческая география | нет | нет | **Getty TGN + WHGazetteer** |
| Биографические БД коллекторов | нет | нет | **Bionomia → Wikidata QID** |
| Архитектура коррекции | один проход LLM | text-only LLM | **двойной агент Extractor/Verifier** |
| Калибровка и curation-free rate | нет | нет | **first-class метрика** |
| End-to-end VLM как ветка сравнения | нет | нет | **отдельный эксперимент (Эксп. 4)** |

---

## 4. Архитектура предлагаемой системы

Система спроектирована как модульный конвейер из четырёх ступеней (Рис. 1), допускающий замену отдельных модулей независимо. Модульность важна для воспроизводимости и для использования будущих, более совершенных моделей без переписывания всей системы [21].

```
┌─────────────────────────────────────────────────────────────────────┐
│  Изображение этикетки (JPG)                                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Ступень A. Предобработка и сегментация                             │
│  ─ детектор компонентов листа (этикетка / штамп / штрих-код /       │
│    isotypus-стикер) — DocLayout-YOLO [41] или YOLOv8/v11 [29]       │
│  ─ deskew, denoise, super-resolution (Real-ESRGAN [90] / SwinIR     │
│    [91]) для затёртых этикеток XIX в.                               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Ступень B. OCR-ансамбль (классические SOTA-движки)                 │
│  ─ Tesseract 5 [11], PaddleOCR [12], Surya [44], docTR [45]         │
│  ─ TrOCR-large-handwritten [13] для рукописных полей                │
│  ─ выход: токены + bbox + per-token confidence                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Ступень C. VLM-нормализатор                                        │
│  ─ модель: Qwen2.5-VL [17] / InternVL [53] / GPT-4o [16]            │
│  ─ вход: изображение + OCR-выдача + JSON-схема Pydantic             │
│  ─ задача: подтвердить/исправить токены, распределить по полям,     │
│    разобрать рукопись там, где OCR дал низкую уверенность           │
│  ─ выход: структурированный JSON + per-field confidence             │
│  ─ JSON-валидность через constrained decoding (Outlines [64])       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Ступень D. Агент-валидатор (tool-using LLM, LangGraph [83])        │
│  ─ инструменты:                                                     │
│      • Index Fungorum [22] / MycoBank [23] / GBIF [24]              │
│      • Bionomia [25] / HUH [70] / Wikidata [75]                     │
│      • GeoNames [72] / Nominatim [73] / TGN [76] / WHGazetteer [77] │
│      • JSTOR Global Plants [71] / GBIF Occurrences [24]             │
│      • web-поиск для редких случаев                                 │
│  ─ независимый Verifier-агент (CriticGPT-style [88])                │
│  ─ проверки: существование таксона, годы жизни коллектора ⊇ дата    │
│    сбора, locality ∈ country, согласованность с дубликатами         │
│  ─ выход: финальная запись DwC [27] + флаги/обоснования             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                  Итоговая таблица (CSV / Parquet)
                  + журнал решений и calibrated confidence
```

**Рис. 1.** Архитектура конвейера. Каждая ступень оценивается отдельно (см. §6), что позволяет количественно атрибутировать прирост качества компоненту.

---

## 5. Гипотезы исследования

Конкретные гипотезы, проверяемые в экспериментах:

- **H1.** Современные классические OCR-движки [11, 12, 44, 45] достигают приемлемого CER на печатных этикетках, но дают низкий F1 по структурированным полям без LLM-нормализации, и заметно деградируют на рукописи и смешанных этикетках. (соответствует наблюдениям [9, 35, 38]).
- **H2.** Мультимодальная VLM, получающая *одновременно* изображение и OCR-выдачу, существенно (≥ 10 п.п. в F1) превосходит чисто OCR-конвейер и одиночную VLM zero-shot, особенно на смешанных и рукописных этикетках [16, 17, 21].
- **H3.** Доступ агента к авторитативным справочникам [22, 23, 25, 72] даёт дополнительный прирост качества преимущественно в *нормализованных* полях (taxonID, QID, GeoNames ID) и сокращает долю записей, требующих ручной проверки куратором.
- **H4.** На печатных этикетках text-only LLM с OCR-входом [33] даёт качество, сопоставимое с multimodal VLM, при многократно меньшей стоимости; на рукописных и смешанных multimodal-режим необходим.
- **H5.** End-to-end VLM (Donut [18] / Florence-2 [50] / GOT-OCR 2.0 [19]) после файнтюна на нашей схеме конкурентоспособна с классическим конвейером на печатных этикетках и принципиально дешевле в inference.
- **H6.** Calibrated per-field confidence позволяет автоматизировать ≥ 50 % записей с точностью, превышающей точность оператора начального уровня, и направлять оставшиеся к куратору (curation-free rate metric).

---

## 6. Программа экспериментов

Эксперименты выстроены по принципу **поступательного наращивания сложности**: каждый шаг сопоставляется с предыдущим на одном «золотом» наборе, что позволяет количественно атрибутировать прирост каждому компоненту.

### Эксперимент 0. Подготовка датасета и инфраструктуры оценки

**Задачи.**
- Сформировать «золотой» набор: ~50 → 200 → 500 этикеток с ручной разметкой по всем полям (стратифицированно по типу текста, эпохе и языку).
- Pydantic-схема [92] полей и валидатор; формат хранения — JSONL.
- Сквозной evaluation-скрипт с метриками §7.
- Документация схемы и инструкции по разметке для согласованности между разметчиками.

### Эксперимент 1. Бейзлайн — классический OCR

**Проверяемая гипотеза.** H1.

**Подэксперименты.**
- 1a. Tesseract 5 [11] (rus+eng+deu+lat) без предобработки.
- 1b. Tesseract 5 + классическая предобработка (deskew, denoise, Sauvola adaptive threshold [93]).
- 1c. PaddleOCR PP-OCRv4 [12] в multilingual-режиме.
- 1d. Surya OCR [44].
- 1e. TrOCR (printed + handwritten checkpoints) [13].
- 1f. Ансамбль: голосование/выбор по confidence.

**Постобработка для извлечения полей.** Только эвристики/regex (поиск дат по шаблонам, заглавные слова после `leg.` / `det.`, и т. п.) — как в исторических работах [37].

**Ожидаемый результат.** CER приемлем для типографских этикеток (≤ 5 %), F1 по полям заметно ниже из-за слабой структуризации; на рукописных и смешанных — значительная деградация. Этот результат — нижний baseline для всей системы.

### Эксперимент 2. Подключение VLM как нормализатора и структуризатора

**Проверяемая гипотеза.** H2.

**Подэксперименты.**
- 2a. VLM zero-shot, только изображение, без OCR (контрольная точка — что умеет VLM сама).
- 2b. VLM + сырой OCR-текст в промпте.
- 2c. VLM + OCR с bbox и confidence (структурированный вход).
- 2d. Few-shot с 3–5 примерами разных типов этикеток (по аналогии с GPT-3-style in-context learning [94]).
- 2e. Self-consistency [81]: k = 5 прогонов с разным сэмплированием, агрегация большинством.

**Сравнение моделей.** Минимум одна open-weight (Qwen2.5-VL [17]) и одна frontier (GPT-4o [16] или Claude 3.5 Sonnet [57]) — чтобы отделить эффект архитектуры от эффекта схемы.

**Ожидаемый результат.** Существенный рост F1 по полям, особенно на смешанных и рукописных этикетках; снижение CER за счёт визуальной коррекции; формализованный JSON-выход.

### Эксперимент 3. Агентная валидация с внешним контекстом

**Проверяемая гипотеза.** H3.

**Подэксперименты — наращивание набора инструментов:**
- 3a. Агент с поиском по таксонам (**Index Fungorum** [22] / **MycoBank** [23] / **GBIF** [24]).
- 3b. + биографические БД коллекторов (**Bionomia → Wikidata QID** [25, 75], **HUH** [70]).
- 3c. + современный геокодинг (**GeoNames** [72], **Nominatim** [73]).
- 3d. + **исторический геокодинг** (**Getty TGN** [76], **WHGazetteer** [77]) для устаревших административных единиц.
- 3e. + **кросс-гербарная сверка дубликатов** в **JSTOR Global Plants** [71] и **GBIF Occurrences** [24] по комбинации (taxon + collector + date) — особенно мощно для типовых образцов.
- 3f. + **двойной агент Extractor/Verifier** [88, 89]: вторая LLM-инстанция, не видящая решение первой, независимо извлекает поля и сравнивает; рассогласования → флаг к ручной проверке.
- 3g. Полный агент со всеми инструментами и циклом самопроверки.

**Метрики этапа.** Доля исправленных ошибок, доля ложных «исправлений» (как в [21], раздел 5.4), вклад в нормализованные поля.

**Ожидаемый результат.** Дальнейший рост качества именно по нормализованным полям и резкое сокращение случаев, требующих ручной проверки.

### Эксперимент 4. End-to-end VLM без классического OCR (контрольная ветка)

**Проверяемая гипотеза.** H5.

**Подэксперименты.**
- 4a. **GOT-OCR 2.0** [19] — выход в markdown → парсер в JSON-схему.
- 4b. **Florence-2** [50] в режиме dense OCR + region grounding.
- 4c. **Qwen2.5-VL** [17] zero-shot с JSON-schema constrained decoding [64].
- 4d. **Donut** [18], файнтюненный на ~500–1000 наших разметок под целевую JSON-схему.
- 4e. Сравнение с лучшим режимом Эксп. 2 на одном gold-наборе.

**Ожидаемый результат.** Понимание, где end-to-end уже конкурирует с конвейером (вероятно, печатные этикетки Vánky), а где OCR-связка ещё необходима (старая рукопись на кириллице, мульти-этикеточные кадры).

### Эксперимент 5. Text-only LLM vs Multimodal VLM (по мотивам [33])

**Проверяемая гипотеза.** H4.

**Подэксперименты.**
- 5a. PaddleOCR + **DeepSeek-V3** [34] (text-only) — воспроизведение пайплайна Wen [33].
- 5b. PaddleOCR + **Qwen2.5-72B-Instruct** [51] (text-only, локально через vLLM [59]).
- 5c. Та же этикетка → **Qwen2.5-VL** (multimodal).
- 5d. Стратифицированное сравнение по подмножествам: printed / typed / handwritten / mixed.
- 5e. Анализ Pareto-границы: качество vs latency vs cost.

**Ожидаемый результат.** Карта «когда multimodal стоит своих денег» — фундаментальный практический результат для масштабирования на 350 тыс. этикеток.

### Эксперимент 6. Сводный анализ и абляции

- Сравнение четырёх режимов (OCR / OCR+VLM / OCR+VLM+Agent / E2E-VLM) на одном gold-наборе.
- Абляции: вклад каждого инструмента агента; чувствительность к выбору VLM; влияние объёма OCR-входа.
- Анализ остаточных ошибок (failure modes): на каких категориях этикеток система всё ещё проигрывает человеку?
- Экономика: latency и стоимость каждого режима в пересчёте на 1 этикетку и 350 тыс. этикеток.

### Эксперимент 7. Калибровка уверенности и curation-free rate

**Проверяемая гипотеза.** H6.

**Подэксперименты.**
- 7a. Per-field confidence из VLM (logprob по полю) + temperature/Platt scaling [80].
- 7b. Self-consistency [81]: k прогонов VLM, дисперсия как мера неуверенности.
- 7c. **LLM-as-a-judge** [82]: отдельная модель оценивает «правильно ли мы извлекли», возвращает score 0–1.
- 7d. Построение reliability-диаграмм и ROC «уверенность vs корректность».
- 7e. Расчёт **curation-free rate**: при пороге τ — какая доля записей не требует ручной правки, и какова при этом фактическая корректность оставшихся.

### Эксперимент 8. Активное обучение и итеративное расширение gold-набора

**Подэксперименты.**
- 8a. Старт с 50 размеченных этикеток.
- 8b. На каждой итерации: прогон системы, расчёт неуверенности (Эксп. 7), выбор top-k неуверенных для разметки по принципам uncertainty sampling [95] с использованием инструментов вроде modAL [96].
- 8c. Дообучение детекторов; пополнение few-shot пула VLM.
- 8d. Кривая «качество vs объём разметки» — сравнить с random sampling baseline.

### Эксперимент 9. Файнтюн end-to-end VLM на собственной схеме и дистилляция

**Подэксперименты.**
- 9a. Подготовка обучающего набора (image → JSON) из верифицированных записей предыдущих экспериментов.
- 9b. LoRA-файнтюн [97] Qwen2.5-VL-7B на нашей схеме.
- 9c. Полный файнтюн Donut [18].
- 9d. Сравнение с frontier-агентом по качеству, latency, cost.
- 9e. **Дистилляция** [98]: учитель — полный агент с инструментами, ученик — компактная локальная VLM.

---

## 7. Метрики оценки

Метрики выбраны с расчётом на (а) сопоставимость с литературой (Hespi [21], Wen [33]), (б) интерпретируемость для куратора и (в) экономическую интерпретируемость при масштабировании.

**Текстовые (символьный и пословный уровень):**
- CER (Character Error Rate), WER (Word Error Rate) по всему тексту этикетки.
- CER/WER отдельно по типографскому и рукописному фрагментам.

**Полевые (на уровне извлечённой записи):**
- Per-field exact match accuracy.
- Per-field F1 (для составных полей: список коллекторов).
- Per-field Gestalt/Ratcliff–Obershelp similarity [99] — для прямой сопоставимости с Hespi [21].

**Нормализованные (entity-resolution):**
- Научное название — попадание в правильный **taxonID Index Fungorum / MycoBank**.
- Дата — точность ISO-8601.
- Коллектор / определитель — попадание в правильный **Wikidata QID** через Bionomia.
- Геолокация — попадание в правильный **admin-0 / admin-1** регион (через GeoNames ID) и расстояние до эталонной точки.
- `host_plant` — попадание в правильный **WFO taxonID**.

**Системные:**
- **Curation-free rate(τ)** — доля записей, не требующих ручной правки при пороге уверенности τ.
- ROC/PR-кривая «уверенность vs корректность».
- **Latency и cost per record**; экстраполяция на 350 тыс. этикеток.
- **Energy per record** (kWh) для on-prem конфигураций.

**Кросс-валидационные:**
- **Duplicate-consistency**: для образцов с дубликатами в GBIF/JSTOR — доля полей, согласованных с дубликатом.
- **Internal-consistency**: годы жизни коллектора ⊇ дата сбора; locality ∈ country.

**Экспертные:**
- Слепая оценка куратором БИН РАН на N образцах — для системы и для человека-оператора начального уровня (human baseline).

---

## 8. Риски и ограничения

- **Малый объём gold-разметки** ⇒ риск переоценки на узком наборе; смягчается стратификацией и итеративным расширением (Эксп. 8).
- **Стоимость VLM и агента** при масштабировании на 350 тыс. этикеток — необходимо измерять и предусматривать каскад «дёшево → дорого» с роутингом по confidence (Эксп. 7); смягчается также дистилляцией (Эксп. 9).
- **Галлюцинации VLM** [100] — особенно опасны для таксонов и коллекторов; смягчаются обязательной сверкой со справочниками в агентном слое (Эксп. 3) и Verifier-агентом.
- **Воспроизводимость** — фиксируем версии моделей, seed-ы, снапшоты внешних БД.
- **Этика, атрибуция и FAIR-принципы** [101] — корректное цитирование коллекторов и определителей через Bionomia; уважение к данным БИН РАН; публикация только тех полей, на которые получено согласие института.

---

## 9. Дорожная карта

1. **Эксп. 0** — подготовка gold-набора (50 → 200 → 500) и инфраструктуры оценки.
2. **Эксп. 1** — классический OCR-бейзлайн (Tesseract / PaddleOCR / Surya / TrOCR + ансамбль).
3. **Эксп. 2** — VLM-надстройка (open-weight + frontier; zero-/few-shot; self-consistency).
4. **Эксп. 3** — агент-валидатор с инструментами (Index Fungorum/MycoBank, Bionomia, GeoNames + WHGazetteer, GBIF/JSTOR кросс-сверка, Extractor/Verifier).
5. **Эксп. 4** — end-to-end VLM (GOT-OCR2, Florence-2, Donut, Qwen-VL) как контрольная ветка.
6. **Эксп. 5** — text-only LLM vs multimodal VLM на разных подмножествах.
7. **Эксп. 6** — сводный анализ и абляции.
8. **Эксп. 7** — калибровка уверенности и curation-free rate.
9. **Эксп. 8** — активное обучение, расширение gold до production-набора.
10. **Эксп. 9** — файнтюн собственной on-prem VLM и дистилляция агента.
11. Пилотный прогон на полной коллекции Smuts (394 образца); отчёт для БИН РАН.
12. Масштабирование на полную микологическую коллекцию (~350 тыс. образцов).

---

## 10. Литература

1. Thiers B. *Index Herbariorum: A global directory of public herbaria and associated staff.* New York Botanical Garden. https://sweetgum.nybg.org/science/ih/ (accessed 2024).
2. Funk V. A., Gostel M., Devine A. et al. *Guidelines for the preparation of digitized natural history collections data.* PhytoKeys, 2018.
3. Lacey E. A., et al. *Climate change, collections and the classroom.* CBE Life Sciences Education, 2017.
4. Davis C. C. *The herbarium of the future.* Trends in Ecology & Evolution, 2023.
5. Kirchhoff A., Bourlat S. J., et al. *Toward a sustainable digital infrastructure for natural history collections.* Database, 2018.
6. Groom Q., et al. *Improved standardization of transcribed digital specimen data.* Database, 2019.
7. Vollmar A., Macklin J. A., Ford L. *Natural history specimen digitization: challenges and concerns.* Biodiversity Informatics, 2010.
8. Walton S., Livermore L., et al. *Landscape analysis for the specimen data refinery.* Research Ideas and Outcomes, 2020.
9. Owen D., Livermore L., et al. *Towards a scientific workflow featuring Natural Language Processing for the digitisation of natural history collections.* Research Ideas and Outcomes, 2020.
10. Guralnick R., et al. *Towards instructable, generalisable VLMs for label transcription of biodiversity specimens.* 2024.
11. Smith R. *An Overview of the Tesseract OCR Engine.* In Proc. ICDAR, 2007. (Tesseract 5, 2023).
12. Du Y., Chen J. et al. *PaddleOCR: Awesome multilingual OCR toolkit.* PP-OCRv4 / PP-StructureV3, 2023–2024. https://github.com/PaddlePaddle/PaddleOCR
13. Li M., Lv T., et al. *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models.* AAAI, 2023.
14. Smith R. W. *History of the Tesseract OCR Engine: What Worked and What Didn't.* DRR, 2013.
15. JaidedAI. *EasyOCR.* https://github.com/JaidedAI/EasyOCR (2020–2024).
16. OpenAI. *GPT-4o / GPT-4.1 system card.* 2024.
17. Bai S., Chen K., et al. *Qwen2.5-VL Technical Report.* arXiv:2502.13923, 2025.
18. Kim G., Hong T., et al. *OCR-Free Document Understanding Transformer (Donut).* arXiv:2111.15664; ECCV 2022.
19. Wei H., Liu C., Chen J., et al. *General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model (GOT-OCR 2.0).* arXiv:2409.01704, 2024.
20. Beyer L., Steiner A., et al. *PaliGemma: A versatile 3B VLM for transfer.* arXiv:2407.07726, 2024.
21. Turnbull R., Fitzgerald E., Thompson K., Birch J. L. *Hespi: A pipeline for automatically detecting information from herbarium specimen sheets.* arXiv:2410.08740; BioScience, 2025. DOI:10.1093/biosci/biaf042.
22. *Index Fungorum.* CABI / Royal Botanic Gardens, Kew. http://www.indexfungorum.org/
23. *MycoBank Database.* International Mycological Association / Westerdijk Institute. https://www.mycobank.org/
24. *GBIF: Global Biodiversity Information Facility.* GBIF Backbone Taxonomy & Occurrence API. https://www.gbif.org/
25. Shorthouse D. P. *Bionomia: linking people and specimens.* https://bionomia.net/
26. Chapman A. D., Wieczorek J. (eds.). *Principles and Methods of Data Cleaning – Primary Species and Species-Occurrence Data.* GBIF, 2020.
27. Wieczorek J., Bloom D., Guralnick R., et al. *Darwin Core: An Evolving Community-Developed Biodiversity Data Standard.* PLoS ONE, 2012.
28. *World Flora Online (WFO).* http://www.worldfloraonline.org/, 2023.
29. Jocher G., Chaurasia A., Qiu J., et al. *Ultralytics YOLOv8 / YOLOv11.* 2023–2024. https://github.com/ultralytics/ultralytics
30. He K., Zhang X., Ren S., Sun J. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
31. Liu Z., Hu H., et al. *Swin Transformer V2: Scaling Up Capacity and Resolution.* CVPR, 2022.
32. Dillen M., Groom Q., et al. *A benchmark dataset of herbarium specimen images with label data.* Biodiversity Data Journal, 2019.
33. Wen J., et al. *Automated information extraction from plant specimen labels using OCR and Large Language Models.* Biodiversity Data Journal, 2025. Code: https://github.com/JW135/Herbarium-OCR-LLM-Pipeline
34. DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv:2412.19437, 2024.
35. Haston E., Cubey R., Harpke D. *Data concepts and their relevance for data capture in large scale digitisation of biological collections.* International Journal of Humanities and Arts Computing, 2012; Haston 2015.
36. Drinkwater R. E., Cubey R. W. N., Haston E. M. *The use of Optical Character Recognition (OCR) in the digitisation of herbarium specimen labels.* PhytoKeys, 2014.
37. Alzuru I., Matsunaga A., Tsugawa M., Fortes J. A. B. *Quality-aware human-machine text extraction for biocollections using OCR.* IEEE eScience, 2016.
38. Milleville K., Verstockt S., et al. *Automatic Extraction of Specimens from Multi-Specimen Herbaria.* J. Comput. Cult. Herit., 2023.
39. Weaver W. N., Smith S. A. *From leaves to labels: Building modular machine-learning networks for rapid herbarium specimen analysis with LeafMachine2 / GPT-4V approaches.* Applications in Plant Sciences, 2023–2024.
40. Zhao Y., Lv W., Xu S., et al. *DETRs Beat YOLOs on Real-time Object Detection (RT-DETR).* CVPR, 2024.
41. Zhao Z., Kang Z., et al. *DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception.* arXiv:2410.12628, 2024.
42. Liu S., Zeng Z., Ren T., et al. *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.* arXiv:2303.05499, 2023.
43. Ravi N., Gabeur V., et al. *SAM 2: Segment Anything in Images and Videos.* Meta AI, 2024.
44. Paruchuri V. *Surya OCR.* https://github.com/VikParuchuri/surya, 2024.
45. Mindee. *docTR: Document Text Recognition.* https://github.com/mindee/doctr, 2021–2024.
46. Kiessling B. *Kraken — a Universal Text Recogniser for HTR.* https://kraken.re/
47. Wick C., Reul C., Puppe F. *Calamari — A High-Performance Tensorflow-based Deep Learning Package for OCR.* Digital Humanities Quarterly, 2020.
48. Puigcerver J. *Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?* (PyLaia) ICDAR, 2017.
49. Kahle P., Colutto S., Hackl G., Mühlberger G. *Transkribus — a Service Platform for Transcription, Recognition and Retrieval of Historical Documents.* ICDAR-OST, 2017.
50. Xiao B., Wu H., et al. *Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks.* arXiv:2311.06242, Microsoft, 2024.
51. Yang A., Bai J., et al. *Qwen2.5 Technical Report.* arXiv:2412.15115, 2024.
52. Blecher L., Cucurull G., Scialom T., Stojnic R. *Nougat: Neural Optical Understanding for Academic Documents.* arXiv:2308.13418, Meta, 2023.
53. Chen Z., Wu J., et al. *InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks; InternVL 2.5 / 3.* CVPR 2024; 2024–2025.
54. Yao Y., Yu T., et al. *MiniCPM-V: A GPT-4V Level MLLM on Your Phone.* arXiv:2408.01800, 2024.
55. Meta AI. *Llama 3.2 Vision.* 2024. https://ai.meta.com/llama/
56. Steiner A., et al. *PaliGemma 2: A Family of Versatile VLMs for Transfer.* arXiv:2412.03555, 2024.
57. Anthropic. *Claude 3.5 / Claude 4 model card.* 2024–2025.
58. Google DeepMind. *Gemini 2.0 / 2.5 Pro technical report.* 2024–2025.
59. Kwon W., Li Z., Zhuang S., et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM).* SOSP, 2023.
60. Zheng L., Yin L., et al. *SGLang: Efficient Execution of Structured Language Model Programs.* arXiv:2312.07104, 2023.
61. Mathew M., Karatzas D., Jawahar C. V. *DocVQA: A Dataset for VQA on Document Images.* WACV, 2021.
62. Fu C., et al. *OCRBench v2: An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning.* 2024.
63. Zhu J., et al. *MMDocBench: Benchmarking Large Vision-Language Models for Fine-Grained Visual Document Understanding.* 2024.
64. Willard B., Louf R. *Efficient Guided Generation for Large Language Models (Outlines).* arXiv:2307.09702, 2023.
65. Liu J. *Instructor: Structured outputs powered by LLMs.* https://github.com/jxnl/instructor, 2023.
66. Little D. P. *Recognition of Latinized Scientific Names with NLP — Quaesitor.* Biodiversity Data Journal, 2020.
67. Zaratiana U., Tomeh N., Holat P., Charnois T. *GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer.* NAACL, 2024.
68. Bánki O., et al. *Catalogue of Life Checklist.* https://www.catalogueoflife.org/
69. Schoch C. L., et al. *NCBI Taxonomy: a comprehensive update on curation, resources and tools.* Database, 2020.
70. Harvard University Herbaria. *Index of Botanists.* https://huh.harvard.edu/botanists
71. *JSTOR Global Plants.* https://plants.jstor.org/
72. Wick M. *GeoNames Geographical Database.* https://www.geonames.org/
73. *Nominatim — OpenStreetMap geocoder.* https://nominatim.org/
74. Mapzen / Linux Foundation. *Pelias geocoder.* https://pelias.io/
75. Vrandečić D., Krötzsch M. *Wikidata: A Free Collaborative Knowledgebase.* CACM, 2014.
76. J. Paul Getty Trust. *Getty Thesaurus of Geographic Names (TGN).* http://www.getty.edu/research/tools/vocabularies/tgn/
77. Mostern R., Berman M. L., et al. *The World Historical Gazetteer.* University of Pittsburgh. https://whgazetteer.org/
78. Zizka A., Silvestro D., et al. *CoordinateCleaner: Standardized cleaning of occurrence records from biological collection databases.* Methods in Ecology and Evolution, 2019.
79. Ribeiro B. R., Velazco S. J. E., et al. *bdc: A toolkit for standardizing, integrating and cleaning biodiversity data.* Methods in Ecology and Evolution, 2022.
80. Guo C., Pleiss G., Sun Y., Weinberger K. Q. *On Calibration of Modern Neural Networks.* ICML, 2017.
81. Wang X., Wei J., et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR, 2023.
82. Zheng L., Chiang W.-L., et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS, 2023.
83. LangChain Inc. *LangGraph: Stateful, multi-actor agents with LLMs.* https://github.com/langchain-ai/langgraph, 2024.
84. *Pydantic-AI.* https://github.com/pydantic/pydantic-ai, 2024.
85. Hugging Face. *smolagents.* https://github.com/huggingface/smolagents, 2024.
86. Liu J., et al. *LlamaIndex: A data framework for LLM applications.* https://github.com/run-llama/llama_index
87. Anthropic. *Model Context Protocol (MCP).* https://modelcontextprotocol.io/, 2024.
88. McAleese N., et al. *LLM Critics Help Catch LLM Bugs (CriticGPT).* OpenAI, 2024.
89. Shinn N., Cassano F., Berman R., et al. *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS, 2023.
90. Wang X., Xie L., Dong C., Shan Y. *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.* ICCV Workshops, 2021.
91. Liang J., Cao J., et al. *SwinIR: Image Restoration Using Swin Transformer.* ICCV Workshops, 2021.
92. *Pydantic v2.* https://docs.pydantic.dev/
93. Sauvola J., Pietikäinen M. *Adaptive document image binarization.* Pattern Recognition, 2000.
94. Brown T., Mann B., et al. *Language Models are Few-Shot Learners (GPT-3).* NeurIPS, 2020.
95. Settles B. *Active Learning Literature Survey.* University of Wisconsin–Madison, 2010.
96. Danka T. *modAL: A modular active learning framework for Python.* JOSS, 2018.
97. Hu E. J., Shen Y., et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022.
98. Hinton G., Vinyals O., Dean J. *Distilling the Knowledge in a Neural Network.* NIPS Deep Learning Workshop, 2014.
99. Ratcliff J. W., Metzener D. E. *Pattern Matching: The Gestalt Approach.* Dr. Dobb's Journal, 1988.
100. Huang L., Yu W., et al. *A Survey on Hallucination in Large Language Models.* arXiv:2311.05232, 2023.
101. Wilkinson M. D., Dumontier M., et al. *The FAIR Guiding Principles for scientific data management and stewardship.* Scientific Data, 2016.
