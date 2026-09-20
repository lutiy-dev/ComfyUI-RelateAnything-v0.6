# ComfyUI-RelateAnything v0.6 — локальный ONNX + существующий SAM3

Подготовлено и проверено 20.09.2026. Это самостоятельная ONNX-обёртка: она не импортирует
`relsgg`, `torch` или `transformers`, не читает `model.pth` / `text_student.pt`, не скачивает
модели при запуске и не меняет настройки SAM3. ComfyUI и SAM3 продолжают использовать свой
существующий Torch; сама ветка RelateAnything исполняется через ONNX Runtime.

## Что в пакете

- `__init__.py` — четыре ноды с отдельными идентификаторами v0.6.
- `requirements.txt` — только `onnxruntime==1.29.0` (CPU-вариант).
- `workflows/RELATE_ANYTHING_SAM3_ONNX_v006.json` — импортируемый UI workflow:
  SOURCE IMAGE → существующий SAM3 → RA Regions → RA ONNX Predict → Preview as Text.
- `workflows/RELATE_ANYTHING_ONNX_DIAGNOSTICS_v006.json` — отдельная диагностика без SAM3/inference.
- `evidence/` — реальные session inputs/outputs, источники, хеши файлов, результаты тестов.

Весов в репозитории нет. Они скачиваются из официального Hugging Face по закреплённой
ревизии; загрузчик сверяет SHA256 всех трёх необходимых файлов:

```powershell
& 'Q:\AI_ArchViz\ComfyUI_windows_portable\python_embeded\python.exe' scripts/download_models.py --output 'F:\ComfyUI\models\relateanything\relsgg-vits16plus'
```

Команда выполняется из папки этого пакета. Можно использовать другой Python и другой путь.
Загрузчик использует только стандартную библиотеку Python, ничего не устанавливает и не
перезаписывает существующий файл с другой контрольной суммой.

## Подтверждённая сигнатура — не предположение

Hugging Face: `maelic/relsgg-vits16plus`, revision
`caf70d3af46478614209d86881a16e601d6bbf61`.

| Вход | Тип | Форма и смысл |
|---|---|---|
| `image` | float32 | `[B,3,448,448]`, RGB в диапазоне 0..1 |
| `boxes` | float32 | `[B,N,4]`, нормализованные **cx,cy,w,h** |
| `box_counts` | int64 | `[B]`, число настоящих регионов до padding |
| `W` | float32 | `[V,512]`, готовые эмбеддинги предикатов из банка |
| `alpha` | float32 | `[V]`, веса выбора экспертов из того же банка |

| Выход | Тип | Форма |
|---|---|---|
| `pred_logits` | float32 | `[B,K,V]` |
| `pair_logits` | float32 | `[B,K]` |
| `sub_idx` | int64 | `[B,K]` |
| `obj_idx` | int64 | `[B,K]` |
| `valid_mask` | bool | `[B,K]` |

Эти данные прочитаны через настоящие `session.get_inputs()` / `session.get_outputs()`.
Нода обрабатывает один исходный RGB-кадр за запуск (`B=1`), до 32 регионов;
боксы дополняются нулями до 32 согласно официальному runtime и sidecar.
Динамическая ось ONNX `num_boxes` не трактуется как обещание поддержки произвольного лимита.

Изображение преобразуется в uint8 RGB, затем растягивается в квадрат 448×448 через
OpenCV INTER_LINEAR и делится на 255. Letterbox, ImageNet mean/std и перестановка RGB→BGR
на входе модели не применяются. Preprocessing побитово сопоставлен с официальным runtime.

**У этого ONNX нет входа масок.** Каждая отдельная SAM3 instance mask превращается в один
pixel-xyxy bounding box, а затем в нормализованный cxcywh. Точная форма маски в ONNX не поступает.
Объединённая маска всех объектов даёт один регион; разделять её на экземпляры автоматически
эта нода не пытается. Перекрывающиеся отдельные маски поддерживаются.

Текст строится из предикатов официального банка (243 готовых имени) и индексов регионов.
Классы вроде «окно»/«автомобиль» модель не придумывает. `SAM3 #N` — исходный индекс маски
или бокса, с нуля; координаты соответствий есть в `regions_debug` / `relations_json`.

Score: `sigmoid(a * (pred_logit + pair_weight * pair_logit) + b)`;
проверенный sidecar задаёт `a=0.5651`, `b=-1.9623`. По умолчанию `pair_weight=1.0`.
Отбрасываются invalid/padded/self пары, для каждой направленной пары выбирается один
лучший предикат, затем применяется порог и общий top-k. Detector/SAM3 scores в ранжирование
не подмешиваются. Per-predicate thresholds из другого режима оценки не применяются.
Изменение `pair_weight` меняет score и смысл калибровки; для первого теста оставьте 1.0.

## Установка без изменения основного стека

В тестовом окружении `import onnxruntime` работает и возвращает версию 1.29.0;
CPU inference прошёл. **Если ваш Runtime уже работает, повторная установка не требуется автоматически.**
Метаданные тестового окружения также содержали `onnxruntime-gpu 1.26.0`; пакеты не удалялись и
не переустанавливались. GPU-выполнение v0.6 не проверено. Начните с CPUExecutionProvider.

1. Клонируйте репозиторий в **новую** папку custom_nodes (или скачайте ZIP и распакуйте):

   ```powershell
   git clone https://github.com/lutiy-dev/ComfyUI-RelateAnything-v0.6.git 'Q:\AI_ArchViz\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-RelateAnything-ONNX-v06'
   ```

   Итоговый путь:

   ```text
   Q:\AI_ArchViz\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-RelateAnything-ONNX-v06\
   ```

   `__init__.py` должен лежать непосредственно в ней. Старую папку
   `ComfyUI-RelateAnything` оставьте как есть. Новые идентификаторы
   `RARegionsV06`, `RAOnnxLoadV06`, `RAOnnxInspectV06`, `RAOnnxPredictV06`
   не конфликтуют с прежними нодами. Это рекомендуемый вариант, сохраняющий старые графы.

2. Положите три **проверенных файла одной ревизии** рядом:

   ```text
   F:\ComfyUI\models\relateanything\relsgg-vits16plus\relateanything.onnx
   F:\ComfyUI\models\relateanything\relsgg-vits16plus\relateanything.json
   F:\ComfyUI\models\relateanything\relsgg-vits16plus\predicate_bank.npz
   ```

   Скачайте их через `scripts/download_models.py` или по официальным ссылкам ниже.
   Существующие `model.pth`, `calibration.json`, остальные веса не заменяйте.
   Можно хранить ONNX в отдельной папке и указать абсолютный путь к нему в ноде загрузки.
   Также допустима отдельная папка на Q:, например
   `Q:\AI_ArchViz\ComfyUI_windows_portable\ComfyUI\models\relateanything\relsgg-vits16plus`.
   В workflow по умолчанию прописан путь на F:.

3. Дождитесь окончания текущих генераций и перезапустите ComfyUI вручную.
   В рамках подготовки пакета сервер не перезапускался, старые файлы не заменялись.

4. Импортируйте диагностический JSON, проверьте путь к `.onnx`, запустите.
   В `Preview as Text` должны появиться реальные inputs/outputs и `inference_ready: true`.
   Это проверка загрузки/контракта, ещё не inference.

5. Импортируйте основной JSON. В Load Image выберите свой **исходный RGB-рендер**;
   тот же выход IMAGE идёт в SAM3, RA Regions и Predict. Не подавайте визуализацию SAM3
   с цветными масками в Predict. Измените SAM3 text_prompt с `window` на нужный объект.
   Нужны минимум два найденных экземпляра/региона.

6. Запустите. Проверьте PreviewImage SAM3, `regions_debug`, затем `RELATIONS TEXT`.
   Строка `No relations passed the selected threshold` означает успешный inference
   без прошедших порог отношений, а не ошибку модели.

### Если всё-таки нужен именно файл на замену

Можно сохранить прежний `__init__.py` **вне `custom_nodes`**, а затем заменить его новым.
Но старые `RALoadModel` / `RAPredictRelations` не являются псевдонимами новых нод:
после замены старые графы потребуют миграции. Используйте приложенный v006 JSON.
Не устанавливайте этот пакет одновременно двумя копиями: идентификаторы v0.6 будут дублироваться.
Для отката при остановленном ComfyUI восстановите сохранённый старый файл; модели не трогайте.

### Другая машина, где ONNX Runtime ещё отсутствует

Сначала проверьте **embedded Python именно этой portable-установки** (PowerShell):

```powershell
& 'Q:\AI_ArchViz\ComfyUI_windows_portable\python_embeded\python.exe' -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

Если импорт работает, сначала попробуйте диагностику; обновление не требуется автоматически.
Если Runtime отсутствует, проверьте план установки:

```powershell
& 'Q:\AI_ArchViz\ComfyUI_windows_portable\python_embeded\python.exe' -m pip install --dry-run -r 'Q:\AI_ArchViz\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-RelateAnything-ONNX-v06\requirements.txt'
```

Устанавливайте только после проверки, что план не заменяет рабочие библиотеки:

```powershell
& 'Q:\AI_ArchViz\ComfyUI_windows_portable\python_embeded\python.exe' -m pip install -r 'Q:\AI_ArchViz\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-RelateAnything-ONNX-v06\requirements.txt'
```

Не устанавливайте CPU/GPU/DirectML distributions поверх друг друга ради этой ноды.
`numpy` и `cv2` используются из уже работающего ComfyUI. Они не закрепляются заново и
не обновляются requirements этого пакета. `onnx` нужен только для некоторых тестовых fixtures,
но не для работы ноды. `relsgg`, transformers, torchvision, text encoders устанавливать не нужно.

## Подключение к вашему готовому SAM3-графу

Подтверждено через текущий `/object_info` и локальный код:
`custom_nodes.comfyui-sam3`, `LoadSAM3Model → SAM3Grounding`.
Входы SAM3/precision не заменяются сторонним сегментатором. Использован известный fp32,
compile=false; для существующего графа сохраняйте свои рабочие настройки.

- Подайте оригинальный IMAGE в `RA Regions v0.6.image` и `RA ONNX Predict v0.6.image`.
- `SAM3Grounding.masks` → `RA Regions v0.6.masks`.
- `RA Regions.regions` → `RA ONNX Predict.regions`.
- `RA Load Local ONNX.ra_onnx_model` → `RA ONNX Predict.ra_onnx_model`.
- `relations_text` → встроенный `PreviewAny` / `Preview as Text`.

Альтернатива: отсоедините masks и подключите `SAM3Grounding.boxes` (STRING, JSON массив
pixel-xyxy) к `RA Regions.boxes_json`. Вход `SAM3_BOXES_PROMPT` не подходит — это prompts,
а не выход детектора. Другие форматы координат автоматически не угадываются.

Можно подать один из `RA_REGIONS`, `masks`, `boxes_json` непосредственно в Predict.
Одновременно подключать несколько источников нельзя. Прямые masks/boxes используют
threshold=0.5, min_area=20, max_regions=32; чтобы менять фильтры и видеть debug, используйте Regions.

MASK может быть `[H,W]`, `[N,H,W]`, `[N,1,H,W]`, `[N,H,W,1]` или списком таких тензоров.
Все маски должны относиться к **одному** source IMAGE того же размера; список кадров не поддерживается.
Порядок экземпляров сохраняется, маленькие маски отбрасываются, первые 32 оставшихся
проходят дальше. Любое ограничение количества отражено в `dropped_by_limit`.

Существующий `RA_REGIONS` поддержан для словаря с `boxes [N,4]` (pixel xyxy), `height`,
`width`; поле masks не используется ONNX. Более 32 регионов из внешнего RA_REGIONS
вызывают ошибку вместо скрытого усечения. `source_indices` необязателен.

## Словарь, диагностика и ошибки

`vocabulary`: точные имена из банка, по одному на строку. Пустое поле выбирает официальные
35 default-предикатов. Произвольные новые строки требуют отдельно рассчитанного и проверенного
банка; v0.6 не запускает текстовый encoder. Например, используйте `to the left of`,
а не `left of`, и `casting shadow on`, а не `casting a shadow on`.
Полный список доступен в диагностике (`available_predicates`).

Любое несовпадение известных SHA256/входов/выходов/sidecar блокирует inference.
Loader всё равно возвращает реальную диагностику для сессии, которую Runtime смог открыть.
Для другого ONNX revision сначала нужен отдельный аудит, а не отключение проверки.
Если сам Runtime не может открыть граф, показывается его ошибка загрузки; session inputs/outputs
до успешного открытия недоступны. Нет fallback на `.pth`, `strict=False` или старые sigmoid-выходы.

Официальный NPZ хранит названия как object arrays. Их чтение с pickle разрешено только
после совпадения SHA256 с проверенным официальным банком; произвольный NPZ не принимается.

CPUExecutionProvider проверен. CUDAExecutionProvider доступен в меню только как ручной выбор:
если отсутствует/не инициализируется, нода выдаёт ошибку. Даже при CUDA некоторые операции
Runtime может исполнять на CPU; полной GPU-поддержки и скорости этот пакет не обещает.

## Что реально проверено

- Реальный HF ONNX (207 881 414 байт), метаданные и банк скачаны по закреплённой ревизии.
- Получены inputs/outputs настоящей ONNX Runtime session, SHA256 сохранены в evidence.
- 8 тестов прошли: сбор списка/батча масок, JSON boxes, валидация источников/геометрии,
  фильтры/индексы, совпадение preprocessing с официальным runtime, неизвестный предикат,
  блокировка неподтверждённого контракта/sidecar, декодирование и настоящий CPU inference.
- Тестовый кадр 128×64 с двумя регионами: `region1 --below [0.3879]--> region0`,
  `region0 --above [0.3477]--> region1`; порог в этом техническом тесте **0.0**.
  При обычном пороге 0.4 эти два тестовых отношения не проходят — это ожидаемо.
- Все три входных пути (RA_REGIONS, masks, boxes_json) дали одинаковый результат.
- Первый вызов Predict после загрузки модели: около 0.49 с на CPU в одном тесте;
  это не устойчивый benchmark и не время SAM3.
- В тестовом процессе отсутствовали импорты torch / transformers / relsgg.
- Оба UI JSON проверены на соответствие типам и связям текущих SAM3/core нод и новых RA-нод.

**Не проверено:** полный новый workflow внутри запущенного ComfyUI, новый SAM3 inference,
качество отношений на вашем архитектурном рендере и CUDA. Пакет подготовлен отдельно:
действующая установка, её процессы, зависимости и файлы моделей не изменялись.

## Официальные источники и воспроизводимость

- [Hugging Face model card](https://huggingface.co/maelic/relsgg-vits16plus/blob/caf70d3af46478614209d86881a16e601d6bbf61/README.md)
- [ONNX sidecar](https://huggingface.co/maelic/relsgg-vits16plus/blob/caf70d3af46478614209d86881a16e601d6bbf61/relateanything.json)
- [Скачать ONNX](https://huggingface.co/maelic/relsgg-vits16plus/resolve/caf70d3af46478614209d86881a16e601d6bbf61/relateanything.onnx)
- [Скачать sidecar JSON](https://huggingface.co/maelic/relsgg-vits16plus/resolve/caf70d3af46478614209d86881a16e601d6bbf61/relateanything.json)
- [Скачать predicate_bank.npz](https://huggingface.co/maelic/relsgg-vits16plus/resolve/caf70d3af46478614209d86881a16e601d6bbf61/predicate_bank.npz)
- [Официальный экспорт](https://github.com/Maelic/RelateAnything/blob/6b9c07c12ff47d4be17fa558b183023dcf829e61/deploy/export_onnx.py)
- [Официальный runtime / make_feed](https://github.com/Maelic/RelateAnything/blob/6b9c07c12ff47d4be17fa558b183023dcf829e61/deploy/runtime.py)
- [Декодирование](https://github.com/Maelic/RelateAnything/blob/6b9c07c12ff47d4be17fa558b183023dcf829e61/deploy/postprocess.py)
- [Score contract](https://github.com/Maelic/RelateAnything/blob/6b9c07c12ff47d4be17fa558b183023dcf829e61/relsgg/scoring.py)

Модель под лицензией DINOv3, согласно model card. Пакет не переиздаёт веса.
Pre/postprocessing согласован с официальным Apache-2.0 проектом RelateAnything;
это сторонняя ComfyUI-обёртка, а не официальный релиз авторов модели.

## Повторение тестов

`tests/test_ra_v06.py` проверяет реальный ONNX inference и обработку входов.
Требуются уже имеющиеся `numpy`, `opencv-python` (либо headless), `onnxruntime` и
`onnx` для малого диагностического тестового графа. `onnx` не является runtime-зависимостью ноды.
Запускайте тесты в подходящем тестовом окружении, не обновляйте основной ComfyUI ради тестов.

```powershell
$env:RA_ONNX_MODEL_DIR = 'F:\ComfyUI\models\relateanything\relsgg-vits16plus'
python tests/test_ra_v06.py
```

Для дополнительного сравнения preprocessing с официальным исходным кодом задайте
`RA_OFFICIAL_REPO` — путь к checkout Maelic/RelateAnything на commit
`6b9c07c12ff47d4be17fa558b183023dcf829e61`. Без него только этот тест будет пропущен.
Новые отчёты пишутся в `test-results/`, опубликованные `evidence/` не перезаписываются.
В evidence сохранён первоначальный результат: все 8 тестов, включая сравнение с upstream, прошли.
