# ComfyUI Fal API

Кастомные ноды ComfyUI для [fal.ai](https://fal.ai) — 44 узла поверх облачных моделей:
видео (Seedance, LTX-2.3, MiniMax H3, Kling, Wan), картинки (FLUX, GPT Image, Qwen,
Seedream, Ideogram, Nano Banana), апскейл и реставрация, сегментация, обучение и
конвертация LoRA. Локальная видеокарта не нужна — всё считается на стороне fal.

Ключевое отличие от обёрток «один эндпоинт — одна нода»: узлы **сами выбирают режим**
по подключённым входам (нет картинки — text-to-video, есть — image-to-video, подключено
видео — reference), проверяют лимиты провайдера **до отправки задания** и показывают
**оценку стоимости** ещё до запуска — в консоли и прямо в заголовке ноды.

## Установка

Через **ComfyUI Manager**: `Custom Nodes Manager` → `Install via Git URL` →
`https://github.com/Panicontrol/ComfyUI_Fal_API`

Вручную:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Panicontrol/ComfyUI_Fal_API
pip install -r ComfyUI_Fal_API/requirements.txt
```

После установки перезапустите ComfyUI.

## Ключ API

Скопируйте `config.ini.example` в `config.ini` и впишите ключ из
[fal.ai/dashboard/keys](https://fal.ai/dashboard/keys):

```ini
[API]
FAL_KEY = ваш-ключ
```

Либо задайте переменную окружения `FAL_KEY`. Файл `config.ini` добавлен в `.gitignore`
и в репозиторий не попадает. На Windows он читается в нескольких кодировках, так что
кириллица в пути не ломает запуск.

## Ноды

### Видео

| Нода | Эндпоинт fal | Что делает |
|---|---|---|
| Seedance 2.0 Text / Image / Reference-to-Video | `bytedance/seedance-2.0/*` (+`fast`, +`mini`) | До 15 с со звуком, до 4K; `fast_mode` и `mini_mode` — дешёвые уровни (до 720p, mini ≈ $0.15/с); reference принимает 9 картинок + 3 видео + 3 аудио |
| Seedance 2.5 Video | `bytedance/seedance-2.5/{text,image}-to-video` (+`flash`) | До 30 с одним дублем, до 4K, первый и последний кадр; `tier` flash — быстрый уровень до 720p; `draft` — черновик в 480p с выходом `draft_id` |
| Seedance 2.5 Reference-to-Video | `bytedance/seedance-2.5/reference-to-video` (+`flash`) | До 30 картинок + 10 видео + 10 аудио, всего 50 файлов (flash: 10 + 5 + 5); `task`: reference, editing (правка видео), extension (продолжение, одно видео до 120 с), auto — см. [лимиты на вход и режим task](#seedance-лимиты-на-вход-и-режим-task) |
| Seedance 2.5 Draft Complete | `bytedance/seedance-2.5/draft/complete` | Досчёт черновика по `draft_id` в 1080p или 4K — тот же дубль; 14 дней, оплачивается сверх черновика |
| Seedance 1.5 Pro Text / Image-to-Video | `fal-ai/bytedance/seedance/v1.5/pro/*` | Дешевле, 4–12 с, до 1080p |
| LTX-2.3 Video LoRA | `fal-ai/ltx-2.3-22b[/distilled]/{text,image,reference-video}-to-video[/lora]` | t2v / i2v / reference, звук, 3 слота LoRA, LoRA движения камеры |
| LTX-2.3 Extend Video | `fal-ai/ltx-2.3-quality/extend-video[/lora]` | Продолжает ролик вперёд или назад |
| LTX-2.3 Reframe | `fal-ai/ltx-2.3/reframe` | Меняет соотношение сторон, дорисовывая кадр |
| MiniMax H3 Video LoRA | `minimax/h3/{text,image}-to-video[/lora]` | До 4K, нативное стерео, открытые веса → свои LoRA |
| MiniMax H3 Reference-to-Video | `minimax/h3/reference-to-video[/lora]` | 9 картинок + 3 видео + 3 аудио |
| MiniMax H3 Max Video | `minimax/h3-max[-turbo]/{text,image}-to-video` | Пост-тренированный fal вариант, быстрее реального времени |
| MiniMax H3 Max Reference-to-Video | `minimax/h3-max/reference-to-video` | То же с мультимодальными референсами |
| MiniMax H3 Max Relight | `minimax/h3-max/relight` | Переосвещение готового видео по картинке сферы со светом; движение, камера и звук сохраняются |
| MiniMax H3 Max Reframe | `minimax/h3-max/reframe-video` | Смена соотношения сторон с дорисовкой краёв, звук и таймлайн сохраняются; 0.25–300 с, $0.05–0.08/с |
| MiniMax H3 Max Recast | `minimax/h3-max/recast` | Замена людей в видео на людей с фото (до 4), движение, камера и звук сохраняются; 5–30 с, $0.30–0.45/с |
| MiniMax H3 Max Extend | `minimax/h3-max{,-turbo}/extend-video` | Продолжение ролика по промпту на 0.71–15 с, с аудио-референсами; весь ролик или только новое. Max $0.05–0.32/с, Turbo вдвое дешевле |
| MiniMax H3 Max Insert Scene | `minimax/h3-max/insert-video` | Вставка новой сцены 5–13 с между `start_time` и `resume_time` исходника, по промпту и референсам. $0.05–0.06/с |
| MiniMax H3 Max 3D-to-Video | `minimax/h3-max/3d-to-video` | Рендер из Blender / превиз → реалистичное видео: камера и тайминг из ролика, вид из референсов. $0.05–0.16/с |
| MiniMax H3 Max Styles | `minimax/h3-max/styles/*` | Видео в готовом стиле: hand-drawn, retro-toon-70s, low-poly, 16bit-pixel, vhs. 768p, $0.08/с |
| Kling Video | `fal-ai/kling-video/<версия>/<тир>/*-to-video` | 2.6 Pro и 2.1 master / pro / standard |
| Wan 2.2 LoRA Video | `fal-ai/wan/v2.2-a14b/*/lora` | t2v / i2v с пользовательскими LoRA |

### Картинки

| Нода | Эндпоинт fal | Что делает |
|---|---|---|
| GPT Image 2 / GPT Image 2 Edit | `openai/gpt-image-2[/edit]` | До 16 референсов, инпейнт по маске |
| GPT Image 2.5 Flare / Sunburst | `openai/gpt-image-2.5/{flare,sunburst}/*` | Качество до `max`, прозрачный фон, отдельный выход `alpha` |
| FLUX.1 LoRA / FLUX.2 LoRA | `fal-ai/flux-lora[/image-to-image]`, `fal-ai/flux-2/lora[/edit]` | Генерация и правка с тремя слотами LoRA |
| Qwen Image Max | `fal-ai/qwen-image-max/{text-to-image,edit}` | t2i и edit, промпт на русском и китайском |
| Qwen-Image Edit LoRA | `fal-ai/qwen-image-edit-plus-lora` | Правка по нескольким картинкам с LoRA |
| Seedream 5.0 Pro | `bytedance/seedream/v5/pro/{text-to-image,edit}` | Генерация и редактирование |
| Ideogram V4 / V3 Edit | `ideogram/v4`, `fal-ai/ideogram/v3/edit` | Сильная типографика, инпейнт |
| Nano Banana 2 / 2.1 / Pro Edit | `fal-ai/nano-banana-{2,pro}/edit`, `google/nano-banana-2.1/edit` | Правка по референсам (до 14); 2.1 считается по токенам, ~$0.04 за 1K |
| Bria Fibo 2.0 Edit | `bria/fibo-2/edit` | Правка по инструкции, до 4 картинок `<image_1>…`, правка по маске; возвращает VGL (JSON-промпт) для доработки. $0.04 |
| FLUX.3 Klein Edit | `blackforestlabs/flux-3/klein/edit-image` | Правка и композиция по 1–10 референсам, раскладка боксами. Плановая цена ~$0.009 за 1K |

### Апскейл и реставрация

| Нода | Эндпоинт fal | Что делает |
|---|---|---|
| Topaz Video Precision | `topaz/upscale/video/precision` | 21 модель без дорисовки: Proteus, Iris, Dione, Artemis, Gaia, Rhea, Theia |
| Topaz Video Generative / Starlight | `topaz/upscale/video/generative` | Диффузионный проход Starlight, Fast 2 вдвое дешевле |
| Topaz Video Creative / Astra 2 | `topaz/upscale/video/creative` | Дорисовка деталей по промпту, самый дорогой тир |
| ByteDance Video Upscale | `fal-ai/bytedance-upscaler/upscale/video` | До 8K, вывод в 10 и 12 бит, пресеты aigc / old_film / ugc |
| Topaz Video Upscale (старый эндпоинт) | `fal-ai/topaz/upscale/video` | Единый эндпоинт прошлого поколения, оставлен для совместимости |
| FLUX Video Upscale | `blackforestlabs/flux-video-upscale` | FLUX 3, режимы precise / creative, вход до 20 с |
| FLUX Vision Upscaler | `fal-ai/flux-vision-upscaler` | До 4x по подписи от VLM, отдаёт подпись выходом |
| LucidFlux Restore / Upscale | `fal-ai/lucidflux` | Реставрация битых и мыльных кадров |

### Интерполяция кадров и HDR

| Нода | Эндпоинт | Что делает |
|---|---|---|
| Topaz Frame Interpolation | `topaz/interpolate/video` | Apollo / Chronos / Aion: смена частоты до 120 fps и слоумо до 8x |
| Frame Interpolation FILM / RIFE | `fal-ai/film/video`, `fal-ai/rife/video` | Дешёвая интерполяция для черновиков и анимации, $0.0013 за секунду вычислений |
| Topaz SDR to HDR | `topaz/sdr-to-hdr/video` | Hyperion 2.5: SDR → HDR10, 10-битный ProRes 422 HQ или H.265 |

### Сегментация, LoRA, утилиты

| Нода | Эндпоинт fal | Что делает |
|---|---|---|
| SAM 2 Image / Video Segment | `fal-ai/sam2/{image,video}` | Маски по точкам и боксам, трекинг по ролику |
| EVF-SAM Text Segment | `fal-ai/evf-sam` | Маска по текстовому описанию объекта |
| FLUX LoRA Trainer | `fal-ai/flux-lora-fast-training`, `fal-ai/flux-lora-portrait-trainer` | Обучение LoRA прямо из IMAGE-батча графа |
| MiniMax H3 LoRA Trainer | `minimax/h3/{t2v,i2v,flf2v,ref2va}/trainer` | LoRA для H3 из папки клипов с капшнами; цена max(100, шагов) × $0.005–0.015 |
| LoRA Convert | локально | fp16 и понижение ранга через SVD, чтобы влезть в лимит fal в 1 ГБ |
| Save Video as PNG Sequence (16-bit) | локально | Видеофайл по `local_path` → PNG-секвенция; 10/12-битный источник сохраняется в 16-битный PNG без потерь |

## Seedance: лимиты на вход и режим task

Лимиты взяты из схем fal на ноябрь 2026. В колонке «Нода» — что нода делает сама до
запуска. Всё остальное проверяет fal, и отказ приходит уже после загрузки файлов.

### Seedance 2.5

| Вход | Лимит fal | Нода |
|---|---|---|
| Картинки-референсы: `image_1`…`image_4`, `image_urls` | До 30 штук, у flash до 10. JPG, PNG, WebP, BMP, TIFF, GIF, HEIC, HEIF, до 30 МБ каждая | Каждый кадр батча — отдельная картинка, грузится как PNG. Лишние сверх лимита отбрасываются |
| Видео-референсы: `video_1`…`video_3`, `video_refs` | До 10 штук, у flash до 5. MP4 или MOV. Каждое 1.8–30.2 с и до 200 МБ, в сумме до 30.2 с. Сторона кадра 300–6000 px, отношение сторон от 0.4 до 2.5, 24–60 fps | Ролик грузится как есть, без перекодировки. Размер кадра, отношение сторон, fps и вес файла проверяются до любой загрузки: не подходит — ошибка с названием причины (23.976 fps проходит как 24). Короче 1.8 с — тоже ошибка. Длиннее 30.2 с или сумма больше 30.2 с — только предупреждение в консоли. Ролики, заданные ссылкой в `video_refs`, проверить нечем — их проверит fal |
| Аудио-референсы: `audio_1`…`audio_3`, `audio_refs` | До 10 штук, у flash до 5. MP3 или WAV. Каждое 1.8–30.2 с и до 15 МБ, в сумме до 30.2 с | Грузится как WAV 16 бит. Длительность нода не проверяет |
| Всего файлов | До 50. Для flash общий лимит fal не называет, по типам выходит 20 | Больше — ошибка до запуска. Нужен хотя бы один референс |
| Первый и последний кадр: `image`, `end_image` у Seedance 2.5 Video | Те же форматы, до 30 МБ. С картинкой отношение сторон всегда берётся из неё | Берётся первый кадр батча; `aspect_ratio` не отправляется |
| `prompt` | Обязателен только без картинки (text-to-video) | — |
| `duration` | 4–30 с или auto | `duration_override` округляется и зажимается в 4–30 |
| `resolution` | 480p, 720p, 1080p, 4k. У flash только 480p и 720p | flash с 1080p или 4k — ошибка до запуска |
| `draft` | Только standard. Черновик всегда 480p, `draft_id` действует 14 дней и только с того же аккаунта fal | flash с `draft` — ошибка до запуска |

На что обратить внимание: видео-референс со стороной кадра больше 6000 px, с отношением
сторон больше 2.5 (например панорама 7204×2304) или на 12 и 16 fps не пройдёт — нода
остановит граф до загрузки. Такой ролик нужно сначала уменьшить, обрезать или поднять
до 24 fps.

### Режим `task` у Seedance 2.5 Reference

| `task` | Что делает | Что важно |
|---|---|---|
| `reference` (по умолчанию) | Медиа — ориентир: персонаж, место, стиль, движение, звук | `duration` и `aspect_ratio` работают как обычно |
| `editing` | Правит ролик из видео-входа по промпту | Нужно видео, без него нода выдаст ошибку. fal сам ставит `duration` и `aspect_ratio` в auto: виджеты `duration`, `duration_override` и `aspect_ratio` не действуют, длина и отношение сторон берутся из ролика. Ролик — до 30.2 с |
| `extension` | Продолжает ролик | Нужно видео. Если оно одно, принимается до 120 с, но в модель уходят только последние 30.2 с. `aspect_ratio` fal ставит в auto, `duration` действует |
| `auto` | Тип задачи не задаётся | Модель сама решает по промпту и референсам, править, продолжать или генерировать |

Общее для всех значений `task`:

- **Ссылки в промпте.** На файлы ссылаются как `@Image1`, `@Video1`, `@Audio1`. Нумерация
  идёт по порядку входов: сначала `image_1`…`image_4` (кадры батча подряд), затем строки
  `image_urls`; у видео и аудио так же.
- **Цена с видео на входе.** Оплачиваются секунды входного видео вместе с выходными, зато
  весь запрос идёт по сниженному тарифу, примерно ×0.6. Правка 10-секундного ролика в 720p —
  это 20 оплаченных секунд по $0.28, около $5.5. Оценка в заголовке ноды длину входа не
  знает; точнее считает строка в консоли.
- **Чего fal не пишет.** Входит ли исходный ролик в результат `extension` и как
  оплачивается вход длиннее 30.2 с, в документации fal не сказано, и на реальных вызовах
  это не проверялось. Проверьте на коротком тесте в 480p.

### Seedance 2.0 и 1.5 Pro

| Вход | Лимит fal | Нода |
|---|---|---|
| Картинки-референсы 2.0 | До 9 штук, до 30 МБ каждая | Лишние отбрасываются |
| Видео-референсы 2.0 | До 3 штук, в сумме 2–15 с и меньше 50 МБ. Кадр — примерно от 480p (640×640) до 720p (834×1112) | Ролик короче 2 с или сумма больше 15 с — ошибка до запуска. Размер кадра и вес файла нода не проверяет |
| Аудио-референсы 2.0 | До 3 штук, в сумме до 15 с, до 15 МБ каждое | Длительность нода не проверяет |
| Всего файлов 2.0 | До 12. Нужна хотя бы одна картинка или видео: одного аудио мало | Не проверяет |
| `prompt` у 2.0 Reference | Обязателен | — |
| `duration` | 2.0: 4–15 с или auto. 1.5 Pro: 4–12 с | `duration_override` зажимается в эти пределы |
| `resolution` | 2.0: 480p, 720p, 1080p, 4k; у fast и mini только 480p и 720p. 1.5 Pro: 480p, 720p, 1080p | fast или mini с 1080p / 4k — нода понижает до 720p и пишет об этом в консоль |
| Первый и последний кадр | 2.0: до 30 МБ. Для 1.5 Pro fal лимитов на файлы не называет | Берётся первый кадр батча |

## Обучение LoRA для MiniMax H3

Нода **MiniMax H3 LoRA Trainer** принимает папку (или готовый `.zip`, или URL архива).
Из папки в архив попадают только файлы, которые нужны выбранному режиму:

```
dataset/
  clip01.mp4              # клипы: .mp4 .mov .avi .mkv, лучше от 10 штук
  clip01.txt              # капшн, необязательно; trigger_phrase добавится в начало
  clip01.first_frame.png  # i2v, flf2v: свой первый кадр (иначе берётся из клипа)
  clip01.last_frame.png   # flf2v: свой последний кадр
  clip01.ref_1.png        # ref2va: референсы ref_1..ref_4 — картинка, видео или звук
```

| Режим | Генерация с LoRA | $ за шаг |
|---|---|---|
| `t2v` | MiniMax H3 Video LoRA без картинки | 0.005 |
| `i2v` | MiniMax H3 Video LoRA с `image` | 0.01 |
| `flf2v` | MiniMax H3 Video LoRA с `image` + `end_image` | 0.01 |
| `ref2va` | MiniMax H3 Reference-to-Video | 0.015 |

Оплачивается max(100, `number_of_steps`) шагов: 2000 шагов по умолчанию в t2v стоят $10.
`number_of_frames` должно давать остаток 5 при делении на 17 (22, 39, 56, 73, 90, 107, 124),
другое значение нода округлит. Картинки как обучающие файлы fal не принимает. Если клип
разрезается на сцены (`split_input_into_scenes`), его доп. файлы не используются. Готовая
LoRA скачивается в `models/loras`, существующий файл с тем же именем не перезаписывается.

## Как это устроено

**Выходы.** Видео-ноды отдают `video` (нативный тип VIDEO), `video_url` и `local_path` —
файл уже скачан в `output`. Картиночные отдают `images` и `image_urls`. У части нод есть
дополнительные выходы: `seed` и `draft_id` у Seedance 2.5, `expanded_prompt` у H3 Max, `caption` у
FLUX Vision Upscaler, `alpha` у GPT Image 2.5.

**Оценка стоимости** считается до запуска по тарифам fal и выводится зелёной строкой в
заголовке ноды (`web/seedance_cost.js`) и в консоль. Для токенных моделей используется
формула провайдера, для посекундных — тир разрешения.

**Устойчивость.** Повторы при временных сбоях шлюза fal (502/503/504) и при обрыве
скачивания с CDN; если результат сгенерирован, но не скачался, ссылка попадает в текст
ошибки, чтобы забрать файл вручную. Битые таймкоды контейнера (типичная беда рендеров из
Unreal и NLE) чинятся перекодировкой через PyAV с запасным вариантом на ffmpeg.

**Проверки до оплаты.** Лимиты длительности и количества референсов, доступность LoRA по
ссылке (закрытые репозитории Hugging Face отдают серверам fal 403 — нода скажет об этом
заранее), размер LoRA против лимита в 1 ГБ, кратность размеров.

**Прогресс** идёт в стандартный прогресс-бар ComfyUI, отмена по Ctrl+C снимает задание и
на стороне fal.

## Требования

Python 3.10+, ComfyUI любой свежей версии, пакеты из `requirements.txt`
(`fal-client`, `requests`, `pillow`). Для перекодировки битых контейнеров желателен
`av` (PyAV) или `ffmpeg` в `PATH` — оба обычно уже стоят вместе с ComfyUI.

## Лицензия

MIT — см. [LICENSE](LICENSE).
