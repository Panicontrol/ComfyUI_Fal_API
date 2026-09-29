# ComfyUI Fal API

Кастомные ноды ComfyUI для [fal.ai](https://fal.ai) — 35 узлов поверх облачных моделей:
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
| Seedance 2.0 Text / Image / Reference-to-Video | `bytedance/seedance-2.0/*` (+`fast`) | До 15 с со звуком; reference принимает 9 картинок + 3 видео + 3 аудио |
| Seedance 2.5 Video | `bytedance/seedance-2.5/{text,image}-to-video` | До 30 с одним дублем, первый и последний кадр |
| Seedance 2.5 Reference-to-Video | `bytedance/seedance-2.5/reference-to-video` | До 30 картинок + 10 видео + 10 аудио, всего 50 файлов |
| Seedance 1.5 Pro Text / Image-to-Video | `fal-ai/bytedance/seedance/v1.5/pro/*` | Дешевле, 4–12 с |
| LTX-2.3 Video LoRA | `fal-ai/ltx-2.3-22b[/distilled]/{text,image,reference-video}-to-video[/lora]` | t2v / i2v / reference, звук, 3 слота LoRA, LoRA движения камеры |
| LTX-2.3 Extend Video | `fal-ai/ltx-2.3-quality/extend-video[/lora]` | Продолжает ролик вперёд или назад |
| LTX-2.3 Reframe | `fal-ai/ltx-2.3/reframe` | Меняет соотношение сторон, дорисовывая кадр |
| MiniMax H3 Video LoRA | `minimax/h3/{text,image}-to-video[/lora]` | До 4K, нативное стерео, открытые веса → свои LoRA |
| MiniMax H3 Reference-to-Video | `minimax/h3/reference-to-video[/lora]` | 9 картинок + 3 видео + 3 аудио |
| MiniMax H3 Max Video | `minimax/h3-max[-turbo]/{text,image}-to-video` | Пост-тренированный fal вариант, быстрее реального времени |
| MiniMax H3 Max Reference-to-Video | `minimax/h3-max/reference-to-video` | То же с мультимодальными референсами |
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
| Nano Banana 2 / Pro Edit | `fal-ai/nano-banana-{2,pro}/edit` | Правка по референсам |

### Апскейл и реставрация

| Нода | Эндпоинт fal | Что делает |
|---|---|---|
| Topaz Video Upscale | `fal-ai/topaz/upscale/video` | 19 моделей, до 8x, интерполяция кадров |
| FLUX Video Upscale | `blackforestlabs/flux-video-upscale` | FLUX 3, режимы precise / creative, вход до 20 с |
| FLUX Vision Upscaler | `fal-ai/flux-vision-upscaler` | До 4x по подписи от VLM, отдаёт подпись выходом |
| LucidFlux Restore / Upscale | `fal-ai/lucidflux` | Реставрация битых и мыльных кадров |

### Сегментация, LoRA, утилиты

| Нода | Эндпоинт fal | Что делает |
|---|---|---|
| SAM 2 Image / Video Segment | `fal-ai/sam2/{image,video}` | Маски по точкам и боксам, трекинг по ролику |
| EVF-SAM Text Segment | `fal-ai/evf-sam` | Маска по текстовому описанию объекта |
| FLUX LoRA Trainer | `fal-ai/flux-lora-fast-training`, `fal-ai/flux-lora-portrait-trainer` | Обучение LoRA прямо из IMAGE-батча графа |
| LoRA Convert | локально | fp16 и понижение ранга через SVD, чтобы влезть в лимит fal в 1 ГБ |

## Как это устроено

**Выходы.** Видео-ноды отдают `video` (нативный тип VIDEO), `video_url` и `local_path` —
файл уже скачан в `output`. Картиночные отдают `images` и `image_urls`. У части нод есть
дополнительные выходы: `seed` у Seedance 2.5, `expanded_prompt` у H3 Max, `caption` у
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
