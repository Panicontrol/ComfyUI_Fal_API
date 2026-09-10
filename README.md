# ComfyUI-fal-Seedance

Кастомные ноды ComfyUI для генерации видео **Seedance (ByteDance)** через **fal.ai** — без локальной видеокарты, всё считается в облаке fal.

## Ноды

| Нода | Эндпоинт fal | Что делает |
|---|---|---|
| Seedance 2.0 Text-to-Video (fal) | `bytedance/seedance-2.0/text-to-video` (+fast) | Видео из текста, до 15 сек, звук |
| Seedance 2.0 Image-to-Video (fal) | `bytedance/seedance-2.0/image-to-video` (+fast) | Оживление картинки, первый+последний кадр, до 1080p |
| Seedance 2.0 Reference-to-Video (fal) | `bytedance/seedance-2.0/reference-to-video` | До 9 картинок + до 3 видео + до 3 аудио как референсы |
| Seedance 1.5 Pro Text-to-Video (fal) | `fal-ai/bytedance/seedance/v1.5/pro/text-to-video` | Дешевле, 4–12 сек |
| Seedance 1.5 Pro Image-to-Video (fal) | `fal-ai/bytedance/seedance/v1.5/pro/image-to-video` | Дешевле, первый+последний кадр |
| Seedance 2.5 Video (fal) | `bytedance/seedance-2.5/{text,image}-to-video` | До 30 с одним дублем, 480p/720p, звук, первый+последний кадр |
| Seedance 2.5 Reference-to-Video (fal) | `bytedance/seedance-2.5/reference-to-video` | До 30 картинок + 10 видео + 10 аудио, всего до 50 файлов |
| LTX-2.3 Video LoRA (fal) | `fal-ai/ltx-2.3-22b/{text,image,reference-video}-to-video[/lora]` (+`/distilled`) | Видео со звуком, 3 слота LoRA, LoRA камеры, режим по подключённым входам |
| LTX-2.3 Extend Video (fal) | `fal-ai/ltx-2.3-quality/extend-video[/lora]` | Продолжает ролик вперёд или назад, с LoRA |
| LTX-2.3 Reframe (fal) | `fal-ai/ltx-2.3/reframe` | Меняет соотношение сторон, дорисовывая кадр (до 60 с) |
| MiniMax H3 Video LoRA (fal) | `minimax/h3/{text,image}-to-video[/lora]` | Hailuo 3.0: до 4K, 5–15 с, нативное стерео-аудио, LoRA |
| MiniMax H3 Reference-to-Video (fal) | `minimax/h3/reference-to-video[/lora]` | До 9 картинок + 3 видео + 3 аудио как референсы |
| MiniMax H3 Max Video (fal) | `minimax/h3-max[-turbo]/{text,image}-to-video` | Пост-тренированный fal вариант H3: точнее по промпту, быстрее реального времени, 480P/768P |
| MiniMax H3 Max Reference-to-Video (fal) | `minimax/h3-max/reference-to-video` | Те же мультимодальные референсы на H3 Max |
| GPT Image 2.5 Flare / Sunburst (fal) | `openai/gpt-image-2.5/{flare,sunburst}/{text-to-image,edit}` | Качество до max, прозрачный фон, до 16 референсов, выход alpha |

Каждая нода возвращает три выхода: `video` (нативный тип VIDEO — подключается к Save Video / Preview Video), `video_url` (ссылка на mp4 на серверах fal) и `local_path` (файл уже скачан в папку output ComfyUI).

## Установка

1. Скопируйте папку `ComfyUI-fal-Seedance` в `ComfyUI/custom_nodes/`
2. Установите зависимости в питон-окружение ComfyUI:

   ```
   pip install -r requirements.txt
   ```

   Для portable-версии ComfyUI на Windows:

   ```
   python_embeded\python.exe -m pip install fal-client requests pillow
   ```

3. Получите API-ключ: зарегистрируйтесь на [fal.ai](https://fal.ai), откройте [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys) → **Add key** → скопируйте ключ. Не забудьте привязать карту / пополнить баланс в разделе Billing — модели платные.

4. Переименуйте `config.ini.example` в `config.ini` и вставьте ключ:

   ```ini
   [API]
   FAL_KEY = ваш-ключ
   ```

   (Либо задайте переменную окружения `FAL_KEY`.)

5. Перезапустите ComfyUI. Ноды появятся в категории **fal/Seedance**.

## Готовые workflow

В папке `workflows/` — три файла, перетащите любой в окно ComfyUI:

- `seedance2_image_to_video.json` — картинка → видео
- `seedance2_text_to_video.json` — текст → видео
- `seedance2_reference_to_video.json` — референсы (картинки/видео/аудио) → видео

## Подсказки

- В Reference-to-Video ссылайтесь на референсы прямо в промпте: `@Image1`, `@Image2`… Порядок: сначала кадры из IMAGE-входа (батч), потом строки из `image_urls`.
- `video_refs` / `audio_refs` — по одному пути или URL на строку; локальные файлы автоматически загружаются в хранилище fal.
- `fast_mode` — быстрее и дешевле, чуть ниже качество.
- `duration: auto` — модель сама выбирает длину (4–15 сек). Тариф посекундный, для 720p со звуком ~$0.30/сек (fast ~$0.24/сек), так что для тестов выгоднее ставить 4–5 сек и 480p.
- Seedance 1.5 Pro заметно дешевле (~$0.26 за 5-секундный ролик 720p со звуком) — удобно для черновиков.
