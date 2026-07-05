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
