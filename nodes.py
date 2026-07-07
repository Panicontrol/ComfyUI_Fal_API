"""
ComfyUI-fal-Seedance
Кастомные ноды для генерации видео Seedance (ByteDance) через fal.ai.

Поддерживаются:
  - Seedance 2.0: Text-to-Video, Image-to-Video (+end frame), Reference-to-Video
    (референсы: до 9 картинок, до 3 видео, до 3 аудио), fast-варианты
  - Seedance 1.5 Pro: Text-to-Video, Image-to-Video

API-ключ: переменная окружения FAL_KEY или файл config.ini рядом с этим файлом:
    [API]
    FAL_KEY = ваш-ключ
"""

import os
import io
import configparser
import tempfile

import numpy as np
import requests

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import fal_client
except ImportError:
    fal_client = None

# Нативный тип VIDEO появился в ComfyUI в 2025 году; на старых сборках
# нода вернёт только URL и локальный путь.
try:
    from comfy_api.input_impl import VideoFromFile
except ImportError:
    try:
        from comfy_api.input_impl.video_types import VideoFromFile
    except ImportError:
        VideoFromFile = None

try:
    import folder_paths
except ImportError:
    folder_paths = None


# ---------------------------------------------------------------------------
# ключ и утилиты
# ---------------------------------------------------------------------------

def _ensure_api_key():
    if os.environ.get("FAL_KEY"):
        return
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
    if os.path.isfile(cfg_path):
        cfg = configparser.ConfigParser()
        # На Windows configparser по умолчанию читает в cp1252 и падает
        # на кириллице — пробуем несколько кодировок.
        for enc in ("utf-8-sig", "utf-8", "cp1251", None):
            try:
                cfg.read(cfg_path, encoding=enc)
                break
            except (UnicodeDecodeError, configparser.Error):
                cfg = configparser.ConfigParser()
                continue
        key = cfg.get("API", "FAL_KEY", fallback="").strip()
        if key and "ваш" not in key and "<" not in key:
            os.environ["FAL_KEY"] = key
            return
    raise RuntimeError(
        "FAL_KEY не найден. Укажите ключ в config.ini (секция [API]) "
        "или в переменной окружения FAL_KEY. Ключ создаётся на "
        "https://fal.ai/dashboard/keys"
    )


def _require_deps():
    if fal_client is None:
        raise RuntimeError(
            "Модуль fal_client не установлен. Выполните: pip install fal-client"
        )
    _ensure_api_key()


def _tensor_batch_to_pil(image_tensor):
    """IMAGE-тензор ComfyUI (B,H,W,C float 0..1) -> список PIL.Image."""
    if Image is None:
        raise RuntimeError("Pillow не установлен: pip install pillow")
    images = []
    arr = image_tensor.cpu().numpy() if hasattr(image_tensor, "cpu") else np.asarray(image_tensor)
    if arr.ndim == 3:
        arr = arr[None, ...]
    for frame in arr:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        images.append(Image.fromarray(frame))
    return images


def _upload_pil(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(buf.read())
        tmp = f.name
    try:
        return fal_client.upload_file(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _upload_image_input(image_tensor, limit=1):
    """Загружает первые `limit` кадров IMAGE-входа в fal storage, возвращает URL'ы."""
    urls = []
    for img in _tensor_batch_to_pil(image_tensor)[:limit]:
        urls.append(_upload_pil(img))
    return urls


def _reencode_video(src_path):
    """Перекодирует видео в чистый H.264 mp4 с корректными тайм-метками.

    Ролики из Unreal/NLE часто имеют битые метаданные длительности:
    локальный плеер их играет, а парсер fal видит «1 кадр» (0.04 с)
    и отклоняет запрос. Возвращает (путь_tmp, длительность_с)."""
    import av
    from fractions import Fraction

    with av.open(src_path) as inp:
        vstream = inp.streams.video[0]
        rate = vstream.average_rate or vstream.guessed_rate or Fraction(24, 1)
        cc = vstream.codec_context
        width = cc.width - (cc.width % 2)
        height = cc.height - (cc.height % 2)

        fd, tmp = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        n = 0
        with av.open(tmp, "w") as out:
            out_v = out.add_stream("h264", rate=rate)
            out_v.width = width
            out_v.height = height
            out_v.pix_fmt = "yuv420p"
            out_v.options = {"crf": "18", "preset": "fast"}
            for frame in inp.decode(vstream):
                frame = frame.reformat(width=width, height=height,
                                       format="yuv420p")
                frame.pts = None
                for pkt in out_v.encode(frame):
                    out.mux(pkt)
                n += 1
            for pkt in out_v.encode():
                out.mux(pkt)
    return tmp, n / float(rate)


def _upload_video_input(video, label="видео"):
    """VIDEO-вход ComfyUI -> URL в fal storage.

    Видео всегда перекодируется перед загрузкой (лечит битые тайм-метки).
    Возвращает (url, длительность_с | None)."""
    if isinstance(video, str) and video.lower().startswith(("http://", "https://")):
        return video, None

    # получаем локальный файл-источник
    src, tmp_src = None, None
    if isinstance(video, str):
        if not os.path.isfile(video):
            raise RuntimeError(f"Видео не найдено: {video}")
        src = video
    else:
        if hasattr(video, "get_stream_source"):
            try:
                s = video.get_stream_source()
                if isinstance(s, str) and os.path.isfile(s):
                    src = s
            except Exception:
                pass
        if src is None:
            fd, tmp_src = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            video.save_to(tmp_src)
            src = tmp_src

    tmp_enc = None
    try:
        try:
            tmp_enc, dur = _reencode_video(src)
            upload_path = tmp_enc
        except ImportError:
            upload_path, dur = src, None  # нет PyAV — грузим как есть
        if dur is not None:
            print(f"[fal] {label}: {dur:.2f} с после перекодировки")
            if dur < 2.0:
                raise RuntimeError(
                    f"Референс-{label} слишком короткое: {dur:.2f} с "
                    f"(fal требует 2–15 с суммарно). Похоже, в видео-вход "
                    f"попал одиночный кадр — подключи полноценный ролик."
                )
        return fal_client.upload_file(upload_path), dur
    finally:
        for p in (tmp_src, tmp_enc):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


def _upload_audio_input(audio):
    """AUDIO-вход ComfyUI ({'waveform': (B,C,T), 'sample_rate': int}) -> URL wav в fal storage."""
    import wave
    wf = audio["waveform"]
    sr = int(audio["sample_rate"])
    arr = wf[0].cpu().numpy() if hasattr(wf, "cpu") else np.asarray(wf)[0]  # (C, T)
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype("<i2")
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with wave.open(tmp, "wb") as w:
            w.setnchannels(pcm.shape[0])
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.T.tobytes())
        return fal_client.upload_file(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _resolve_media_list(text, limit):
    """Многострочное поле: каждая строка — URL или локальный путь.
    Локальные файлы загружаются в fal storage."""
    urls = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith(("http://", "https://", "data:")):
            urls.append(line)
        elif os.path.isfile(line):
            urls.append(fal_client.upload_file(line))
        else:
            raise RuntimeError(f"Файл не найден и это не URL: {line}")
        if len(urls) >= limit:
            break
    return urls


def _run_request(endpoint, arguments, est_seconds=120):
    """Отправляет запрос на fal и ждёт результат, показывая прогресс в ComfyUI.

    Прогресс-бар ноды: реальный процент из логов fal (если модель его пишет),
    иначе — оценка по прошедшему времени относительно est_seconds.
    Нажатие Cancel в ComfyUI отменяет задачу и на стороне fal."""
    import re
    import time

    def _fal_error_text(exc):
        s = str(exc)
        msgs = re.findall(r"['\"]msg['\"]:\s*['\"]([^'\"]+)['\"]", s)
        return "; ".join(msgs) if msgs else s

    try:
        from comfy.utils import ProgressBar
        pbar = ProgressBar(100)
    except Exception:
        pbar = None
    try:
        import comfy.model_management as mm
    except ImportError:
        mm = None

    def set_progress(value):
        if pbar is not None:
            pbar.update_absolute(int(min(99, max(0, value))), 100)

    try:
        handler = fal_client.submit(endpoint, arguments=arguments)
    except Exception as e:
        raise RuntimeError(f"fal отклонил запрос: {_fal_error_text(e)}") from e
    start = time.time()
    seen_logs = set()
    percent_re = re.compile(r"(\d{1,3})\s*%")
    log_percent = 0
    last_queue_pos = None

    while True:
        # реагируем на Cancel в ComfyUI
        if mm is not None:
            try:
                mm.throw_exception_if_processing_interrupted()
            except Exception:
                try:
                    handler.cancel()
                    print(f"[fal {endpoint}] задача отменена")
                except Exception:
                    pass
                raise

        status = handler.status(with_logs=True)
        status_name = type(status).__name__

        for log in (getattr(status, "logs", None) or []):
            msg = (log or {}).get("message", "")
            if msg and msg not in seen_logs:
                seen_logs.add(msg)
                print(f"[fal {endpoint}] {msg}")
                m = percent_re.search(msg)
                if m:
                    log_percent = max(log_percent, min(100, int(m.group(1))))

        if status_name == "Completed":
            break
        if status_name == "Queued":
            pos = getattr(status, "position", None)
            if pos != last_queue_pos:
                last_queue_pos = pos
                print(f"[fal {endpoint}] в очереди, позиция: {pos}")
            set_progress(2)
        else:  # InProgress
            elapsed = time.time() - start
            estimated = 5 + (elapsed / max(est_seconds, 1)) * 90
            set_progress(max(estimated, log_percent))
        time.sleep(2)

    try:
        result = handler.get()
    except Exception as e:
        raise RuntimeError(f"fal вернул ошибку: {_fal_error_text(e)}") from e
    if pbar is not None:
        pbar.update_absolute(100, 100)
    return result


def _est_video_seconds(duration, fast=False, ref=False):
    """Грубая оценка времени генерации для прогресс-бара, в секундах."""
    try:
        dur = int(duration)
    except (TypeError, ValueError):
        dur = 6  # auto
    est = 45 + dur * 18
    if ref:
        est += 60
    if fast:
        est *= 0.5
    return est


# ---------------------------------------------------------------------------
# предварительная оценка стоимости
# ---------------------------------------------------------------------------

_RES_H = {"480p": 480, "720p": 720, "1080p": 1080}

# Seedance: токены = w*h*24*сек/1024
_SEEDANCE2_RATE = 0.014 / 1000       # $ за токен (standard)
_SEEDANCE2_FAST_RATE = 0.0112 / 1000  # $ за токен (fast, ~$0.2419/с на 720p)
_SEEDANCE15_AUDIO_RATE = 2.4 / 1e6    # $ за токен со звуком
_SEEDANCE15_RATE = 1.2 / 1e6          # $ за токен без звука

# GPT Image 2: $ за изображение (пиксели -> {low, medium, high})
_GPT_PRICES = [
    (1024 * 768, {"low": 0.005, "medium": 0.037, "high": 0.145}),
    (1024 * 1024, {"low": 0.006, "medium": 0.053, "high": 0.211}),
    (1024 * 1536, {"low": 0.005, "medium": 0.042, "high": 0.165}),
    (1920 * 1080, {"low": 0.005, "medium": 0.040, "high": 0.158}),
    (2560 * 1440, {"low": 0.007, "medium": 0.056, "high": 0.222}),
    (3840 * 2160, {"low": 0.012, "medium": 0.101, "high": 0.401}),
]
_GPT_PRESET_PX = {
    "auto": 1024 * 1024,
    "square": 1024 * 1024,
    "square_hd": 1536 * 1536,
    "portrait_4_3": 768 * 1024,
    "portrait_16_9": 1080 * 1920,
    "landscape_4_3": 1024 * 768,
    "landscape_16_9": 1920 * 1080,
}


def _px_dims(resolution, aspect_ratio):
    h = _RES_H.get(resolution, 720)
    ar = aspect_ratio if aspect_ratio and aspect_ratio != "auto" else "16:9"
    try:
        aw, ah = ar.split(":")
        w = int(h * int(aw) / int(ah))
    except (ValueError, ZeroDivisionError):
        w = int(h * 16 / 9)
    return w, h


def _video_cost_text(endpoint, args):
    w, h = _px_dims(args.get("resolution", "720p"), args.get("aspect_ratio"))
    tokens_per_sec = w * h * 24 / 1024
    if "seedance-2.0" in endpoint:
        per_sec = tokens_per_sec * (_SEEDANCE2_FAST_RATE if "/fast/" in endpoint
                                    else _SEEDANCE2_RATE)
        lo, hi = 4, 15
    else:
        per_sec = tokens_per_sec * (_SEEDANCE15_AUDIO_RATE
                                    if args.get("generate_audio", True)
                                    else _SEEDANCE15_RATE)
        lo, hi = 4, 12
    dur = args.get("duration")
    if dur in (None, "auto"):
        return f"~${per_sec * lo:.2f}–${per_sec * hi:.2f} (auto, {lo}–{hi} с)"
    return f"~${per_sec * int(dur):.2f} ({dur} с)"


def _gpt_cost_text(args):
    size = args.get("image_size", "auto")
    if isinstance(size, dict):
        px = size.get("width", 1024) * size.get("height", 1024)
    else:
        px = _GPT_PRESET_PX.get(size, 1024 * 1024)
    prices = min(_GPT_PRICES, key=lambda row: abs(row[0] - px))[1]
    quality = args.get("quality", "high")
    if quality == "auto":
        quality = "high"
    n = args.get("num_images", 1)
    return f"~${prices[quality] * n:.3f} ({n} шт, {quality})"


def _run(endpoint, arguments):
    print(f"[fal {endpoint}] ориентировочная стоимость: "
          f"{_video_cost_text(endpoint, arguments)}")
    est = _est_video_seconds(
        arguments.get("duration"),
        fast="/fast/" in endpoint,
        ref="reference" in endpoint,
    )
    result = _run_request(endpoint, arguments, est)
    video = (result or {}).get("video") or {}
    url = video.get("url")
    if not url:
        raise RuntimeError(f"fal не вернул видео: {result}")
    return url


def _download(url, prefix):
    out_dir = folder_paths.get_output_directory() if folder_paths else tempfile.gettempdir()
    os.makedirs(out_dir, exist_ok=True)
    idx = 0
    while True:
        path = os.path.join(out_dir, f"{prefix}_{idx:05d}.mp4")
        if not os.path.exists(path):
            break
        idx += 1
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return path


def _finish(url, prefix):
    path = _download(url, prefix)
    video_obj = VideoFromFile(path) if VideoFromFile else None
    return (video_obj, url, path)


DURATIONS_20 = ["auto"] + [str(i) for i in range(4, 16)]      # Seedance 2.0: auto, 4..15
DURATIONS_15 = [str(i) for i in range(4, 13)]                 # Seedance 1.5: 4..12
ASPECTS_20 = ["auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]
ASPECTS_15 = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]

RETURN_TYPES = ("VIDEO", "STRING", "STRING")
RETURN_NAMES = ("video", "video_url", "local_path")
CATEGORY = "fal/Seedance"


def _seed_arg(args, seed):
    if seed is not None and seed >= 0:
        args["seed"] = seed


DURATION_OVERRIDE_INPUT = ("FLOAT", {
    "default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1,
    "forceInput": True,
    "tooltip": "Если подключено и > 0 — перекрывает виджет duration. "
               "Секунды, округляются и зажимаются в допустимый диапазон.",
})


def _duration_value(duration, duration_override, lo, hi):
    """Выбор длительности: подключённый duration_override (секунды, FLOAT)
    имеет приоритет над комбо-виджетом."""
    if duration_override is not None and duration_override > 0:
        return str(max(lo, min(hi, int(round(duration_override)))))
    return duration


# ---------------------------------------------------------------------------
# Seedance 2.0
# ---------------------------------------------------------------------------

class Seedance2TextToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "resolution": (["720p", "480p"], {"default": "720p"}),
                "duration": (DURATIONS_20, {"default": "auto"}),
                "aspect_ratio": (ASPECTS_20, {"default": "16:9"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "fast_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "duration_override": DURATION_OVERRIDE_INPUT,
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
        }

    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, prompt, resolution, duration, aspect_ratio,
                 generate_audio, fast_mode, duration_override=0.0, seed=-1):
        _require_deps()
        endpoint = ("bytedance/seedance-2.0/fast/text-to-video"
                    if fast_mode else "bytedance/seedance-2.0/text-to-video")
        args = {
            "prompt": prompt,
            "resolution": resolution,
            "duration": _duration_value(duration, duration_override, 4, 15),
            "aspect_ratio": aspect_ratio,
            "generate_audio": generate_audio,
        }
        _seed_arg(args, seed)
        return _finish(_run(endpoint, args), "seedance2_t2v")


class Seedance2ImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "resolution": (["720p", "480p", "1080p"], {"default": "720p"}),
                "duration": (DURATIONS_20, {"default": "auto"}),
                "aspect_ratio": (ASPECTS_20, {"default": "auto"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "fast_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "end_image": ("IMAGE",),
                "duration_override": DURATION_OVERRIDE_INPUT,
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
        }

    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, image, prompt, resolution, duration, aspect_ratio,
                 generate_audio, fast_mode, end_image=None,
                 duration_override=0.0, seed=-1):
        _require_deps()
        endpoint = ("bytedance/seedance-2.0/fast/image-to-video"
                    if fast_mode else "bytedance/seedance-2.0/image-to-video")
        args = {
            "prompt": prompt,
            "image_url": _upload_image_input(image, 1)[0],
            "resolution": resolution,
            "duration": _duration_value(duration, duration_override, 4, 15),
            "aspect_ratio": aspect_ratio,
            "generate_audio": generate_audio,
        }
        if end_image is not None:
            args["end_image_url"] = _upload_image_input(end_image, 1)[0]
        _seed_arg(args, seed)
        return _finish(_run(endpoint, args), "seedance2_i2v")


class Seedance2ReferenceToVideo:
    """Мультимодальный режим: до 9 референс-картинок (@Image1, @Image2 ... в промпте),
    до 3 референс-видео и до 3 аудио.

    Картинки — входы image_1..image_4 (IMAGE, батч учитывается целиком) и/или
    список URL в image_urls. Видео — входы video_1..video_3 (VIDEO, нативная нода
    Load Video) и/или пути/URL в video_refs. Аудио — входы audio_1..audio_3 (AUDIO)
    и/или пути/URL в audio_refs. Нумерация @ImageN идёт по порядку: image_1..image_4,
    затем строки image_urls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "resolution": (["720p", "480p", "1080p"], {"default": "720p"}),
                "duration": (DURATIONS_20, {"default": "auto"}),
                "aspect_ratio": (ASPECTS_20, {"default": "auto"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "video_1": ("VIDEO",),
                "video_2": ("VIDEO",),
                "video_3": ("VIDEO",),
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "image_urls": ("STRING", {"multiline": True, "default": ""}),
                "video_refs": ("STRING", {"multiline": True, "default": ""}),
                "audio_refs": ("STRING", {"multiline": True, "default": ""}),
                "duration_override": DURATION_OVERRIDE_INPUT,
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
        }

    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, prompt, resolution, duration, aspect_ratio, generate_audio,
                 image_1=None, image_2=None, image_3=None, image_4=None,
                 video_1=None, video_2=None, video_3=None,
                 audio_1=None, audio_2=None, audio_3=None,
                 image_urls="", video_refs="", audio_refs="",
                 duration_override=0.0, seed=-1):
        _require_deps()

        img_urls = []
        for img in (image_1, image_2, image_3, image_4):
            if img is not None and len(img_urls) < 9:
                img_urls += _upload_image_input(img, 9 - len(img_urls))
        img_urls += _resolve_media_list(image_urls, 9 - len(img_urls))

        vid_urls, vid_total = [], 0.0
        for i, vid in enumerate((video_1, video_2, video_3), 1):
            if vid is not None and len(vid_urls) < 3:
                url, d = _upload_video_input(vid, label=f"video_{i}")
                if d:
                    vid_total += d
                vid_urls.append(url)
        if vid_total > 15.0:
            raise RuntimeError(
                f"Суммарная длительность референс-видео {vid_total:.1f} с — "
                f"больше лимита fal (2–15 с суммарно). Подрежь ролики."
            )
        vid_urls += _resolve_media_list(video_refs, 3 - len(vid_urls))

        aud_urls = []
        for aud in (audio_1, audio_2, audio_3):
            if aud is not None and len(aud_urls) < 3:
                aud_urls.append(_upload_audio_input(aud))
        aud_urls += _resolve_media_list(audio_refs, 3 - len(aud_urls))

        args = {
            "prompt": prompt,
            "resolution": resolution,
            "duration": _duration_value(duration, duration_override, 4, 15),
            "aspect_ratio": aspect_ratio,
            "generate_audio": generate_audio,
        }
        if img_urls:
            args["image_urls"] = img_urls
        if vid_urls:
            args["video_urls"] = vid_urls
        if aud_urls:
            args["audio_urls"] = aud_urls
        _seed_arg(args, seed)
        return _finish(_run("bytedance/seedance-2.0/reference-to-video", args),
                       "seedance2_ref")


# ---------------------------------------------------------------------------
# GPT Image 2 (OpenAI через fal)
# ---------------------------------------------------------------------------

GPT_IMAGE_SIZES = ["auto", "square_hd", "square", "portrait_4_3", "portrait_16_9",
                   "landscape_4_3", "landscape_16_9"]
GPT_QUALITY = ["high", "auto", "low", "medium"]


def _gpt_image_size(image_size, custom_width, custom_height):
    if custom_width > 0 and custom_height > 0:
        return {"width": custom_width, "height": custom_height}
    return image_size


def _mask_to_url(mask):
    """MASK ComfyUI (B,H,W, 1=редактируемая область) -> RGBA PNG,
    где редактируемая область прозрачна (конвенция OpenAI)."""
    arr = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[0]
    alpha = ((1.0 - np.clip(arr, 0, 1)) * 255).astype(np.uint8)
    h, w = alpha.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = alpha
    return _upload_pil(Image.fromarray(rgba, mode="RGBA"))


def _download_images_as_tensor(urls):
    """Скачивает картинки и собирает IMAGE-батч ComfyUI."""
    import torch
    pils = []
    for u in urls:
        resp = requests.get(u, timeout=300)
        resp.raise_for_status()
        pils.append(Image.open(io.BytesIO(resp.content)).convert("RGB"))
    base = pils[0].size
    arrs = []
    for p in pils:
        if p.size != base:
            p = p.resize(base, Image.LANCZOS)
        arrs.append(np.asarray(p).astype(np.float32) / 255.0)
    return torch.from_numpy(np.stack(arrs))


def _run_gpt_image(endpoint, args):
    print(f"[fal {endpoint}] ориентировочная стоимость: {_gpt_cost_text(args)}")
    result = _run_request(endpoint, args, est_seconds=45 * args.get("num_images", 1))
    urls = [img["url"] for img in (result or {}).get("images", []) if img.get("url")]
    if not urls:
        raise RuntimeError(f"fal не вернул изображения: {result}")
    return _download_images_as_tensor(urls), "\n".join(urls)


class GPTImage2TextToImage:
    """Универсальная нода GPT Image 2. Без подключённых картинок — чистый
    text-to-image. Если подключены image_1..image_4 / mask / extra_image_urls —
    нода автоматически использует edit-эндпоинт (референсы + инпейнт)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image_size": (GPT_IMAGE_SIZES, {"default": "landscape_4_3"}),
                "quality": (GPT_QUALITY, {"default": "high"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "mask": ("MASK",),
                "extra_image_urls": ("STRING", {"multiline": True, "default": ""}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "image_urls")
    FUNCTION = "generate"
    CATEGORY = "fal/GPT Image"

    def generate(self, prompt, image_size, quality, num_images,
                 image_1=None, image_2=None, image_3=None, image_4=None,
                 mask=None, extra_image_urls="",
                 custom_width=0, custom_height=0):
        _require_deps()
        urls = []
        for img in (image_1, image_2, image_3, image_4):
            if img is not None:
                urls += _upload_image_input(img, 16 - len(urls))
        urls += _resolve_media_list(extra_image_urls, 16 - len(urls))

        args = {
            "prompt": prompt,
            "image_size": _gpt_image_size(image_size, custom_width, custom_height),
            "quality": quality,
            "num_images": num_images,
            "output_format": "png",
        }
        if urls:  # есть референсы -> edit-эндпоинт
            args["image_urls"] = urls
            if mask is not None:
                args["mask_url"] = _mask_to_url(mask)
            return _run_gpt_image("openai/gpt-image-2/edit", args)
        if mask is not None:
            raise RuntimeError(
                "Маска подключена, но нет ни одной картинки: для инпейнта "
                "подключи изображение в image_1."
            )
        return _run_gpt_image("openai/gpt-image-2", args)


class GPTImage2Edit:
    """Редактирование/композиция: до 4 картинок сокетами + список URL,
    опциональная маска (белое = область, которую нужно изменить)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image_size": (GPT_IMAGE_SIZES, {"default": "auto"}),
                "quality": (GPT_QUALITY, {"default": "high"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "mask": ("MASK",),
                "extra_image_urls": ("STRING", {"multiline": True, "default": ""}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "image_urls")
    FUNCTION = "generate"
    CATEGORY = "fal/GPT Image"

    def generate(self, image_1, prompt, image_size, quality, num_images,
                 image_2=None, image_3=None, image_4=None, mask=None,
                 extra_image_urls="", custom_width=0, custom_height=0):
        _require_deps()
        urls = []
        for img in (image_1, image_2, image_3, image_4):
            if img is not None:
                urls += _upload_image_input(img, 16 - len(urls))
        urls += _resolve_media_list(extra_image_urls, 16 - len(urls))

        args = {
            "prompt": prompt,
            "image_urls": urls,
            "image_size": _gpt_image_size(image_size, custom_width, custom_height),
            "quality": quality,
            "num_images": num_images,
            "output_format": "png",
        }
        if mask is not None:
            args["mask_url"] = _mask_to_url(mask)
        return _run_gpt_image("openai/gpt-image-2/edit", args)


# ---------------------------------------------------------------------------
# Seedance 1.5 Pro
# ---------------------------------------------------------------------------

class Seedance15ProTextToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "resolution": (["720p", "480p"], {"default": "720p"}),
                "duration": (DURATIONS_15, {"default": "5"}),
                "aspect_ratio": (ASPECTS_15, {"default": "16:9"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "camera_fixed": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "duration_override": DURATION_OVERRIDE_INPUT,
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
        }

    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, prompt, resolution, duration, aspect_ratio,
                 generate_audio, camera_fixed, duration_override=0.0, seed=-1):
        _require_deps()
        args = {
            "prompt": prompt,
            "resolution": resolution,
            "duration": _duration_value(duration, duration_override, 4, 12),
            "aspect_ratio": aspect_ratio,
            "generate_audio": generate_audio,
            "camera_fixed": camera_fixed,
        }
        _seed_arg(args, seed)
        return _finish(_run("fal-ai/bytedance/seedance/v1.5/pro/text-to-video", args),
                       "seedance15_t2v")


class Seedance15ProImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "resolution": (["720p", "480p"], {"default": "720p"}),
                "duration": (DURATIONS_15, {"default": "5"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "camera_fixed": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "end_image": ("IMAGE",),
                "duration_override": DURATION_OVERRIDE_INPUT,
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
        }

    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, image, prompt, resolution, duration,
                 generate_audio, camera_fixed, end_image=None,
                 duration_override=0.0, seed=-1):
        _require_deps()
        args = {
            "prompt": prompt,
            "image_url": _upload_image_input(image, 1)[0],
            "resolution": resolution,
            "duration": _duration_value(duration, duration_override, 4, 12),
            "generate_audio": generate_audio,
            "camera_fixed": camera_fixed,
        }
        if end_image is not None:
            args["end_image_url"] = _upload_image_input(end_image, 1)[0]
        _seed_arg(args, seed)
        return _finish(_run("fal-ai/bytedance/seedance/v1.5/pro/image-to-video", args),
                       "seedance15_i2v")


NODE_CLASS_MAPPINGS = {
    "Seedance2TextToVideo_fal": Seedance2TextToVideo,
    "Seedance2ImageToVideo_fal": Seedance2ImageToVideo,
    "Seedance2ReferenceToVideo_fal": Seedance2ReferenceToVideo,
    "Seedance15ProTextToVideo_fal": Seedance15ProTextToVideo,
    "Seedance15ProImageToVideo_fal": Seedance15ProImageToVideo,
    "GPTImage2TextToImage_fal": GPTImage2TextToImage,
    "GPTImage2Edit_fal": GPTImage2Edit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Seedance2TextToVideo_fal": "Seedance 2.0 Text-to-Video (fal)",
    "Seedance2ImageToVideo_fal": "Seedance 2.0 Image-to-Video (fal)",
    "Seedance2ReferenceToVideo_fal": "Seedance 2.0 Reference-to-Video (fal)",
    "Seedance15ProTextToVideo_fal": "Seedance 1.5 Pro Text-to-Video (fal)",
    "Seedance15ProImageToVideo_fal": "Seedance 1.5 Pro Image-to-Video (fal)",
    "GPTImage2TextToImage_fal": "GPT Image 2 (fal)",
    "GPTImage2Edit_fal": "GPT Image 2 Edit (fal)",
}
