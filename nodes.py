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


def _probe_video_info(path):
    """(длительность_с, ширина, высота) по метаданным контейнера — так видео
    увидит fal. Любое поле может быть None."""
    dur = w = h = None
    try:
        import av
        with av.open(path) as c:
            if c.duration:
                dur = c.duration / 1_000_000.0  # микросекунды
            v = c.streams.video[0]
            if dur is None and v.duration and v.time_base:
                dur = float(v.duration * v.time_base)
            w, h = v.codec_context.width, v.codec_context.height
    except Exception:
        pass
    return dur, w, h


def _probe_container_duration(path):
    return _probe_video_info(path)[0]


def _ffmpeg_exe():
    import shutil
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _mktmp_mp4():
    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    return tmp


def _reencode_video(src_path):
    """Перекодирует видео в чистый H.264 mp4 с корректными тайм-метками.

    Ролики из Unreal/NLE часто имеют битые pts: локальный плеер их играет,
    а парсер fal видит «1 кадр» (0.04 с) и отклоняет запрос.
    После кодирования длительность проверяется по метаданным контейнера
    (как её увидит fal); если она всё ещё битая — запасной путь через ffmpeg.
    Возвращает (путь_tmp, длительность_с)."""
    import av
    from fractions import Fraction

    with av.open(src_path) as inp:
        vstream = inp.streams.video[0]
        rate = vstream.average_rate or vstream.guessed_rate or Fraction(24, 1)
        cc = vstream.codec_context
        width = cc.width - (cc.width % 2)
        height = cc.height - (cc.height % 2)

        tmp = _mktmp_mp4()
        time_base = Fraction(rate.denominator, rate.numerator)
        n = 0
        with av.open(tmp, "w") as out:
            out_v = out.add_stream("h264", rate=rate)
            out_v.width = width
            out_v.height = height
            out_v.pix_fmt = "yuv420p"
            out_v.options = {"crf": "18", "preset": "fast"}
            try:
                out_v.codec_context.time_base = time_base
            except Exception:
                pass
            for frame in inp.decode(vstream):
                frame = frame.reformat(width=width, height=height,
                                       format="yuv420p")
                # явные тайм-метки: кадр i в момент i/fps
                frame.pts = n
                frame.time_base = time_base
                for pkt in out_v.encode(frame):
                    out.mux(pkt)
                n += 1
            for pkt in out_v.encode():
                out.mux(pkt)

    expected = n / float(rate)
    got = _probe_container_duration(tmp)
    if got is not None and abs(got - expected) < max(0.5, expected * 0.2):
        return tmp, expected

    # PyAV не справился — пробуем ffmpeg (идёт в комплекте VideoHelperSuite)
    print(f"[fal] контейнер после PyAV: {got} с вместо {expected:.2f} с, "
          f"пробую ffmpeg")
    exe = _ffmpeg_exe()
    if exe:
        import subprocess
        tmp2 = _mktmp_mp4()
        cmd = [exe, "-y", "-r", str(rate), "-i", src_path, "-an",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               "-preset", "fast", "-movflags", "+faststart", tmp2]
        proc = subprocess.run(cmd, capture_output=True)
        got2 = _probe_container_duration(tmp2) if proc.returncode == 0 else None
        if got2 is not None and abs(got2 - expected) < max(0.5, expected * 0.2):
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return tmp2, expected
        try:
            os.unlink(tmp2)
        except OSError:
            pass
    raise RuntimeError(
        f"Не удалось получить видео с корректной длительностью: контейнер "
        f"показывает {got} с при ожидаемых {expected:.2f} с. Прогони ролик "
        f"через ffmpeg вручную: ffmpeg -r {rate} -i вход.mp4 -c:v libx264 "
        f"-pix_fmt yuv420p выход.mp4"
    )


def _upload_video_input(video, label="видео", min_duration=2.0):
    """VIDEO-вход ComfyUI -> URL в fal storage.

    Здоровые файлы грузятся как есть; если метаданные контейнера битые
    (Unreal/NLE) — видео перекодируется. Возвращает (url, dur, w, h)."""
    if isinstance(video, str) and video.lower().startswith(("http://", "https://")):
        return video, None, None, None

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
        dur, w, h = _probe_video_info(src)
        if dur is not None and dur > 0.5:
            # контейнер здоровый — грузим оригинал без перекодировки
            print(f"[fal] {label}: {dur:.2f} с, {w}x{h} (без перекодировки)")
        else:
            try:
                tmp_enc, dur = _reencode_video(src)
                _, w, h = _probe_video_info(tmp_enc)
                src = tmp_enc
                print(f"[fal] {label}: {dur:.2f} с после перекодировки")
            except ImportError:
                dur = None  # нет PyAV — грузим как есть
        if min_duration and dur is not None and dur < min_duration:
            raise RuntimeError(
                f"Референс-{label} слишком короткое: {dur:.2f} с "
                f"(fal требует 2–15 с суммарно). Похоже, в видео-вход "
                f"попал одиночный кадр — подключи полноценный ролик."
            )
        return fal_client.upload_file(src), dur, w, h
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
    Локальные файлы загружаются в fal storage. Строки, не похожие ни на путь,
    ни на URL (например «0» от сдвига виджетов старой ноды), пропускаются
    с предупреждением."""
    urls = []
    for line in (str(text) if text is not None else "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith(("http://", "https://", "data:")):
            urls.append(line)
        elif os.path.isfile(line):
            urls.append(fal_client.upload_file(line))
        elif any(ch in line for ch in "/\\.") :
            raise RuntimeError(f"Файл не найден и это не URL: {line}")
        else:
            print(f"[fal] строка «{line}» не похожа на путь или URL — пропускаю "
                  f"(если нода старая, пересоздай её: Fix node / удалить и добавить)")
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

    def _is_transient(exc):
        """Временный сбой на стороне fal — есть смысл повторить."""
        s = str(exc).lower()
        return any(t in s for t in (
            "downstream_service_unavailable", "downstream service unavailable",
            "502", "503", "504", "gateway timeout", "service unavailable",
            "bad gateway", "timed out", "timeout", "connection",
        ))

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
    transient_hits = 0
    MAX_TRANSIENT = 8  # переживаем ~кратковременные сбои шлюза fal

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

        try:
            status = handler.status(with_logs=True)
            transient_hits = 0
        except Exception as e:
            if _is_transient(e) and transient_hits < MAX_TRANSIENT:
                transient_hits += 1
                wait = min(30, 3 * transient_hits)
                print(f"[fal {endpoint}] временный сбой fal "
                      f"({_fal_error_text(e)}), повтор {transient_hits}/"
                      f"{MAX_TRANSIENT} через {wait} с")
                time.sleep(wait)
                continue
            raise RuntimeError(f"fal вернул ошибку: {_fal_error_text(e)}") from e
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

    # получаем результат — с повтором на временных сбоях шлюза
    # (задача уже посчитана на fal, разовый 504 не должен её терять)
    result = None
    for attempt in range(MAX_TRANSIENT):
        try:
            result = handler.get()
            break
        except Exception as e:
            if _is_transient(e) and attempt < MAX_TRANSIENT - 1:
                wait = min(30, 3 * (attempt + 1))
                print(f"[fal {endpoint}] временный сбой при получении "
                      f"результата ({_fal_error_text(e)}), повтор "
                      f"{attempt + 1}/{MAX_TRANSIENT} через {wait} с")
                time.sleep(wait)
                continue
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


# Обычный FLOAT-виджет (НЕ forceInput: скрытые входы ломают порядок
# widgets_values при перезагрузке окна — seed превращался в NaN).
# Провод подключается прямо к точке слева от виджета.
DURATION_OVERRIDE_INPUT = ("FLOAT", {
    "default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1,
    "tooltip": "Если > 0 (или подключён провод) — перекрывает виджет "
               "duration. Секунды, округляются и зажимаются в допустимый "
               "диапазон.",
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
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "duration_override": DURATION_OVERRIDE_INPUT,
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
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "duration_override": DURATION_OVERRIDE_INPUT,
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
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "duration_override": DURATION_OVERRIDE_INPUT,
                # в конце списка, чтобы не сдвигать виджеты сохранённых нод
                "fast_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Fast-тир: ~$0.24/с вместо ~$0.30/с, быстрее, "
                               "но максимум 720p"}),
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
                 duration_override=0.0, seed=-1, fast_mode=False):
        _require_deps()
        if fast_mode and resolution == "1080p":
            print("[fal] fast-тир reference-to-video поддерживает максимум "
                  "720p — понижаю разрешение")
            resolution = "720p"

        img_urls = []
        for img in (image_1, image_2, image_3, image_4):
            if img is not None and len(img_urls) < 9:
                img_urls += _upload_image_input(img, 9 - len(img_urls))
        img_urls += _resolve_media_list(image_urls, 9 - len(img_urls))

        vid_urls, vid_total = [], 0.0
        for i, vid in enumerate((video_1, video_2, video_3), 1):
            if vid is not None and len(vid_urls) < 3:
                url, d, _, _ = _upload_video_input(vid, label=f"video_{i}")
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
        endpoint = ("bytedance/seedance-2.0/fast/reference-to-video"
                    if fast_mode else "bytedance/seedance-2.0/reference-to-video")
        return _finish(_run(endpoint, args), "seedance2_ref")


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
                # порядок виджетов: custom_* раньше extra_image_urls, чтобы
                # старые ноды на канвасе не ловили сдвиг значений
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
                "extra_image_urls": ("STRING", {"multiline": True, "default": ""}),
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
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "duration_override": DURATION_OVERRIDE_INPUT,
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
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "duration_override": DURATION_OVERRIDE_INPUT,
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


# ---------------------------------------------------------------------------
# Qwen Image Max (Alibaba)
# ---------------------------------------------------------------------------

QWEN_SIZES = ["square_hd", "square", "portrait_4_3", "portrait_16_9",
              "landscape_4_3", "landscape_16_9"]
_QWEN_PRICE = 0.075  # $ за изображение


class QwenImageMax:
    """Qwen Image Max: без картинок — text-to-image, с подключёнными
    image_1..image_3 — edit (инструкции по референсам, до 3 картинок).
    Промпт до 800 символов, поддерживает русский/английский/китайский."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image_size": (QWEN_SIZES, {"default": "landscape_16_9"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "enable_prompt_expansion": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "LLM-дораскрытие промпта на стороне Qwen"}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
                "extra_image_urls": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "image_urls")
    FUNCTION = "generate"
    CATEGORY = "fal/Qwen"

    def generate(self, prompt, image_size, num_images, enable_prompt_expansion,
                 image_1=None, image_2=None, image_3=None, negative_prompt="",
                 seed=-1, custom_width=0, custom_height=0, extra_image_urls=""):
        _require_deps()
        urls = []
        for img in (image_1, image_2, image_3):
            if img is not None and len(urls) < 3:
                urls += _upload_image_input(img, 3 - len(urls))
        urls += _resolve_media_list(extra_image_urls, 3 - len(urls))

        size = ({"width": custom_width, "height": custom_height}
                if custom_width > 0 and custom_height > 0 else image_size)
        args = {
            "prompt": prompt[:800],
            "image_size": size,
            "num_images": num_images,
            "enable_prompt_expansion": enable_prompt_expansion,
            "output_format": "png",
        }
        if negative_prompt.strip():
            args["negative_prompt"] = negative_prompt[:500]
        _seed_arg(args, seed)

        if urls:
            args["image_urls"] = urls
            endpoint = "fal-ai/qwen-image-max/edit"
        else:
            endpoint = "fal-ai/qwen-image-max/text-to-image"
        print(f"[fal {endpoint}] ориентировочная стоимость: "
              f"~${_QWEN_PRICE * num_images:.3f} ({num_images} шт)")
        result = _run_request(endpoint, args, est_seconds=30 * num_images)
        img_urls = [i["url"] for i in (result or {}).get("images", [])
                    if i.get("url")]
        if not img_urls:
            raise RuntimeError(f"fal не вернул изображения: {result}")
        return _download_images_as_tensor(img_urls), "\n".join(img_urls)


# ---------------------------------------------------------------------------
# Nano Banana (Google) — Edit
# ---------------------------------------------------------------------------

NB_MODELS = {
    "nano-banana-2": "fal-ai/nano-banana-2/edit",
    "nano-banana-pro": "fal-ai/nano-banana-pro/edit",
}
NB_ASPECTS = ["auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1",
              "4:5", "3:4", "2:3", "9:16"]
NB_RESOLUTIONS = ["1K", "0.5K", "2K", "4K"]

# $ за изображение: базовая цена и множители по разрешению
_NB_PRICE = {
    "nano-banana-2": {"0.5K": 0.06, "1K": 0.08, "2K": 0.12, "4K": 0.16},
    "nano-banana-pro": {"0.5K": 0.15, "1K": 0.15, "2K": 0.15, "4K": 0.30},
}


class NanoBananaEdit:
    """Редактирование/композиция картинок моделями Google Nano Banana 2 /
    Nano Banana Pro через fal. До 4 референсов сокетами + список URL.
    В промпте можно ссылаться на референсы по порядку подачи."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (list(NB_MODELS), {"default": "nano-banana-2"}),
                "resolution": (NB_RESOLUTIONS, {"default": "1K"}),
                "aspect_ratio": (NB_ASPECTS, {"default": "auto"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "thinking_level": (["off", "minimal", "high"], {
                    "default": "off",
                    "tooltip": "Только nano-banana-2; high +$0.002/запрос"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "extra_image_urls": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "image_urls", "description")
    FUNCTION = "generate"
    CATEGORY = "fal/Nano Banana"

    def generate(self, image_1, prompt, model, resolution, aspect_ratio,
                 num_images, image_2=None, image_3=None, image_4=None,
                 system_prompt="", thinking_level="off", seed=-1,
                 extra_image_urls=""):
        _require_deps()
        urls = []
        for img in (image_1, image_2, image_3, image_4):
            if img is not None:
                urls += _upload_image_input(img, 14 - len(urls))
        urls += _resolve_media_list(extra_image_urls, 14 - len(urls))

        # 0.5K поддерживает только nano-banana-2
        if model == "nano-banana-pro" and resolution == "0.5K":
            resolution = "1K"

        args = {
            "prompt": prompt,
            "image_urls": urls,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "num_images": num_images,
            "output_format": "png",
        }
        if system_prompt.strip():
            args["system_prompt"] = system_prompt
        if model == "nano-banana-2" and thinking_level != "off":
            args["thinking_level"] = thinking_level
        _seed_arg(args, seed)

        price = _NB_PRICE[model].get(resolution, 0.08) * num_images
        endpoint = NB_MODELS[model]
        print(f"[fal {endpoint}] ориентировочная стоимость: ~${price:.2f} "
              f"({num_images} шт, {resolution})")
        result = _run_request(endpoint, args, est_seconds=30 * num_images)
        img_urls = [i["url"] for i in (result or {}).get("images", [])
                    if i.get("url")]
        if not img_urls:
            raise RuntimeError(f"fal не вернул изображения: {result}")
        tensor = _download_images_as_tensor(img_urls)
        return (tensor, "\n".join(img_urls),
                (result or {}).get("description", "") or "")


# ---------------------------------------------------------------------------
# Topaz Video Upscale
# ---------------------------------------------------------------------------

TOPAZ_MODELS = ["Proteus", "Artemis HQ", "Artemis MQ", "Artemis LQ",
                "Nyx", "Nyx Fast", "Nyx XL", "Nyx HF",
                "Gaia HQ", "Gaia CG", "Gaia 2",
                "Starlight Precise 1", "Starlight Precise 2",
                "Starlight Precise 2.5", "Starlight HQ", "Starlight Mini",
                "Starlight Sharp", "Starlight Fast 1", "Starlight Fast 2"]

# Тумблер тонких настроек: -1 = не отправлять (fal возьмёт дефолт модели)
_TOPAZ_TUNE = ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                         "tooltip": "-1 = дефолт выбранной модели"})


def _topaz_cost_text(dur, w, h, factor, target_fps, model):
    if not dur or not w or not h:
        return "оценка недоступна (не удалось прочитать метаданные)"
    out_h = h * factor
    if out_h <= 720:
        per_sec = 0.01
    elif out_h <= 1080:
        per_sec = 0.02
    else:
        per_sec = 0.08
    if target_fps and target_fps >= 60:
        per_sec *= 2
    if model == "Gaia 2":
        per_sec /= 2
    return (f"~${dur * per_sec:.2f} "
            f"({dur:.1f} с, выход ~{int(w*factor)}x{int(out_h)})")


class TopazVideoUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "model": (TOPAZ_MODELS, {"default": "Proteus"}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0,
                                             "max": 8.0, "step": 0.5}),
                "H264_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "H264 совместимее (превью в браузере); "
                               "false = H265, меньше размер"}),
            },
            "optional": {
                "target_fps": ("INT", {
                    "default": 0, "min": 0, "max": 120,
                    "tooltip": "0 = не интерполировать кадры; "
                               ">0 = включить интерполяцию до этого fps"}),
                "compression": _TOPAZ_TUNE,
                "noise": _TOPAZ_TUNE,
                "halo": _TOPAZ_TUNE,
                "grain": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 0.1,
                                    "step": 0.01,
                                    "tooltip": "-1 = дефолт модели"}),
                "recover_detail": _TOPAZ_TUNE,
            },
        }

    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES
    FUNCTION = "upscale"
    CATEGORY = "fal/Topaz"

    def upscale(self, video, model, upscale_factor, H264_output,
                target_fps=0, compression=-1.0, noise=-1.0, halo=-1.0,
                grain=-1.0, recover_detail=-1.0):
        _require_deps()
        url, dur, w, h = _upload_video_input(video, label="видео для апскейла",
                                             min_duration=None)
        print(f"[fal fal-ai/topaz/upscale/video] ориентировочная стоимость: "
              f"{_topaz_cost_text(dur, w, h, upscale_factor, target_fps, model)}")
        args = {
            "video_url": url,
            "model": model,
            "upscale_factor": upscale_factor,
            "H264_output": H264_output,
        }
        if target_fps and target_fps > 0:
            args["target_fps"] = target_fps
        for name, val in (("compression", compression), ("noise", noise),
                          ("halo", halo), ("grain", grain),
                          ("recover_detail", recover_detail)):
            if val is not None and val >= 0:
                args[name] = val
        est = 60 + (dur or 10) * 15
        result = _run_request("fal-ai/topaz/upscale/video", args, est)
        out = (result or {}).get("video") or {}
        if not out.get("url"):
            raise RuntimeError(f"fal не вернул видео: {result}")
        return _finish(out["url"], "topaz_upscale")


# ---------------------------------------------------------------------------
# Модели с поддержкой пользовательских LoRA (FLUX / Wan / Qwen-Image Edit)
# ---------------------------------------------------------------------------

_LORA_URL_CACHE = {}  # (abspath, mtime, size) -> URL в fal storage


def _lora_choices():
    """Список LoRA из папки ComfyUI/models/loras для выпадающего списка."""
    try:
        import folder_paths
        return ["none"] + list(folder_paths.get_filename_list("loras"))
    except Exception:
        return ["none"]


_FAL_LORA_LIMIT = 1024 ** 3  # fal не принимает LoRA больше 1 ГБ


def _upload_lora(ref):
    """ref — URL или локальный путь к .safetensors. URL возвращается как есть,
    локальный файл грузится в fal storage (с кэшем по mtime/размеру)."""
    if ref.lower().startswith(("http://", "https://", "data:")):
        return ref
    if not os.path.isfile(ref):
        raise RuntimeError(f"Файл LoRA не найден и это не URL: {ref}")
    st = os.stat(ref)
    # проверяем лимит fal ДО долгой загрузки
    if st.st_size > _FAL_LORA_LIMIT:
        raise RuntimeError(
            f"LoRA {os.path.basename(ref)} весит {st.st_size / 1024**3:.2f} ГБ — "
            f"fal не принимает LoRA больше 1 ГБ. Похоже, это не обычная LoRA, "
            f"а полный файн-тюн или очень высокий ранг. Уменьши ранг при "
            f"обучении, сконвертируй в fp16/квантуй, либо используй LoRA "
            f"поменьше. (Проверка до загрузки — время и трафик не потрачены.)"
        )
    key = (os.path.abspath(ref), int(st.st_mtime), st.st_size)
    if key in _LORA_URL_CACHE:
        return _LORA_URL_CACHE[key]
    print(f"[fal] загружаю LoRA {os.path.basename(ref)} "
          f"({st.st_size / 1e6:.0f} МБ) в fal storage — первый раз может занять время")
    url = fal_client.upload_file(ref)
    _LORA_URL_CACHE[key] = url
    return url


def _lora_slot_inputs(n=3):
    """n слотов LoRA для секции optional INPUT_TYPES."""
    choices = _lora_choices()
    d = {}
    for i in range(1, n + 1):
        d[f"lora_{i}"] = (choices, {"default": "none"})
        d[f"lora_{i}_url"] = ("STRING", {
            "default": "",
            "tooltip": "URL весов (Civitai/HF) или путь к файлу; "
                       "перекрывает выбор из списка"})
        d[f"lora_{i}_scale"] = ("FLOAT", {
            "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05})
    return d


def _build_loras(kw, n=3, extra=None):
    """Собирает массив loras из kwargs слотов. extra — доп. поля в каждый
    элемент (например {'transformer': 'both'} для Wan 2.2)."""
    out = []
    for i in range(1, n + 1):
        name = kw.get(f"lora_{i}", "none")
        url = (kw.get(f"lora_{i}_url", "") or "").strip()
        scale = kw.get(f"lora_{i}_scale", 1.0)
        ref = None
        if url:
            ref = url
        elif name and name != "none":
            try:
                import folder_paths
                ref = folder_paths.get_full_path("loras", name) or name
            except Exception:
                ref = name
        if not ref:
            continue
        item = {"path": _upload_lora(ref), "scale": float(scale)}
        if extra:
            item.update(extra)
        out.append(item)
    return out


IMG_SIZES = ["landscape_16_9", "landscape_4_3", "square_hd", "square",
             "portrait_4_3", "portrait_16_9"]


def _img_size(preset, cw, ch):
    return {"width": cw, "height": ch} if cw > 0 and ch > 0 else preset


class FluxLoraImage:
    """FLUX.1 [dev] с пользовательскими LoRA. Без картинки — text-to-image,
    с подключённой image — image-to-image."""

    @classmethod
    def INPUT_TYPES(cls):
        opt = {
            "image": ("IMAGE",),
            "strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0,
                                   "step": 0.01,
                                   "tooltip": "Сила для image-to-image"}),
            "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 60}),
            "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0,
                                         "step": 0.1}),
            "custom_width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32}),
            "custom_height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32}),
            "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
        }
        opt.update(_lora_slot_inputs(3))
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image_size": (IMG_SIZES, {"default": "landscape_16_9"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
            "optional": opt,
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "image_urls")
    FUNCTION = "generate"
    CATEGORY = "fal/LoRA"

    def generate(self, prompt, image_size, num_images, image=None, strength=0.85,
                 num_inference_steps=28, guidance_scale=3.5,
                 custom_width=0, custom_height=0, seed=-1, **kw):
        _require_deps()
        loras = _build_loras(kw, 3)
        args = {
            "prompt": prompt,
            "image_size": _img_size(image_size, custom_width, custom_height),
            "num_images": num_images,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "output_format": "png",
        }
        if loras:
            args["loras"] = loras
        _seed_arg(args, seed)
        if image is not None:
            args["image_url"] = _upload_image_input(image, 1)[0]
            args["strength"] = strength
            endpoint = "fal-ai/flux-lora/image-to-image"
        else:
            endpoint = "fal-ai/flux-lora"
        print(f"[fal {endpoint}] LoRA: {len(loras)} шт")
        result = _run_request(endpoint, args, est_seconds=15 * num_images)
        urls = [i["url"] for i in (result or {}).get("images", []) if i.get("url")]
        if not urls:
            raise RuntimeError(f"fal не вернул изображения: {result}")
        return _download_images_as_tensor(urls), "\n".join(urls)


class QwenImageEditLora:
    """Qwen-Image Edit с пользовательскими LoRA (до 3). Редактирование картинки
    по инструкции — в отличие от закрытой Qwen Image Max, здесь свои LoRA."""

    @classmethod
    def INPUT_TYPES(cls):
        opt = {
            "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 60}),
            "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0,
                                         "step": 0.1}),
            "acceleration": (["none", "regular", "high"], {"default": "none"}),
            "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
            "custom_width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32}),
            "custom_height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32}),
            "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
        }
        opt.update(_lora_slot_inputs(3))
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image_size": (IMG_SIZES, {"default": "square_hd"}),
            },
            "optional": opt,
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "image_urls")
    FUNCTION = "generate"
    CATEGORY = "fal/LoRA"

    def generate(self, image, prompt, image_size, negative_prompt="",
                 num_inference_steps=30, guidance_scale=4.0, acceleration="none",
                 num_images=1, custom_width=0, custom_height=0, seed=-1, **kw):
        _require_deps()
        loras = _build_loras(kw, 3)
        args = {
            "prompt": prompt,
            "image_url": _upload_image_input(image, 1)[0],
            "image_size": _img_size(image_size, custom_width, custom_height),
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "acceleration": acceleration,
            "num_images": num_images,
            "output_format": "png",
        }
        if negative_prompt.strip():
            args["negative_prompt"] = negative_prompt
        if loras:
            args["loras"] = loras
        _seed_arg(args, seed)
        print(f"[fal fal-ai/qwen-image-edit-lora] LoRA: {len(loras)} шт")
        result = _run_request("fal-ai/qwen-image-edit-lora", args,
                              est_seconds=20 * num_images)
        urls = [i["url"] for i in (result or {}).get("images", []) if i.get("url")]
        if not urls:
            raise RuntimeError(f"fal не вернул изображения: {result}")
        return _download_images_as_tensor(urls), "\n".join(urls)


class WanLoraVideo:
    """Wan 2.2 A14B с пользовательскими LoRA. Без картинки — text-to-video,
    с подключённой image — image-to-video. Цена ~$0.1/сек."""

    @classmethod
    def INPUT_TYPES(cls):
        opt = {
            "image": ("IMAGE",),
            "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            "num_frames": ("INT", {"default": 81, "min": 17, "max": 161}),
            "frames_per_second": ("INT", {"default": 16, "min": 4, "max": 60}),
            "num_inference_steps": ("INT", {"default": 27, "min": 1, "max": 50}),
            "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0,
                                         "step": 0.1}),
            "lora_transformer": (["both", "high", "low"], {
                "default": "both",
                "tooltip": "К какому эксперту Wan 2.2 применять LoRA "
                           "(high/low noise). both — безопаснее для готовых LoRA"}),
            "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
        }
        opt.update(_lora_slot_inputs(3))
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "resolution": (["720p", "580p", "480p"], {"default": "720p"}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
            },
            "optional": opt,
        }

    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = "fal/LoRA"

    def generate(self, prompt, resolution, aspect_ratio, image=None,
                 negative_prompt="", num_frames=81, frames_per_second=16,
                 num_inference_steps=27, guidance_scale=3.5,
                 lora_transformer="both", seed=-1, **kw):
        _require_deps()
        loras = _build_loras(kw, 3, extra={"transformer": lora_transformer})
        args = {
            "prompt": prompt,
            "resolution": resolution,
            "num_frames": num_frames,
            "frames_per_second": frames_per_second,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        if negative_prompt.strip():
            args["negative_prompt"] = negative_prompt
        if loras:
            args["loras"] = loras
        _seed_arg(args, seed)
        if image is not None:
            args["image_url"] = _upload_image_input(image, 1)[0]
            endpoint = "fal-ai/wan/v2.2-a14b/image-to-video/lora"
        else:
            args["aspect_ratio"] = aspect_ratio
            endpoint = "fal-ai/wan/v2.2-a14b/text-to-video/lora"
        secs = num_frames / max(frames_per_second, 1)
        print(f"[fal {endpoint}] LoRA: {len(loras)} шт, "
              f"~{secs:.1f} с видео, ориентировочная стоимость ~${secs * 0.1:.2f}")
        return _finish(_run_wan(endpoint, args), "wan_lora")


def _run_wan(endpoint, args):
    est = args.get("num_frames", 81) / max(args.get("frames_per_second", 16), 1) * 20 + 30
    result = _run_request(endpoint, args, est)
    video = (result or {}).get("video") or {}
    if not video.get("url"):
        raise RuntimeError(f"fal не вернул видео: {result}")
    return video["url"]


NODE_CLASS_MAPPINGS = {
    "Seedance2TextToVideo_fal": Seedance2TextToVideo,
    "Seedance2ImageToVideo_fal": Seedance2ImageToVideo,
    "Seedance2ReferenceToVideo_fal": Seedance2ReferenceToVideo,
    "Seedance15ProTextToVideo_fal": Seedance15ProTextToVideo,
    "Seedance15ProImageToVideo_fal": Seedance15ProImageToVideo,
    "GPTImage2TextToImage_fal": GPTImage2TextToImage,
    "GPTImage2Edit_fal": GPTImage2Edit,
    "TopazVideoUpscale_fal": TopazVideoUpscale,
    "NanoBananaEdit_fal": NanoBananaEdit,
    "QwenImageMax_fal": QwenImageMax,
    "FluxLoraImage_fal": FluxLoraImage,
    "QwenImageEditLora_fal": QwenImageEditLora,
    "WanLoraVideo_fal": WanLoraVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Seedance2TextToVideo_fal": "Seedance 2.0 Text-to-Video (fal)",
    "Seedance2ImageToVideo_fal": "Seedance 2.0 Image-to-Video (fal)",
    "Seedance2ReferenceToVideo_fal": "Seedance 2.0 Reference-to-Video (fal)",
    "Seedance15ProTextToVideo_fal": "Seedance 1.5 Pro Text-to-Video (fal)",
    "Seedance15ProImageToVideo_fal": "Seedance 1.5 Pro Image-to-Video (fal)",
    "GPTImage2TextToImage_fal": "GPT Image 2 (fal)",
    "GPTImage2Edit_fal": "GPT Image 2 Edit (fal)",
    "TopazVideoUpscale_fal": "Topaz Video Upscale (fal)",
    "NanoBananaEdit_fal": "Nano Banana 2 / Pro Edit (fal)",
    "QwenImageMax_fal": "Qwen Image Max (fal)",
    "FluxLoraImage_fal": "FLUX LoRA (fal)",
    "QwenImageEditLora_fal": "Qwen-Image Edit LoRA (fal)",
    "WanLoraVideo_fal": "Wan 2.2 LoRA Video (fal)",
}
