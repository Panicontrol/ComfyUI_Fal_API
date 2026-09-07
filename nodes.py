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
import re
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


def _split_media_refs(text):
    """Разбор поля со ссылками: и многострочного, и однострочного.
    Разделители — перевод строки, запятая и точка с запятой. Строка вида
    data:... не режется по запятой, иначе развалится base64."""
    out = []
    for line in (str(text) if text is not None else "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("data:"):
            out.append(line)          # data-URI содержит запятую — не трогаем
            continue
        for part in re.split(r"[,;]", line):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _resolve_media_list(text, limit):
    """Поле со ссылками: URL или локальный путь, по одному на строку либо
    несколько через запятую.
    Локальные файлы загружаются в fal storage. Строки, не похожие ни на путь,
    ни на URL (например «0» от сдвига виджетов старой ноды), пропускаются
    с предупреждением."""
    urls = []
    if limit is not None and limit <= 0:
        return urls
    for line in _split_media_refs(text):
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
        text = "; ".join(msgs) if msgs else s
        low = s.lower()
        if ("content_policy_violation" in low or "partner_validation_failed" in low
                or "likeness" in low):
            text += ("\n  >> Это блокировка модерации провайдера (ByteDance/fal), "
                     "а не сбой ноды: контент отклонён из-за образа реального "
                     "человека / переноса личности. Повтор не поможет — модель "
                     "не обрабатывает такой материал. Используй синтетических "
                     "персонажей или контент, который проходит их проверку.")
        if "could not get lora" in low or "failed to download file" in low:
            text += ("\n  >> fal не смог скачать LoRA по ссылке. Чаще всего это "
                     "закрытый (gated) репозиторий Hugging Face: файл отдаётся "
                     "только после входа в аккаунт и принятия лицензии, а у "
                     "серверов fal токена нет — они получают 403. Скачай "
                     ".safetensors вручную, положи в ComfyUI/models/loras и "
                     "выбери его в слоте lora_N — нода зальёт файл в fal сама. "
                     "Если файл тяжелее 1 ГБ, прогони его через ноду "
                     "LoRA Convert (fp16 / меньший ранг).")
        return text

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


def _download(url, prefix, ext=None):
    out_dir = folder_paths.get_output_directory() if folder_paths else tempfile.gettempdir()
    os.makedirs(out_dir, exist_ok=True)
    if not ext:
        # расширение берём из URL (без query) — mp4/webm/mov/gif
        ext = os.path.splitext(url.split("?")[0].split("#")[0])[1].lower()
        if ext not in (".mp4", ".webm", ".mov", ".gif", ".mkv"):
            ext = ".mp4"
    idx = 0
    while True:
        path = os.path.join(out_dir, f"{prefix}_{idx:05d}{ext}")
        if not os.path.exists(path):
            break
        idx += 1
    import time
    last = None
    for i in range(5):
        try:
            resp = requests.get(url, stream=True, timeout=(15, 300))
            resp.raise_for_status()
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            return path
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            last = e
            wait = min(20, 3 * (i + 1))
            print(f"[fal] сбой загрузки видео ({type(e).__name__}), "
                  f"повтор {i + 1}/5 через {wait} с")
            time.sleep(wait)
    raise RuntimeError(
        f"Видео сгенерировано, но не скачалось с CDN fal ({type(last).__name__}). "
        f"Ссылка (выход video_url): {url}")


def _finish(url, prefix, ext=None):
    path = _download(url, prefix, ext)
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


def _download_bytes(url, attempts=5):
    """Скачивает URL с повтором на таймаутах/обрывах CDN fal."""
    import time
    last = None
    for i in range(attempts):
        try:
            resp = requests.get(url, timeout=(15, 120), stream=True)
            resp.raise_for_status()
            return resp.content
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            last = e
            wait = min(20, 3 * (i + 1))
            print(f"[fal] сбой загрузки результата ({type(e).__name__}), "
                  f"повтор {i + 1}/{attempts} через {wait} с")
            time.sleep(wait)
    raise last


def _download_images_as_tensor(urls):
    """Скачивает картинки и собирает IMAGE-батч ComfyUI. Если CDN fal не
    отдаёт файл после повторов — ошибка со ссылками, чтобы забрать вручную
    (результат уже сгенерирован и оплачен)."""
    import torch
    pils = []
    for u in urls:
        try:
            data = _download_bytes(u)
        except Exception as e:
            raise RuntimeError(
                "Результат сгенерирован, но не скачался с CDN fal "
                f"({type(e).__name__}). Файлы доступны по ссылкам (выход "
                f"image_urls / открой в браузере):\n  " + "\n  ".join(urls)
            ) from e
        pils.append(Image.open(io.BytesIO(data)).convert("RGB"))
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
# Ideogram (V4 text-to-image, V3 edit/inpaint)
# ---------------------------------------------------------------------------

IDEOGRAM_SIZES = ["square_hd", "square", "portrait_4_3", "portrait_16_9",
                  "landscape_4_3", "landscape_16_9"]
IDEOGRAM_SPEED = ["BALANCED", "TURBO", "QUALITY"]
_IDEOGRAM_RATE = {"TURBO": 0.03, "BALANCED": 0.06, "QUALITY": 0.10}  # $ за МП


def _mask_to_bw_url(mask):
    """MASK ComfyUI (B,H,W, 1=зона правки) -> ч/б RGB PNG, белое = править."""
    arr = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[0]
    g = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1)
    return _upload_pil(Image.fromarray(rgb, mode="RGB"))


def _ideogram_cost(size, cw, ch, speed, num_images, expand):
    mp = (cw * ch / 1e6) if (cw > 0 and ch > 0) else 1.0  # пресеты ~1 МП
    cost = _IDEOGRAM_RATE.get(speed, 0.06) * mp * num_images
    if expand:
        cost += 0.03
    return cost


class IdeogramImage:
    """Ideogram через fal. Без картинки — text-to-image (V4). С подключёнными
    image + mask — инпейнт/редактирование (V3 edit): белое на маске = зона правки."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image_size": (IDEOGRAM_SIZES, {"default": "square_hd"}),
                "rendering_speed": (IDEOGRAM_SPEED, {"default": "BALANCED"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "enable_prompt_expansion": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "MagicPrompt — дораскрытие промпта (+$0.03)"}),
                "style_preset": ("STRING", {
                    "default": "",
                    "tooltip": "Только для edit (V3): напр. OIL_PAINTING, "
                               "WATERCOLOR, POP_ART; пусто = без пресета"}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "image_urls")
    FUNCTION = "generate"
    CATEGORY = "fal/Ideogram"

    def generate(self, prompt, image_size, rendering_speed, num_images,
                 image=None, mask=None, enable_prompt_expansion=True,
                 style_preset="", custom_width=0, custom_height=0, seed=-1):
        _require_deps()
        if image is not None:
            if mask is None:
                raise RuntimeError(
                    "Для редактирования Ideogram (V3 edit) нужна маска: подключи "
                    "MASK (белое = зона, которую перерисовать). Без картинки — "
                    "это text-to-image V4.")
            args = {
                "prompt": prompt,
                "image_url": _upload_image_input(image, 1)[0],
                "mask_url": _mask_to_bw_url(mask),
                "rendering_speed": rendering_speed,
                "num_images": num_images,
                "expand_prompt": enable_prompt_expansion,
            }
            if style_preset.strip():
                args["style_preset"] = style_preset.strip()
            _seed_arg(args, seed)
            endpoint = "fal-ai/ideogram/v3/edit"
        else:
            size = ({"width": custom_width, "height": custom_height}
                    if custom_width > 0 and custom_height > 0 else image_size)
            args = {
                "prompt": prompt,
                "image_size": size,
                "rendering_speed": rendering_speed,
                "num_images": num_images,
                "enable_prompt_expansion": enable_prompt_expansion,
                "output_format": "png",
            }
            _seed_arg(args, seed)
            endpoint = "ideogram/v4"
        cost = _ideogram_cost(image_size, custom_width, custom_height,
                              rendering_speed, num_images, enable_prompt_expansion)
        print(f"[fal {endpoint}] ориентировочная стоимость: ~${cost:.3f} "
              f"({num_images} шт, {rendering_speed})")
        result = _run_request(endpoint, args, est_seconds=15 * num_images)
        img_urls = [i["url"] for i in (result or {}).get("images", [])
                    if i.get("url")]
        if not img_urls:
            raise RuntimeError(f"fal не вернул изображения: {result}")
        return _download_images_as_tensor(img_urls), "\n".join(img_urls)


# ---------------------------------------------------------------------------
# Seedream 5.0 Pro (ByteDance)
# ---------------------------------------------------------------------------

SEEDREAM_SIZES = ["auto_2K", "auto_1K", "square_hd", "square",
                  "portrait_4_3", "portrait_16_9", "landscape_4_3",
                  "landscape_16_9"]


def _seedream_cost(size, cw, ch, num_images, n_inputs=0):
    """Seedream 5 Pro: ≤1536² — $0.0675, до 2048² — $0.135; edit +$0.0045/вход."""
    if cw > 0 and ch > 0:
        big = cw * ch > 1536 * 1536
    else:
        big = size == "auto_2K"
    per = (0.135 if big else 0.0675) + 0.0045 * n_inputs
    return per * num_images


class SeedreamV5Pro:
    """Seedream 5.0 Pro (ByteDance) через fal. Без картинок — text-to-image,
    с подключёнными image_1..image_4 / extra_image_urls — edit (до 10 референсов)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image_size": (SEEDREAM_SIZES, {"default": "auto_2K"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "extra_image_urls": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "image_urls")
    FUNCTION = "generate"
    CATEGORY = "fal/Seedream"

    def generate(self, prompt, image_size, num_images, image_1=None, image_2=None,
                 image_3=None, image_4=None, custom_width=0, custom_height=0,
                 seed=-1, extra_image_urls=""):
        _require_deps()
        urls = []
        for img in (image_1, image_2, image_3, image_4):
            if img is not None and len(urls) < 10:
                urls += _upload_image_input(img, 10 - len(urls))
        urls += _resolve_media_list(extra_image_urls, 10 - len(urls))

        size = ({"width": custom_width, "height": custom_height}
                if custom_width > 0 and custom_height > 0 else image_size)
        args = {
            "prompt": prompt,
            "image_size": size,
            "num_images": num_images,
            "output_format": "png",
        }
        _seed_arg(args, seed)
        if urls:
            args["image_urls"] = urls
            endpoint = "bytedance/seedream/v5/pro/edit"
        else:
            endpoint = "bytedance/seedream/v5/pro/text-to-image"
        cost = _seedream_cost(image_size, custom_width, custom_height,
                              num_images, len(urls))
        print(f"[fal {endpoint}] ориентировочная стоимость: ~${cost:.3f} "
              f"({num_images} шт, референсов: {len(urls)})")
        result = _run_request(endpoint, args, est_seconds=25 * num_images)
        img_urls = [i["url"] for i in (result or {}).get("images", [])
                    if i.get("url")]
        if not img_urls:
            raise RuntimeError(f"fal не вернул изображения: {result}")
        return _download_images_as_tensor(img_urls), "\n".join(img_urls)


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
# Kling (Kuaishou) — видео
# ---------------------------------------------------------------------------

# model -> (версия, тир, поддержка text-to-video, поддержка нативного аудио)
KLING_MODELS = {
    "2.6 pro": ("v2.6", "pro", False, True),
    "2.1 master": ("v2.1", "master", True, False),
    "2.1 pro": ("v2.1", "pro", True, False),
    "2.1 standard": ("v2.1", "standard", True, False),
}


class KlingVideo:
    """Kling (Kuaishou) через fal. Без картинки — text-to-video (2.1),
    с подключённой image — image-to-video (+end-кадр). Kling 2.6 Pro —
    только image-to-video, зато с нативным аудио."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (list(KLING_MODELS), {"default": "2.6 pro"}),
                "duration": (["5", "10"], {"default": "5"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
                "generate_audio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Нативное аудио — только Kling 2.6 Pro (цена x2)"}),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blur, distort, and low quality"}),
                "cfg_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                                        "step": 0.05}),
            },
        }

    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = "fal/Kling"

    def generate(self, prompt, model, duration, image=None, end_image=None,
                 aspect_ratio="16:9", generate_audio=False,
                 negative_prompt="blur, distort, and low quality", cfg_scale=0.5):
        _require_deps()
        ver, tier, t2v_ok, audio_ok = KLING_MODELS[model]
        args = {"prompt": prompt, "duration": duration, "cfg_scale": cfg_scale}
        if negative_prompt.strip():
            args["negative_prompt"] = negative_prompt

        if image is not None:
            args["image_url"] = _upload_image_input(image, 1)[0]
            if end_image is not None:
                args["tail_image_url"] = _upload_image_input(end_image, 1)[0]
            if audio_ok and generate_audio:
                args["generate_audio"] = True
            endpoint = f"fal-ai/kling-video/{ver}/{tier}/image-to-video"
        else:
            if not t2v_ok:
                raise RuntimeError(
                    f"Kling {model} работает только в image-to-video — подключи "
                    f"image. Для text-to-video выбери 2.1 master/pro/standard.")
            args["aspect_ratio"] = aspect_ratio
            endpoint = f"fal-ai/kling-video/{ver}/{tier}/text-to-video"

        secs = int(duration)
        if ver == "v2.6":  # известный тариф
            per = 0.14 if (audio_ok and generate_audio) else 0.07
            cost = f"~${per * secs:.2f}"
        else:
            cost = "тариф см. на fal"
        print(f"[fal {endpoint}] ориентировочная стоимость: {cost} ({secs} с)")
        result = _run_request(endpoint, args, est_seconds=secs * 30 + 60)
        video = (result or {}).get("video") or {}
        if not video.get("url"):
            raise RuntimeError(f"fal не вернул видео: {result}")
        return _finish(video["url"], "kling")


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


def _normalize_lora_url(url):
    """Ссылка на страницу файла -> ссылка на сам файл (Hugging Face blob/resolve)."""
    if "huggingface.co" in url and "/blob/" in url:
        url = url.replace("/blob/", "/resolve/", 1)
    return url


def _check_lora_url(url):
    """Проверяем доступность LoRA ДО отправки задания: fal качает файл со своей
    стороны без токенов, и закрытые репозитории Hugging Face отдают ему 403.
    Лучше упасть здесь с понятным текстом, чем через минуту очереди."""
    try:
        r = requests.head(url, allow_redirects=True, timeout=20)
        if r.status_code in (405, 501):  # HEAD не поддержан — пробуем поток
            r = requests.get(url, stream=True, timeout=20)
            r.close()
        code, ctype = r.status_code, (r.headers.get("Content-Type") or "").lower()
    except requests.RequestException:
        return  # сети нет или хост капризничает — не мешаем, решит fal
    if code in (401, 403):
        raise RuntimeError(
            f"LoRA по ссылке недоступна без авторизации (HTTP {code}):\n  {url}\n"
            f"Скорее всего это закрытый (gated) репозиторий Hugging Face — файл "
            f"отдаётся только после входа и принятия лицензии. У серверов fal "
            f"токена нет, поэтому скачать они не смогут.\n"
            f"Что делать: открой страницу модели, прими условия, скачай "
            f".safetensors вручную и положи в ComfyUI/models/loras — затем выбери "
            f"его в слоте lora_N (нода сама зальёт файл в fal storage). Если файл "
            f"тяжелее 1 ГБ, сначала прогони его через ноду LoRA Convert.")
    if code == 404:
        raise RuntimeError(f"LoRA по ссылке не найдена (HTTP 404):\n  {url}")
    if ctype.startswith("text/html"):
        raise RuntimeError(
            f"По ссылке отдаётся веб-страница, а не файл весов:\n  {url}\n"
            f"Нужна прямая ссылка на .safetensors (на Hugging Face это кнопка "
            f"download или адрес вида .../resolve/main/имя.safetensors).")


def _upload_lora(ref):
    """ref — URL или локальный путь к .safetensors. URL проверяется и уходит в
    fal как есть, локальный файл грузится в fal storage (кэш по mtime/размеру)."""
    if ref.lower().startswith(("http://", "https://")):
        ref = _normalize_lora_url(ref)
        _check_lora_url(ref)
        return ref
    if ref.lower().startswith("data:"):
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

# пиксели пресетов fal (для оценки стоимости по мегапикселям)
_PRESET_PX = {
    "square_hd": 1024 * 1024, "square": 512 * 512,
    "landscape_4_3": 1024 * 768, "portrait_4_3": 768 * 1024,
    "landscape_16_9": 1024 * 576, "portrait_16_9": 576 * 1024,
}
_FLUX2_RATE_MP = 0.021  # $ за мегапиксель выхода


def _img_size(preset, cw, ch):
    return {"width": cw, "height": ch} if cw > 0 and ch > 0 else preset


def _preset_px(preset, cw, ch):
    if cw > 0 and ch > 0:
        return cw * ch
    return _PRESET_PX.get(preset, 1024 * 1024)


class LoraConvert:
    """Локальный конвертер LoRA (без fal): кастует веса в fp16/bf16 и по желанию
    снижает ранг через SVD, чтобы уложиться в лимит fal (1 ГБ). Поддерживает
    kohya (lora_down/lora_up/alpha) и PEFT (lora_A/lora_B, adapter_model.safetensors).
    Результат сохраняется в models/loras; путь можно скормить в lora_*_url."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora": (_lora_choices(), {"default": "none"}),
                "dtype": (["fp16", "bf16"], {"default": "fp16"}),
                "target_rank": ("INT", {
                    "default": 0, "min": 0, "max": 320,
                    "tooltip": "0 = не трогать ранг (только fp16). "
                               ">0 = снизить ранг линейных слоёв через SVD"}),
            },
            "optional": {
                "lora_path": ("STRING", {
                    "default": "",
                    "tooltip": "Путь к .safetensors; перекрывает выбор из списка"}),
                "output_name": ("STRING", {
                    "default": "",
                    "tooltip": "Имя результата без пути; пусто = авто"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_path",)
    FUNCTION = "convert"
    CATEGORY = "fal/LoRA"
    OUTPUT_NODE = True

    # (down_suffix, up_suffix, alpha_suffix|None)
    _PAIRS = [(".lora_down.weight", ".lora_up.weight", ".alpha"),
              (".lora_A.weight", ".lora_B.weight", None)]

    def _reduce(self, down, up, target_rank):
        import torch
        if down.dim() != 2 or up.dim() != 2:
            return down, up, down.shape[0], False  # conv/4D — не трогаем
        r = down.shape[0]
        if target_rank <= 0 or target_rank >= r:
            return down, up, r, False
        M = up.float() @ down.float()                 # [out, in]
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        r2 = min(target_rank, S.shape[0])
        s = torch.sqrt(S[:r2])
        up2 = (U[:, :r2] * s.unsqueeze(0))            # [out, r2]
        down2 = (s.unsqueeze(1) * Vh[:r2, :])         # [r2, in]
        return down2, up2, r2, True

    def convert(self, lora, dtype, target_rank, lora_path="", output_name=""):
        import torch
        from safetensors.torch import load_file, save_file

        src = lora_path.strip()
        if not src:
            if not lora or lora == "none":
                raise RuntimeError("Не выбрана LoRA: укажи файл в списке или lora_path")
            import folder_paths
            src = folder_paths.get_full_path("loras", lora) or lora
        if not os.path.isfile(src):
            raise RuntimeError(f"Файл LoRA не найден: {src}")

        state = load_file(src)
        keys = set(state.keys())
        td = torch.float16 if dtype == "fp16" else torch.bfloat16
        reduced_pairs = 0

        if target_rank > 0:
            for down_sfx, up_sfx, alpha_sfx in self._PAIRS:
                for k in list(keys):
                    if not k.endswith(down_sfx):
                        continue
                    prefix = k[: -len(down_sfx)]
                    up_key = prefix + up_sfx
                    if up_key not in state:
                        continue
                    old_r = int(state[k].shape[0])  # до замены!
                    down2, up2, r2, changed = self._reduce(
                        state[k], state[up_key], target_rank)
                    if not changed:
                        continue
                    state[k] = down2
                    state[up_key] = up2
                    reduced_pairs += 1
                    # kohya: alpha/rank должен остаться прежним
                    if alpha_sfx:
                        ak = prefix + alpha_sfx
                        if ak in state:
                            state[ak] = (state[ak].float() * r2 / old_r)

        # каст всех float-тензоров в целевой тип
        for k in list(state.keys()):
            if state[k].is_floating_point():
                state[k] = state[k].to(td).contiguous()

        loras_dir = self._loras_dir()
        base = os.path.splitext(os.path.basename(src))[0]
        if output_name.strip():
            out_name = output_name.strip()
            if not out_name.endswith(".safetensors"):
                out_name += ".safetensors"
        else:
            tag = f"_{dtype}" + (f"_rank{target_rank}" if reduced_pairs else "")
            out_name = f"{base}{tag}.safetensors"
        out_path = os.path.join(loras_dir, out_name)
        save_file(state, out_path)

        new_size = os.path.getsize(out_path)
        old_size = os.path.getsize(src)
        print(f"[LoRA convert] {os.path.basename(src)} "
              f"{old_size/1024**2:.0f} МБ -> {out_name} {new_size/1024**2:.0f} МБ "
              f"({dtype}, слоёв со сниженным рангом: {reduced_pairs})")
        if new_size > _FAL_LORA_LIMIT:
            print(f"[LoRA convert] ВНИМАНИЕ: результат всё ещё > 1 ГБ — "
                  f"снизь target_rank (например 32–64)")
        return (out_path,)

    @staticmethod
    def _loras_dir():
        try:
            import folder_paths
            dirs = folder_paths.get_folder_paths("loras")
            if dirs:
                os.makedirs(dirs[0], exist_ok=True)
                return dirs[0]
        except Exception:
            pass
        d = os.path.join(tempfile.gettempdir(), "loras_out")
        os.makedirs(d, exist_ok=True)
        return d


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


class Flux2LoraImage:
    """FLUX.2 [dev] с пользовательскими LoRA. Без картинок — text-to-image;
    с подключёнными image_1..image_3 — edit (до 3 референсов).
    ВНИМАНИЕ: LoRA для FLUX.2 не совместимы с FLUX.1 — нужны обученные под FLUX.2."""

    @classmethod
    def INPUT_TYPES(cls):
        opt = {
            "image_1": ("IMAGE",),
            "image_2": ("IMAGE",),
            "image_3": ("IMAGE",),
            "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 60}),
            "guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0,
                                         "step": 0.1}),
            "acceleration": (["none", "regular", "high"], {"default": "regular"}),
            "custom_width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32}),
            "custom_height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32}),
            "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
        }
        opt.update(_lora_slot_inputs(3))
        opt["extra_image_urls"] = ("STRING", {"multiline": True, "default": ""})
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

    def generate(self, prompt, image_size, num_images, image_1=None, image_2=None,
                 image_3=None, num_inference_steps=28, guidance_scale=2.5,
                 acceleration="regular", custom_width=0, custom_height=0,
                 seed=-1, extra_image_urls="", **kw):
        _require_deps()
        loras = _build_loras(kw, 3)
        urls = []
        for img in (image_1, image_2, image_3):
            if img is not None and len(urls) < 3:
                urls += _upload_image_input(img, 3 - len(urls))
        urls += _resolve_media_list(extra_image_urls, 3 - len(urls))
        args = {
            "prompt": prompt,
            "image_size": _img_size(image_size, custom_width, custom_height),
            "num_images": num_images,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "acceleration": acceleration,
            "output_format": "png",
        }
        if loras:
            args["loras"] = loras
        _seed_arg(args, seed)
        if urls:
            args["image_urls"] = urls
            endpoint = "fal-ai/flux-2/lora/edit"
        else:
            endpoint = "fal-ai/flux-2/lora"
        mp = _preset_px(image_size, custom_width, custom_height) / 1e6
        cost = _FLUX2_RATE_MP * mp * num_images
        print(f"[fal {endpoint}] картинок: {len(urls)}, LoRA: {len(loras)} шт, "
              f"ориентировочная стоимость: ~${cost:.3f} ({mp:.2f} МП x{num_images})")
        result = _run_request(endpoint, args, est_seconds=15 * num_images)
        out = [i["url"] for i in (result or {}).get("images", []) if i.get("url")]
        if not out:
            raise RuntimeError(f"fal не вернул изображения: {result}")
        return _download_images_as_tensor(out), "\n".join(out)


class FluxLoraTrainer:
    """Обучение FLUX LoRA на fal из IMAGE-батча. fast — универсальный,
    portrait — под лица/людей. Кадры сохраняются в PNG, при наличии — капшны
    (по строке на кадр), зипуются и грузятся в fal. Готовая LoRA скачивается
    в models/loras и сразу пригодна для ноды FLUX.1 LoRA.
    ВНИМАНИЕ: обучение платное и идёт несколько минут."""

    _TRAINERS = {
        "fast": "fal-ai/flux-lora-fast-training",
        "portrait": "fal-ai/flux-lora-portrait-trainer",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "trigger": ("STRING", {"default": "",
                                       "tooltip": "слово-активатор (имя персонажа/стиля)"}),
                "trainer": (list(cls._TRAINERS), {"default": "fast"}),
                "steps": ("INT", {"default": 1000, "min": 100, "max": 6000,
                                  "tooltip": "fast ~1000, portrait ~2500"}),
            },
            "optional": {
                "captions": ("STRING", {"multiline": True, "default": "",
                                        "tooltip": "по строке на кадр; пусто = "
                                                   "авто-капшн (или только trigger)"}),
                "is_style": ("BOOLEAN", {"default": False,
                                         "tooltip": "fast: обучение стиля, "
                                                    "без масок/авто-капшнов"}),
                "create_masks": ("BOOLEAN", {"default": True,
                                             "tooltip": "сегментация субъекта"}),
                "learning_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.01,
                                            "step": 0.00001,
                                            "tooltip": "0 = дефолт тренера"}),
                "output_name": ("STRING", {"default": "",
                                           "tooltip": "имя .safetensors; пусто = авто"}),
                "auto_download": ("BOOLEAN", {"default": True,
                                              "tooltip": "скачать LoRA в models/loras"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("lora_path", "lora_url", "config_url")
    FUNCTION = "train"
    CATEGORY = "fal/Train"
    OUTPUT_NODE = True

    def train(self, images, trigger, trainer, steps, captions="",
              is_style=False, create_masks=True, learning_rate=0.0,
              output_name="", auto_download=True):
        _require_deps()
        import zipfile
        pils = _tensor_batch_to_pil(images)
        n = len(pils)
        min_n = 10 if trainer == "portrait" else 4
        if n < min_n:
            print(f"[fal train] предупреждение: {n} кадров — тренер '{trainer}' "
                  f"рекомендует минимум {min_n}. Качество может пострадать.")
        cap_lines = [c.strip() for c in (captions or "").splitlines()]

        tmpdir = tempfile.mkdtemp(prefix="flux_train_")
        zip_path = os.path.join(tempfile.gettempdir(),
                                f"flux_train_{os.getpid()}.zip")
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for i, img in enumerate(pils):
                    name = f"img_{i:03d}"
                    p = os.path.join(tmpdir, name + ".png")
                    img.save(p, "PNG")
                    z.write(p, name + ".png")
                    if i < len(cap_lines) and cap_lines[i]:
                        cp = os.path.join(tmpdir, name + ".txt")
                        with open(cp, "w", encoding="utf-8") as f:
                            f.write(cap_lines[i])
                        z.write(cp, name + ".txt")
            print(f"[fal train] загружаю датасет: {n} кадров, "
                  f"капшнов: {sum(1 for c in cap_lines[:n] if c)}")
            data_url = fal_client.upload_file(zip_path)
        finally:
            for f in os.listdir(tmpdir):
                try:
                    os.unlink(os.path.join(tmpdir, f))
                except OSError:
                    pass
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass
            try:
                os.unlink(zip_path)
            except OSError:
                pass

        endpoint = self._TRAINERS[trainer]
        args = {"images_data_url": data_url, "steps": steps}
        if trainer == "portrait":
            if trigger.strip():
                args["trigger_phrase"] = trigger.strip()
            if learning_rate > 0:
                args["learning_rate"] = learning_rate
            if create_masks:
                args["create_masks"] = True
        else:  # fast
            if trigger.strip():
                args["trigger_word"] = trigger.strip()
            args["is_style"] = is_style
            if not is_style:
                args["create_masks"] = create_masks
            if learning_rate > 0:
                args["learning_rate"] = learning_rate

        print(f"[fal {endpoint}] старт обучения: {steps} шагов "
              f"(это займёт несколько минут, задача платная)")
        result = _run_request(endpoint, args, est_seconds=steps * 0.6 + 180)
        lora = (result or {}).get("diffusers_lora_file") or {}
        lora_url = lora.get("url")
        cfg_url = ((result or {}).get("config_file") or {}).get("url", "")
        if not lora_url:
            raise RuntimeError(f"fal не вернул LoRA: {result}")

        local_path = ""
        if auto_download:
            base = output_name.strip() or (trigger.strip() or "flux") + f"_{trainer}"
            if not base.endswith(".safetensors"):
                base += ".safetensors"
            loras_dir = LoraConvert._loras_dir()
            local_path = os.path.join(loras_dir, base)
            data = _download_bytes(lora_url)
            with open(local_path, "wb") as f:
                f.write(data)
            print(f"[fal train] LoRA сохранена: {local_path} "
                  f"({len(data)/1e6:.0f} МБ)")
        return (local_path, lora_url, cfg_url)


class QwenImageEditLora:
    """Qwen-Image Edit Plus (2509) с пользовательскими LoRA (до 3). Принимает
    несколько картинок (image + image_2 + image_3 + URL) — редактирование и
    композиция по нескольким референсам. В отличие от закрытой Qwen Image Max,
    здесь свои LoRA."""

    @classmethod
    def INPUT_TYPES(cls):
        opt = {
            "image_2": ("IMAGE",),
            "image_3": ("IMAGE",),
            "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 60}),
            "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0,
                                         "step": 0.1}),
            "acceleration": (["none", "regular"], {"default": "none"}),
            "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
            "custom_width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32}),
            "custom_height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32}),
            "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
        }
        opt.update(_lora_slot_inputs(3))
        opt["extra_image_urls"] = ("STRING", {"multiline": True, "default": ""})
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

    def generate(self, image, prompt, image_size, image_2=None, image_3=None,
                 negative_prompt="", num_inference_steps=30, guidance_scale=4.0,
                 acceleration="none", num_images=1, custom_width=0,
                 custom_height=0, seed=-1, extra_image_urls="", **kw):
        _require_deps()
        loras = _build_loras(kw, 3)
        urls = []
        for img in (image, image_2, image_3):
            if img is not None and len(urls) < 10:
                urls += _upload_image_input(img, 10 - len(urls))
        urls += _resolve_media_list(extra_image_urls, 10 - len(urls))
        args = {
            "prompt": prompt,
            "image_urls": urls,
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
        print(f"[fal fal-ai/qwen-image-edit-plus-lora] картинок: {len(urls)}, "
              f"LoRA: {len(loras)} шт")
        result = _run_request("fal-ai/qwen-image-edit-plus-lora", args,
                              est_seconds=20 * num_images)
        out = [i["url"] for i in (result or {}).get("images", []) if i.get("url")]
        if not out:
            raise RuntimeError(f"fal не вернул изображения: {result}")
        return _download_images_as_tensor(out), "\n".join(out)


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


# ---------------------------------------------------------------------------
# Сегментация: SAM 2 (image/video) и EVF-SAM (по тексту)
# ---------------------------------------------------------------------------

def _parse_points(text):
    """Строки 'x,y[,label[,frame]]' -> список PointPrompt (label 1=объект,0=фон)."""
    pts = []
    for line in (text or "").splitlines():
        line = line.strip().replace(";", ",")
        if not line:
            continue
        p = [s for s in (x.strip() for x in line.split(",")) if s != ""]
        if len(p) < 2:
            continue
        pts.append({
            "x": int(float(p[0])), "y": int(float(p[1])),
            "label": int(float(p[2])) if len(p) >= 3 else 1,
            "frame_index": int(float(p[3])) if len(p) >= 4 else 0,
        })
    return pts


def _parse_boxes(text):
    """Строки 'x_min,y_min,x_max,y_max[,frame]' -> список BoxPrompt."""
    boxes = []
    for line in (text or "").splitlines():
        line = line.strip().replace(";", ",")
        if not line:
            continue
        p = [s for s in (x.strip() for x in line.split(",")) if s != ""]
        if len(p) < 4:
            continue
        boxes.append({
            "x_min": int(float(p[0])), "y_min": int(float(p[1])),
            "x_max": int(float(p[2])), "y_max": int(float(p[3])),
            "frame_index": int(float(p[4])) if len(p) >= 5 else 0,
        })
    return boxes


def _single_url(result):
    for key in ("image", "mask", "file"):
        v = (result or {}).get(key)
        if isinstance(v, dict) and v.get("url"):
            return v["url"]
    imgs = (result or {}).get("images")
    if imgs and isinstance(imgs, list) and imgs[0].get("url"):
        return imgs[0]["url"]
    return None


def _download_mask_and_image(url):
    """Скачанный PNG -> (IMAGE (1,H,W,3), MASK (1,H,W))."""
    import torch
    data = _download_bytes(url)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.asarray(img).astype(np.float32) / 255.0
    mask = np.asarray(img.convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(rgb)[None, ...], torch.from_numpy(mask)[None, ...]


class SAM2Image:
    """SAM 2 — сегментация картинки по точкам/боксам. Точки: строки 'x,y,label'
    (label 1=объект, 0=фон); боксы: 'x_min,y_min,x_max,y_max'. Координаты — в
    пикселях входной картинки. Выход: визуализация (IMAGE) и маска (MASK)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "points": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "по строке: x,y  или  x,y,label"}),
                "boxes": ("STRING", {"multiline": True, "default": "",
                                     "tooltip": "по строке: x_min,y_min,x_max,y_max"}),
                "apply_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "True — наложить маску на картинку; "
                               "False — вернуть саму маску (для выхода MASK)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "segment"
    CATEGORY = "fal/Segment"

    def segment(self, image, points="", boxes="", apply_mask=False):
        _require_deps()
        pts, bxs = _parse_points(points), _parse_boxes(boxes)
        if not pts and not bxs:
            raise RuntimeError(
                "SAM 2 нужна хотя бы одна подсказка: добавь точку 'x,y' в points "
                "или бокс в boxes (координаты в пикселях картинки).")
        args = {"image_url": _upload_image_input(image, 1)[0],
                "apply_mask": apply_mask, "output_format": "png"}
        if pts:
            args["prompts"] = pts
        if bxs:
            args["box_prompts"] = bxs
        print(f"[fal fal-ai/sam2/image] точек: {len(pts)}, боксов: {len(bxs)}")
        result = _run_request("fal-ai/sam2/image", args, est_seconds=20)
        url = _single_url(result)
        if not url:
            raise RuntimeError(f"fal не вернул результат: {result}")
        return _download_mask_and_image(url)


class SAM2Video:
    """SAM 2 — сегментация и трекинг объекта в видео. Укажи точку/бокс на кадре
    (frame_index, по умолчанию 0), модель протянет маску через всё видео."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"video": ("VIDEO",)},
            "optional": {
                "points": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "x,y,label,frame — по строке"}),
                "boxes": ("STRING", {"multiline": True, "default": "",
                                     "tooltip": "x_min,y_min,x_max,y_max,frame"}),
                "apply_mask": ("BOOLEAN", {"default": True,
                                           "tooltip": "наложить маску на видео"}),
            },
        }

    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES
    FUNCTION = "segment"
    CATEGORY = "fal/Segment"

    def segment(self, video, points="", boxes="", apply_mask=True):
        _require_deps()
        pts, bxs = _parse_points(points), _parse_boxes(boxes)
        if not pts and not bxs:
            raise RuntimeError(
                "SAM 2 video нужна подсказка: точка 'x,y' или бокс на кадре "
                "(frame_index, по умолчанию 0).")
        url, _, _, _ = _upload_video_input(video, label="видео для сегментации",
                                           min_duration=None)
        args = {"video_url": url, "apply_mask": apply_mask}
        if pts:
            args["prompts"] = pts
        if bxs:
            args["box_prompts"] = bxs
        print(f"[fal fal-ai/sam2/video] точек: {len(pts)}, боксов: {len(bxs)}")
        result = _run_request("fal-ai/sam2/video", args, est_seconds=180)
        video_out = (result or {}).get("video") or {}
        if not video_out.get("url"):
            raise RuntimeError(f"fal не вернул видео: {result}")
        return _finish(video_out["url"], "sam2")


class EVFSAM:
    """EVF-SAM — сегментация по текстовому описанию. Напиши, что выделить
    ('hair', 'lips', 'the person', 'красная машина'). Возвращает бинарную маску.
    semantic_type=True — для частей тела/лица."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "semantic_type": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "семантический режим — для частей тела/лица"}),
                "revert_mask": ("BOOLEAN", {"default": False,
                                            "tooltip": "инвертировать маску"}),
                "fill_holes": ("BOOLEAN", {"default": False}),
                "expand_mask": ("INT", {"default": 0, "min": 0, "max": 128,
                                        "tooltip": "расширить маску на N пикселей"}),
                "blur_mask": ("INT", {"default": 0, "min": 0, "max": 99, "step": 2,
                                      "tooltip": "размытие краёв (нечётное ядро)"}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "image")
    FUNCTION = "segment"
    CATEGORY = "fal/Segment"

    def segment(self, image, prompt, negative_prompt="", semantic_type=False,
                revert_mask=False, fill_holes=False, expand_mask=0, blur_mask=0):
        _require_deps()
        args = {
            "image_url": _upload_image_input(image, 1)[0],
            "prompt": prompt,
            "mask_only": True,
            "semantic_type": semantic_type,
            "revert_mask": revert_mask,
            "fill_holes": fill_holes,
        }
        if negative_prompt.strip():
            args["negative_prompt"] = negative_prompt
        if expand_mask > 0:
            args["expand_mask"] = expand_mask
        if blur_mask > 0:
            args["blur_mask"] = blur_mask if blur_mask % 2 == 1 else blur_mask + 1
        print(f"[fal fal-ai/evf-sam] сегментация по тексту: «{prompt[:40]}»")
        result = _run_request("fal-ai/evf-sam", args, est_seconds=20)
        url = _single_url(result)
        if not url:
            raise RuntimeError(f"fal не вернул результат: {result}")
        img, mask = _download_mask_and_image(url)
        return mask, img


# ---------------------------------------------------------------------------
# LTX-2.3 (Lightricks): text/image/reference-to-video + LoRA, extend, reframe
# ---------------------------------------------------------------------------

_LTX_SIZES = ["auto", "landscape_16_9", "landscape_4_3", "square_hd", "square",
              "portrait_4_3", "portrait_16_9"]

# Стандартные пресеты ImageSize у fal (для видео обычно задают custom_width/height)
_LTX_WH = {
    "landscape_16_9": (1024, 576), "landscape_4_3": (1024, 768),
    "square_hd": (1024, 1024), "square": (512, 512),
    "portrait_4_3": (768, 1024), "portrait_16_9": (576, 1024),
}

_LTX_CAMERA = ["none", "dolly_in", "dolly_out", "dolly_left", "dolly_right",
               "jib_up", "jib_down", "static"]

# $ за мегапиксель сгенерированного видео (ширина x высота x кадры)
_LTX_RATE_MP = {"quality": 0.001605, "distilled": 0.001205}
_LTX_MODELS = ["22B quality", "22B distilled (быстрее и дешевле)"]


def _ltx_frames(n):
    """LTX работает с кадрами вида 8k+1 — подгоняем ближайшее допустимое."""
    n = max(9, int(n))
    return int(round((n - 1) / 8.0)) * 8 + 1


def _ltx_dims(video_size, cw, ch, image=None):
    """Разрешение, из которого считается цена (и что уйдёт в video_size)."""
    if cw > 0 and ch > 0:
        return int(cw), int(ch)
    if video_size == "auto":
        if image is not None:
            try:
                return int(image.shape[2]), int(image.shape[1])  # (B,H,W,C)
            except Exception:
                pass
        return 1280, 720  # оценка: реальный размер выберет модель
    return _LTX_WH.get(video_size, (1024, 576))


_LTX_OUTPUT = {
    "mp4 (H.264)": "X264 (.mp4)",
    "webm (VP9)": "VP9 (.webm)",
    "mov (ProRes 4444, без потерь)": "PRORES4444 (.mov)",
    "gif": "GIF (.gif)",
}
_LTX_OUTPUT_EXT = {"X264 (.mp4)": ".mp4", "VP9 (.webm)": ".webm",
                   "PRORES4444 (.mov)": ".mov", "GIF (.gif)": ".gif"}


def _ltx_num(args, key, value, sentinel=-1.0):
    """Кладёт число в args только если оно не равно «не трогать» (-1).
    Ноль у некоторых параметров LTX осмысленный, поэтому сентинел именно -1."""
    if value is not None and value > sentinel:
        args[key] = float(value)


def _ltx_flag(args, key, value):
    """Тумблер из трёх состояний: default / on / off."""
    if value == "on":
        args[key] = True
    elif value == "off":
        args[key] = False


def _ltx_advanced_inputs():
    """Продвинутые ручки денойзинга и сэмплера. ВСЕГДА добавляются в конец
    optional — иначе поедут widgets_values в уже сохранённых воркфлоу."""
    f = lambda tip, mx=20.0: ("FLOAT", {"default": -1.0, "min": -1.0, "max": mx,
                                        "step": 0.05, "tooltip": tip + " (-1 — по умолчанию модели)"})
    return {
        "use_multiscale": (["default", "on", "off"], {
            "default": "default",
            "tooltip": "многомасштабная генерация: лучше связность, дольше счёт"}),
        "use_restart_sampling": (["default", "on", "off"], {
            "default": "default",
            "tooltip": "подмешивание шума на каждом шаге: больше деталей, "
                       "но менее предсказуемо"}),
        "gradient_estimation_gamma": ("FLOAT", {
            "default": -1.0, "min": -1.0, "max": 10.0, "step": 0.1,
            "tooltip": "градиент денойзинга (по умолчанию 2). 0 — отключить "
                       "полностью, -1 — не трогать"}),
        "video_stg_scale": f("STG видео: пространственно-временная направляющая"),
        "video_rescaling_scale": f("баланс CFG/STG для видео (по умолчанию 0.7)", 1.0),
        "video_modality_scale": f("вес видео относительно звука (по умолчанию 3)"),
        "audio_cfg_scale": f("сила направляющей для звука (по умолчанию 7)"),
        "audio_stg_scale": f("STG звука"),
        "audio_rescaling_scale": f("баланс CFG/STG для звука (по умолчанию 0.7)", 1.0),
        "audio_modality_scale": f("вес звука относительно видео (по умолчанию 3)"),
        "distill_lora_first_pass_scale": f("distill-LoRA, первый проход (0.2)", 2.0),
        "distill_lora_second_pass_scale": f("distill-LoRA, следующие проходы (0.5)", 2.0),
        "video_output_type": (list(_LTX_OUTPUT), {
            "default": "mp4 (H.264)",
            "tooltip": "ProRes 4444 — для монтажа и композа без потерь "
                       "(файл в разы тяжелее)"}),
    }


def _ltx_apply_advanced(args, kw):
    """Переносит продвинутые параметры из kwargs в тело запроса."""
    _ltx_flag(args, "use_multiscale", kw.get("use_multiscale", "default"))
    _ltx_flag(args, "use_restart_sampling", kw.get("use_restart_sampling", "default"))
    _ltx_num(args, "gradient_estimation_gamma",
             kw.get("gradient_estimation_gamma", -1.0))
    for key in ("video_stg_scale", "video_rescaling_scale", "video_modality_scale",
                "audio_cfg_scale", "audio_stg_scale", "audio_rescaling_scale",
                "audio_modality_scale", "distill_lora_first_pass_scale",
                "distill_lora_second_pass_scale"):
        _ltx_num(args, key, kw.get(key, -1.0))
    out = _LTX_OUTPUT.get(kw.get("video_output_type", "mp4 (H.264)"), "X264 (.mp4)")
    if out != "X264 (.mp4)":
        args["video_output_type"] = out
    return _LTX_OUTPUT_EXT.get(out, ".mp4")


def _ltx_cost(w, h, frames, distilled):
    rate = _LTX_RATE_MP["distilled" if distilled else "quality"]
    return rate * (w * h * frames) / 1e6


class LTX23Video:
    """LTX-2.3 22B — видео со звуком и пользовательскими LoRA.

    Режим выбирается по подключённым входам: ничего — text-to-video,
    image — image-to-video (первый кадр), video — reference-to-video.
    LoRA-слоты работают во всех режимах (эндпоинт /lora подставляется сам).
    Цена: $0.001605 (quality) / $0.001205 (distilled) за мегапиксель
    ширина x высота x кадры."""

    @classmethod
    def INPUT_TYPES(cls):
        opt = {
            "image": ("IMAGE", {"tooltip": "первый кадр -> image-to-video"}),
            "end_image": ("IMAGE", {"tooltip": "последний кадр (необязательно)"}),
            "video": ("VIDEO", {"tooltip": "видео-референс -> reference-to-video"}),
            "audio": ("AUDIO", {"tooltip": "своя звуковая дорожка (для режима с видео)"}),
            "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            "custom_width": ("INT", {"default": 0, "min": 0, "max": 3840, "step": 8,
                                     "tooltip": "0 — брать пресет video_size"}),
            "custom_height": ("INT", {"default": 0, "min": 0, "max": 3840, "step": 8}),
            "num_inference_steps": ("INT", {
                "default": 0, "min": 0, "max": 60,
                "tooltip": "0 — по умолчанию модели (40 у quality)"}),
            "video_cfg_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0,
                                          "step": 0.1, "tooltip": "0 — по умолчанию"}),
            "image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                         "step": 0.05}),
            "video_strength": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "сила привязки к видео-референсу; меньше — больше свободы"}),
            "camera_lora": (_LTX_CAMERA, {"default": "none",
                                          "tooltip": "встроенная LoRA движения камеры"}),
            "camera_lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0,
                                            "step": 0.05}),
            "acceleration": (["default", "none", "regular", "high", "full"],
                             {"default": "default"}),
            "scheduler": (["default", "ltx2", "linear_quadratic", "beta"],
                          {"default": "default"}),
            "video_quality": (["high", "maximum", "medium", "low"], {"default": "high"}),
            "enable_prompt_expansion": ("BOOLEAN", {"default": True}),
            "enable_safety_checker": ("BOOLEAN", {"default": True}),
        }
        opt.update(_lora_slot_inputs(3))
        opt.update(_ltx_advanced_inputs())
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (_LTX_MODELS, {"default": _LTX_MODELS[0]}),
                "num_frames": ("INT", {
                    "default": 121, "min": 9, "max": 481, "step": 8,
                    "tooltip": "121 кадр @24 fps = ~5 с. Округляется до 8k+1"}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60}),
                "video_size": (_LTX_SIZES, {"default": "auto"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
            "optional": opt,
        }

    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = "fal/LTX"

    def generate(self, prompt, model, num_frames=121, fps=24, video_size="auto",
                 generate_audio=True, seed=-1, image=None, end_image=None,
                 video=None, audio=None, negative_prompt="",
                 custom_width=0, custom_height=0, num_inference_steps=0,
                 video_cfg_scale=0.0, image_strength=1.0, video_strength=1.0,
                 camera_lora="none", camera_lora_scale=1.0,
                 acceleration="default", scheduler="default",
                 video_quality="high", enable_prompt_expansion=True,
                 enable_safety_checker=True, **kw):
        _require_deps()
        distilled = "distilled" in model
        loras = _build_loras(kw, 3)
        frames = _ltx_frames(num_frames)

        args = {
            "prompt": prompt,
            "num_frames": frames,
            "fps": float(fps),
            "generate_audio": bool(generate_audio),
            "video_quality": video_quality,
            "enable_prompt_expansion": bool(enable_prompt_expansion),
            "enable_safety_checker": bool(enable_safety_checker),
        }
        if custom_width > 0 and custom_height > 0:
            args["video_size"] = {"width": int(custom_width), "height": int(custom_height)}
        elif video_size != "auto":
            args["video_size"] = video_size
        if negative_prompt.strip():
            args["negative_prompt"] = negative_prompt
        if num_inference_steps > 0:
            args["num_inference_steps"] = int(num_inference_steps)
        if video_cfg_scale > 0:
            args["video_cfg_scale"] = float(video_cfg_scale)
        if acceleration != "default":
            args["acceleration"] = acceleration
        if scheduler != "default":
            args["scheduler"] = scheduler
        if camera_lora != "none":
            args["camera_lora"] = camera_lora
            args["camera_lora_scale"] = float(camera_lora_scale)
        if loras:
            args["loras"] = loras
        ext = _ltx_apply_advanced(args, kw)
        _seed_arg(args, seed)

        base = "fal-ai/ltx-2.3-22b" + ("/distilled" if distilled else "")
        if video is not None:
            args["video_url"] = _upload_video_input(video, "видео-референс",
                                                    min_duration=0.0)[0]
            args["video_strength"] = float(video_strength)
            task = "/reference-video-to-video"
            if audio is not None:
                args["audio_url"] = _upload_audio_input(audio)
            if image is not None:
                args["image_url"] = _upload_image_input(image, 1)[0]
        elif image is not None:
            args["image_url"] = _upload_image_input(image, 1)[0]
            args["image_strength"] = float(image_strength)
            task = "/image-to-video"
        else:
            task = "/text-to-video"
        if end_image is not None:
            args["end_image_url"] = _upload_image_input(end_image, 1)[0]
        endpoint = base + task + ("/lora" if loras else "")

        w, h = _ltx_dims(video_size, custom_width, custom_height, image)
        secs = frames / max(fps, 1)
        print(f"[fal {endpoint}] {w}x{h}, {frames} кадров (~{secs:.1f} с), "
              f"LoRA: {len(loras)} шт, ориентировочная стоимость "
              f"~${_ltx_cost(w, h, frames, distilled):.2f}")
        result = _run_request(endpoint, args, est_seconds=secs * 25 + 40)
        url = ((result or {}).get("video") or {}).get("url")
        if not url:
            raise RuntimeError(f"fal не вернул видео: {result}")
        return _finish(url, "ltx23", ext)


class LTX23ExtendVideo:
    """LTX-2.3 Extend — продолжает существующее видео вперёд или назад,
    с поддержкой LoRA. Берёт последние (или первые) num_context_frames кадров
    как контекст и дорисовывает num_frames кадров."""

    @classmethod
    def INPUT_TYPES(cls):
        opt = {
            "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            "extend_direction": (["forward", "backward"], {
                "default": "forward",
                "tooltip": "forward — продолжить с конца, backward — дорисовать начало"}),
            "num_context_frames": ("INT", {
                "default": 25, "min": 1, "max": 121,
                "tooltip": "сколько кадров исходника взять как контекст"}),
            "frames_per_second": ("INT", {"default": 24, "min": 8, "max": 60}),
            "num_inference_steps": ("INT", {"default": 0, "min": 0, "max": 60,
                                            "tooltip": "0 — по умолчанию (15)"}),
            "guidance_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0,
                                         "step": 0.1, "tooltip": "0 — по умолчанию (1)"}),
            "video_quality": (["high", "maximum", "medium", "low"], {"default": "high"}),
            "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
        }
        opt.update(_lora_slot_inputs(3))
        opt.update(_ltx_advanced_inputs())
        return {
            "required": {
                "video": ("VIDEO",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "num_frames": ("INT", {
                    "default": 121, "min": 9, "max": 481, "step": 8,
                    "tooltip": "сколько кадров сгенерировать (с учётом контекста)"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
            },
            "optional": opt,
        }

    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES
    FUNCTION = "extend"
    CATEGORY = "fal/LTX"

    def extend(self, video, prompt, num_frames=121, generate_audio=True,
               negative_prompt="", extend_direction="forward",
               num_context_frames=25, frames_per_second=24,
               num_inference_steps=0, guidance_scale=0.0,
               video_quality="high", seed=-1, **kw):
        _require_deps()
        loras = _build_loras(kw, 3)
        frames = _ltx_frames(num_frames)
        url_in, dur, w, h = _upload_video_input(video, "видео", min_duration=0.0)
        args = {
            "video_url": url_in,
            "prompt": prompt,
            "num_frames": frames,
            "num_context_frames": int(num_context_frames),
            "extend_direction": extend_direction,
            "frames_per_second": int(frames_per_second),
            "generate_audio": bool(generate_audio),
            "video_quality": video_quality,
        }
        if negative_prompt.strip():
            args["negative_prompt"] = negative_prompt
        if num_inference_steps > 0:
            args["num_inference_steps"] = int(num_inference_steps)
        if guidance_scale > 0:
            args["guidance_scale"] = float(guidance_scale)
        if loras:
            args["loras"] = loras
        ext = _ltx_apply_advanced(args, kw)
        _seed_arg(args, seed)

        endpoint = "fal-ai/ltx-2.3-quality/extend-video" + ("/lora" if loras else "")
        secs = frames / max(frames_per_second, 1)
        cost = _ltx_cost(w or 1280, h or 720, frames, False)
        print(f"[fal {endpoint}] {extend_direction}, +{frames} кадров (~{secs:.1f} с), "
              f"LoRA: {len(loras)} шт, ориентировочная стоимость ~${cost:.2f}")
        result = _run_request(endpoint, args, est_seconds=secs * 25 + 40)
        url = ((result or {}).get("video") or {}).get("url")
        if not url:
            raise RuntimeError(f"fal не вернул видео: {result}")
        return _finish(url, "ltx23_extend", ext)


class LTX23Reframe:
    """LTX-2.3 Reframe — меняет соотношение сторон видео, дорисовывая кадр
    (вертикаль из горизонтали и наоборот). Максимум 60 секунд исходника."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "aspect_ratio": (["9:16", "16:9", "1:1", "4:5", "5:4"],
                                 {"default": "9:16"}),
                "resolution": (["1080p", "720p"], {"default": "1080p"}),
            },
        }

    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES
    FUNCTION = "reframe"
    CATEGORY = "fal/LTX"

    def reframe(self, video, aspect_ratio="9:16", resolution="1080p"):
        _require_deps()
        url_in, dur, w, h = _upload_video_input(video, "видео", min_duration=0.0)
        if dur is not None and dur > 60:
            raise RuntimeError(
                f"Reframe принимает максимум 60 с, а на входе {dur:.1f} с — "
                f"нарежь видео короче.")
        args = {"video_url": url_in, "aspect_ratio": aspect_ratio,
                "resolution": resolution}
        print(f"[fal fal-ai/ltx-2.3/reframe] {aspect_ratio} @ {resolution}"
              + (f", исходник {dur:.1f} с" if dur else ""))
        result = _run_request("fal-ai/ltx-2.3/reframe", args,
                              est_seconds=(dur or 5) * 20 + 40)
        url = ((result or {}).get("video") or {}).get("url")
        if not url:
            raise RuntimeError(f"fal не вернул видео: {result}")
        return _finish(url, "ltx23_reframe")


# ---------------------------------------------------------------------------
# MiniMax H3 (Hailuo 3.0): text/image-to-video + мультимодальный reference,
# нативное стерео-аудио, открытые веса -> поддержка пользовательских LoRA
# ---------------------------------------------------------------------------

_H3_RES = ["2K", "768P", "4K"]
_H3_ASPECTS = ["16:9", "21:9", "4:3", "1:1", "3:4", "9:16"]
# $ за секунду видео по разрешению
_H3_RATE = {"768P": 0.08, "2K": 0.13, "4K": 0.16}


def _h3_cost(resolution, duration):
    return _H3_RATE.get(resolution, 0.13) * max(1, int(duration))


_H3_FPS = 24  # частота у H3 фиксированная


_H3_FRAMES_INPUT = ("INT", {
    "default": 0, "min": 0, "max": 400,
    "tooltip": "0 — не использовать. Длительность в кадрах: пересчитывается в "
               "секунды по 24 fps и округляется. API принимает только целые "
               "секунды, поэтому точного совпадения с плейтом может не быть — "
               "нода напишет в лог, сколько кадров получится на самом деле"})


def _h3_duration(duration, duration_override, duration_frames=0):
    """Секунды для H3. Приоритет: duration_override (провод, секунды) ->
    duration_frames (кадры при 24 fps) -> виджет duration. Лимит 5..15 с."""
    frames = int(duration_frames or 0)
    from_frames = False
    if duration_override is not None and duration_override > 0:
        duration = int(round(duration_override))
        if frames > 0:
            print("[fal H3] заданы и duration_override, и duration_frames — "
                  "беру duration_override (секунды)")
    elif frames > 0:
        duration = int(round(frames / _H3_FPS))
        from_frames = True
    dur = max(5, min(15, int(duration)))
    if from_frames:
        out = dur * _H3_FPS
        if out == frames:
            print(f"[fal H3] {frames} кадров = ровно {dur} с при {_H3_FPS} fps")
        else:
            word = "короче" if out < frames else "длиннее"
            print(f"[fal H3] запрошено {frames} кадров -> {dur} с = {out} кадров "
                  f"при {_H3_FPS} fps, это на {abs(out - frames)} кадр(ов) {word}. "
                  f"API принимает только целые секунды — подрежь результат в "
                  f"монтаже или возьми на секунду больше с запасом")
    return dur


def _h3_collect_refs(images, videos, audios,
                     image_urls="", video_refs="", audio_refs=""):
    """Сборка мультимодальных референсов H3 / H3 Max в три массива URL.
    Лимиты одинаковые у обеих моделей: до 9 картинок, до 3 видео и до 3 аудио,
    всего не больше 12 файлов; видео суммарно не длиннее 15 с; аудио не может
    быть единственным референсом."""
    img_urls = []
    for img in images:
        if img is not None and len(img_urls) < 9:
            img_urls += _upload_image_input(img, 9 - len(img_urls))
    img_urls += _resolve_media_list(image_urls, 9 - len(img_urls))

    vid_urls, vid_total = [], 0.0
    for i, vid in enumerate(videos, 1):
        if vid is not None and len(vid_urls) < 3:
            url, d, _, _ = _upload_video_input(vid, label=f"video_{i}")
            if d:
                vid_total += d
            vid_urls.append(url)
    if vid_total > 15.0:
        raise RuntimeError(
            f"Суммарная длительность референс-видео {vid_total:.1f} с — "
            f"больше лимита H3 (2–15 с суммарно). Подрежь ролики.")
    vid_urls += _resolve_media_list(video_refs, 3 - len(vid_urls))

    aud_urls = []
    for aud in audios:
        if aud is not None and len(aud_urls) < 3:
            aud_urls.append(_upload_audio_input(aud))
    aud_urls += _resolve_media_list(audio_refs, 3 - len(aud_urls))

    if not img_urls and not vid_urls:
        raise RuntimeError(
            "H3 не принимает аудио как единственный референс: подключи "
            "хотя бы одну картинку или видео." if aud_urls else
            "Не задано ни одного референса — подключи картинку, видео "
            "или укажи ссылки в image_urls / video_refs.")
    total = len(img_urls) + len(vid_urls) + len(aud_urls)
    if total > 12:
        raise RuntimeError(
            f"Всего референсов {total}, а H3 принимает максимум 12 файлов "
            f"(из них до 9 картинок, до 3 видео и до 3 аудио).")
    return img_urls, vid_urls, aud_urls


class MinimaxH3Video:
    """MiniMax H3 (Hailuo 3.0) — 24 fps, до 4K, нативное стерео-аудио
    (музыка, диалоги, фоли, эмбиенс) прямо из промпта.

    Без картинки — text-to-video. С подключённой image — image-to-video,
    end_image задаёт последний кадр (генерация между двумя ключевыми кадрами);
    в этом режиме пропорции берутся из картинки, а не из aspect_ratio.
    Веса модели открытые, поэтому работают пользовательские LoRA.
    Цена: $0.08/с (768P), $0.13/с (2K), $0.16/с (4K)."""

    @classmethod
    def INPUT_TYPES(cls):
        opt = {
            "image": ("IMAGE", {"tooltip": "первый кадр -> image-to-video"}),
            "end_image": ("IMAGE", {"tooltip": "последний кадр (нужна image)"}),
            "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            "enable_prompt_expansion": ("BOOLEAN", {
                "default": True,
                "tooltip": "расширение промпта через VLM: помогает коротким "
                           "описаниям, мешает точным"}),
            "enable_safety_checker": ("BOOLEAN", {"default": True}),
            "duration_override": DURATION_OVERRIDE_INPUT,
        }
        opt.update(_lora_slot_inputs(3))
        opt["duration_frames"] = _H3_FRAMES_INPUT
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "звук описывается здесь же: музыка, реплики, "
                               "шумы. Для длинных сцен помогает тайминг "
                               "вида «[0–2 s] ...»"}),
                "duration": ("INT", {"default": 5, "min": 5, "max": 15}),
                "resolution": (_H3_RES, {"default": "2K"}),
                "aspect_ratio": (_H3_ASPECTS, {
                    "default": "16:9",
                    "tooltip": "игнорируется в режиме image-to-video — "
                               "пропорции берутся из картинки"}),
            },
            "optional": opt,
        }

    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = "fal/MiniMax"

    def generate(self, prompt, duration=5, resolution="2K", aspect_ratio="16:9",
                 image=None, end_image=None, seed=-1,
                 enable_prompt_expansion=True, enable_safety_checker=True,
                 duration_override=0.0, duration_frames=0, **kw):
        _require_deps()
        loras = _build_loras(kw, 3)
        dur = _h3_duration(duration, duration_override, duration_frames)
        args = {
            "prompt": prompt,
            "duration": dur,
            "resolution": resolution,
            "enable_prompt_expansion": bool(enable_prompt_expansion),
            "enable_safety_checker": bool(enable_safety_checker),
        }
        if loras:
            args["loras"] = loras
        _seed_arg(args, seed)

        if image is not None:
            args["image_url"] = _upload_image_input(image, 1)[0]
            if end_image is not None:
                args["end_image_url"] = _upload_image_input(end_image, 1)[0]
            task = "image-to-video"
        else:
            if end_image is not None:
                raise RuntimeError(
                    "end_image задаёт последний кадр и работает только вместе "
                    "с image — подключи стартовый кадр во вход image.")
            args["aspect_ratio"] = aspect_ratio
            task = "text-to-video"
        endpoint = f"minimax/h3/{task}" + ("/lora" if loras else "")

        print(f"[fal {endpoint}] {resolution}, {dur} с, LoRA: {len(loras)} шт, "
              f"ориентировочная стоимость ~${_h3_cost(resolution, dur):.2f}")
        result = _run_request(endpoint, args, est_seconds=dur * 20 + 40)
        url = ((result or {}).get("video") or {}).get("url")
        if not url:
            raise RuntimeError(f"fal не вернул видео: {result}")
        return _finish(url, "minimax_h3")


class MinimaxH3Reference:
    """MiniMax H3 Reference-to-Video — мультимодальные референсы: до 9 картинок,
    до 3 видео и до 3 аудио (всего не больше 12 файлов).

    В промпте на них ссылаются как «Image 1», «Video 1», «Audio 1» — нумерация
    идёт по порядку: сначала входы image_1..image_4, затем строки image_urls.
    Видео и аудио — по 2–15 с каждое, суммарно не больше 15 с. Аудио не может
    быть единственным референсом. Типовые задачи: перенос движения с плейта,
    замена хромакея, клонирование голоса на персонажа."""

    @classmethod
    def INPUT_TYPES(cls):
        opt = {
            "image_1": ("IMAGE",), "image_2": ("IMAGE",),
            "image_3": ("IMAGE",), "image_4": ("IMAGE",),
            "video_1": ("VIDEO",), "video_2": ("VIDEO",), "video_3": ("VIDEO",),
            "audio_1": ("AUDIO",), "audio_2": ("AUDIO",), "audio_3": ("AUDIO",),
            # однострочные: несколько ссылок разделяются запятой
            "image_urls": ("STRING", {"default": "", "tooltip":
                                      "ссылки или пути, через запятую"}),
            "video_refs": ("STRING", {"default": "", "tooltip":
                                      "ссылки или пути, через запятую"}),
            "audio_refs": ("STRING", {"default": "", "tooltip":
                                      "ссылки или пути, через запятую"}),
            "enable_prompt_expansion": ("BOOLEAN", {"default": True}),
            "enable_safety_checker": ("BOOLEAN", {"default": True}),
            "duration_override": DURATION_OVERRIDE_INPUT,
        }
        opt.update(_lora_slot_inputs(3))
        opt["duration_frames"] = _H3_FRAMES_INPUT
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "дай каждому референсу роль: «Image 1 — костюм», "
                               "«Video 1 — движение камеры»"}),
                "duration": ("INT", {"default": 5, "min": 5, "max": 15}),
                "resolution": (_H3_RES, {"default": "2K"}),
                "aspect_ratio": (["adaptive"] + _H3_ASPECTS, {
                    "default": "adaptive",
                    "tooltip": "adaptive — взять пропорции из референсов"}),
            },
            "optional": opt,
        }

    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = "fal/MiniMax"

    def generate(self, prompt, duration=5, resolution="2K",
                 aspect_ratio="adaptive",
                 image_1=None, image_2=None, image_3=None, image_4=None,
                 video_1=None, video_2=None, video_3=None,
                 audio_1=None, audio_2=None, audio_3=None,
                 image_urls="", video_refs="", audio_refs="",
                 enable_prompt_expansion=True, enable_safety_checker=True,
                 duration_override=0.0, duration_frames=0, **kw):
        _require_deps()
        loras = _build_loras(kw, 3)
        dur = _h3_duration(duration, duration_override, duration_frames)

        img_urls, vid_urls, aud_urls = _h3_collect_refs(
            (image_1, image_2, image_3, image_4),
            (video_1, video_2, video_3), (audio_1, audio_2, audio_3),
            image_urls, video_refs, audio_refs)

        args = {
            "prompt": prompt,
            "duration": dur,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "enable_prompt_expansion": bool(enable_prompt_expansion),
            "enable_safety_checker": bool(enable_safety_checker),
        }
        if img_urls:
            args["reference_image_urls"] = img_urls
        if vid_urls:
            args["reference_video_urls"] = vid_urls
        if aud_urls:
            args["reference_audio_urls"] = aud_urls
        if loras:
            args["loras"] = loras

        endpoint = "minimax/h3/reference-to-video" + ("/lora" if loras else "")
        print(f"[fal {endpoint}] {resolution}, {dur} с, референсы: "
              f"{len(img_urls)} картинок / {len(vid_urls)} видео / "
              f"{len(aud_urls)} аудио, LoRA: {len(loras)} шт, "
              f"ориентировочная стоимость ~${_h3_cost(resolution, dur):.2f}")
        result = _run_request(endpoint, args, est_seconds=dur * 20 + 40)
        url = ((result or {}).get("video") or {}).get("url")
        if not url:
            raise RuntimeError(f"fal не вернул видео: {result}")
        return _finish(url, "minimax_h3_ref")


# ---------------------------------------------------------------------------
# MiniMax H3 Max — пост-тренированный fal вариант H3: сильнее следует промпту,
# быстрее реального времени. Turbo — ещё быстрее и вдвое дешевле.
# ---------------------------------------------------------------------------

_H3MAX_RES = ["768P", "480P"]
_H3MAX_MODELS = ["H3 Max", "H3 Max Turbo (быстрее и вдвое дешевле)"]
_H3MAX_EXPANSION = ["balanced", "quality"]
_H3MAX_ASPECTS = _H3_ASPECTS

# $ за секунду видео (базовый тариф после стартовой промо-скидки)
_H3MAX_RATE = {
    "max": {"480P": 0.05, "768P": 0.08},
    "turbo": {"480P": 0.025, "768P": 0.04},
}

# у H3 Max, в отличие от H3, есть ещё и текст раскрытого промпта на выходе
_H3MAX_RETURN_TYPES = RETURN_TYPES + ("STRING",)
_H3MAX_RETURN_NAMES = RETURN_NAMES + ("expanded_prompt",)


def _h3max_tier(model):
    return "turbo" if "Turbo" in model else "max"


def _h3max_cost(model, resolution, duration):
    table = _H3MAX_RATE[_h3max_tier(model)]
    return table.get(resolution, table["768P"]) * max(1, int(duration))


def _h3max_finish(result, prefix, endpoint):
    """Общий разбор ответа H3 Max: видео + раскрытый промпт."""
    url = ((result or {}).get("video") or {}).get("url")
    if not url:
        raise RuntimeError(f"fal не вернул видео: {result}")
    expanded = (result or {}).get("expanded_prompt") or ""
    if expanded:
        print(f"[fal {endpoint}] промпт после раскрытия: {expanded[:300]}"
              f"{'…' if len(expanded) > 300 else ''}")
    timings = (result or {}).get("timings")
    if timings:
        print(f"[fal {endpoint}] тайминги: {timings}")
    return _finish(url, prefix) + (expanded,)


class MinimaxH3MaxVideo:
    """MiniMax H3 Max — пост-тренированный fal вариант H3: лучше следует
    промпту и аккуратнее по картинке, считается быстрее реального времени
    (5 с в 768P — за пару секунд). Звук синхронный, как у обычного H3.

    Без картинки — text-to-video, с подключённой image — image-to-video,
    end_image задаёт последний кадр. В режиме с картинкой пропорции берутся
    из неё, aspect_ratio не отправляется.

    Отличия от обычного H3: только 480P и 768P (нет 2K/4K), нет поддержки
    LoRA, вместо тумблера расширения промпта — режим balanced/quality.
    Цена: $0.05/$0.08 за секунду (480P/768P), Turbo — вдвое дешевле."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "звук описывается здесь же: музыка, реплики, шумы"}),
                "model": (_H3MAX_MODELS, {"default": _H3MAX_MODELS[0]}),
                "duration": ("INT", {"default": 5, "min": 5, "max": 15}),
                "resolution": (_H3MAX_RES, {"default": "768P"}),
                "aspect_ratio": (_H3MAX_ASPECTS, {
                    "default": "16:9",
                    "tooltip": "игнорируется в режиме image-to-video — "
                               "пропорции берутся из картинки"}),
                "prompt_expansion_mode": (_H3MAX_EXPANSION, {
                    "default": "balanced",
                    "tooltip": "balanced — раскрытие промпта за ~1 с; "
                               "quality — тщательнее, но добавляет ~30 с"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "первый кадр -> image-to-video"}),
                "end_image": ("IMAGE", {"tooltip": "последний кадр (нужна image)"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "duration_override": DURATION_OVERRIDE_INPUT,
                "duration_frames": _H3_FRAMES_INPUT,
            },
        }

    RETURN_TYPES = _H3MAX_RETURN_TYPES
    RETURN_NAMES = _H3MAX_RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = "fal/MiniMax"

    def generate(self, prompt, model=_H3MAX_MODELS[0], duration=5,
                 resolution="768P", aspect_ratio="16:9",
                 prompt_expansion_mode="balanced", image=None, end_image=None,
                 seed=-1, enable_safety_checker=True,
                 duration_override=0.0, duration_frames=0):
        _require_deps()
        dur = _h3_duration(duration, duration_override, duration_frames)
        args = {
            "prompt": prompt,
            "duration": dur,
            "resolution": resolution,
            "prompt_expansion_mode": prompt_expansion_mode,
            "enable_safety_checker": bool(enable_safety_checker),
        }
        _seed_arg(args, seed)

        if image is not None:
            args["image_url"] = _upload_image_input(image, 1)[0]
            if end_image is not None:
                args["end_image_url"] = _upload_image_input(end_image, 1)[0]
            task = "image-to-video"
        else:
            if end_image is not None:
                raise RuntimeError(
                    "end_image задаёт последний кадр и работает только вместе "
                    "с image — подключи стартовый кадр во вход image.")
            args["aspect_ratio"] = aspect_ratio
            task = "text-to-video"
        base = "minimax/h3-max-turbo" if _h3max_tier(model) == "turbo" else "minimax/h3-max"
        endpoint = f"{base}/{task}"

        print(f"[fal {endpoint}] {resolution}, {dur} с, "
              f"раскрытие промпта: {prompt_expansion_mode}, ориентировочная "
              f"стоимость ~${_h3max_cost(model, resolution, dur):.2f}")
        result = _run_request(endpoint, args, est_seconds=max(20, dur * 2 + 10))
        return _h3max_finish(result, "minimax_h3max", endpoint)


class MinimaxH3MaxReference:
    """MiniMax H3 Max Reference-to-Video — мультимодальные референсы на
    пост-тренированной модели: до 9 картинок, до 3 видео и до 3 аудио
    (всего не больше 12 файлов).

    В промпте на них ссылаются как «Image 1», «Video 1», «Audio 1» — нумерация
    идёт по порядку: сначала входы image_1..image_4, затем строки image_urls.
    Видео и аудио — по 2–15 с каждое. Аудио не может быть единственным
    референсом. У Turbo этого режима нет, только у обычного H3 Max."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "дай каждому референсу роль: «Image 1 — костюм», "
                               "«Video 1 — движение камеры»"}),
                "duration": ("INT", {"default": 5, "min": 5, "max": 15}),
                "resolution": (_H3MAX_RES, {"default": "768P"}),
                "aspect_ratio": (["adaptive"] + _H3MAX_ASPECTS, {
                    "default": "adaptive",
                    "tooltip": "adaptive — взять пропорции из референсов"}),
                "prompt_expansion_mode": (_H3MAX_EXPANSION, {
                    "default": "balanced",
                    "tooltip": "quality тщательнее, но добавляет ~30 с"}),
            },
            "optional": {
                "image_1": ("IMAGE",), "image_2": ("IMAGE",),
                "image_3": ("IMAGE",), "image_4": ("IMAGE",),
                "video_1": ("VIDEO",), "video_2": ("VIDEO",), "video_3": ("VIDEO",),
                "audio_1": ("AUDIO",), "audio_2": ("AUDIO",), "audio_3": ("AUDIO",),
                "image_urls": ("STRING", {"default": "", "tooltip":
                                          "ссылки или пути, через запятую"}),
                "video_refs": ("STRING", {"default": "", "tooltip":
                                          "ссылки или пути, через запятую"}),
                "audio_refs": ("STRING", {"default": "", "tooltip":
                                          "ссылки или пути, через запятую"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "duration_override": DURATION_OVERRIDE_INPUT,
                "duration_frames": _H3_FRAMES_INPUT,
            },
        }

    RETURN_TYPES = _H3MAX_RETURN_TYPES
    RETURN_NAMES = _H3MAX_RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = "fal/MiniMax"

    def generate(self, prompt, duration=5, resolution="768P",
                 aspect_ratio="adaptive", prompt_expansion_mode="balanced",
                 image_1=None, image_2=None, image_3=None, image_4=None,
                 video_1=None, video_2=None, video_3=None,
                 audio_1=None, audio_2=None, audio_3=None,
                 image_urls="", video_refs="", audio_refs="", seed=-1,
                 enable_safety_checker=True, duration_override=0.0,
                 duration_frames=0):
        _require_deps()
        dur = _h3_duration(duration, duration_override, duration_frames)
        img_urls, vid_urls, aud_urls = _h3_collect_refs(
            (image_1, image_2, image_3, image_4),
            (video_1, video_2, video_3), (audio_1, audio_2, audio_3),
            image_urls, video_refs, audio_refs)

        args = {
            "prompt": prompt,
            "duration": dur,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "prompt_expansion_mode": prompt_expansion_mode,
            "enable_safety_checker": bool(enable_safety_checker),
        }
        if img_urls:
            args["reference_image_urls"] = img_urls
        if vid_urls:
            args["reference_video_urls"] = vid_urls
        if aud_urls:
            args["reference_audio_urls"] = aud_urls
        _seed_arg(args, seed)

        endpoint = "minimax/h3-max/reference-to-video"
        print(f"[fal {endpoint}] {resolution}, {dur} с, референсы: "
              f"{len(img_urls)} картинок / {len(vid_urls)} видео / "
              f"{len(aud_urls)} аудио, ориентировочная стоимость "
              f"~${_h3max_cost('H3 Max', resolution, dur):.2f}")
        result = _run_request(endpoint, args, est_seconds=max(20, dur * 2 + 10))
        return _h3max_finish(result, "minimax_h3max_ref", endpoint)


NODE_CLASS_MAPPINGS = {
    "Seedance2TextToVideo_fal": Seedance2TextToVideo,
    "Seedance2ImageToVideo_fal": Seedance2ImageToVideo,
    "Seedance2ReferenceToVideo_fal": Seedance2ReferenceToVideo,
    "Seedance15ProTextToVideo_fal": Seedance15ProTextToVideo,
    "Seedance15ProImageToVideo_fal": Seedance15ProImageToVideo,
    "GPTImage2TextToImage_fal": GPTImage2TextToImage,
    "GPTImage2Edit_fal": GPTImage2Edit,
    "TopazVideoUpscale_fal": TopazVideoUpscale,
    "KlingVideo_fal": KlingVideo,
    "SAM2Image_fal": SAM2Image,
    "SAM2Video_fal": SAM2Video,
    "EVFSAM_fal": EVFSAM,
    "NanoBananaEdit_fal": NanoBananaEdit,
    "QwenImageMax_fal": QwenImageMax,
    "SeedreamV5Pro_fal": SeedreamV5Pro,
    "IdeogramImage_fal": IdeogramImage,
    "FluxLoraImage_fal": FluxLoraImage,
    "Flux2LoraImage_fal": Flux2LoraImage,
    "FluxLoraTrainer_fal": FluxLoraTrainer,
    "QwenImageEditLora_fal": QwenImageEditLora,
    "WanLoraVideo_fal": WanLoraVideo,
    "LoraConvert_fal": LoraConvert,
    "LTX23Video_fal": LTX23Video,
    "LTX23ExtendVideo_fal": LTX23ExtendVideo,
    "LTX23Reframe_fal": LTX23Reframe,
    "MinimaxH3Video_fal": MinimaxH3Video,
    "MinimaxH3Reference_fal": MinimaxH3Reference,
    "MinimaxH3MaxVideo_fal": MinimaxH3MaxVideo,
    "MinimaxH3MaxReference_fal": MinimaxH3MaxReference,
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
    "KlingVideo_fal": "Kling Video (fal)",
    "SAM2Image_fal": "SAM 2 Image Segment (fal)",
    "SAM2Video_fal": "SAM 2 Video Segment (fal)",
    "EVFSAM_fal": "EVF-SAM Text Segment (fal)",
    "NanoBananaEdit_fal": "Nano Banana 2 / Pro Edit (fal)",
    "QwenImageMax_fal": "Qwen Image Max (fal)",
    "SeedreamV5Pro_fal": "Seedream 5.0 Pro (fal)",
    "IdeogramImage_fal": "Ideogram V4 / V3 Edit (fal)",
    "FluxLoraImage_fal": "FLUX.1 LoRA (fal)",
    "Flux2LoraImage_fal": "FLUX.2 LoRA (fal)",
    "FluxLoraTrainer_fal": "FLUX LoRA Trainer (fal)",
    "QwenImageEditLora_fal": "Qwen-Image Edit LoRA (fal)",
    "WanLoraVideo_fal": "Wan 2.2 LoRA Video (fal)",
    "LoraConvert_fal": "LoRA Convert fp16 / уменьшить ранг",
    "LTX23Video_fal": "LTX-2.3 Video LoRA (fal)",
    "LTX23ExtendVideo_fal": "LTX-2.3 Extend Video (fal)",
    "LTX23Reframe_fal": "LTX-2.3 Reframe (fal)",
    "MinimaxH3Video_fal": "MiniMax H3 Video LoRA (fal)",
    "MinimaxH3Reference_fal": "MiniMax H3 Reference-to-Video (fal)",
    "MinimaxH3MaxVideo_fal": "MiniMax H3 Max Video (fal)",
    "MinimaxH3MaxReference_fal": "MiniMax H3 Max Reference-to-Video (fal)",
}
