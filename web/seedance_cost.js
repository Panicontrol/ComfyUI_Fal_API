// Живая оценка стоимости генерации на нодах fal/Seedance и fal/GPT Image.
// Цена рисуется в заголовке ноды и пересчитывается при каждой перерисовке
// по текущим значениям виджетов — ещё до запуска генерации.
import { app } from "../../scripts/app.js";

const RES_H = { "480p": 480, "720p": 720, "1080p": 1080 };

// Seedance: токены = w*h*24*сек/1024
const S2_RATE = 0.014 / 1000;        // $ за токен (standard, звук бесплатно)
const S2_FAST_RATE = 0.0112 / 1000;  // $ за токен (fast)
const S15_AUDIO_RATE = 2.4 / 1e6;    // $ за токен со звуком
const S15_RATE = 1.2 / 1e6;          // $ за токен без звука

// GPT Image 2: $ за изображение (пиксели -> {low, medium, high})
const GPT_PRICES = [
  [1024 * 768,  { low: 0.005, medium: 0.037, high: 0.145 }],
  [1024 * 1024, { low: 0.006, medium: 0.053, high: 0.211 }],
  [1024 * 1536, { low: 0.005, medium: 0.042, high: 0.165 }],
  [1920 * 1080, { low: 0.005, medium: 0.040, high: 0.158 }],
  [2560 * 1440, { low: 0.007, medium: 0.056, high: 0.222 }],
  [3840 * 2160, { low: 0.012, medium: 0.101, high: 0.401 }],
];
const GPT_PRESET_PX = {
  auto: 1024 * 1024, square: 1024 * 1024, square_hd: 1536 * 1536,
  portrait_4_3: 768 * 1024, portrait_16_9: 1080 * 1920,
  landscape_4_3: 1024 * 768, landscape_16_9: 1920 * 1080,
};

function widgetValues(node) {
  const vals = {};
  for (const w of node.widgets || []) vals[w.name] = w.value;
  return vals;
}

function pxDims(resolution, aspect) {
  const h = RES_H[resolution] || 720;
  let ar = aspect && aspect !== "auto" ? aspect : "16:9";
  const parts = ar.split(":").map(Number);
  const w = parts.length === 2 && parts[0] > 0 && parts[1] > 0
    ? Math.round((h * parts[0]) / parts[1])
    : Math.round((h * 16) / 9);
  return [w, h];
}

function fmt(x) {
  return "$" + (x < 0.1 ? x.toFixed(3) : x.toFixed(2));
}

function seedanceCost(v, { v15 = false, durMax = 15 } = {}) {
  const [w, h] = pxDims(v.resolution, v.aspect_ratio);
  const tokensPerSec = (w * h * 24) / 1024;
  let rate;
  if (v15) rate = v.generate_audio ? S15_AUDIO_RATE : S15_RATE;
  else rate = v.fast_mode ? S2_FAST_RATE : S2_RATE; // у reference fast_mode нет -> standard
  const perSec = tokensPerSec * rate;
  const dur = v.duration;
  if (dur === "auto" || dur === undefined || dur === null) {
    return `≈ ${fmt(perSec * 4)}–${fmt(perSec * durMax)}`;
  }
  return `≈ ${fmt(perSec * parseInt(dur, 10))}`;
}

function gptCost(v) {
  let px;
  if (v.custom_width > 0 && v.custom_height > 0) px = v.custom_width * v.custom_height;
  else px = GPT_PRESET_PX[v.image_size] || 1024 * 1024;
  let best = GPT_PRICES[0];
  for (const row of GPT_PRICES) {
    if (Math.abs(row[0] - px) < Math.abs(best[0] - px)) best = row;
  }
  const q = v.quality === "auto" || !v.quality ? "high" : v.quality;
  const n = v.num_images || 1;
  return `≈ ${fmt(best[1][q] * n)}`;
}

// Nano Banana: $ за изображение по модели и разрешению
const NB_PRICE = {
  "nano-banana-2":   { "0.5K": 0.06, "1K": 0.08, "2K": 0.12, "4K": 0.16 },
  "nano-banana-pro": { "0.5K": 0.15, "1K": 0.15, "2K": 0.15, "4K": 0.30 },
};

function nanoBananaCost(v) {
  const table = NB_PRICE[v.model] || NB_PRICE["nano-banana-2"];
  const per = table[v.resolution] ?? 0.08;
  return `≈ ${fmt(per * (v.num_images || 1))}`;
}

const CALCS = {
  Seedance2TextToVideo_fal: (v) => seedanceCost(v, { durMax: 15 }),
  Seedance2ImageToVideo_fal: (v) => seedanceCost(v, { durMax: 15 }),
  Seedance2ReferenceToVideo_fal: (v) => seedanceCost(v, { durMax: 15 }),
  Seedance15ProTextToVideo_fal: (v) => seedanceCost(v, { v15: true, durMax: 12 }),
  Seedance15ProImageToVideo_fal: (v) => seedanceCost(v, { v15: true, durMax: 12 }),
  GPTImage2TextToImage_fal: gptCost,
  GPTImage2Edit_fal: gptCost,
  NanoBananaEdit_fal: nanoBananaCost,
  QwenImageMax_fal: (v) => `≈ ${fmt(0.075 * (v.num_images || 1))}`,
  IdeogramImage_fal: (v) => {
    const rate = { TURBO: 0.03, BALANCED: 0.06, QUALITY: 0.10 }[v.rendering_speed] ?? 0.06;
    const mp = (v.custom_width > 0 && v.custom_height > 0)
      ? (v.custom_width * v.custom_height) / 1e6 : 1.0;
    let c = rate * mp * (v.num_images || 1);
    if (v.enable_prompt_expansion) c += 0.03;
    return `≈ ${fmt(c)}`;
  },
  SeedreamV5Pro_fal: (v) => {
    const big = (v.custom_width > 0 && v.custom_height > 0)
      ? v.custom_width * v.custom_height > 1536 * 1536
      : v.image_size === "auto_2K";
    return `≈ ${fmt((big ? 0.135 : 0.0675) * (v.num_images || 1))}`;
  },
  Flux2LoraImage_fal: (v) => {
    const presetPx = {
      square_hd: 1024 * 1024, square: 512 * 512,
      landscape_4_3: 1024 * 768, portrait_4_3: 768 * 1024,
      landscape_16_9: 1024 * 576, portrait_16_9: 576 * 1024,
    };
    const px = (v.custom_width > 0 && v.custom_height > 0)
      ? v.custom_width * v.custom_height
      : (presetPx[v.image_size] ?? 1024 * 1024);
    return `≈ ${fmt(0.021 * (px / 1e6) * (v.num_images || 1))}`;
  },
  KlingVideo_fal: (v) => {
    if (v.model !== "2.6 pro") return "";  // тариф 2.1 не фиксирован
    const per = v.generate_audio ? 0.14 : 0.07;
    return `≈ ${fmt(per * parseInt(v.duration || "5", 10))}`;
  },
  LTX23Video_fal: (v) => {
    const wh = {
      landscape_16_9: [1024, 576], landscape_4_3: [1024, 768],
      square_hd: [1024, 1024], square: [512, 512],
      portrait_4_3: [768, 1024], portrait_16_9: [576, 1024],
    };
    let w, h;
    if (v.custom_width > 0 && v.custom_height > 0) {
      w = v.custom_width; h = v.custom_height;
    } else if (v.video_size === "auto" || !wh[v.video_size]) {
      w = 1280; h = 720;                       // реальный размер выберет модель
    } else {
      [w, h] = wh[v.video_size];
    }
    const frames = Math.round(((v.num_frames || 121) - 1) / 8) * 8 + 1;
    const rate = String(v.model || "").includes("distilled") ? 0.001205 : 0.001605;
    return `≈ ${fmt(rate * (w * h * frames) / 1e6)}`;
  },
  LTX23ExtendVideo_fal: (v) => {
    const frames = Math.round(((v.num_frames || 121) - 1) / 8) * 8 + 1;
    return `≈ ${fmt(0.001605 * (1280 * 720 * frames) / 1e6)}`;  // по 720p
  },
  WanLoraVideo_fal: (v) => {
    const fps = v.frames_per_second || 16;
    const secs = (v.num_frames || 81) / (fps || 16);
    return `≈ ${fmt(secs * 0.1)}`;
  },
};

app.registerExtension({
  name: "fal.seedance.costEstimate",
  beforeRegisterNodeDef(nodeType, nodeData) {
    const calc = CALCS[nodeData.name];
    if (!calc) return;
    const origDraw = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      origDraw?.apply(this, arguments);
      if (this.flags.collapsed) return;
      let text;
      try {
        text = calc(widgetValues(this));
      } catch (e) {
        return;
      }
      ctx.save();
      ctx.font = "bold 12px Arial";
      ctx.fillStyle = "#8f8";
      ctx.textAlign = "right";
      // в правой части заголовка ноды
      ctx.fillText(text, this.size[0] - 12, -9);
      ctx.restore();
    };
  },
});
