import { RENDER_CONST } from "./constants";
import type { TelemetryAgent, TelemetryPayload, TelemetryThreat } from "../types/telemetry";

export function getCanvasSize(canvas: HTMLCanvasElement, fallback: number) {
  const parent = canvas.parentElement;
  const rect = parent ? parent.getBoundingClientRect() : canvas.getBoundingClientRect();
  const size = Math.floor(Math.min(rect.width || fallback, rect.height || fallback));
  const finalSize = size > 4 ? size - 4 : fallback;
  return finalSize;
}

export function prepareCanvas(canvas: HTMLCanvasElement, size: number) {
  const dpr = window.devicePixelRatio || 1;
  const scaled = Math.max(1, Math.floor(size * dpr));
  if (canvas.width !== scaled || canvas.height !== scaled) {
    canvas.width = scaled;
    canvas.height = scaled;
  }
  canvas.style.width = `${size}px`;
  canvas.style.height = `${size}px`;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;
  return ctx;
}

export function getSceneEditorMetrics(rect: DOMRect) {
  const width = Math.max(1, Math.floor(rect.width || 300));
  const height = Math.max(1, Math.floor(rect.height || 300));
  const size = Math.min(width, height);
  return {
    width,
    height,
    size,
    offsetX: (width - size) / 2,
    offsetY: (height - size) / 2,
  };
}

export function prepareCanvasStatic(canvas: HTMLCanvasElement) {
  const rect = canvas.getBoundingClientRect();
  const metrics = getSceneEditorMetrics(rect);
  const dpr = window.devicePixelRatio || 1;
  const scaledW = Math.max(1, Math.floor(metrics.width * dpr));
  const scaledH = Math.max(1, Math.floor(metrics.height * dpr));
  if (canvas.width !== scaledW || canvas.height !== scaledH) {
    canvas.width = scaledW;
    canvas.height = scaledH;
  }
  canvas.style.width = `${metrics.width}px`;
  canvas.style.height = `${metrics.height}px`;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;
  return { ctx, ...metrics };
}

function worldToScreen(x: number, y: number, field: number, size: number) {
  const scale = size / (field * RENDER_CONST.ZOOM_OUT);
  const offset = (size - field * scale) / 2;
  const sx = x * scale + offset;
  const sy = (field - y) * scale + offset;
  return [sx, sy, scale, offset] as const;
}

export function drawWindWidget(canvas: HTMLCanvasElement | null, windVec: number[] | null) {
  if (!canvas) return;
  const size = Math.min(canvas.width || 72, canvas.height || 72);
  const ctx = prepareCanvas(canvas, size);
  if (!ctx) return;
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = "#0b1220";
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = "rgba(148,163,184,0.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(size / 2, 6);
  ctx.lineTo(size / 2, size - 6);
  ctx.moveTo(6, size / 2);
  ctx.lineTo(size - 6, size / 2);
  ctx.stroke();
  if (!windVec || windVec.length !== 2) return;
  const vx = Number(windVec[0]);
  const vy = Number(windVec[1]);
  if (!Number.isFinite(vx) || !Number.isFinite(vy)) return;
  const mag = Math.hypot(vx, vy);
  const len = Math.min(size * 0.35, mag * RENDER_CONST.WIND_SCALE);
  const angle = Math.atan2(-vy, vx);
  const cx = size / 2;
  const cy = size / 2;
  const ex = cx + Math.cos(angle) * len;
  const ey = cy + Math.sin(angle) * len;
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(ex, ey);
  ctx.stroke();
  const head = 6;
  ctx.fillStyle = "#38bdf8";
  ctx.beginPath();
  ctx.moveTo(ex, ey);
  ctx.lineTo(ex - Math.cos(angle - Math.PI / 6) * head, ey - Math.sin(angle - Math.PI / 6) * head);
  ctx.lineTo(ex - Math.cos(angle + Math.PI / 6) * head, ey - Math.sin(angle + Math.PI / 6) * head);
  ctx.closePath();
  ctx.fill();
}

export function drawWindOverlay(ctx: CanvasRenderingContext2D, size: number, windVec: number[]) {
  const vx = Number(windVec[0]);
  const vy = Number(windVec[1]);
  if (!Number.isFinite(vx) || !Number.isFinite(vy)) return;
  const mag = Math.hypot(vx, vy);
  const box = 60;
  const pad = 8;
  const cx = pad + box / 2;
  const cy = size - pad - box / 2;
  ctx.save();
  ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
  ctx.strokeStyle = "rgba(148,163,184,0.4)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(pad, size - pad - box, box, box);
  ctx.fill();
  ctx.stroke();
  const len = Math.min(box * 0.35, mag * RENDER_CONST.WIND_SCALE);
  const angle = Math.atan2(-vy, vx);
  const ex = cx + Math.cos(angle) * len;
  const ey = cy + Math.sin(angle) * len;
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(ex, ey);
  ctx.stroke();
  ctx.fillStyle = "#38bdf8";
  ctx.beginPath();
  ctx.moveTo(ex, ey);
  ctx.lineTo(ex - Math.cos(angle - Math.PI / 6) * 6, ey - Math.sin(angle - Math.PI / 6) * 6);
  ctx.lineTo(ex - Math.cos(angle + Math.PI / 6) * 6, ey - Math.sin(angle + Math.PI / 6) * 6);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "#e2e8f0";
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.fillText(`${mag.toFixed(2)} m/s`, pad + 6, size - pad - 6);
  ctx.restore();
}

export function drawTelemetry(canvas: HTMLCanvasElement | null, telemetry: TelemetryPayload | null, selectedIdx: number) {
  if (!canvas || !telemetry) return;
  const field = telemetry.field_size || 100;
  const size = getCanvasSize(canvas, telemetry.screen_size || canvas.width || 800);
  const ctx = prepareCanvas(canvas, size);
  if (!ctx) return;
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = "#0b1220";
  ctx.fillRect(0, 0, size, size);

  if (telemetry.walls) {
    ctx.fillStyle = "rgba(100,116,139,0.65)";
    telemetry.walls.forEach((w: number[]) => {
      const [x1, y1, x2, y2] = w;
      const p1 = worldToScreen(x1, y1, field, size);
      const p2 = worldToScreen(x2, y2, field, size);
      const rx = Math.min(p1[0], p2[0]);
      const ry = Math.min(p1[1], p2[1]);
      const rw = Math.abs(p2[0] - p1[0]);
      const rh = Math.abs(p2[1] - p1[1]);
      ctx.fillRect(rx, ry, rw, rh);
      ctx.strokeStyle = "rgba(15, 23, 42, 0.9)";
      ctx.lineWidth = 2;
      ctx.lineJoin = "miter";
      ctx.strokeRect(rx, ry, rw, rh);
    });
  }

  if (telemetry.oracle_path && telemetry.oracle_path.length > 1) {
    ctx.strokeStyle = "rgba(147, 51, 234, 0.9)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    telemetry.oracle_path.forEach((p: number[], idx: number) => {
      const pt = worldToScreen(p[0], p[1], field, size);
      const x = pt[0];
      const y = pt[1];
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  if (telemetry.threats) {
    telemetry.threats.forEach((t: TelemetryThreat) => {
      const pt = worldToScreen(t.pos[0], t.pos[1], field, size);
      const r = (t.radius || 1) * pt[2];
      ctx.fillStyle = "rgba(248, 113, 113, 0.35)";
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], r, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(248, 113, 113, 0.75)";
      ctx.stroke();
    });
  }

  if (telemetry.target_pos) {
    const pt = worldToScreen(telemetry.target_pos[0], telemetry.target_pos[1], field, size);
    ctx.fillStyle = "rgba(34, 197, 94, 0.95)";
    ctx.beginPath();
    ctx.arc(pt[0], pt[1], 6, 0, Math.PI * 2);
    ctx.fill();
  }

  if (telemetry.agents) {
    telemetry.agents.forEach((a: TelemetryAgent) => {
      const pt = worldToScreen(a.pos[0], a.pos[1], field, size);
      const isDead = a.alive === false;
      if (isDead) {
        ctx.fillStyle = "rgba(100,116,139,0.55)";
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "rgba(248,113,113,0.9)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], 6.5, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(pt[0] - 5, pt[1] - 5);
        ctx.lineTo(pt[0] + 5, pt[1] + 5);
        ctx.moveTo(pt[0] + 5, pt[1] - 5);
        ctx.lineTo(pt[0] - 5, pt[1] + 5);
        ctx.stroke();
        ctx.lineWidth = 1;
      } else {
        ctx.fillStyle = a.index === selectedIdx ? "#38bdf8" : "#e2e8f0";
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], 5, 0, Math.PI * 2);
        ctx.fill();
        if (a.vel) {
          const vx = a.vel[0];
          const vy = a.vel[1];
          const ex = pt[0] + vx * 2;
          const ey = pt[1] - vy * 2;
          ctx.strokeStyle = "rgba(148,163,184,0.7)";
          ctx.beginPath();
          ctx.moveTo(pt[0], pt[1]);
          ctx.lineTo(ex, ey);
          ctx.stroke();
        }
      }
    });
  }

  if (telemetry.wind) {
    drawWindOverlay(ctx, size, telemetry.wind);
  }
}

export function drawGrid(canvas: HTMLCanvasElement | null, grid: number[][] | null, attention: number[][] | null) {
  if (!canvas || !grid) return;
  const size = Math.min(canvas.width || RENDER_CONST.GRID_SIZE, canvas.height || RENDER_CONST.GRID_SIZE);
  const ctx = prepareCanvas(canvas, size);
  if (!ctx) return;
  const h = grid.length;
  const w = grid[0]?.length || 0;
  const cell = size / Math.max(1, w);
  ctx.clearRect(0, 0, size, size);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = grid[y][x];
      const c = Math.max(0, Math.min(1, v));
      const shade = Math.floor(20 + c * 180);
      ctx.fillStyle = `rgb(${shade}, ${shade}, ${shade})`;
      ctx.fillRect(x * cell, y * cell, cell, cell);
    }
  }
  if (attention) {
    const ah = attention.length;
    const aw = attention[0]?.length || 0;
    const scaleX = w / Math.max(1, aw);
    const scaleY = h / Math.max(1, ah);
    ctx.globalAlpha = 0.6;
    for (let y = 0; y < ah; y++) {
      for (let x = 0; x < aw; x++) {
        const v = attention[y][x];
        const c = Math.max(0, Math.min(1, v));
        ctx.fillStyle = `rgba(56, 189, 248, ${c})`;
        ctx.fillRect(x * scaleX * cell, y * scaleY * cell, scaleX * cell, scaleY * cell);
      }
    }
    ctx.globalAlpha = 1;
  }
}

export function pickAgentFromTelemetry(
  telemetry: TelemetryPayload | null,
  evt: MouseEvent,
  canvas: HTMLCanvasElement | null,
) {
  if (!telemetry || !telemetry.agents || !canvas) return null;
  const rect = canvas.getBoundingClientRect();
  const px = evt.clientX - rect.left;
  const py = evt.clientY - rect.top;
  const field = telemetry.field_size || 100;
  const size = Math.max(1, Math.min(rect.width, rect.height));
  const scale = size / (field * RENDER_CONST.ZOOM_OUT);
  const offset = (size - field * scale) / 2;
  const worldX = (px - offset) / scale;
  const worldY = field - (py - offset) / scale;
  let bestIdx = null;
  let bestDist = Infinity;
  telemetry.agents.forEach((a: TelemetryAgent) => {
    const dx = a.pos[0] - worldX;
    const dy = a.pos[1] - worldY;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestDist) {
      bestDist = d2;
      bestIdx = a.index;
    }
  });
  return bestIdx;
}
