import { RENDER_CONST } from "./constants";
import type { TelemetryAgent, TelemetryPayload, TelemetryThreat } from "../types/telemetry";

export function drawMinimap(canvas: HTMLCanvasElement | null, telemetry: TelemetryPayload | null, selectedIdx: number) {
  if (!canvas || !telemetry) return;
  const field = telemetry.field_size || 100;
  const size = RENDER_CONST.MINIMAP_SIZE;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== size * dpr || canvas.height !== size * dpr) {
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = "rgba(10, 14, 25, 0.95)";
  ctx.fillRect(0, 0, size, size);

  const scale = size / (field * RENDER_CONST.ZOOM_OUT);
  const offset = (size - field * scale) / 2;

  const toScreen = (pos: number[]) => {
    const x = pos[0] * scale + offset;
    const y = (field - pos[1]) * scale + offset;
    return [x, y] as const;
  };

  if (telemetry.walls) {
    ctx.fillStyle = "rgba(100,116,139,0.65)";
    telemetry.walls.forEach((w: number[]) => {
      const [x1, y1, x2, y2] = w;
      const p1 = toScreen([x1, y1]);
      const p2 = toScreen([x2, y2]);
      const rx = Math.min(p1[0], p2[0]);
      const ry = Math.min(p1[1], p2[1]);
      const rw = Math.abs(p2[0] - p1[0]);
      const rh = Math.abs(p2[1] - p1[1]);
      ctx.fillRect(rx, ry, rw, rh);
    });
  }

  if (telemetry.threats) {
    telemetry.threats.forEach((t: TelemetryThreat) => {
      const pt = toScreen(t.pos);
      const r = (t.radius || 1) * scale;
      ctx.fillStyle = "rgba(248, 113, 113, 0.35)";
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], r, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  if (telemetry.target_pos) {
    const pt = toScreen(telemetry.target_pos);
    ctx.fillStyle = "rgba(34, 197, 94, 0.9)";
    ctx.beginPath();
    ctx.arc(pt[0], pt[1], 3, 0, Math.PI * 2);
    ctx.fill();
  }

  if (telemetry.agents) {
    telemetry.agents.forEach((a: TelemetryAgent) => {
      const pt = toScreen(a.pos);
      ctx.fillStyle = a.index === selectedIdx ? "#38bdf8" : "#e2e8f0";
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], 2.5, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  ctx.strokeStyle = "rgba(148,163,184,0.4)";
  ctx.lineWidth = 1;
  ctx.strokeRect(1, 1, size - 2, size - 2);
}
