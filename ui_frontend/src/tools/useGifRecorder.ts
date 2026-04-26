import { useCallback, useMemo, useRef, useState } from "react";

type RecorderState = {
  active: boolean;
  frames: ImageData[];
  fps: number;
  side: "left" | "right";
  lastCapture: number;
  width: number;
  height: number;
  captureCanvas: HTMLCanvasElement | null;
  url: string | null;
};

type RecorderAPI = {
  status: string;
  url: string | null;
  downloadName: string;
  recording: boolean;
  side: "left" | "right";
  fps: number;
  setSide: (side: "left" | "right") => void;
  setFps: (fps: number) => void;
  start: () => void;
  stop: () => void;
  capture: (now: number, canvasLeft: HTMLCanvasElement | null, canvasRight: HTMLCanvasElement | null) => void;
};

function resolveSourceCanvas(
  side: "left" | "right",
  canvasLeft: HTMLCanvasElement | null,
  canvasRight: HTMLCanvasElement | null,
) {
  if (side === "right" && canvasRight) return canvasRight;
  return canvasLeft;
}

export function useGifRecorder(): RecorderAPI {
  const [status, setStatus] = useState("");
  const [url, setUrl] = useState<string | null>(null);
  const [downloadName, setDownloadName] = useState("");
  const [recording, setRecording] = useState(false);
  const [side, setSide] = useState<"left" | "right">("left");
  const [fps, setFps] = useState(20);
  const stateRef = useRef<RecorderState>({
    active: false,
    frames: [],
    fps: 20,
    side: "left",
    lastCapture: 0,
    width: 0,
    height: 0,
    captureCanvas: null,
    url: null,
  });
  const lastStatusUpdateRef = useRef(0);

  const start = useCallback(() => {
    if (!window.SimpleGifEncoder) {
      setStatus("GIF error: encoder missing");
      return;
    }
    const state = stateRef.current;
    state.active = true;
    state.frames = [];
    state.fps = Math.max(1, fps || 20);
    state.side = side;
    state.lastCapture = 0;
    state.width = 0;
    state.height = 0;
    if (state.url) {
      URL.revokeObjectURL(state.url);
      state.url = null;
    }
    setUrl(null);
    setDownloadName("");
    setRecording(true);
    setStatus("GIF: recording...");
  }, [fps, side]);

  const stop = useCallback(() => {
    const state = stateRef.current;
    if (!state.active) {
      setStatus("GIF error: not recording");
      return;
    }
    state.active = false;
    setRecording(false);
    const frames = state.frames.slice();
    if (!frames.length) {
      setStatus("GIF error: no frames");
      return;
    }
    const width = state.width;
    const height = state.height;
    const fpsValue = state.fps;
    setStatus("GIF: encoding...");
    setTimeout(() => {
      try {
        const blob = window.SimpleGifEncoder.encode(frames, width, height, fpsValue);
        const nextUrl = URL.createObjectURL(blob);
        state.url = nextUrl;
        setUrl(nextUrl);
        const name = `swarm_${state.side}_${Date.now()}.gif`;
        setDownloadName(name);
        setStatus(`GIF ready (${frames.length} frames)`);
      } catch (err) {
        console.error("GIF encode failed:", err);
        setStatus("GIF error: encode failed");
      }
    }, 30);
  }, []);

  const capture = useCallback(
    (now: number, canvasLeft: HTMLCanvasElement | null, canvasRight: HTMLCanvasElement | null) => {
      const state = stateRef.current;
      if (!state.active) return;
      const interval = 1000 / state.fps;
      if (now - state.lastCapture < interval) return;
      const src = resolveSourceCanvas(state.side, canvasLeft, canvasRight);
      if (!src) return;
      const width = src.width || 0;
      const height = src.height || 0;
      if (width <= 0 || height <= 0) return;
      const maxSize = 480;
      const scale = Math.min(1, maxSize / Math.max(width, height));
      const outW = Math.max(1, Math.floor(width * scale));
      const outH = Math.max(1, Math.floor(height * scale));
      if (!state.captureCanvas) {
        state.captureCanvas = document.createElement("canvas");
      }
      const cap = state.captureCanvas;
      if (cap.width !== outW || cap.height !== outH) {
        cap.width = outW;
        cap.height = outH;
      }
      const ctx = cap.getContext("2d");
      if (!ctx) return;
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, outW, outH);
      ctx.drawImage(src, 0, 0, outW, outH);
      const img = ctx.getImageData(0, 0, outW, outH);
      state.frames.push(img);
      state.width = outW;
      state.height = outH;
      state.lastCapture = now;
      if (now - lastStatusUpdateRef.current > 500) {
        lastStatusUpdateRef.current = now;
        setStatus(`GIF: recording (${state.frames.length})`);
      }
    },
    [],
  );

  return useMemo(
    () => ({
      status,
      url,
      downloadName,
      recording,
      side,
      fps,
      setSide,
      setFps,
      start,
      stop,
      capture,
    }),
    [status, url, downloadName, recording, side, fps, start, stop, capture],
  );
}
