/// <reference types="vite/client" />

interface Window {
  UIBridge?: {
    send?: (msg: { type: "control"; action: string; [key: string]: unknown }) => void;
    resetCharts?: () => void;
    resetTimeline?: () => void;
    onState?: (data: unknown) => void;
  };
  SimpleGifEncoder?: {
    encode: (frames: ImageData[], width: number, height: number, fps: number) => Blob;
  };
}
