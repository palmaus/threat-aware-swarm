import { decode } from "@msgpack/msgpack";
import { useEffect, useRef } from "react";
import { useTelemetryStore } from "../store/telemetryStore";
import { useUIStore } from "../store/uiStore";
import { applySceneToEditor, setSceneError, setSceneId, setSceneText } from "../tools/sceneEditor";
import type { ServerMessage } from "../types/telemetry";

type ControlMessage = {
  type: "control";
  action: string;
  [key: string]: unknown;
};

type UIBridge = {
  send?: (msg: ControlMessage) => void;
  resetCharts?: () => void;
  resetTimeline?: () => void;
};

function resolveWsUrl() {
  const wsScheme = location.protocol === "https:" ? "wss" : "ws";
  const wsHost = location.hostname === "0.0.0.0" ? `localhost:${location.port}` : location.host;
  return `${wsScheme}://${wsHost}/ws`;
}

async function decodeWsMessage(event: MessageEvent): Promise<ServerMessage | null> {
  if (typeof event.data === "string") {
    try {
      return JSON.parse(event.data);
    } catch (err) {
      console.error("WS JSON parse failed:", err);
      return null;
    }
  }
  let buffer: ArrayBuffer;
  try {
    buffer = event.data instanceof ArrayBuffer ? event.data : await event.data.arrayBuffer();
  } catch (err) {
    console.error("WS buffer read failed:", err);
    return null;
  }
  try {
    const decoded = decode(new Uint8Array(buffer), { useMaps: false });
    if (decoded && typeof decoded === "object") return decoded as ServerMessage;
  } catch (err) {
    // молча пробуем JSON fallback
  }
  try {
    const text = new TextDecoder("utf-8").decode(new Uint8Array(buffer));
    return JSON.parse(text);
  } catch (err) {
    console.error("WS binary decode failed:", err);
    return null;
  }
}

export function useTelemetrySocket() {
  const setInit = useTelemetryStore((s) => s.setInit);
  const setState = useTelemetryStore((s) => s.setState);
  const setError = useTelemetryStore((s) => s.setError);
  const pushFrame = useTelemetryStore((s) => s.pushFrame);
  const setLoading = useTelemetryStore((s) => s.setLoading);
  const resetHistory = useTelemetryStore((s) => s.resetHistory);
  const resetCharts = useTelemetryStore((s) => s.resetCharts);
  const setCompare = useUIStore((s) => s.setCompare);
  const schemaVersionRef = useRef<string | null>(null);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let firstFrameSeen = false;

    const ensureBridge = () => {
      const w = window as unknown as { UIBridge?: UIBridge };
      w.UIBridge = w.UIBridge || {};
      w.UIBridge.send = (msg: ControlMessage) => {
        if (!ws || ws.readyState !== 1) return;
        ws.send(JSON.stringify(msg));
      };
      w.UIBridge.resetCharts = () => resetCharts();
      w.UIBridge.resetTimeline = () => resetHistory();
    };

    const handleMessage = (msg: ServerMessage) => {
      if (!msg) return;
      if (msg.type === "init") {
        setInit(msg);
        if (msg.state) {
          setState(msg.state);
          if (msg.state.compare !== undefined) setCompare(!!msg.state.compare);
        }
        setLoading(false);
        return;
      }
      if (msg.type === "state") {
        setState(msg.state);
        if (msg.state?.compare !== undefined) setCompare(!!msg.state.compare);
        return;
      }
      if (msg.type === "frame") {
        const framePayload = msg.telemetry || msg;
        const expectedVersion = schemaVersionRef.current;
        const actualVersion = framePayload?.schema_version || null;
        const left = framePayload?.left || null;
        const right = framePayload?.right || null;
        const leftVersion = left?.schema_version || null;
        const rightVersion = right?.schema_version || null;
        if (expectedVersion) {
          if (actualVersion && expectedVersion !== actualVersion) {
            setError(`Telemetry schema mismatch: expected ${expectedVersion}, got ${actualVersion}`);
            return;
          }
          if (left && !leftVersion) {
            setError(`Telemetry schema mismatch: expected ${expectedVersion}, got empty`);
            return;
          }
          if (leftVersion && expectedVersion !== leftVersion) {
            setError(`Telemetry schema mismatch: expected ${expectedVersion}, got ${leftVersion}`);
            return;
          }
          if (right && !rightVersion) {
            setError(`Telemetry schema mismatch: expected ${expectedVersion}, got empty`);
            return;
          }
          if (rightVersion && expectedVersion !== rightVersion) {
            setError(`Telemetry schema mismatch: expected ${expectedVersion}, got ${rightVersion}`);
            return;
          }
        }
        pushFrame(framePayload);
        if (!firstFrameSeen) {
          firstFrameSeen = true;
          setLoading(false);
        }
        return;
      }
      if (msg.type === "scenes") {
        setInit({ scenes: msg.scenes || [] });
        return;
      }
      if (msg.type === "scene_parsed") {
        if (msg.scene) applySceneToEditor(msg.scene);
        return;
      }
      if (msg.type === "scene_exported") {
        setSceneText(msg.text || "");
        return;
      }
      if (msg.type === "scene_saved") {
        if (msg.scene_id) setSceneId(msg.scene_id);
        return;
      }
      if (msg.type === "scene_error") {
        if (msg.error) setSceneError(msg.error);
      }
    };

    const connect = () => {
      ws = new WebSocket(resolveWsUrl());
      ws.binaryType = "arraybuffer";
      ws.onopen = () => {
        ensureBridge();
      };
      ws.onmessage = async (event) => {
        const msg = await decodeWsMessage(event);
        handleMessage(msg);
      };
      ws.onclose = () => {
        if (ws) ws = null;
      };
      ws.onerror = () => {
        // ошибки подключения не фейлим, UI продолжает жить
      };
    };

    const loadSchema = async () => {
      try {
        const resp = await fetch("/static/telemetry_schema.json");
        if (!resp.ok) return;
        const payload = await resp.json();
        const version = payload?.schema_version || null;
        if (version) schemaVersionRef.current = String(version);
      } catch {
        // игнорируем ошибки схемы, UI продолжает работать
      }
    };

    ensureBridge();
    loadSchema();
    connect();

    return () => {
      if (ws) ws.close();
      const w = window as unknown as { UIBridge?: UIBridge };
      if (w.UIBridge) {
        w.UIBridge.send = undefined;
      }
    };
  }, [resetCharts, resetHistory, setCompare, setError, setInit, setLoading, setState, pushFrame]);
}
