import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import { useUIStore } from "./store/uiStore";
import { resetChartsAndTimeline, sendControl } from "./bridge/control";
import { useTelemetrySocket } from "./bridge/telemetrySocket";
import { useTelemetryStore } from "./store/telemetryStore";
import { useCharts } from "./bridge/useCharts";
import { drawGrid, drawTelemetry, drawWindWidget, pickAgentFromTelemetry } from "./renderer/canvas";
import { drawMinimap } from "./renderer/minimap";
import { initSceneEditor, setSceneId } from "./tools/sceneEditor";
import { useGifRecorder } from "./tools/useGifRecorder";
import type { TelemetryAgent, TelemetryPayload, TelemetryStats } from "./types/telemetry";
import "./tools/gif_encoder.js";

function formatStats(stats: TelemetryStats | null | undefined, nAgents?: number) {
  if (!stats) return "";
  const items: string[] = [];
  if (nAgents !== undefined) items.push(`agents: ${nAgents}`);
  if (Number.isFinite(stats.alive) && nAgents) {
    items.push(`alive: ${stats.alive}/${nAgents} (${((stats.alive / nAgents) * 100).toFixed(1)}%)`);
  }
  if (Number.isFinite(stats.finished) && nAgents) {
    items.push(`finished: ${stats.finished}/${nAgents} (${((stats.finished / nAgents) * 100).toFixed(1)}%)`);
  }
  if (Number.isFinite(stats.in_goal) && nAgents) {
    items.push(`in_goal: ${stats.in_goal}/${nAgents}`);
  }
  if (Number.isFinite(stats.mean_dist)) items.push(`mean_dist: ${Number(stats.mean_dist).toFixed(2)}`);
  if (Number.isFinite(stats.mean_path_ratio)) items.push(`path_ratio: ${Number(stats.mean_path_ratio).toFixed(3)}`);
  if (Number.isFinite(stats.mean_risk)) items.push(`risk_p: ${Number(stats.mean_risk).toFixed(4)}`);
  return items.join("\n").trim();
}

function formatAgentInfo(telemetry: TelemetryPayload | null | undefined, selectedIdx: number) {
  if (!telemetry || !telemetry.agents) return "";
  const agent = telemetry.agents.find((a: TelemetryAgent) => a.index === selectedIdx);
  if (!agent) return "";
  const items: string[] = [];
  if (agent.pos) {
    const posText = agent.pos.map((v: number) => (Number.isFinite(v) ? Number(v).toFixed(2) : "--")).join(", ");
    items.push(`pos: [${posText}]`);
  }
  if (agent.vel) {
    const velText = agent.vel.map((v: number) => (Number.isFinite(v) ? Number(v).toFixed(2) : "--")).join(", ");
    items.push(`vel: [${velText}]`);
  }
  if (Number.isFinite(agent.dist)) items.push(`dist: ${Number(agent.dist).toFixed(2)}`);
  if (Number.isFinite(agent.energy)) items.push(`energy: ${Number(agent.energy).toFixed(2)}`);
  if (Number.isFinite(agent.risk_p)) items.push(`risk: ${Number(agent.risk_p).toFixed(4)}`);
  if (agent.alive !== undefined) items.push(`alive: ${agent.alive}`);
  if (agent.finished !== undefined) items.push(`finished: ${agent.finished}`);
  if (agent.in_goal !== undefined) items.push(`in_goal: ${agent.in_goal}`);
  if (agent.threat_collided !== undefined) items.push(`threat_hit: ${agent.threat_collided}`);
  return items.join("\n");
}

function formatWindText(windVec: number[] | null) {
  if (!windVec || windVec.length !== 2) return "--";
  const vx = Number(windVec[0]);
  const vy = Number(windVec[1]);
  if (!Number.isFinite(vx) || !Number.isFinite(vy)) return "--";
  const mag = Math.hypot(vx, vy);
  return `${mag.toFixed(2)} m/s`;
}

function computeEnergyDisplay(telemetry: TelemetryPayload | null | undefined, selectedIdx: number) {
  const agent = telemetry?.agents?.find((a: TelemetryAgent) => a.index === selectedIdx) || null;
  const levelRaw = agent?.energy_level ?? telemetry?.stats?.mean_energy_level ?? null;
  const energyAbs = agent?.energy ?? telemetry?.stats?.mean_energy ?? null;
  if (!Number.isFinite(levelRaw)) {
    return { percent: 0, text: "--", gradient: "linear-gradient(90deg, #64748b, #94a3b8)" };
  }
  const level = Math.max(0, Math.min(1, Number(levelRaw)));
  let gradient = "linear-gradient(90deg, #22c55e, #4ade80)";
  if (level < 0.3) gradient = "linear-gradient(90deg, #ef4444, #f87171)";
  else if (level < 0.6) gradient = "linear-gradient(90deg, #f59e0b, #fbbf24)";
  const text = Number.isFinite(energyAbs)
    ? `${(level * 100).toFixed(0)}% (${Number(energyAbs).toFixed(1)})`
    : `${(level * 100).toFixed(0)}%`;
  return { percent: level * 100, text, gradient };
}

const DEFAULT_TUNABLES = {
  threat_speed: 1.0,
  threat_mode: "static",
  wall_friction: 0.0,
  w_risk: 2.0,
  w_progress: 5.0,
  w_wall: 20.0,
  max_accel: 8.0,
  drag: 0.05,
  obs_noise_target: 0.0,
  obs_noise_vel: 0.0,
  obs_noise_grid: 0.0,
  domain_randomization: false,
  dr_max_speed_min: -1.0,
  dr_max_speed_max: -1.0,
  dr_drag_min: -1.0,
  dr_drag_max: -1.0,
};

export default function App() {
  const toolsTab = useUIStore((s) => s.toolsTab);
  const setToolsTab = useUIStore((s) => s.setToolsTab);
  const compare = useUIStore((s) => s.compare);
  const setCompare = useUIStore((s) => s.setCompare);
  const uiMode = useUIStore((s) => s.uiMode);
  const setUiMode = useUIStore((s) => s.setUiMode);
  const agentIdx = useUIStore((s) => s.agentIdx);
  const setAgentIdx = useUIStore((s) => s.setAgentIdx);

  const policies = useTelemetryStore((s) => s.policies);
  const scenes = useTelemetryStore((s) => s.scenes);
  const models = useTelemetryStore((s) => s.models);
  const controlState = useTelemetryStore((s) => s.controlState);
  const patchControlState = useTelemetryStore((s) => s.patchControlState);
  const historyFrames = useTelemetryStore((s) => s.historyFrames);
  const timelineIndex = useTelemetryStore((s) => s.timelineIndex);
  const timelineLocked = useTelemetryStore((s) => s.timelineLocked);
  const markers = useTelemetryStore((s) => s.markers);
  const loading = useTelemetryStore((s) => s.loading);
  const lastError = useTelemetryStore((s) => s.lastError);
  const setTimelineIndex = useTelemetryStore((s) => s.setTimelineIndex);
  const setTimelineLive = useTelemetryStore((s) => s.setTimelineLive);

  useTelemetrySocket();
  useCharts();

  const canvasLeftRef = useRef<HTMLCanvasElement | null>(null);
  const canvasRightRef = useRef<HTMLCanvasElement | null>(null);
  const minimapRef = useRef<HTMLCanvasElement | null>(null);
  const gridLeftRef = useRef<HTMLCanvasElement | null>(null);
  const gridRightRef = useRef<HTMLCanvasElement | null>(null);
  const windLeftRef = useRef<HTMLCanvasElement | null>(null);
  const windRightRef = useRef<HTMLCanvasElement | null>(null);
  const sceneEditorRootRef = useRef<HTMLDivElement | null>(null);
  const errorRef = useRef<HTMLDivElement | null>(null);

  const [modelPath, setModelPath] = useState("");
  const [modelSelect, setModelSelect] = useState("");
  const gif = useGifRecorder();

  const handleTabChange = (value: string) => {
    setToolsTab(value);
    if (value === "editor") {
      window.dispatchEvent(new CustomEvent("ui:tab-editor"));
    }
  };

  useEffect(() => {
    const mode = document.body.dataset.mode === "demo" ? "demo" : "research";
    setUiMode(mode);
  }, [setUiMode]);

  useEffect(() => {
    document.body.dataset.mode = uiMode;
  }, [uiMode]);

  useEffect(() => {
    document.body.classList.toggle("compare", compare);
  }, [compare]);

  useEffect(() => {
    if (sceneEditorRootRef.current) {
      initSceneEditor(sendControl, sceneEditorRootRef.current, errorRef.current);
    }
  }, [toolsTab]);

  useEffect(() => {
    if (controlState?.left?.model_path !== undefined) {
      setModelPath(controlState.left.model_path || "");
    }
  }, [controlState?.left?.model_path]);

  const activeFrame = useMemo(() => {
    if (!historyFrames.length) return null;
    if (timelineLocked) return historyFrames[Math.min(timelineIndex, historyFrames.length - 1)];
    return historyFrames[historyFrames.length - 1];
  }, [historyFrames, timelineIndex, timelineLocked]);

  const activeLeft = activeFrame?.left ?? null;
  const activeRight = activeFrame?.right ?? null;

  const agentMax = Math.max(
    0,
    (controlState?.left?.n_agents ?? activeLeft?.agents?.length ?? 1) - 1,
  );

  useEffect(() => {
    if (agentIdx > agentMax) setAgentIdx(agentMax);
  }, [agentIdx, agentMax, setAgentIdx]);

  const statsLeftText = useMemo(
    () => formatStats(activeLeft?.stats, activeLeft?.agents?.length),
    [activeLeft],
  );
  const statsRightText = useMemo(
    () => formatStats(activeRight?.stats, activeRight?.agents?.length),
    [activeRight],
  );
  const agentInfoLeftText = useMemo(
    () => formatAgentInfo(activeLeft, agentIdx),
    [activeLeft, agentIdx],
  );
  const agentInfoRightText = useMemo(
    () => formatAgentInfo(activeRight, agentIdx),
    [activeRight, agentIdx],
  );
  const windLeftText = useMemo(() => formatWindText(activeLeft?.wind ?? null), [activeLeft]);
  const windRightText = useMemo(() => formatWindText(activeRight?.wind ?? null), [activeRight]);
  const energyLeft = useMemo(() => computeEnergyDisplay(activeLeft, agentIdx), [activeLeft, agentIdx]);
  const energyRight = useMemo(() => computeEnergyDisplay(activeRight, agentIdx), [activeRight, agentIdx]);

  const renderFrame = useCallback(() => {
    if (activeLeft && canvasLeftRef.current) {
      drawTelemetry(canvasLeftRef.current, activeLeft, agentIdx);
    }
    if (compare && activeRight && canvasRightRef.current) {
      drawTelemetry(canvasRightRef.current, activeRight, agentIdx);
    }
    if (!compare && canvasRightRef.current) {
      const ctx = canvasRightRef.current.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, canvasRightRef.current.width, canvasRightRef.current.height);
    }
    if (activeLeft && minimapRef.current) drawMinimap(minimapRef.current, activeLeft, agentIdx);
    if (gridLeftRef.current) {
      drawGrid(gridLeftRef.current, activeLeft?.agent_grid || null, activeLeft?.agent_attention || null);
    }
    if (gridRightRef.current) {
      drawGrid(gridRightRef.current, activeRight?.agent_grid || null, activeRight?.agent_attention || null);
    }
    drawWindWidget(windLeftRef.current, activeLeft?.wind || null);
    drawWindWidget(windRightRef.current, activeRight?.wind || null);
    gif.capture(performance.now(), canvasLeftRef.current, canvasRightRef.current);
  }, [activeLeft, activeRight, compare, agentIdx, gif]);

  useEffect(() => {
    renderFrame();
  }, [renderFrame]);

  useEffect(() => {
    return () => {
      if (tuneTimer.current) window.clearTimeout(tuneTimer.current);
    };
  }, []);

  const handleCanvasClick = useCallback(
    (evt: React.MouseEvent<HTMLCanvasElement>, telemetry: TelemetryPayload | null, canvas: HTMLCanvasElement | null) => {
      const picked = pickAgentFromTelemetry(telemetry, evt.nativeEvent as MouseEvent, canvas);
      if (picked === null || picked === undefined) return;
      setAgentIdx(picked);
      sendControl({ type: "control", action: "agent", value: picked });
    },
    [setAgentIdx],
  );

  const summary = useMemo(() => {
    const leftStats = activeLeft?.stats || {};
    const rightStats = activeRight?.stats || {};
    const leftAgents = activeLeft?.agents?.length || 0;
    const rightAgents = activeRight?.agents?.length || 0;

    const fmtPair = (lVal: string, rVal: string) => (compare ? `${lVal} | ${rVal}` : `${lVal}`);
    const fmtPct = (val: number | undefined, total: number) => {
      if (!Number.isFinite(val) || !total) return "--";
      return `${val}/${total} (${((val / total) * 100).toFixed(0)}%)`;
    };
    const fmtNum = (val: number | undefined, digits: number) =>
      Number.isFinite(val) ? Number(val).toFixed(digits) : "--";

    return {
      step: fmtPair(String(leftStats.step ?? "--"), String(rightStats.step ?? "--")),
      alive: fmtPair(fmtPct(leftStats.alive, leftAgents), fmtPct(rightStats.alive, rightAgents)),
      finish: fmtPair(fmtPct(leftStats.finished, leftAgents), fmtPct(rightStats.finished, rightAgents)),
      dist: fmtPair(fmtNum(leftStats.mean_dist, 1), fmtNum(rightStats.mean_dist, 1)),
      risk: fmtPair(fmtNum(leftStats.mean_risk, 3), fmtNum(rightStats.mean_risk, 3)),
      path: fmtPair(fmtNum(leftStats.mean_path_ratio, 3), fmtNum(rightStats.mean_path_ratio, 3)),
    };
  }, [activeLeft, activeRight, compare]);

  const timelineMax = Math.max(0, historyFrames.length - 1);
  const timelineValue = timelineLocked ? Math.min(timelineIndex, timelineMax) : timelineMax;
  const timelineLabel =
    historyFrames.length === 0
      ? "live"
      : timelineLocked && timelineValue < timelineMax
        ? `${timelineValue + 1} / ${historyFrames.length}`
        : "live";

  const overlay = controlState?.left?.overlay || {};
  const oracle = controlState?.left?.oracle || {};
  const baseTunables = useMemo(
    () => ({ ...DEFAULT_TUNABLES, ...(controlState?.left?.tunables || {}) }),
    [controlState],
  );

  const updateOverlay = useCallback(
    (partial: Record<string, unknown>) => {
      const next = { ...overlay, ...partial };
      patchControlState({ left: { ...controlState?.left, overlay: next } });
      return next;
    },
    [overlay, patchControlState, controlState],
  );

  const updateOracle = useCallback(
    (partial: Record<string, unknown>) => {
      const next = { ...oracle, ...partial };
      patchControlState({ left: { ...controlState?.left, oracle: next } });
      return next;
    },
    [oracle, patchControlState, controlState],
  );

  const tuneTimer = useRef<number | null>(null);
  const scheduleTune = useCallback(
    (next: Record<string, unknown>) => {
      if (tuneTimer.current) window.clearTimeout(tuneTimer.current);
      tuneTimer.current = window.setTimeout(() => {
        sendControl({ type: "control", action: "tune", params: next });
      }, 120);
    },
    [],
  );

  const updateTunables = useCallback(
    (partial: Record<string, unknown>) => {
      const next = { ...baseTunables, ...partial };
      patchControlState({ left: { ...controlState?.left, tunables: next } });
      scheduleTune(next);
      return next;
    },
    [baseTunables, patchControlState, controlState, scheduleTune],
  );

  useEffect(() => {
    let timer: number | null = null;
    const onResize = () => {
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        renderFrame();
      }, 120);
    };
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      if (timer) window.clearTimeout(timer);
    };
  }, [renderFrame]);

  return (
    <>
      <div id="app">
        <header id="topbar">
          <div className="brand">
            <div className="brand-title">Threat-Aware Swarm</div>
            <div className="brand-sub">Mission Control</div>
          </div>
          <div className="topbar-group">
            <label className="toggle">
              <input
                id="compare"
                type="checkbox"
                checked={compare}
                onChange={(e) => {
                  const next = e.target.checked;
                  setCompare(next);
                  resetChartsAndTimeline();
                  sendControl({ type: "control", action: "compare", value: next });
                }}
              />{" "}
              Compare
            </label>
            <label className="field">
              <span>Mode</span>
              <select
                id="uiMode"
                value={uiMode}
                onChange={(e) => {
                  const next = e.target.value === "demo" ? "demo" : "research";
                  setUiMode(next);
                }}
              >
                <option value="demo">Demo</option>
                <option value="research">Research</option>
              </select>
            </label>
          </div>
          <div className="topbar-group">
            <label className="field">
              <span>Policy L</span>
              <select
                id="policyLeft"
                value={controlState?.left?.policy ?? ""}
                onChange={(e) => {
                  const name = e.target.value;
                  patchControlState({ left: { ...controlState?.left, policy: name } });
                  sendControl({ type: "control", action: "policy", name, side: "left" });
                }}
              >
                <option value="">(loading...)</option>
                {policies.map((p) => (
                  <option key={`policy-left-${p}`} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
            <label id="policyRightWrap" className="field compare-only">
              <span>Policy R</span>
              <select
                id="policyRight"
                value={controlState?.right?.policy ?? ""}
                onChange={(e) => {
                  const name = e.target.value;
                  patchControlState({ right: { ...controlState?.right, policy: name } });
                  sendControl({ type: "control", action: "policy", name, side: "right" });
                }}
              >
                <option value="">(loading...)</option>
                {policies.map((p) => (
                  <option key={`policy-right-${p}`} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="topbar-group">
            <label className="field">
              <span>Scene</span>
              <select
                id="scene"
                value={controlState?.left?.scene ?? ""}
                onChange={(e) => {
                  const name = e.target.value;
                  resetChartsAndTimeline();
                  if (name) setSceneId(name);
                  patchControlState({ left: { ...controlState?.left, scene: name } });
                  sendControl({ type: "control", action: "scene", name });
                }}
              >
                <option value="">(loading...)</option>
                {scenes.map((s) => (
                  <option key={`scene-${s}`} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Seed</span>
              <input
                id="seed"
                type="number"
                value={controlState?.seed ?? 0}
                onChange={(e) => {
                  const value = parseInt(e.target.value, 10);
                  patchControlState({ seed: value });
                  sendControl({ type: "control", action: "seed", value });
                }}
              />
            </label>
          </div>
          <div className="topbar-group">
            <label className="field">
              <span>FPS</span>
              <input
                id="fps"
                type="range"
                min={1}
                max={60}
                value={controlState?.fps ?? 20}
                onChange={(e) => {
                  const value = parseInt(e.target.value, 10);
                  patchControlState({ fps: value });
                  sendControl({ type: "control", action: "fps", value });
                }}
              />
            </label>
            <div className="actions">
              <button id="pause" onClick={() => sendControl({ type: "control", action: "pause" })}>
                {controlState?.paused ? "Resume" : "Pause"}
              </button>
              <button id="step" onClick={() => sendControl({ type: "control", action: "step" })}>
                Step
              </button>
              <button
                id="reset"
                onClick={() => {
                  resetChartsAndTimeline();
                  sendControl({ type: "control", action: "reset" });
                }}
              >
                Reset
              </button>
              <button
                id="newMap"
                onClick={() => {
                  resetChartsAndTimeline();
                  sendControl({ type: "control", action: "new_map" });
                }}
              >
                New Map
              </button>
            </div>
          </div>
          <div id="runSummary" className="run-summary">
            <div className="summary-item">
              <span className="label">Step</span>
              <span className="value" id="sumStep">
                {summary.step}
              </span>
            </div>
            <div className="summary-item">
              <span className="label">Alive</span>
              <span className="value" id="sumAlive">
                {summary.alive}
              </span>
            </div>
            <div className="summary-item">
              <span className="label">Finish</span>
              <span className="value" id="sumFinish">
                {summary.finish}
              </span>
            </div>
            <div className="summary-item">
              <span className="label">Dist</span>
              <span className="value" id="sumDist">
                {summary.dist}
              </span>
            </div>
            <div className="summary-item">
              <span className="label">Risk</span>
              <span className="value" id="sumRisk">
                {summary.risk}
              </span>
            </div>
            <div className="summary-item">
              <span className="label">Path</span>
              <span className="value" id="sumPath">
                {summary.path}
              </span>
            </div>
          </div>
        </header>

        <div className="app-body">
          <aside id="inspector" className="sidebar research-only">
            <h3>Inspector</h3>
            <div className="card">
              <div className="card-title">Selected Agent</div>
              <div className="small">
                Agent: <span id="agentIdxLabel">{agentIdx}</span>
              </div>
              <input
                id="agentIdx"
                type="range"
                min={0}
                max={agentMax}
                value={agentIdx}
                onChange={(e) => {
                  const next = parseInt(e.target.value, 10) || 0;
                  setAgentIdx(next);
                  sendControl({ type: "control", action: "agent", value: next });
                }}
              />
              <div className="two-col inspector">
                <div>
                  <div className="small">Left grid</div>
                  <canvas id="gridLeft" ref={gridLeftRef} width={132} height={132}></canvas>
                  <pre id="agentInfoLeft" className="small">{agentInfoLeftText}</pre>
                </div>
                <div className="compare-only">
                  <div className="small">Right grid</div>
                  <canvas id="gridRight" ref={gridRightRef} width={132} height={132}></canvas>
                  <pre id="agentInfoRight" className="small">{agentInfoRightText}</pre>
                </div>
              </div>
            </div>
            <div className="card">
              <div className="card-title">Stats</div>
              <div className="two-col">
                <pre id="statsLeft">{statsLeftText}</pre>
                <pre id="statsRight" className="compare-only">{statsRightText}</pre>
              </div>
            </div>
            <div className="card">
              <div className="card-title">Sensors</div>
              <div className="two-col">
                <div className="sensor-card">
                  <div className="sensor-title">Left</div>
                  <div className="sensor-item">
                    <div className="sensor-label">Wind</div>
                    <div className="wind-box">
                      <canvas id="windLeft" ref={windLeftRef} width={72} height={72}></canvas>
                      <div id="windLeftText" className="small">
                        {windLeftText}
                      </div>
                    </div>
                  </div>
                  <div className="sensor-item">
                    <div className="sensor-label">Energy</div>
                    <div className="energy-bar">
                      <div
                        id="energyLeftFill"
                        className="energy-fill"
                        style={{ width: `${energyLeft.percent}%`, background: energyLeft.gradient }}
                      ></div>
                    </div>
                    <div id="energyLeftText" className="small">
                      {energyLeft.text}
                    </div>
                  </div>
                </div>
                <div className="sensor-card compare-only">
                  <div className="sensor-title">Right</div>
                  <div className="sensor-item">
                    <div className="sensor-label">Wind</div>
                    <div className="wind-box">
                      <canvas id="windRight" ref={windRightRef} width={72} height={72}></canvas>
                      <div id="windRightText" className="small">
                        {windRightText}
                      </div>
                    </div>
                  </div>
                  <div className="sensor-item">
                    <div className="sensor-label">Energy</div>
                    <div className="energy-bar">
                      <div
                        id="energyRightFill"
                        className="energy-fill"
                        style={{ width: `${energyRight.percent}%`, background: energyRight.gradient }}
                      ></div>
                    </div>
                    <div id="energyRightText" className="small">
                      {energyRight.text}
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div className="card">
              <div className="card-title">View</div>
              <label>
                <input
                  id="toggleGrid"
                  type="checkbox"
                  checked={!!overlay.show_grid}
                  onChange={(e) => {
                    const value = e.target.checked;
                    updateOverlay({ show_grid: value });
                    sendControl({ type: "control", action: "toggle", name: "show_grid", value });
                  }}
                />{" "}
                Show Grid
              </label>
              <br />
              <label>
                <input
                  id="toggleTrails"
                  type="checkbox"
                  checked={overlay.show_trails ?? true}
                  onChange={(e) => {
                    const value = e.target.checked;
                    updateOverlay({ show_trails: value });
                    sendControl({ type: "control", action: "toggle", name: "show_trails", value });
                  }}
                />{" "}
                Show Trails
              </label>
              <br />
              <label>
                <input
                  id="toggleThreats"
                  type="checkbox"
                  checked={overlay.show_threats ?? true}
                  onChange={(e) => {
                    const value = e.target.checked;
                    updateOverlay({ show_threats: value });
                    sendControl({ type: "control", action: "toggle", name: "show_threats", value });
                  }}
                />{" "}
                Show Threats
              </label>
              <br />
              <label>
                <input
                  id="toggleAttention"
                  type="checkbox"
                  checked={!!overlay.show_attention}
                  onChange={(e) => {
                    const value = e.target.checked;
                    updateOverlay({ show_attention: value });
                    sendControl({ type: "control", action: "toggle", name: "show_attention", value });
                  }}
                />{" "}
                Show Agent Attention
              </label>
              <label className="field">
                <span>Attention Channel</span>
                <select
                  id="attentionChannel"
                  value={overlay.attention_channel || "sum"}
                  onChange={(e) => {
                    const value = e.target.value;
                    updateOverlay({ attention_channel: value });
                    sendControl({ type: "control", action: "attention_channel", value });
                  }}
                >
                  <option value="sum">Sum</option>
                  <option value="x">Action X</option>
                  <option value="y">Action Y</option>
                </select>
              </label>
            </div>
            <div className="card">
              <div className="card-title">Oracle</div>
              <label>
                <input
                  id="toggleOracle"
                  type="checkbox"
                  checked={!!oracle.enabled}
                  onChange={(e) => {
                    const enabled = e.target.checked;
                    updateOracle({ enabled });
                    sendControl({ type: "control", action: "oracle", enabled });
                  }}
                />{" "}
                Oracle path
              </label>
              <br />
              <label>
                <input
                  id="toggleOracleAsync"
                  type="checkbox"
                  checked={oracle.async !== undefined ? !!oracle.async : true}
                  onChange={(e) => {
                    const async = e.target.checked;
                    updateOracle({ async });
                    sendControl({ type: "control", action: "oracle", async });
                  }}
                />{" "}
                Oracle async
              </label>
              <br />
              <label className="field">
                <span>Update interval</span>
                <input
                  id="oracleInterval"
                  type="number"
                  min={1}
                  step={1}
                  value={oracle.update_interval ?? 10}
                  onChange={(e) => {
                    const interval = Math.max(1, parseInt(e.target.value, 10) || 1);
                    updateOracle({ update_interval: interval });
                    sendControl({ type: "control", action: "oracle", interval });
                  }}
                />
              </label>
            </div>
            <div className="card">
              <div className="card-title">System</div>
              <div id="error" ref={errorRef} className="error">
                {lastError}
              </div>
            </div>
          </aside>

          <main className="center-stage">
            <section id="canvasArea">
              <div id="canvasStack">
                <div className="panel" id="panelLeft">
                  <canvas
                    id="canvasLeft"
                    ref={canvasLeftRef}
                    onClick={(e) => handleCanvasClick(e, activeLeft, canvasLeftRef.current)}
                  ></canvas>
                </div>
                <div className="panel compare-only" id="panelRight">
                  <canvas
                    id="canvasRight"
                    ref={canvasRightRef}
                    onClick={(e) => handleCanvasClick(e, activeRight, canvasRightRef.current)}
                  ></canvas>
                </div>
              </div>
              <div id="runSummaryOverlay" className="overlay-card summary-overlay">
                <div className="overlay-title">Run Summary</div>
                <div className="summary-grid">
                  {[
                    { label: "Step", value: summary.step },
                    { label: "Alive", value: summary.alive },
                    { label: "Finish", value: summary.finish },
                    { label: "Dist", value: summary.dist },
                    { label: "Risk", value: summary.risk },
                    { label: "Path", value: summary.path },
                  ].map((item) => (
                    <div key={`summary-${item.label}`} className="summary-row">
                      <span className="label">{item.label}</span>
                      <span className="value">{item.value}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div id="legend" className="overlay-card">
                <div className="overlay-title">Legend</div>
                <div className="legend-row">
                  <span className="dot agent"></span> Agent
                </div>
                <div className="legend-row">
                  <span className="dot goal"></span> Goal
                </div>
                <div className="legend-row">
                  <span className="dot threat"></span> Threat
                </div>
                <div className="legend-row">
                  <span className="dot oracle"></span> Oracle
                </div>
              </div>
              <div id="minimap" className="overlay-card">
                <div className="overlay-title">Mini Map</div>
                <canvas id="minimapCanvas" ref={minimapRef} width={140} height={140}></canvas>
              </div>
            </section>

            <section id="timelinePanel" className="research-only bottom-panel">
              <div className="dock-grid">
                <div className="card timeline-card">
                  <div className="card-title">Timeline</div>
                  <div className="timeline-wrap">
                    <input
                      id="timeline"
                      type="range"
                      min={0}
                      max={timelineMax}
                      value={timelineValue}
                      onChange={(e) => {
                        const idx = parseInt(e.target.value, 10) || 0;
                        setTimelineIndex(idx);
                      }}
                    />
                    <div id="timelineMarkers">
                      {markers.map((m, idx) => {
                        const left = timelineMax > 0 ? (m.index / timelineMax) * 100 : 0;
                        return (
                          <div
                            key={`marker-${idx}-${m.type}-${m.side}-${m.index}`}
                            className={`timeline-marker ${m.type} ${m.side}`}
                            style={{ left: `${left}%` }}
                          />
                        );
                      })}
                    </div>
                  </div>
                  <div className="two-col">
                    <div className="small">
                      Frame: <span id="timelineLabel">{timelineLabel}</span>
                    </div>
                    <button
                      id="timelineLive"
                      type="button"
                      onClick={() => {
                        setTimelineLive();
                      }}
                    >
                      Live
                    </button>
                  </div>
                </div>
                <div className="card charts-card">
                  <div className="card-title">Charts</div>
                  <div className="charts-row">
                    <canvas id="chartPath" className="chart"></canvas>
                    <canvas id="chartRisk" className="chart"></canvas>
                    <canvas id="chartEnergy" className="chart"></canvas>
                  </div>
                </div>
              </div>
            </section>
          </main>

          <aside id="toolsSidebar" className="sidebar research-only">
            <div className="sidebar-header">
              <h3>Tools</h3>
            </div>

            <Tabs.Root className="tabs-root" value={toolsTab} onValueChange={handleTabChange}>
              <Tabs.List className="tabs">
                <Tabs.Trigger className="tab-btn" value="models">
                  Models
                </Tabs.Trigger>
                <Tabs.Trigger className="tab-btn" value="editor">
                  Editor
                </Tabs.Trigger>
                <Tabs.Trigger className="tab-btn" value="tuning">
                  Tune
                </Tabs.Trigger>
              </Tabs.List>

              <div className="tabs-content">
                <Tabs.Content id="tab-models" className="tab-pane" value="models" forceMount>
                <div className="card">
                  <div className="card-title">Models</div>
                  <label>PPO model</label>
                  <select
                    id="modelSelect"
                    value={modelSelect}
                    onChange={(e) => setModelSelect(e.target.value)}
                  >
                    <option value="">(loading...)</option>
                    {models.map((m) => (
                      <option key={`model-${m}`} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                  <input
                    id="modelPath"
                    type="text"
                    placeholder="runs/run_*/models/final.zip"
                    value={modelPath}
                    onChange={(e) => setModelPath(e.target.value)}
                  />
                  <div className="two-col">
                    <button
                      id="loadModelLeft"
                      onClick={() => {
                        const value = modelPath || modelSelect || "";
                        sendControl({
                          type: "control",
                          action: "model",
                          value,
                          deterministic: controlState?.left?.deterministic ?? true,
                          side: "left",
                        });
                      }}
                    >
                      Load Left
                    </button>
                    <button
                      id="loadModelRight"
                      className="compare-only"
                      onClick={() => {
                        const value = modelPath || modelSelect || "";
                        sendControl({
                          type: "control",
                          action: "model",
                          value,
                          deterministic: controlState?.right?.deterministic ?? true,
                          side: "right",
                        });
                      }}
                    >
                      Load Right
                    </button>
                  </div>
                  <div className="two-col">
                    <label>
                      <input
                        id="detLeft"
                        type="checkbox"
                        checked={controlState?.left?.deterministic ?? true}
                        onChange={(e) => {
                          const value = e.target.checked;
                          patchControlState({ left: { ...controlState?.left, deterministic: value } });
                          sendControl({ type: "control", action: "deterministic", value, side: "left" });
                        }}
                      />{" "}
                      Deterministic L
                    </label>
                    <label className="compare-only">
                      <input
                        id="detRight"
                        type="checkbox"
                        checked={controlState?.right?.deterministic ?? true}
                        onChange={(e) => {
                          const value = e.target.checked;
                          patchControlState({ right: { ...controlState?.right, deterministic: value } });
                          sendControl({ type: "control", action: "deterministic", value, side: "right" });
                        }}
                      />{" "}
                      Deterministic R
                    </label>
                  </div>
                </div>
                <div className="card">
                  <div className="card-title">GIF export</div>
                  <div className="two-col">
                    <select
                      id="gifSide"
                      value={gif.side}
                      onChange={(e) => gif.setSide(e.target.value === "right" ? "right" : "left")}
                    >
                      <option value="left">left</option>
                      <option value="right">right</option>
                    </select>
                    <input
                      id="gifFps"
                      type="number"
                      min={1}
                      max={60}
                      step={1}
                      value={gif.fps}
                      onChange={(e) => gif.setFps(parseInt(e.target.value, 10) || 1)}
                    />
                  </div>
                  <div className="two-col">
                    <button id="gifStart" onClick={gif.start}>Record GIF</button>
                    <button id="gifStop" onClick={gif.stop}>Stop</button>
                  </div>
                  <div className="small" id="gifStatus">{gif.status}</div>
                  <a
                    id="gifLink"
                    className="small"
                    href={gif.url || "#"}
                    target="_blank"
                    style={{ display: gif.url ? "" : "none" }}
                    download={gif.downloadName || undefined}
                  >
                    Download GIF
                  </a>
                </div>
                </Tabs.Content>

                <Tabs.Content id="tab-editor" className="tab-pane" value="editor" forceMount>
                <div ref={sceneEditorRootRef}>
                <div className="card">
                  <div className="card-title">Scene Editor</div>
                  <canvas id="sceneEditor" width={280} height={280}></canvas>
                  <div className="two-col mt-2">
                    <select id="sceneTool">
                      <option value="start">Start</option>
                      <option value="goal">Goal</option>
                      <option value="wall">Wall</option>
                      <option value="threat">Threat</option>
                      <option value="move">Move</option>
                      <option value="erase">Erase</option>
                    </select>
                    <button id="sceneClear">Clear</button>
                  </div>

                  <div className="editor-settings mt-2">
                    <label>
                      <input id="sceneSnap" type="checkbox" defaultChecked /> Snap to grid
                    </label>
                    <label>Grid step</label>
                    <input id="sceneSnapStep" type="number" defaultValue={2} step={0.5} />
                    <label>Field size</label>
                    <input id="sceneField" type="number" defaultValue={100} />
                    <label>Max steps</label>
                    <input id="sceneMaxSteps" type="number" defaultValue={600} />
                    <label>Start sigma</label>
                    <input id="sceneStartSigma" type="number" defaultValue={2.0} step={0.1} />
                    <label>Threat radius</label>
                    <input id="sceneThreatRadius" type="number" defaultValue={12} step={0.5} />
                    <label>Threat intensity</label>
                    <input id="sceneThreatIntensity" type="number" defaultValue={0.1} step={0.01} />
                    <label>
                      <input id="sceneWindEnabled" type="checkbox" defaultChecked /> Wind enabled
                    </label>
                    <label>Wind theta</label>
                    <input id="sceneWindTheta" type="number" defaultValue={0.15} step={0.01} />
                    <label>Wind sigma</label>
                    <input id="sceneWindSigma" type="number" defaultValue={0.3} step={0.01} />
                    <label>Wind seed</label>
                    <input id="sceneWindSeed" type="number" placeholder="auto" />
                    <label>Threat type</label>
                    <select id="sceneThreatType">
                      <option value="static">static</option>
                      <option value="linear">linear</option>
                      <option value="brownian">brownian</option>
                      <option value="chaser">chaser</option>
                    </select>
                    <label id="sceneThreatSpeedLabel">Threat speed</label>
                    <input id="sceneThreatSpeed" type="number" defaultValue={2.0} step={0.1} />
                    <label id="sceneThreatAngleLabel">Threat angle</label>
                    <input id="sceneThreatAngle" type="number" defaultValue={0} step={5} />
                    <label id="sceneThreatNoiseLabel">Noise scale</label>
                    <input id="sceneThreatNoise" type="number" defaultValue={0.5} step={0.1} />
                    <label id="sceneThreatVisionLabel">Vision radius</label>
                    <input id="sceneThreatVision" type="number" defaultValue={30} step={1} />
                    <label>
                      <input id="sceneOracleBlock" type="checkbox" /> Oracle block
                    </label>
                  </div>

                  <hr className="divider" />

                  <label>Scene ID</label>
                  <input id="sceneId" type="text" placeholder="custom_scene" />
                  <div className="two-col mt-2">
                    <button id="scenePreview">Preview</button>
                    <button id="sceneSave">Save</button>
                    <button id="sceneDelete">Delete</button>
                    <button id="sceneRefresh">Refresh</button>
                  </div>

                  <hr className="divider" />
                  <label>Import / Export</label>
                  <select id="sceneFormat" className="mt-2">
                    <option value="json">JSON</option>
                    <option value="yaml">YAML</option>
                  </select>
                  <textarea id="sceneText" rows={4} className="mt-2" placeholder="Paste JSON/YAML here..."></textarea>
                  <input id="sceneFile" type="file" accept=".json,.yaml,.yml" className="mt-2" />
                  <div className="two-col mt-2">
                    <button id="sceneImport">Import</button>
                    <button id="sceneExport">Export</button>
                  </div>
                </div>
                </div>
                </Tabs.Content>

                <Tabs.Content id="tab-tuning" className="tab-pane" value="tuning" forceMount>
                <div className="card">
                  <div className="card-title">Tunables</div>
                  <div className="small mb-2">Applied live or on next reset.</div>
                  <label>Threat speed</label>
                  <input
                    id="tuneThreatSpeed"
                    type="range"
                    min={0}
                    max={3}
                    step={0.1}
                    value={baseTunables.threat_speed}
                    onChange={(e) => updateTunables({ threat_speed: parseFloat(e.target.value) })}
                  />
                  <label>Threat mode</label>
                  <select
                    id="tuneThreatMode"
                    value={baseTunables.threat_mode}
                    onChange={(e) => updateTunables({ threat_mode: e.target.value })}
                  >
                    <option value="static">Static</option>
                    <option value="dynamic">Dynamic</option>
                    <option value="mixed">Mixed</option>
                  </select>
                  <label>Wall friction</label>
                  <input
                    id="tuneWallFriction"
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={baseTunables.wall_friction}
                    onChange={(e) => updateTunables({ wall_friction: parseFloat(e.target.value) })}
                  />
                  <label>Reward w_risk</label>
                  <input
                    id="tuneWRisk"
                    type="range"
                    min={0}
                    max={5}
                    step={0.1}
                    value={baseTunables.w_risk}
                    onChange={(e) => updateTunables({ w_risk: parseFloat(e.target.value) })}
                  />
                  <label>Reward w_progress</label>
                  <input
                    id="tuneWProgress"
                    type="range"
                    min={0}
                    max={10}
                    step={0.2}
                    value={baseTunables.w_progress}
                    onChange={(e) => updateTunables({ w_progress: parseFloat(e.target.value) })}
                  />
                  <label>Reward w_wall</label>
                  <input
                    id="tuneWWall"
                    type="range"
                    min={0}
                    max={30}
                    step={1}
                    value={baseTunables.w_wall}
                    onChange={(e) => updateTunables({ w_wall: parseFloat(e.target.value) })}
                  />
                  <label>Physics mode</label>
                  <select id="tunePhysicsMode" value="dynamic" disabled>
                    <option value="dynamic">dynamic</option>
                  </select>
                  <label>Max accel</label>
                  <input
                    id="tuneMaxAccel"
                    type="number"
                    step={0.5}
                    value={baseTunables.max_accel}
                    onChange={(e) => updateTunables({ max_accel: parseFloat(e.target.value) || 0 })}
                  />
                  <label>Drag</label>
                  <input
                    id="tuneDrag"
                    type="number"
                    step={0.01}
                    value={baseTunables.drag}
                    onChange={(e) => updateTunables({ drag: parseFloat(e.target.value) || 0 })}
                  />
                  <label>Obs noise: target</label>
                  <input
                    id="tuneObsNoiseTarget"
                    type="number"
                    step={0.01}
                    value={baseTunables.obs_noise_target}
                    onChange={(e) => updateTunables({ obs_noise_target: parseFloat(e.target.value) || 0 })}
                  />
                  <label>Obs noise: vel</label>
                  <input
                    id="tuneObsNoiseVel"
                    type="number"
                    step={0.01}
                    value={baseTunables.obs_noise_vel}
                    onChange={(e) => updateTunables({ obs_noise_vel: parseFloat(e.target.value) || 0 })}
                  />
                  <label>Obs noise: grid</label>
                  <input
                    id="tuneObsNoiseGrid"
                    type="number"
                    step={0.01}
                    value={baseTunables.obs_noise_grid}
                    onChange={(e) => updateTunables({ obs_noise_grid: parseFloat(e.target.value) || 0 })}
                  />
                  <label className="mt-2">
                    <input
                      id="tuneDomainRand"
                      type="checkbox"
                      checked={!!baseTunables.domain_randomization}
                      onChange={(e) => updateTunables({ domain_randomization: e.target.checked })}
                    />{" "}
                    Domain randomization
                  </label>
                  <label>Max speed min</label>
                  <input
                    id="tuneMaxSpeedMin"
                    type="number"
                    step={0.1}
                    value={baseTunables.dr_max_speed_min}
                    onChange={(e) => updateTunables({ dr_max_speed_min: parseFloat(e.target.value) || 0 })}
                  />
                  <label>Max speed max</label>
                  <input
                    id="tuneMaxSpeedMax"
                    type="number"
                    step={0.1}
                    value={baseTunables.dr_max_speed_max}
                    onChange={(e) => updateTunables({ dr_max_speed_max: parseFloat(e.target.value) || 0 })}
                  />
                  <label>Drag min</label>
                  <input
                    id="tuneDragMin"
                    type="number"
                    step={0.01}
                    value={baseTunables.dr_drag_min}
                    onChange={(e) => updateTunables({ dr_drag_min: parseFloat(e.target.value) || 0 })}
                  />
                  <label>Drag max</label>
                  <input
                    id="tuneDragMax"
                    type="number"
                    step={0.01}
                    value={baseTunables.dr_drag_max}
                    onChange={(e) => updateTunables({ dr_drag_max: parseFloat(e.target.value) || 0 })}
                  />
                </div>
                </Tabs.Content>
              </div>
            </Tabs.Root>
          </aside>
        </div>
      </div>

      <div id="loadingOverlay" className={`loading-overlay${loading ? "" : " hidden"}`}>
        <div className="loading-content">
          <div className="loading-title">Threat-Aware Swarm</div>
          <div className="loading-sub">Mission Control loading...</div>
          <div className="loading-bar">
            <div className="loading-bar-fill"></div>
          </div>
        </div>
      </div>
    </>
  );
}
