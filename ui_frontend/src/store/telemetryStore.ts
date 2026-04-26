import { create } from "zustand";
import type {
  ControlState,
  InitMessage,
  TelemetryAgent,
  TelemetryFrame,
  TelemetryPayload,
} from "../types/telemetry";

const HISTORY_MAX = 500;
const CHART_MAX = 200;

type Marker = {
  index: number;
  type: string;
  side: "left" | "right";
};

type ChartData = {
  labels: number[];
  pathL: Array<number | null>;
  pathR: Array<number | null>;
  riskL: Array<number | null>;
  riskR: Array<number | null>;
  energyL: Array<number | null>;
  energyR: Array<number | null>;
};

type AgentMarkerState = Pick<TelemetryAgent, "alive" | "finished" | "in_goal" | "threat_collided">;

type TelemetryState = {
  loading: boolean;
  policies: string[];
  scenes: string[];
  models: string[];
  controlState: ControlState;
  lastError: string;
  telemetryLeft: TelemetryPayload | null;
  telemetryRight: TelemetryPayload | null;
  historyFrames: TelemetryFrame[];
  timelineIndex: number;
  timelineLocked: boolean;
  markers: Marker[];
  chartData: ChartData;
  setInit: (msg: InitMessage) => void;
  setState: (state: ControlState | null | undefined) => void;
  patchControlState: (partial: Partial<ControlState>) => void;
  setError: (message: string) => void;
  pushFrame: (frame: TelemetryFrame | { telemetry?: TelemetryFrame }) => void;
  resetHistory: () => void;
  setTimelineIndex: (index: number) => void;
  setTimelineLive: () => void;
  resetCharts: () => void;
  setLoading: (value: boolean) => void;
};

const prevLeft = new Map<number, AgentMarkerState>();
const prevRight = new Map<number, AgentMarkerState>();

function normalizeFrame(frame: TelemetryFrame | { telemetry?: TelemetryFrame }) {
  const candidate = frame as { telemetry?: TelemetryFrame } & TelemetryFrame;
  if (candidate && candidate.telemetry) return candidate.telemetry;
  return candidate;
}

function recordSideMarkers(
  agents: TelemetryAgent[] | null | undefined,
  prevMap: Map<number, AgentMarkerState>,
  side: "left" | "right",
  frameIndex: number,
  markers: Marker[],
) {
  if (!agents) return markers;
  const next = markers.slice();
  agents.forEach((agent) => {
    const prev = prevMap.get(agent.index) || {};
    if (prev.alive && !agent.alive) next.push({ index: frameIndex, type: "death", side });
    if (!prev.finished && agent.finished) next.push({ index: frameIndex, type: "finish", side });
    if (!prev.in_goal && agent.in_goal) next.push({ index: frameIndex, type: "goal", side });
    if (!prev.threat_collided && agent.threat_collided) next.push({ index: frameIndex, type: "hit", side });
    prevMap.set(agent.index, {
      alive: agent.alive,
      finished: agent.finished,
      in_goal: agent.in_goal,
      threat_collided: agent.threat_collided,
    });
  });
  return next;
}

function createEmptyChartData(): ChartData {
  return {
    labels: [],
    pathL: [],
    pathR: [],
    riskL: [],
    riskR: [],
    energyL: [],
    energyR: [],
  };
}

export const useTelemetryStore = create<TelemetryState>((set, get) => ({
  loading: true,
  policies: [],
  scenes: [],
  models: [],
  controlState: {},
  lastError: "",
  telemetryLeft: null,
  telemetryRight: null,
  historyFrames: [],
  timelineIndex: 0,
  timelineLocked: false,
  markers: [],
  chartData: createEmptyChartData(),
  setInit: (msg) =>
    set((state) => ({
      policies: Array.isArray(msg?.policies) ? msg.policies : state.policies,
      scenes: Array.isArray(msg?.scenes) ? msg.scenes : state.scenes,
      models: Array.isArray(msg?.models) ? msg.models : state.models,
    })),
  setState: (state) =>
    set({
      controlState: state || {},
      lastError: state?.last_error || "",
    }),
  setError: (message) =>
    set({
      lastError: message || "",
    }),
  patchControlState: (partial) =>
    set((state) => ({
      controlState: { ...state.controlState, ...(partial || {}) },
    })),
  pushFrame: (frame) => {
    const payload = normalizeFrame(frame);
    const left = payload?.left || null;
    const right = payload?.right || null;
    const history = get().historyFrames.slice();
    history.push(payload);
    if (history.length > HISTORY_MAX) history.shift();

    const frameIndex = Math.max(0, history.length - 1);
    let markers = get().markers;
    markers = recordSideMarkers(left?.agents, prevLeft, "left", frameIndex, markers);
    markers = recordSideMarkers(right?.agents, prevRight, "right", frameIndex, markers);

    const chart = { ...get().chartData };
    const label = chart.labels.length;
    chart.labels = chart.labels.concat(label);
    chart.pathL = chart.pathL.concat(left?.stats?.mean_path_ratio ?? null);
    chart.pathR = chart.pathR.concat(right?.stats?.mean_path_ratio ?? null);
    chart.riskL = chart.riskL.concat(left?.stats?.mean_risk ?? null);
    chart.riskR = chart.riskR.concat(right?.stats?.mean_risk ?? null);
    chart.energyL = chart.energyL.concat(left?.stats?.mean_energy_level ?? null);
    chart.energyR = chart.energyR.concat(right?.stats?.mean_energy_level ?? null);
    if (chart.labels.length > CHART_MAX) {
      chart.labels = chart.labels.slice(-CHART_MAX);
      chart.pathL = chart.pathL.slice(-CHART_MAX);
      chart.pathR = chart.pathR.slice(-CHART_MAX);
      chart.riskL = chart.riskL.slice(-CHART_MAX);
      chart.riskR = chart.riskR.slice(-CHART_MAX);
      chart.energyL = chart.energyL.slice(-CHART_MAX);
      chart.energyR = chart.energyR.slice(-CHART_MAX);
    }

    const locked = get().timelineLocked;
    const nextIndex = locked ? Math.min(get().timelineIndex, history.length - 1) : history.length - 1;
    set({
      telemetryLeft: left,
      telemetryRight: right,
      historyFrames: history,
      timelineIndex: nextIndex,
      markers,
      chartData: chart,
    });
  },
  resetHistory: () => {
    prevLeft.clear();
    prevRight.clear();
    set({
      historyFrames: [],
      timelineIndex: 0,
      timelineLocked: false,
      markers: [],
    });
  },
  setTimelineIndex: (index) =>
    set({
      timelineLocked: true,
      timelineIndex: Math.max(0, index),
    }),
  setTimelineLive: () =>
    set((state) => ({
      timelineLocked: false,
      timelineIndex: Math.max(0, state.historyFrames.length - 1),
    })),
  resetCharts: () => set({ chartData: createEmptyChartData() }),
  setLoading: (value) => set({ loading: value }),
}));
