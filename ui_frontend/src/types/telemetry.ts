export type Vec2 = [number, number];

export type TelemetryStats = {
  step: number;
  alive: number | null;
  finished: number | null;
  in_goal: number | null;
  mean_dist: number | null;
  mean_risk: number | null;
  mean_path_ratio: number | null;
  mean_threat_collisions: number | null;
  mean_energy: number | null;
  mean_energy_level: number | null;
};

export type TelemetryAgent = {
  id: string;
  index: number;
  pos: Vec2;
  vel: Vec2;
  alive: boolean;
  finished: boolean;
  in_goal: boolean;
  dist: number | null;
  risk_p: number | null;
  path_ratio: number | null;
  collided: boolean;
  threat_collided: boolean;
  min_dist_to_threat: number | null;
  energy: number | null;
  energy_level: number | null;
  action: Vec2 | null;
};

export type TelemetryThreat = {
  pos: Vec2;
  radius: number;
  intensity: number;
  kind: string;
  dynamic: boolean;
  oracle_block: boolean;
};

export type AgentObs = {
  vector?: number[];
  to_target?: number[];
  vel?: number[];
  walls?: number[];
  last_action?: number[];
  measured_accel?: number[];
  energy_level?: number;
};

export type TelemetryPayload = {
  schema_version: string;
  stats: TelemetryStats;
  agents: TelemetryAgent[];
  threats: TelemetryThreat[];
  walls: number[][];
  oracle_path: Vec2[];
  field_size: number;
  goal_radius: number;
  target_pos: Vec2 | null;
  screen_size: number;
  grid_res: number;
  agent_index: number;
  agent_id: string | null;
  agent_grid: number[][] | null;
  agent_obs: AgentObs | null;
  agent_attention?: number[][] | null;
  wind: Vec2 | null;
};

export type TelemetryFrame = {
  schema_version?: string;
  left?: TelemetryPayload | null;
  right?: TelemetryPayload | null;
};

export type ControlSideState = {
  policy?: string;
  scene?: string;
  seed?: number;
  n_agents?: number;
  model_path?: string;
  deterministic?: boolean;
  overlay?: Record<string, unknown>;
  tunables?: Record<string, unknown>;
  oracle?: Record<string, unknown>;
};

export type ControlState = {
  left?: ControlSideState;
  right?: ControlSideState;
  seed?: number;
  fps?: number;
  paused?: boolean;
  compare?: boolean;
  last_error?: string;
  [key: string]: unknown;
};

export type InitMessage = {
  policies?: string[];
  scenes?: string[];
  models?: string[];
  state?: ControlState;
};

export type ServerMessage = {
  type?: string;
  telemetry?: TelemetryFrame;
  state?: ControlState;
  policies?: string[];
  scenes?: string[];
  models?: string[];
  [key: string]: unknown;
};
