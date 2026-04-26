import { create } from "zustand";

type UIState = {
  toolsTab: string;
  setToolsTab: (tab: string) => void;
  compare: boolean;
  setCompare: (value: boolean) => void;
  uiMode: "demo" | "research";
  setUiMode: (value: "demo" | "research") => void;
  agentIdx: number;
  setAgentIdx: (value: number) => void;
};

export const useUIStore = create<UIState>((set) => ({
  toolsTab: "models",
  setToolsTab: (tab) => set({ toolsTab: tab }),
  compare: false,
  setCompare: (value) => set({ compare: value }),
  uiMode: "research",
  setUiMode: (value) => set({ uiMode: value }),
  agentIdx: 0,
  setAgentIdx: (value) => set({ agentIdx: value }),
}));
