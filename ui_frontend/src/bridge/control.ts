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

function getBridge(): UIBridge | undefined {
  return (window as unknown as { UIBridge?: UIBridge }).UIBridge;
}

export function sendControl(msg: ControlMessage) {
  const bridge = getBridge();
  if (bridge && typeof bridge.send === "function") {
    bridge.send(msg);
  } else {
    console.warn("UIBridge.send недоступен", msg);
  }
}

export function resetChartsAndTimeline() {
  const bridge = getBridge();
  if (bridge?.resetCharts) bridge.resetCharts();
  if (bridge?.resetTimeline) bridge.resetTimeline();
}
