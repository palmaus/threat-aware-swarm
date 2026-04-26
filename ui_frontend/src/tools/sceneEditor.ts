import { getSceneEditorMetrics, prepareCanvasStatic } from "../renderer/canvas";

type SceneEditorRefs = {
  sceneEditor: HTMLCanvasElement | null;
  sceneTool: HTMLSelectElement | null;
  sceneClear: HTMLButtonElement | null;
  sceneSnap: HTMLInputElement | null;
  sceneSnapStep: HTMLInputElement | null;
  sceneField: HTMLInputElement | null;
  sceneMaxSteps: HTMLInputElement | null;
  sceneStartSigma: HTMLInputElement | null;
  sceneThreatRadius: HTMLInputElement | null;
  sceneThreatIntensity: HTMLInputElement | null;
  sceneWindEnabled: HTMLInputElement | null;
  sceneWindTheta: HTMLInputElement | null;
  sceneWindSigma: HTMLInputElement | null;
  sceneWindSeed: HTMLInputElement | null;
  sceneThreatType: HTMLSelectElement | null;
  sceneThreatSpeedLabel: HTMLElement | null;
  sceneThreatSpeed: HTMLInputElement | null;
  sceneThreatAngleLabel: HTMLElement | null;
  sceneThreatAngle: HTMLInputElement | null;
  sceneThreatNoiseLabel: HTMLElement | null;
  sceneThreatNoise: HTMLInputElement | null;
  sceneThreatVisionLabel: HTMLElement | null;
  sceneThreatVision: HTMLInputElement | null;
  sceneOracleBlock: HTMLInputElement | null;
  sceneId: HTMLInputElement | null;
  scenePreview: HTMLButtonElement | null;
  sceneSave: HTMLButtonElement | null;
  sceneDelete: HTMLButtonElement | null;
  sceneRefresh: HTMLButtonElement | null;
  sceneFormat: HTMLSelectElement | null;
  sceneText: HTMLTextAreaElement | null;
  sceneFile: HTMLInputElement | null;
  sceneImport: HTMLButtonElement | null;
  sceneExport: HTMLButtonElement | null;
};

type SceneEditorApi = {
  applyScene: (scene: any) => void;
  setSceneText: (text: string) => void;
  setSceneId: (sceneId: string) => void;
  setError: (text: string) => void;
  redraw: () => void;
};

const CONST = {
  EDITOR_PAD: 6,
};

const HANDLE_PX = 6;

let refs: SceneEditorRefs | null = null;
let editorScene: any = null;
let wallDrag: any = null;
let dragItem: any = null;
let selectedWallIndex: number | null = null;
let initialized = false;
let sendControl: ((payload: any) => void) | null = null;
let errorEl: HTMLElement | null = null;

function resolveEl<T extends HTMLElement>(root: HTMLElement | null, id: string): T | null {
  if (root) {
    return root.querySelector(`#${id}`) as T | null;
  }
  return document.getElementById(id) as T | null;
}

function defaultEditorScene() {
  const field = parseFloat(refs?.sceneField?.value || "100") || 100;
  const wind = {
    enabled: refs?.sceneWindEnabled?.checked ?? true,
    ou_theta: parseFloat(refs?.sceneWindTheta?.value || "0.15") || 0.15,
    ou_sigma: parseFloat(refs?.sceneWindSigma?.value || "0.3") || 0.3,
  };
  const windSeedRaw = refs?.sceneWindSeed?.value;
  if (windSeedRaw !== undefined && windSeedRaw !== null && String(windSeedRaw).trim() !== "") {
    const parsed = parseInt(windSeedRaw, 10);
    if (!Number.isNaN(parsed)) wind.seed = parsed;
  }
  return {
    id: refs?.sceneId?.value || "",
    field_size: field,
    start_center: [field * 0.1, field * 0.1],
    start_sigma: parseFloat(refs?.sceneStartSigma?.value || "2.0") || 2.0,
    target_pos: [field * 0.9, field * 0.9],
    walls: [],
    threats: [],
    max_steps: parseInt(refs?.sceneMaxSteps?.value || "600", 10) || 600,
    wind,
  };
}

function editorField() {
  return parseFloat(refs?.sceneField?.value || editorScene?.field_size || "100") || 100;
}

function clampValue(val: number, minV: number, maxV: number) {
  return Math.max(minV, Math.min(maxV, val));
}

function snapValue(val: number) {
  if (!refs?.sceneSnap || !refs.sceneSnap.checked) return val;
  const step = parseFloat(refs.sceneSnapStep?.value || "1") || 1;
  if (step <= 0) return val;
  return Math.round(val / step) * step;
}

function snapPoint(pt: number[], field: number) {
  const x = snapValue(pt[0]);
  const y = snapValue(pt[1]);
  return [clampValue(x, 0, field), clampValue(y, 0, field)];
}

function editorToCanvas(p: number[], field: number, size: number, offsetX = 0, offsetY = 0) {
  const scale = (size - 2 * CONST.EDITOR_PAD) / field;
  const x = p[0] * scale + CONST.EDITOR_PAD + offsetX;
  const y = (field - p[1]) * scale + CONST.EDITOR_PAD + offsetY;
  return [x, y, scale] as const;
}

function canvasToEditor(x: number, y: number, field: number, size: number, offsetX = 0, offsetY = 0) {
  const scale = (size - 2 * CONST.EDITOR_PAD) / field;
  const wx = (x - offsetX - CONST.EDITOR_PAD) / scale;
  const wy = field - (y - offsetY - CONST.EDITOR_PAD) / scale;
  return [Math.max(0, Math.min(field, wx)), Math.max(0, Math.min(field, wy))];
}

function drawSceneEditor() {
  if (!refs?.sceneEditor) return;
  const prepared = prepareCanvasStatic(refs.sceneEditor);
  if (!prepared) return;
  const { ctx, size, offsetX, offsetY, width, height } = prepared;
  const field = editorField();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0b1220";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(148,163,184,0.4)";
  ctx.lineWidth = 1;
  ctx.strokeRect(
    CONST.EDITOR_PAD + offsetX,
    CONST.EDITOR_PAD + offsetY,
    size - 2 * CONST.EDITOR_PAD,
    size - 2 * CONST.EDITOR_PAD,
  );

  if (editorScene?.walls) {
    ctx.fillStyle = "rgba(100,116,139,0.6)";
    editorScene.walls.forEach((w: number[]) => {
      const p1 = editorToCanvas([w[0], w[1]], field, size, offsetX, offsetY);
      const p2 = editorToCanvas([w[2], w[3]], field, size, offsetX, offsetY);
      const x = Math.min(p1[0], p2[0]);
      const y = Math.min(p1[1], p2[1]);
      const wdt = Math.abs(p2[0] - p1[0]);
      const hgt = Math.abs(p2[1] - p1[1]);
      ctx.fillRect(x, y, wdt, hgt);
      ctx.strokeStyle = "rgba(15, 23, 42, 0.9)";
      ctx.strokeRect(x, y, wdt, hgt);
    });
  }

  if (selectedWallIndex !== null && editorScene?.walls && editorScene.walls[selectedWallIndex]) {
    const w = editorScene.walls[selectedWallIndex];
    const corners = [
      [w[0], w[1]],
      [w[2], w[1]],
      [w[2], w[3]],
      [w[0], w[3]],
    ];
    ctx.fillStyle = "#facc15";
    corners.forEach((c: number[]) => {
      const pt = editorToCanvas(c, field, size, offsetX, offsetY);
      ctx.fillRect(pt[0] - HANDLE_PX, pt[1] - HANDLE_PX, HANDLE_PX * 2, HANDLE_PX * 2);
      ctx.strokeStyle = "#0f172a";
      ctx.strokeRect(pt[0] - HANDLE_PX, pt[1] - HANDLE_PX, HANDLE_PX * 2, HANDLE_PX * 2);
    });
  }

  if (editorScene?.threats) {
    editorScene.threats.forEach((t: any) => {
      const pt = editorToCanvas(t.pos, field, size, offsetX, offsetY);
      const r = (t.radius || 1) * pt[2];
      ctx.fillStyle = "rgba(248,113,113,0.35)";
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], r, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(248,113,113,0.9)";
      ctx.stroke();
    });
  }

  if (editorScene?.start_center) {
    const pt = editorToCanvas(editorScene.start_center, field, size, offsetX, offsetY);
    ctx.fillStyle = "#38bdf8";
    ctx.beginPath();
    ctx.arc(pt[0], pt[1], 6, 0, Math.PI * 2);
    ctx.fill();
  }

  if (editorScene?.target_pos) {
    const pt = editorToCanvas(editorScene.target_pos, field, size, offsetX, offsetY);
    ctx.fillStyle = "#22c55e";
    ctx.beginPath();
    ctx.arc(pt[0], pt[1], 6, 0, Math.PI * 2);
    ctx.fill();
  }

  if (wallDrag) {
    const p1 = editorToCanvas(wallDrag.start, field, size, offsetX, offsetY);
    const p2 = editorToCanvas(wallDrag.end, field, size, offsetX, offsetY);
    const x = Math.min(p1[0], p2[0]);
    const y = Math.min(p1[1], p2[1]);
    const wdt = Math.abs(p2[0] - p1[0]);
    const hgt = Math.abs(p2[1] - p1[1]);
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, wdt, hgt);
  }
}

function syncEditorSceneFromInputs() {
  if (!refs?.sceneMaxSteps || !refs.sceneStartSigma || !refs.sceneId) return;
  editorScene.field_size = editorField();
  editorScene.max_steps = parseInt(refs.sceneMaxSteps.value || "600", 10) || 600;
  editorScene.start_sigma = parseFloat(refs.sceneStartSigma.value || "2.0") || 2.0;
  editorScene.id = refs.sceneId.value || "";
  editorScene.wind = {
    enabled: refs.sceneWindEnabled?.checked ?? false,
    ou_theta: parseFloat(refs.sceneWindTheta?.value || "0.15") || 0.15,
    ou_sigma: parseFloat(refs.sceneWindSigma?.value || "0.3") || 0.3,
  };
  const seedRaw = refs.sceneWindSeed?.value;
  if (seedRaw !== undefined && seedRaw !== null && String(seedRaw).trim() !== "") {
    const parsed = parseInt(seedRaw, 10);
    if (!Number.isNaN(parsed)) editorScene.wind.seed = parsed;
  }
}

function normalizeWallsInPlace() {
  if (!editorScene.walls) editorScene.walls = [];
  editorScene.walls = editorScene.walls.map((w: number[]) => {
    const x1 = Math.min(w[0], w[2]);
    const x2 = Math.max(w[0], w[2]);
    const y1 = Math.min(w[1], w[3]);
    const y2 = Math.max(w[1], w[3]);
    return [x1, y1, x2, y2];
  });
}

function resetEditor() {
  editorScene = defaultEditorScene();
  selectedWallIndex = null;
  drawSceneEditor();
}

function pickEditorItem(world: number[]) {
  const field = editorField();
  const snapWorld = world;
  const start = editorScene.start_center;
  if (start) {
    const dx = start[0] - snapWorld[0];
    const dy = start[1] - snapWorld[1];
    if (Math.sqrt(dx * dx + dy * dy) <= field * 0.015) return { type: "start" };
  }
  const goal = editorScene.target_pos;
  if (goal) {
    const dx = goal[0] - snapWorld[0];
    const dy = goal[1] - snapWorld[1];
    if (Math.sqrt(dx * dx + dy * dy) <= field * 0.015) return { type: "goal" };
  }
  if (editorScene.threats) {
    for (let i = editorScene.threats.length - 1; i >= 0; i--) {
      const t = editorScene.threats[i];
      const dx = t.pos[0] - snapWorld[0];
      const dy = t.pos[1] - snapWorld[1];
      const r = t.radius || 1;
      if (Math.sqrt(dx * dx + dy * dy) <= r) return { type: "threat", index: i };
    }
  }
  if (editorScene.walls) {
    for (let i = editorScene.walls.length - 1; i >= 0; i--) {
      const w = editorScene.walls[i];
      if (snapWorld[0] >= w[0] && snapWorld[0] <= w[2] && snapWorld[1] >= w[1] && snapWorld[1] <= w[3]) {
        return { type: "wall", index: i };
      }
    }
  }
  return null;
}

function pickWallHandle(
  screenX: number,
  screenY: number,
  field: number,
  size: number,
  wall: number[],
  offsetX = 0,
  offsetY = 0,
) {
  const corners = [
    [wall[0], wall[1]],
    [wall[2], wall[1]],
    [wall[2], wall[3]],
    [wall[0], wall[3]],
  ];
  for (let i = 0; i < corners.length; i++) {
    const pt = editorToCanvas(corners[i], field, size, offsetX, offsetY);
    const dx = pt[0] - screenX;
    const dy = pt[1] - screenY;
    if (Math.sqrt(dx * dx + dy * dy) <= HANDLE_PX * 1.5) return { index: i, corner: corners[i] };
  }
  return null;
}

function updateThreatFields() {
  if (!refs?.sceneThreatType) return;
  const type = refs.sceneThreatType.value;
  const showSpeed = type === "linear" || type === "brownian" || type === "chaser";
  const showAngle = type === "linear";
  const showNoise = type === "brownian";
  const showVision = type === "chaser";
  if (refs.sceneThreatSpeedLabel) refs.sceneThreatSpeedLabel.style.display = showSpeed ? "" : "none";
  if (refs.sceneThreatSpeed) refs.sceneThreatSpeed.style.display = showSpeed ? "" : "none";
  if (refs.sceneThreatAngleLabel) refs.sceneThreatAngleLabel.style.display = showAngle ? "" : "none";
  if (refs.sceneThreatAngle) refs.sceneThreatAngle.style.display = showAngle ? "" : "none";
  if (refs.sceneThreatNoiseLabel) refs.sceneThreatNoiseLabel.style.display = showNoise ? "" : "none";
  if (refs.sceneThreatNoise) refs.sceneThreatNoise.style.display = showNoise ? "" : "none";
  if (refs.sceneThreatVisionLabel) refs.sceneThreatVisionLabel.style.display = showVision ? "" : "none";
  if (refs.sceneThreatVision) refs.sceneThreatVision.style.display = showVision ? "" : "none";
}

export function initSceneEditor(
  send: (payload: any) => void,
  root: HTMLElement | null = null,
  errorTarget: HTMLElement | null = null,
) {
  if (initialized) return;
  initialized = true;
  sendControl = send;
  refs = {
    sceneEditor: resolveEl(root, "sceneEditor") as HTMLCanvasElement | null,
    sceneTool: resolveEl(root, "sceneTool") as HTMLSelectElement | null,
    sceneClear: resolveEl(root, "sceneClear") as HTMLButtonElement | null,
    sceneSnap: resolveEl(root, "sceneSnap") as HTMLInputElement | null,
    sceneSnapStep: resolveEl(root, "sceneSnapStep") as HTMLInputElement | null,
    sceneField: resolveEl(root, "sceneField") as HTMLInputElement | null,
    sceneMaxSteps: resolveEl(root, "sceneMaxSteps") as HTMLInputElement | null,
    sceneStartSigma: resolveEl(root, "sceneStartSigma") as HTMLInputElement | null,
    sceneThreatRadius: resolveEl(root, "sceneThreatRadius") as HTMLInputElement | null,
    sceneThreatIntensity: resolveEl(root, "sceneThreatIntensity") as HTMLInputElement | null,
    sceneWindEnabled: resolveEl(root, "sceneWindEnabled") as HTMLInputElement | null,
    sceneWindTheta: resolveEl(root, "sceneWindTheta") as HTMLInputElement | null,
    sceneWindSigma: resolveEl(root, "sceneWindSigma") as HTMLInputElement | null,
    sceneWindSeed: resolveEl(root, "sceneWindSeed") as HTMLInputElement | null,
    sceneThreatType: resolveEl(root, "sceneThreatType") as HTMLSelectElement | null,
    sceneThreatSpeedLabel: resolveEl(root, "sceneThreatSpeedLabel"),
    sceneThreatSpeed: resolveEl(root, "sceneThreatSpeed") as HTMLInputElement | null,
    sceneThreatAngleLabel: resolveEl(root, "sceneThreatAngleLabel"),
    sceneThreatAngle: resolveEl(root, "sceneThreatAngle") as HTMLInputElement | null,
    sceneThreatNoiseLabel: resolveEl(root, "sceneThreatNoiseLabel"),
    sceneThreatNoise: resolveEl(root, "sceneThreatNoise") as HTMLInputElement | null,
    sceneThreatVisionLabel: resolveEl(root, "sceneThreatVisionLabel"),
    sceneThreatVision: resolveEl(root, "sceneThreatVision") as HTMLInputElement | null,
    sceneOracleBlock: resolveEl(root, "sceneOracleBlock") as HTMLInputElement | null,
    sceneId: resolveEl(root, "sceneId") as HTMLInputElement | null,
    scenePreview: resolveEl(root, "scenePreview") as HTMLButtonElement | null,
    sceneSave: resolveEl(root, "sceneSave") as HTMLButtonElement | null,
    sceneDelete: resolveEl(root, "sceneDelete") as HTMLButtonElement | null,
    sceneRefresh: resolveEl(root, "sceneRefresh") as HTMLButtonElement | null,
    sceneFormat: resolveEl(root, "sceneFormat") as HTMLSelectElement | null,
    sceneText: resolveEl(root, "sceneText") as HTMLTextAreaElement | null,
    sceneFile: resolveEl(root, "sceneFile") as HTMLInputElement | null,
    sceneImport: resolveEl(root, "sceneImport") as HTMLButtonElement | null,
    sceneExport: resolveEl(root, "sceneExport") as HTMLButtonElement | null,
  };
  errorEl = errorTarget || resolveEl(null, "error");

  editorScene = defaultEditorScene();
  updateThreatFields();
  drawSceneEditor();

  if (refs.sceneEditor) {
    refs.sceneEditor.addEventListener("mousedown", (evt) => {
      const rect = refs?.sceneEditor?.getBoundingClientRect();
      if (!rect) return;
      const { size, offsetX, offsetY } = getSceneEditorMetrics(rect);
      const field = editorField();
      const world = canvasToEditor(evt.offsetX, evt.offsetY, field, size, offsetX, offsetY);
      const snapWorld = snapPoint(world, field);
      const tool = refs?.sceneTool?.value || "move";

      if (tool === "wall") {
        wallDrag = { start: snapWorld, end: snapWorld };
      } else if (tool === "erase") {
        const hit = pickEditorItem(snapWorld);
        if (hit?.type === "wall" && editorScene.walls) {
          editorScene.walls.splice(hit.index, 1);
          selectedWallIndex = null;
          drawSceneEditor();
        } else if (hit?.type === "threat" && editorScene.threats) {
          editorScene.threats.splice(hit.index, 1);
          drawSceneEditor();
        }
      } else if (tool === "threat") {
        const radius = parseFloat(refs?.sceneThreatRadius?.value || "8") || 8;
        const intensity = parseFloat(refs?.sceneThreatIntensity?.value || "0.1") || 0.1;
        const tType = refs?.sceneThreatType?.value || "static";
        const threat: any = {
          pos: snapWorld,
          radius,
          intensity,
          type: tType,
        };
        if (tType === "linear" || tType === "brownian" || tType === "chaser") {
          threat.speed = parseFloat(refs?.sceneThreatSpeed?.value || "2") || 2;
        }
        if (tType === "linear") {
          threat.angle = parseFloat(refs?.sceneThreatAngle?.value || "0") || 0;
        }
        if (tType === "brownian") {
          threat.noise_scale = parseFloat(refs?.sceneThreatNoise?.value || "0.5") || 0.5;
        }
        if (tType === "chaser") {
          threat.vision_radius = parseFloat(refs?.sceneThreatVision?.value || "30") || 30;
        }
        if (refs?.sceneOracleBlock?.checked) threat.oracle_block = true;
        editorScene.threats = editorScene.threats || [];
        editorScene.threats.push(threat);
        drawSceneEditor();
      } else if (tool === "start") {
        editorScene.start_center = snapWorld;
        drawSceneEditor();
      } else if (tool === "goal") {
        editorScene.target_pos = snapWorld;
        drawSceneEditor();
      } else if (tool === "move") {
        const hit = pickEditorItem(snapWorld);
        if (hit) {
          if (hit.type === "wall" && editorScene.walls) {
            selectedWallIndex = hit.index;
            const wall = editorScene.walls[hit.index];
            const handle = pickWallHandle(evt.offsetX, evt.offsetY, field, size, wall, offsetX, offsetY);
            if (handle) {
              dragItem = { type: "wall_handle", index: hit.index, cornerIndex: handle.index };
            } else {
              dragItem = { type: "wall", index: hit.index, offset: [snapWorld[0] - wall[0], snapWorld[1] - wall[1]] };
            }
          } else if (hit.type === "threat") {
            dragItem = { type: "threat", index: hit.index };
          } else if (hit.type === "start") {
            dragItem = { type: "start", offset: [snapWorld[0] - editorScene.start_center[0], snapWorld[1] - editorScene.start_center[1]] };
          } else if (hit.type === "goal") {
            dragItem = { type: "goal", offset: [snapWorld[0] - editorScene.target_pos[0], snapWorld[1] - editorScene.target_pos[1]] };
          }
        }
      }
    });

    refs.sceneEditor.addEventListener("mousemove", (evt) => {
      if (!refs?.sceneEditor) return;
      const rect = refs.sceneEditor.getBoundingClientRect();
      const { size, offsetX, offsetY } = getSceneEditorMetrics(rect);
      const field = editorField();
      const world = canvasToEditor(evt.offsetX, evt.offsetY, field, size, offsetX, offsetY);
      const snapWorld = snapPoint(world, field);
      if (wallDrag) {
        wallDrag.end = snapWorld;
        drawSceneEditor();
      }
      if (dragItem) {
        if (dragItem.type === "wall_handle") {
          const wall = editorScene.walls[dragItem.index];
          const next = [...wall];
          const idx = dragItem.cornerIndex;
          if (idx === 0) {
            next[0] = snapWorld[0];
            next[1] = snapWorld[1];
          } else if (idx === 1) {
            next[2] = snapWorld[0];
            next[1] = snapWorld[1];
          } else if (idx === 2) {
            next[2] = snapWorld[0];
            next[3] = snapWorld[1];
          } else if (idx === 3) {
            next[0] = snapWorld[0];
            next[3] = snapWorld[1];
          }
          editorScene.walls[dragItem.index] = next;
          normalizeWallsInPlace();
        } else if (dragItem.type === "wall") {
          const wall = editorScene.walls[dragItem.index];
          const next = [
            snapWorld[0] - dragItem.offset[0],
            snapWorld[1] - dragItem.offset[1],
            snapWorld[0] - dragItem.offset[0] + (wall[2] - wall[0]),
            snapWorld[1] - dragItem.offset[1] + (wall[3] - wall[1]),
          ];
          editorScene.walls[dragItem.index] = next;
          normalizeWallsInPlace();
        } else if (dragItem.type === "threat") {
          const next = snapWorld;
          editorScene.threats[dragItem.index].pos = next;
        } else if (dragItem.type === "start") {
          const next = [snapWorld[0] - dragItem.offset[0], snapWorld[1] - dragItem.offset[1]];
          editorScene.start_center = snapPoint(next, field);
        } else if (dragItem.type === "goal") {
          const next = [snapWorld[0] - dragItem.offset[0], snapWorld[1] - dragItem.offset[1]];
          editorScene.target_pos = snapPoint(next, field);
        }
        drawSceneEditor();
      }
    });

    window.addEventListener("mouseup", () => {
      if (wallDrag) {
        const start = wallDrag.start;
        const end = wallDrag.end;
        wallDrag = null;
        editorScene.walls.push([start[0], start[1], end[0], end[1]]);
        normalizeWallsInPlace();
        drawSceneEditor();
      }
      if (dragItem) dragItem = null;
    });
  }

  window.addEventListener("ui:tab-editor", () => {
    setTimeout(() => drawSceneEditor(), 10);
  });

  refs.sceneClear && (refs.sceneClear.onclick = () => resetEditor());
  refs.sceneField && (refs.sceneField.onchange = () => {
    syncEditorSceneFromInputs();
    drawSceneEditor();
  });
  refs.sceneMaxSteps && (refs.sceneMaxSteps.onchange = () => syncEditorSceneFromInputs());
  refs.sceneStartSigma && (refs.sceneStartSigma.onchange = () => syncEditorSceneFromInputs());
  refs.sceneId && (refs.sceneId.onchange = () => syncEditorSceneFromInputs());
  refs.sceneSnap && (refs.sceneSnap.onchange = () => drawSceneEditor());
  refs.sceneSnapStep && (refs.sceneSnapStep.onchange = () => drawSceneEditor());
  refs.sceneThreatType && (refs.sceneThreatType.onchange = () => updateThreatFields());
  refs.sceneWindEnabled && (refs.sceneWindEnabled.onchange = () => syncEditorSceneFromInputs());
  refs.sceneWindTheta && (refs.sceneWindTheta.onchange = () => syncEditorSceneFromInputs());
  refs.sceneWindSigma && (refs.sceneWindSigma.onchange = () => syncEditorSceneFromInputs());
  refs.sceneWindSeed && (refs.sceneWindSeed.onchange = () => syncEditorSceneFromInputs());

  refs.scenePreview && (refs.scenePreview.onclick = () => {
    syncEditorSceneFromInputs();
    sendControl?.({ type: "control", action: "scene_preview", scene: editorScene });
  });
  refs.sceneSave && (refs.sceneSave.onclick = () => {
    syncEditorSceneFromInputs();
    sendControl?.({ type: "control", action: "scene_save", scene: editorScene });
  });
  refs.sceneDelete && (refs.sceneDelete.onclick = () => {
    const sid = refs?.sceneId?.value || "";
    if (sid) sendControl?.({ type: "control", action: "scene_delete", scene_id: sid });
  });
  refs.sceneRefresh && (refs.sceneRefresh.onclick = () => sendControl?.({ type: "control", action: "scene_refresh" }));
  refs.sceneExport && (refs.sceneExport.onclick = () => {
    syncEditorSceneFromInputs();
    const format = refs?.sceneFormat?.value || "json";
    sendControl?.({ type: "control", action: "scene_export", scene: editorScene, format });
  });
  refs.sceneImport && (refs.sceneImport.onclick = () => {
    const text = refs?.sceneText?.value || "";
    sendControl?.({ type: "control", action: "scene_parse", text });
  });
  refs.sceneFile && (refs.sceneFile.onchange = () => {
    const file = refs?.sceneFile?.files && refs.sceneFile.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (refs?.sceneText) refs.sceneText.value = String(reader.result || "");
      sendControl?.({ type: "control", action: "scene_parse", text: refs?.sceneText?.value || "" });
    };
    reader.readAsText(file);
  });
}

export function applySceneToEditor(scene: any) {
  if (!scene || !refs) return;
  editorScene = scene;
  selectedWallIndex = null;
  if (refs.sceneField) refs.sceneField.value = scene.field_size ?? 100;
  if (refs.sceneMaxSteps) refs.sceneMaxSteps.value = scene.max_steps ?? 600;
  if (refs.sceneStartSigma) refs.sceneStartSigma.value = scene.start_sigma ?? 2.0;
  if (refs.sceneId) refs.sceneId.value = scene.id ?? "";
  const wind = scene.wind || {};
  if (refs.sceneWindEnabled) refs.sceneWindEnabled.checked = !!wind.enabled;
  if (refs.sceneWindTheta) refs.sceneWindTheta.value = wind.ou_theta ?? 0.15;
  if (refs.sceneWindSigma) refs.sceneWindSigma.value = wind.ou_sigma ?? 0.3;
  if (refs.sceneWindSeed) refs.sceneWindSeed.value = wind.seed ?? "";
  drawSceneEditor();
}

export function setSceneText(text: string) {
  if (refs?.sceneText) refs.sceneText.value = text;
}

export function setSceneId(sceneId: string) {
  if (refs?.sceneId) refs.sceneId.value = sceneId;
}

export function setSceneError(text: string) {
  if (errorEl) errorEl.textContent = text;
}

export function sceneEditorApi(): SceneEditorApi {
  return {
    applyScene: applySceneToEditor,
    setSceneText,
    setSceneId,
    setError: setSceneError,
    redraw: drawSceneEditor,
  };
}
