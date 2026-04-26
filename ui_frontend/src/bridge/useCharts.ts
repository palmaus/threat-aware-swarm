import { useEffect, useRef } from "react";
import Chart from "chart.js/auto";
import { useTelemetryStore } from "../store/telemetryStore";

type ChartBundle = {
  path?: Chart;
  risk?: Chart;
  energy?: Chart;
};

export function useCharts() {
  const chartData = useTelemetryStore((s) => s.chartData);
  const chartsRef = useRef<ChartBundle>({});

  useEffect(() => {
    const pathCanvas = document.getElementById("chartPath") as HTMLCanvasElement | null;
    const riskCanvas = document.getElementById("chartRisk") as HTMLCanvasElement | null;
    const energyCanvas = document.getElementById("chartEnergy") as HTMLCanvasElement | null;
    if (!pathCanvas || !riskCanvas || !energyCanvas) return;

    chartsRef.current.path = new Chart(pathCanvas, {
      type: "line",
      data: {
        labels: chartData.labels,
        datasets: [
          { label: "Path Ratio L", data: chartData.pathL, borderColor: "#38bdf8", tension: 0.2, pointRadius: 0 },
          { label: "Path Ratio R", data: chartData.pathR, borderColor: "#a78bfa", tension: 0.2, pointRadius: 0 },
        ],
      },
      options: {
        responsive: true,
        animation: false,
        plugins: { legend: { display: true } },
        scales: { x: { display: false } },
      },
    });

    chartsRef.current.risk = new Chart(riskCanvas, {
      type: "line",
      data: {
        labels: chartData.labels,
        datasets: [
          { label: "Risk L", data: chartData.riskL, borderColor: "#f87171", tension: 0.2, pointRadius: 0 },
          { label: "Risk R", data: chartData.riskR, borderColor: "#fb7185", tension: 0.2, pointRadius: 0 },
        ],
      },
      options: {
        responsive: true,
        animation: false,
        plugins: { legend: { display: true } },
        scales: { x: { display: false } },
      },
    });

    chartsRef.current.energy = new Chart(energyCanvas, {
      type: "line",
      data: {
        labels: chartData.labels,
        datasets: [
          { label: "Energy L", data: chartData.energyL, borderColor: "#facc15", tension: 0.2, pointRadius: 0 },
          { label: "Energy R", data: chartData.energyR, borderColor: "#f59e0b", tension: 0.2, pointRadius: 0 },
        ],
      },
      options: {
        responsive: true,
        animation: false,
        plugins: { legend: { display: true } },
        scales: { x: { display: false } },
      },
    });

    return () => {
      chartsRef.current.path?.destroy();
      chartsRef.current.risk?.destroy();
      chartsRef.current.energy?.destroy();
      chartsRef.current = {};
    };
  }, []);

  useEffect(() => {
    const charts = chartsRef.current;
    if (charts.path) {
      charts.path.data.labels = chartData.labels;
      charts.path.data.datasets[0].data = chartData.pathL;
      charts.path.data.datasets[1].data = chartData.pathR;
      charts.path.update();
    }
    if (charts.risk) {
      charts.risk.data.labels = chartData.labels;
      charts.risk.data.datasets[0].data = chartData.riskL;
      charts.risk.data.datasets[1].data = chartData.riskR;
      charts.risk.update();
    }
    if (charts.energy) {
      charts.energy.data.labels = chartData.labels;
      charts.energy.data.datasets[0].data = chartData.energyL;
      charts.energy.data.datasets[1].data = chartData.energyR;
      charts.energy.update();
    }
  }, [chartData]);
}
