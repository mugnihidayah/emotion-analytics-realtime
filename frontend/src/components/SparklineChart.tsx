'use client';

import React, { useRef, useEffect } from 'react';

interface SparklineChartProps {
  data: number[];
}

export default function SparklineChart({ data }: SparklineChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // High DPI support
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;

    // Clear
    ctx.clearRect(0, 0, w, h);

    if (data.length < 2) return;

    const padding = { top: 4, bottom: 4, left: 0, right: 0 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;

    // Normalize scores (0-100)
    const step = chartW / (data.length - 1);
    const points: [number, number][] = data.map((val, i) => [
      padding.left + i * step,
      padding.top + chartH - (val / 100) * chartH,
    ]);

    // Gradient fill under the line
    const gradient = ctx.createLinearGradient(0, padding.top, 0, h);
    gradient.addColorStop(0, 'rgba(6, 182, 212, 0.2)');
    gradient.addColorStop(1, 'rgba(6, 182, 212, 0.0)');

    ctx.beginPath();
    ctx.moveTo(points[0][0], h);
    points.forEach(([x, y]) => ctx.lineTo(x, y));
    ctx.lineTo(points[points.length - 1][0], h);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);

    // Smooth curve using quadratic bezier
    for (let i = 1; i < points.length; i++) {
      const cpX = (points[i - 1][0] + points[i][0]) / 2;
      const cpY1 = points[i - 1][1];
      const cpY2 = points[i][1];
      ctx.quadraticCurveTo(cpX, cpY1, (cpX + points[i][0]) / 2, (cpY1 + cpY2) / 2);
    }
    // Connect to last point
    ctx.lineTo(points[points.length - 1][0], points[points.length - 1][1]);

    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Dot on last point
    const lastPt = points[points.length - 1];
    ctx.beginPath();
    ctx.arc(lastPt[0], lastPt[1], 4, 0, Math.PI * 2);
    ctx.fillStyle = '#06b6d4';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(lastPt[0], lastPt[1], 7, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.3)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Grid lines (subtle)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const gy = padding.top + (chartH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, gy);
      ctx.lineTo(w - padding.right, gy);
      ctx.stroke();
    }
  }, [data]);

  return (
    <div className="sparkline-section">
      <h3>Engagement Trend</h3>
      <canvas
        ref={canvasRef}
        className="sparkline-canvas"
        style={{ width: '100%', height: 100 }}
      />
    </div>
  );
}
