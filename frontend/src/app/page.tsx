'use client';

import { useState, useCallback, useRef } from 'react';
import type { EmotionResult } from '@/lib/types';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useCamera } from '@/hooks/useCamera';
import CameraFeed from '@/components/CameraFeed';
import AnalyticsPanel from '@/components/AnalyticsPanel';
import StatusIndicator from '@/components/StatusIndicator';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:7860/ws/emotion';
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:7860';
const CAMERA_FPS = Number(process.env.NEXT_PUBLIC_CAMERA_FPS || 4);
const CAMERA_QUALITY = Number(process.env.NEXT_PUBLIC_CAMERA_QUALITY || 0.45);
const CAMERA_WIDTH = Number(process.env.NEXT_PUBLIC_CAMERA_WIDTH || 480);
const CAMERA_HEIGHT = Number(process.env.NEXT_PUBLIC_CAMERA_HEIGHT || 360);

export default function HomePage() {
  const [latestResult, setLatestResult] = useState<EmotionResult | null>(null);
  const [scoreHistory, setScoreHistory] = useState<number[]>(new Array(60).fill(0));
  const [device, setDevice] = useState<string | null>(null);

  // Hidden canvas for frame capture
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);

  // WebSocket
  const handleMessage = useCallback((data: EmotionResult) => {
    setLatestResult(data);
    setScoreHistory((prev) => {
      const next = [...prev, data.score];
      return next.slice(-100); // Keep last 100 data points
    });
  }, []);

  const { status: wsStatus, connect, disconnect: wsDisconnect, send, canSend } = useWebSocket({
    url: WS_URL,
    onMessage: handleMessage,
  });

  // Camera
  const handleFrame = useCallback(
    (base64: string) => {
      send(base64);
    },
    [send]
  );

  const { videoRef, isActive: cameraActive, error: cameraError, start: startCamera, stop: stopCamera } =
    useCamera({
      fps: CAMERA_FPS,
      quality: CAMERA_QUALITY,
      width: CAMERA_WIDTH,
      height: CAMERA_HEIGHT,
      shouldCapture: canSend,
      onFrame: handleFrame,
    });

  // Start everything
  const handleStart = useCallback(async () => {
    // Fetch device info
    try {
      const res = await fetch(`${API_URL}/api/health`);
      const data = await res.json();
      setDevice(data.device || 'unknown');
    } catch {
      setDevice('unknown');
    }

    connect();
    await startCamera();
  }, [connect, startCamera]);

  // Stop everything
  const handleStop = useCallback(() => {
    stopCamera();
    wsDisconnect();
    setLatestResult(null);
  }, [stopCamera, wsDisconnect]);

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="dashboard-header fade-in">
        <h1>🧠 Emotion Analytics Dashboard</h1>
        <p>Real-time emotion analysis powered by YOLOv8 + ResNet-SE CNN + EMA smoothing</p>
      </header>

      {/* Main Grid */}
      <div className="dashboard-grid fade-in" style={{ animationDelay: '0.1s' }}>
        {/* Left — Camera */}
        <div>
          <CameraFeed
            videoRef={videoRef}
            isActive={cameraActive}
            error={cameraError}
            latestResult={latestResult}
            onStart={handleStart}
            onStop={handleStop}
          />

          {/* Status bar */}
          <div className="glass-card" style={{ marginTop: 16, padding: 0 }}>
            <StatusIndicator
              wsStatus={wsStatus}
              device={device}
              cameraActive={cameraActive}
            />
          </div>
        </div>

        {/* Right — Analytics */}
        <AnalyticsPanel
          latestResult={latestResult}
          scoreHistory={scoreHistory}
          device={device}
        />
      </div>

      {/* Hidden canvas for frame capture */}
      <canvas ref={captureCanvasRef} style={{ display: 'none' }} />
    </div>
  );
}
