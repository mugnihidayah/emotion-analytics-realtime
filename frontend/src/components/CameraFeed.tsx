'use client';

import React, { useRef, useEffect } from 'react';
import type { EmotionResult } from '@/lib/types';

interface CameraFeedProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isActive: boolean;
  error: string | null;
  latestResult: EmotionResult | null;
  onStart: () => void;
  onStop: () => void;
}

export default function CameraFeed({
  videoRef,
  isActive,
  error,
  latestResult,
  onStart,
  onStop,
}: CameraFeedProps) {
  const overlayRef = useRef<HTMLCanvasElement>(null);

  // Draw bounding box + label overlay
  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Match canvas size to video display
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Stop here if camera is off or no result
    if (!isActive || !latestResult || !latestResult.bbox) return;

    const [x1, y1, x2, y2] = latestResult.bbox;

    // Scale bbox from 640x480 to canvas size (video is mirrored)
    const scaleX = canvas.width / 640;
    const scaleY = canvas.height / 480;

    // Mirror the x coordinates since the video feed is mirrored
    const drawX1 = canvas.width - x2 * scaleX;
    const drawY1 = y1 * scaleY;
    const drawX2 = canvas.width - x1 * scaleX;
    const drawY2 = y2 * scaleY;

    const w = drawX2 - drawX1;
    const h = drawY2 - drawY1;

    // Color based on engagement
    const isHigh = latestResult.score > 60;
    const color = isHigh ? '#22c55e' : latestResult.score > 30 ? '#eab308' : '#ef4444';

    // Corner lines (modern look instead of full rectangle)
    const cornerLen = Math.min(w, h) * 0.2;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';

    // Top-left
    ctx.beginPath();
    ctx.moveTo(drawX1, drawY1 + cornerLen);
    ctx.lineTo(drawX1, drawY1);
    ctx.lineTo(drawX1 + cornerLen, drawY1);
    ctx.stroke();

    // Top-right
    ctx.beginPath();
    ctx.moveTo(drawX2 - cornerLen, drawY1);
    ctx.lineTo(drawX2, drawY1);
    ctx.lineTo(drawX2, drawY1 + cornerLen);
    ctx.stroke();

    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(drawX1, drawY2 - cornerLen);
    ctx.lineTo(drawX1, drawY2);
    ctx.lineTo(drawX1 + cornerLen, drawY2);
    ctx.stroke();

    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(drawX2 - cornerLen, drawY2);
    ctx.lineTo(drawX2, drawY2);
    ctx.lineTo(drawX2, drawY2 - cornerLen);
    ctx.stroke();

    // Emotion label
    const label = latestResult.emotion.toUpperCase();
    ctx.font = '600 14px Inter, sans-serif';
    const textMetrics = ctx.measureText(label);
    const textW = textMetrics.width + 16;
    const textH = 24;

    // Background pill
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.85;
    const pillX = drawX1;
    const pillY = drawY1 - textH - 6;
    ctx.beginPath();
    ctx.roundRect(pillX, pillY, textW, textH, 6);
    ctx.fill();
    ctx.globalAlpha = 1;

    // Text
    ctx.fillStyle = '#ffffff';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, pillX + 8, pillY + textH / 2);
  }, [latestResult, isActive, videoRef]);

  return (
    <div className="glass-card" style={{ padding: 0, overflow: 'hidden' }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '16px 20px',
        borderBottom: '1px solid var(--border-subtle)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: '1.1rem' }}>🎥</span>
          <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>Live Camera Feed</span>
        </div>
        {isActive && (
          <div className="live-badge">
            <span className="dot" />
            LIVE
          </div>
        )}
      </div>

      {/* Video area */}
      <div className="camera-container">
        <video
          ref={videoRef}
          playsInline
          muted
          style={{ display: isActive ? 'block' : 'none' }}
        />
        <canvas ref={overlayRef} />

        {!isActive && (
          <div className="camera-placeholder">
            <div className="icon">📷</div>
            {error ? (
              <p style={{ color: '#ef4444', maxWidth: 280, textAlign: 'center' }}>{error}</p>
            ) : (
              <p>Grant camera access to start emotion analysis</p>
            )}
            <button className="camera-start-btn" onClick={onStart} id="start-camera-btn">
              Start Camera
            </button>
          </div>
        )}
      </div>

      {/* Controls */}
      {isActive && (
        <div style={{ padding: '12px 20px', borderTop: '1px solid var(--border-subtle)' }}>
          <button
            onClick={onStop}
            id="stop-camera-btn"
            style={{
              padding: '8px 20px',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: 'var(--radius-sm)',
              background: 'rgba(239, 68, 68, 0.1)',
              color: '#ef4444',
              fontWeight: 500,
              fontSize: '0.82rem',
              cursor: 'pointer',
              transition: 'background 0.2s',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(239, 68, 68, 0.2)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'rgba(239, 68, 68, 0.1)')}
          >
            Stop Camera
          </button>
        </div>
      )}
    </div>
  );
}
