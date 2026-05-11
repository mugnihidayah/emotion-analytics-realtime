'use client';

import React from 'react';
import type { EmotionResult } from '@/lib/types';
import EngagementBar from './EngagementBar';
import EmotionDisplay from './EmotionDisplay';
import SparklineChart from './SparklineChart';

interface AnalyticsPanelProps {
  latestResult: EmotionResult | null;
  scoreHistory: number[];
  device: string | null;
}

export default function AnalyticsPanel({ latestResult, scoreHistory, device }: AnalyticsPanelProps) {
  const score = latestResult?.score ?? 0;
  const emotion = latestResult?.emotion ?? 'Initializing...';
  const probabilities = latestResult?.probabilities ?? null;

  return (
    <div className="glass-card" style={{ padding: 0, overflow: 'hidden' }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '16px 20px',
        borderBottom: '1px solid var(--border-subtle)',
      }}>
        <span style={{ fontSize: '1.1rem' }}>📊</span>
        <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>Analytics Insight</span>
      </div>

      {/* Engagement */}
      <EngagementBar score={score} />

      {/* Emotion */}
      <EmotionDisplay emotion={emotion} probabilities={probabilities} />

      {/* Sparkline */}
      <SparklineChart data={scoreHistory} />

      {/* System Info */}
      <div className="analytics-info">
        <div className="analytics-info-row">
          <span className="label">Inference</span>
          <span className="value">YOLOv8 + ResNet-SE CNN</span>
        </div>
        <div className="analytics-info-row">
          <span className="label">Smoothing</span>
          <span className="value">EMA (5-frame avg)</span>
        </div>
        <div className="analytics-info-row">
          <span className="label">Device</span>
          <span className="value">{device ?? 'Unknown'}</span>
        </div>
      </div>
    </div>
  );
}
