'use client';

import React from 'react';

interface EngagementBarProps {
  score: number;
}

export default function EngagementBar({ score }: EngagementBarProps) {
  const level = score > 60 ? 'high' : score > 30 ? 'mid' : 'low';
  const color =
    level === 'high' ? '#22c55e' : level === 'mid' ? '#eab308' : '#ef4444';

  return (
    <div className="engagement-section">
      <h3>Engagement Score</h3>
      <div
        className="engagement-score"
        style={{ color }}
      >
        {Math.round(score)}
        <span style={{ fontSize: '1.25rem', fontWeight: 400, marginLeft: 2 }}>%</span>
      </div>
      <div className="engagement-bar-track">
        <div
          className={`engagement-bar-fill ${level}`}
          style={{ width: `${Math.min(score, 100)}%` }}
        />
      </div>
      <div
        style={{
          marginTop: 8,
          fontSize: '0.75rem',
          color: 'var(--text-muted)',
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <span>Low</span>
        <span>
          {level === 'high' ? '🔥 High Engagement' : level === 'mid' ? '😐 Moderate' : '📉 Low'}
        </span>
        <span>High</span>
      </div>
    </div>
  );
}
