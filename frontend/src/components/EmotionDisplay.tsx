'use client';

import React from 'react';
import { EMOTION_EMOJIS, EMOTION_COLORS } from '@/lib/types';

interface EmotionDisplayProps {
  emotion: string;
  probabilities: Record<string, number> | null;
}

const ORDERED_LABELS = ['Happy', 'Surprise', 'Neutral', 'Sad', 'Angry', 'Fear', 'Disgust'];

export default function EmotionDisplay({ emotion, probabilities }: EmotionDisplayProps) {
  const emoji = EMOTION_EMOJIS[emotion] || '❓';

  return (
    <div className="emotion-display">
      <h3>Detected Emotion</h3>

      {/* Current emotion */}
      <div className="emotion-current">
        <span className="emotion-emoji">{emoji}</span>
        <span
          className="emotion-label"
          style={{ color: EMOTION_COLORS[emotion] || 'var(--text-primary)' }}
        >
          {emotion}
        </span>
      </div>

      {/* Probability distribution */}
      {probabilities && (
        <div className="emotion-prob-list">
          {ORDERED_LABELS.map((label) => {
            const prob = probabilities[label] ?? 0;
            const isActive = label === emotion;

            return (
              <div key={label} className="emotion-prob-row">
                <span className="emotion-prob-name">{label}</span>
                <div className="emotion-prob-bar-track">
                  <div
                    className={`emotion-prob-bar-fill ${isActive ? 'active' : ''}`}
                    style={{
                      width: `${(prob * 100).toFixed(0)}%`,
                      background: isActive
                        ? EMOTION_COLORS[label] || 'var(--accent-primary)'
                        : undefined,
                    }}
                  />
                </div>
                <span className="emotion-prob-value">{(prob * 100).toFixed(0)}%</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
