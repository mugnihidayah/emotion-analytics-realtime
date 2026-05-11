'use client';

import React from 'react';
import type { ConnectionStatus } from '@/lib/types';

interface StatusIndicatorProps {
  wsStatus: ConnectionStatus;
  device: string | null;
  cameraActive: boolean;
  inferenceMs?: number;
}

const STATUS_LABELS: Record<ConnectionStatus, string> = {
  idle: 'Not connected',
  connecting: 'Connecting to server...',
  connected: 'Connected',
  disconnected: 'Disconnected — reconnecting...',
  error: 'Connection error',
};

export default function StatusIndicator({
  wsStatus,
  device,
  cameraActive,
  inferenceMs,
}: StatusIndicatorProps) {
  const dotClass =
    wsStatus === 'connected'
      ? 'connected'
      : wsStatus === 'connecting'
        ? 'connecting'
        : 'disconnected';

  return (
    <div className="status-bar">
      <span className={`status-dot ${dotClass}`} />
      <span>{STATUS_LABELS[wsStatus]}</span>
      <span style={{ marginLeft: 'auto', opacity: 0.6 }}>
        {device ? `Device: ${device}` : ''}
        {cameraActive ? ' | Camera active' : ''}
        {typeof inferenceMs === 'number' ? ` | Inference: ${Math.round(inferenceMs)}ms` : ''}
      </span>
    </div>
  );
}
