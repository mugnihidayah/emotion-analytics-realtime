// Shared TypeScript types for Emotion Analytics

export interface EmotionResult {
  emotion: string;
  score: number;
  bbox: [number, number, number, number] | null;
  probabilities: Record<string, number> | null;
}

export type ConnectionStatus = 'idle' | 'connecting' | 'connected' | 'disconnected' | 'error';

export const EMOTION_EMOJIS: Record<string, string> = {
  Angry: '😠',
  Disgust: '🤢',
  Fear: '😨',
  Happy: '😊',
  Neutral: '😐',
  Sad: '😢',
  Surprise: '😲',
  'No Face': '👤',
  Error: '⚠️',
  'Initializing...': '⏳',
};

export const EMOTION_COLORS: Record<string, string> = {
  Angry: '#ef4444',
  Disgust: '#a855f7',
  Fear: '#f97316',
  Happy: '#22c55e',
  Neutral: '#94a3b8',
  Sad: '#3b82f6',
  Surprise: '#eab308',
};
