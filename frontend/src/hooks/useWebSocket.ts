'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import type { EmotionResult, ConnectionStatus } from '@/lib/types';

interface UseWebSocketOptions {
  url: string;
  onMessage?: (data: EmotionResult) => void;
}

const MAX_BUFFERED_BYTES = 512 * 1024;
const RESPONSE_TIMEOUT_MS = 5000;
type WebSocketPayload = string | Blob | ArrayBuffer;

export function useWebSocket({ url, onMessage }: UseWebSocketOptions) {
  const [status, setStatus] = useState<ConnectionStatus>('idle');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const responseTimeout = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const intentionalClose = useRef(false);
  const awaitingResponse = useRef(false);
  const maxReconnectAttempts = 10;

  const clearPendingResponse = useCallback(() => {
    awaitingResponse.current = false;
    if (responseTimeout.current) {
      clearTimeout(responseTimeout.current);
      responseTimeout.current = null;
    }
  }, []);

  const canSend = useCallback(() => {
    const ws = wsRef.current;
    return Boolean(
      ws &&
        ws.readyState === WebSocket.OPEN &&
        !awaitingResponse.current &&
        ws.bufferedAmount < MAX_BUFFERED_BYTES
    );
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    intentionalClose.current = false;
    setStatus('connecting');

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus('connected');
        reconnectAttempts.current = 0;
        clearPendingResponse();
        console.log('[WS] Connected to', url);
      };

      ws.onmessage = (event) => {
        clearPendingResponse();

        try {
          const data: EmotionResult = JSON.parse(event.data);
          onMessage?.(data);
        } catch (err) {
          console.error('[WS] Failed to parse message:', err);
        }
      };

      ws.onclose = () => {
        clearPendingResponse();

        // Skip reconnect if user intentionally disconnected
        if (intentionalClose.current) return;

        setStatus('disconnected');
        console.log('[WS] Disconnected');

        // Auto-reconnect with exponential backoff
        if (reconnectAttempts.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * 2 ** reconnectAttempts.current, 15000);
          reconnectAttempts.current++;
          console.log(`[WS] Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current})`);
          reconnectTimeout.current = setTimeout(connect, delay);
        }
      };

      ws.onerror = () => {
        clearPendingResponse();

        if (intentionalClose.current) return;
        setStatus('error');
        console.error('[WS] Connection error');
      };
    } catch (err) {
      clearPendingResponse();
      setStatus('error');
      console.error('[WS] Failed to create WebSocket:', err);
    }
  }, [url, onMessage, clearPendingResponse]);

  const disconnect = useCallback(() => {
    intentionalClose.current = true;
    clearPendingResponse();
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }
    reconnectAttempts.current = maxReconnectAttempts;
    wsRef.current?.close();
    wsRef.current = null;
    setStatus('idle');
  }, [clearPendingResponse]);

  const send = useCallback((data: WebSocketPayload) => {
    if (!canSend()) {
      return false;
    }

    try {
      awaitingResponse.current = true;
      wsRef.current?.send(data);
      responseTimeout.current = setTimeout(() => {
        clearPendingResponse();
      }, RESPONSE_TIMEOUT_MS);
      return true;
    } catch (err) {
      clearPendingResponse();
      setStatus('error');
      console.error('[WS] Failed to send frame:', err);
      return false;
    }
  }, [canSend, clearPendingResponse]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      intentionalClose.current = true;
      clearPendingResponse();
      if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
      wsRef.current?.close();
    };
  }, [clearPendingResponse]);

  return { status, connect, disconnect, send, canSend };
}
