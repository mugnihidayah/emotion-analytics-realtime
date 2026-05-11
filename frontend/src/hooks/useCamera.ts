'use client';

import { useState, useRef, useCallback, useEffect } from 'react';

interface UseCameraOptions {
  fps?: number;
  quality?: number;
  onFrame?: (base64: string) => void;
}

export function useCamera({ fps = 8, quality = 0.6, onFrame }: UseCameraOptions = {}) {
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const onFrameRef = useRef(onFrame);

  // Keep onFrame ref current to avoid stale closures in setInterval
  useEffect(() => {
    onFrameRef.current = onFrame;
  }, [onFrame]);

  const start = useCallback(async () => {
    try {
      setError(null);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsActive(true);

      // Create offscreen canvas for frame capture (no DOM attachment needed)
      const captureCanvas = document.createElement('canvas');

      // Frame capture loop
      intervalRef.current = setInterval(() => {
        if (!videoRef.current) return;

        const video = videoRef.current;
        if (video.readyState < video.HAVE_CURRENT_DATA) return;

        captureCanvas.width = video.videoWidth || 640;
        captureCanvas.height = video.videoHeight || 480;

        const ctx = captureCanvas.getContext('2d');
        if (!ctx) return;

        // Draw mirrored frame
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(video, -captureCanvas.width, 0, captureCanvas.width, captureCanvas.height);
        ctx.restore();

        // Convert to base64 JPEG
        const base64 = captureCanvas.toDataURL('image/jpeg', quality);
        onFrameRef.current?.(base64);
      }, 1000 / fps);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Camera access denied';
      setError(msg);
      console.error('[Camera]', msg);
    }
  }, [fps, quality]);

  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsActive(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  return { videoRef, isActive, error, start, stop };
}
