'use client';

import { useState, useRef, useCallback, useEffect } from 'react';

interface UseCameraOptions {
  fps?: number;
  quality?: number;
  width?: number;
  height?: number;
  shouldCapture?: () => boolean;
  onFrame?: (frame: Blob) => void;
}

export function useCamera({
  fps = 4,
  quality = 0.45,
  width = 480,
  height = 360,
  shouldCapture,
  onFrame,
}: UseCameraOptions = {}) {
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const onFrameRef = useRef(onFrame);
  const shouldCaptureRef = useRef(shouldCapture);
  const encodingRef = useRef(false);

  // Keep onFrame ref current to avoid stale closures in setInterval
  useEffect(() => {
    onFrameRef.current = onFrame;
    shouldCaptureRef.current = shouldCapture;
  }, [onFrame, shouldCapture]);

  const start = useCallback(async () => {
    try {
      setError(null);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: width }, height: { ideal: height }, facingMode: 'user' },
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
        if (encodingRef.current) return;
        if (shouldCaptureRef.current && !shouldCaptureRef.current()) return;
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

        encodingRef.current = true;
        captureCanvas.toBlob(
          (blob) => {
            encodingRef.current = false;
            if (!blob) return;
            if (shouldCaptureRef.current && !shouldCaptureRef.current()) return;
            onFrameRef.current?.(blob);
          },
          'image/jpeg',
          quality
        );
      }, 1000 / fps);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Camera access denied';
      setError(msg);
      console.error('[Camera]', msg);
    }
  }, [fps, quality, width, height]);

  const stop = useCallback(() => {
    encodingRef.current = false;

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
      encodingRef.current = false;
      if (intervalRef.current) clearInterval(intervalRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  return { videoRef, isActive, error, start, stop };
}
