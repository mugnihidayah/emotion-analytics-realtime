import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Emotion Analytics Dashboard | Real-time AI",
  description:
    "Real-time facial emotion recognition and engagement analytics powered by YOLOv8, CNN, and EMA smoothing. Detect emotions with live camera feed.",
  keywords: ["emotion detection", "facial recognition", "AI analytics", "real-time", "engagement"],
  openGraph: {
    title: "Emotion Analytics Dashboard",
    description: "Real-time facial emotion recognition and engagement analytics powered by deep learning.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
