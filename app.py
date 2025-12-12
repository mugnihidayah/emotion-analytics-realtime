# app.py
from collections import deque
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import av
import numpy as np
from logic import EmotionAnalyzer


# ==========================================================
#                   MODERN FULL-WIDTH UI CSS
# ==========================================================
st.markdown(
    """
<style>

/* Make page full width */
.main {
    max-width: 1600px;
    padding-left: 40px;
    padding-right: 40px;
    margin: 0 auto;
}

/* Tight top spacing */
.block-container {
    padding-top: 1rem;
}

/* Flex layout for dashboard */
.dashboard-row {
    display: flex;
    gap: 32px;
    width: 100%;
}

/* Column widths */
.left-col {
    flex: 3;
}
.right-col {
    flex: 1.5;
}

/* Card style */
.st-card {
    background: rgba(25, 26, 30, 0.85);
    border-radius: 22px;
    padding: 25px;
    box-shadow: 0 4px 22px rgba(0,0,0,0.3);
    backdrop-filter: blur(8px);
}

/* Typography */
.st-title {
    font-size: 2.7rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    text-align: center;
}
.st-desc {
    text-align: center;
    color: #cfcfcf;
    font-size: 1.05rem;
}
.st-sub {
    font-size: 1.4rem;
    font-weight: 600;
    color: #ececec;
}
.st-info {
    color: #cacaca;
    font-size: 1rem;
}

hr {
    border: 0;
    border-top: 1px solid rgba(255,255,255,0.15);
    margin: 20px 0;
}

</style>
""",
    unsafe_allow_html=True,
)


# LOAD ANALYZER
@st.cache_resource
def load_analyzer():
    return EmotionAnalyzer()


analyzer = load_analyzer()

# PAGE HEADER
st.markdown(
    "<h1 class='st-title'>🧠 Emotion Analytics Dashboard</h1>", unsafe_allow_html=True
)
st.markdown(
    "<p class='st-desc'>Real-time emotion analysis powered by YOLO + CNN + EMA smoothing</p>",
    unsafe_allow_html=True,
)
st.write("")


# VIDEO PROCESSOR
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_emotion = "Initializing..."
        self.last_score = 0.0
        self.bbox = None
        self.has_bbox = False

        # History Score for real-time graphic
        self.score_history = deque([0] * 100, maxlen=100)

    def draw_sparkline(self, img, scores, x, y, w, h):
        """Menggambar grafik garis sederhana (Sparkline) di atas frame"""
        if len(scores) < 2:
            return img

        # Graphic Background
        sub_img = img[y : y + h, x : x + w]
        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 0.7, black_rect, 0.3, 1.0)
        img[y : y + h, x : x + w] = res

        # Normalize the graph points to fit in the box
        norm_scores = [h - int((s / 100) * h) for s in scores]

        # Line
        step = w / (len(scores) - 1)
        for i in range(len(norm_scores) - 1):
            pt1 = (int(x + i * step), int(y + norm_scores[i]))
            pt2 = (int(x + (i + 1) * step), int(y + norm_scores[i + 1]))

            # Line color: Green if rising, Red if falling
            color = (0, 255, 255)
            cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

        return img

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            self.frame_count += 1

            # AI INFERENCE
            if self.frame_count % 5 == 0:
                processed, emo, score, new_bbox = analyzer.process_frame(img)
                self.last_emotion = emo
                self.last_score = score

                # Update Graph Data
                self.score_history.append(score)

                # Smooth BBox (EMA)
                if new_bbox is not None:
                    new_bbox = np.array(new_bbox).astype(float)
                    if self.bbox is None:
                        self.bbox = new_bbox
                    else:
                        self.bbox = 0.7 * self.bbox + 0.3 * new_bbox
                    self.has_bbox = True

                output = img
            else:
                output = img

            # VISUALIZATION LAYER

            # Face Bounding Box
            if self.has_bbox and self.bbox is not None:
                x1, y1, x2, y2 = self.bbox.astype(int)

                # Color changes based on Engagement: Green (High), Gray (Low)
                color = (0, 255, 0) if self.last_score > 60 else (100, 100, 100)
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

                # Emotion labels above the head
                cv2.putText(
                    output,
                    self.last_emotion.upper(),
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            # Bar Engagement
            cv2.putText(
                output,
                f"Engagement: {int(self.last_score)}%",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            bar_width = int((self.last_score / 100) * 200)
            cv2.rectangle(
                output, (20, 50), (220, 65), (50, 50, 50), -1
            )  # Background bar
            cv2.rectangle(
                output, (20, 50), (20 + bar_width, 65), (0, 255, 0), -1
            )  # Fill bar

            # Real-time Graphic
            cv2.putText(
                output,
                "Live Trend (30s)",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            output = self.draw_sparkline(
                output, list(self.score_history), 20, 100, 200, 50
            )

            return av.VideoFrame.from_ndarray(output, format="bgr24")

        except Exception as e:
            print(f"Error: {e}")
            return frame


# DASHBOARD LAYOUT
st.markdown("<div class='dashboard-row'>", unsafe_allow_html=True)

# LEFT COLUMN
st.markdown("<div class='left-col'>", unsafe_allow_html=True)
st.markdown("<div class='st-card'>", unsafe_allow_html=True)

st.markdown("<h3 class='st-sub'>🎥 Live Camera Feed</h3>", unsafe_allow_html=True)

webrtc_streamer(
    key="emotion-modern-ui",
    video_processor_factory=EmotionProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


# RIGHT COLUMN
st.markdown("<div class='right-col'>", unsafe_allow_html=True)
st.markdown("<div class='st-card'>", unsafe_allow_html=True)

st.markdown("<h3 class='st-sub'>📊 Analytics Insight</h3>", unsafe_allow_html=True)

st.markdown(
    """
<div class='st-info'>
Facial expression inference with EMA stabilization and YOLOv8 face detection.
<br><br>
<b>High Engagement:</b> Happy / Surprise<br>
<b>Medium Engagement:</b> Neutral<br>
<b>Low Engagement:</b> Sad / Angry / Fear
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"<hr><div class='st-info'>Running on: <b>{analyzer.device}</b></div>",
    unsafe_allow_html=True,
)

# CPU Disclaimer
if str(analyzer.device) == "cpu":
    st.warning(
        "⚠️ The demo runs on a cloud CPU. For optimal performance, run it locally with a CUDA GPU."
    )

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Close row
st.markdown("</div>", unsafe_allow_html=True)
