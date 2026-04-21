"""
ManowBot – ผู้ช่วยดึงออเดอร์ไลฟ์สด
PyQt5 GUI  ·  yt-dlp Audio Stream  ·  Groq AI Pipeline
"""

import sys
import os
import re
import json
import wave
import subprocess
from io import BytesIO
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq
import requests

from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QUrl, QTimer, QSize, QPropertyAnimation,
    QEasingCurve, pyqtProperty,
)
from PyQt5.QtGui import QFont, QColor, QPainter, QLinearGradient, QBrush, QPen, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QTextEdit, QLabel, QSplitter, QFrame,
    QGraphicsDropShadowEffect, QScrollArea,
)
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings

load_dotenv()

# ══════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 5
MAX_BUFFER_LEN = 300
FIREBASE_URL = os.getenv("FIREBASE_URL", "")
FIREBASE_AUTH = os.getenv("FIREBASE_AUTH", "")


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════

def extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=)([\w-]{11})", r"youtu\.be/([\w-]{11})",
        r"embed/([\w-]{11})", r"live/([\w-]{11})", r"shorts/([\w-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ══════════════════════════════════════════════════════════════
#  AI Functions
# ══════════════════════════════════════════════════════════════

def transcribe_audio(wav_bytes: bytes) -> str:
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        transcription = client.audio.transcriptions.create(
            file=("chunk.wav", wav_bytes),
            model="whisper-large-v3",
            language="th",
            response_format="text",
        )
        return transcription.strip() if transcription else ""
    except Exception as e:
        print(f"Error in transcribe_audio: {e}")
        return ""


def extract_data(text: str) -> dict | None:
    if not text:
        return None

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    system_prompt = """\
คุณเป็น AI ขั้นเทพสำหรับดึงข้อมูลออเดอร์ไลฟ์สดเสื้อผ้าสาวอวบ "ร้านมะนาวแซ่บ"
หน้าที่ของคุณคือดึง "รายการที่" (item), "ไซส์/สัดส่วน" (size) และ "ราคา" (price) จากข้อความ

กฎเหล็ก (ต้องวิเคราะห์ข้อบกพร่องของระบบพิมพ์ตามเสียงด้วย):
1. item: รหัสสินค้า (โดยปกติมีตั้งแต่ 1-60)
2. price: ราคา (โดยปกติเป็นหลักสิบ เช่น 20, 30, 50, 60, 80, 100)
3. size: สัดส่วนและไซส์ เช่น XL, 2XL (ถ้าได้ยิน 2x ให้แปลว่า 2XL), 3XL หรือ อก 48 ยาว 30 (ถ้าไม่พูดให้เป็น null)

⚠️ กฎพิเศษสำหรับการแก้คำเพี้ยน (สำคัญมาก):
- "ตัวเลขฟิวชั่น": ระบบเสียงมักถอดรหัส+ราคาติดกัน เช่น "1160" ให้คุณแยกเป็น รหัส 11 ราคา 60, "3860" คือ รหัส 38 ราคา 60, "750" คือ รหัส 7 ราคา 50, "480" คือ รหัส 4 ราคา 80
- "อก กลายเป็น 6": คำว่า "อก" มักถูกพิมพ์ผิดเป็นเลข "6" เช่น "648" แปลว่า "อก 48" หรือ "654" แปลว่า "อก 54" ห้ามนำเลข 6 นำหน้ามาเป็นรหัสสินค้าเด็ดขาด!
- "ราคาสุดท้าย": หากแม่ค้ามีการเปลี่ยนใจลดราคา เช่น "ตอนแรกขาย 100 เหลือ 80" หรือ "เอาไป 50 บาทพอ" ให้ใช้ "ราคาสุดท้าย" (ราคาที่ถูกกว่า) เสมอ

ตัวอย่างการทำงาน:
- "ราย การ ที่ 3860 บาท 64 8 ยาว 30" → {"item": 38, "size": "อก 48 ยาว 30", "price": 60}
- "ราย การ ที่ 1160 บาท ไซส์ 2x" → {"item": 11, "size": "2XL", "price": 60}
- "ตัว นี้ 80 บาท เอา ไป 50 บาท พอ 654" → {"item": null, "size": "อก 54", "price": 50} 

- ตอบกลับเป็น JSON เท่านั้น: {"item": number, "size": string|null, "price": number}
- ถ้าหา item หรือ price ไม่เจอเลย ให้ตอบ: null"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        if data and data.get("item") is not None and data.get("price") is not None:
            return data
        return None
    except Exception as e:
        print(f"Error in extract_data: {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  Audio Worker Thread
# ══════════════════════════════════════════════════════════════

class AudioWorker(QThread):
    log_signal = pyqtSignal(str, str)       # (msg, level)
    order_signal = pyqtSignal(dict)
    status_signal = pyqtSignal(str)         # status text

    def __init__(self, youtube_url: str):
        super().__init__()
        self.youtube_url = youtube_url
        self._running = True
        self._process = None

    def stop(self):
        self._running = False
        if self._process:
            try:
                self._process.terminate()
            except Exception:
                pass

    def run(self):
        import yt_dlp

        self.status_signal.emit("connecting")
        self.log_signal.emit("กำลังดึง Audio Stream URL ...", "info")

        try:
            ydl_opts = {"format": "bestaudio/best", "quiet": True, "no_warnings": True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.youtube_url, download=False)
                audio_url = info["url"]
        except Exception as e:
            self.log_signal.emit(f"yt-dlp error: {e}", "error")
            self.status_signal.emit("error")
            return

        self.log_signal.emit("เริ่มดึงเสียงจาก Stream ผ่าน FFmpeg ...", "info")
        try:
            self._process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-re",
                    "-reconnect", "1",
                    "-reconnect_streamed", "1",
                    "-reconnect_delay_max", "5",
                    "-i", audio_url,
                    "-f", "s16le",
                    "-acodec", "pcm_s16le",
                    "-ar", str(SAMPLE_RATE),
                    "-ac", str(CHANNELS),
                    "-loglevel", "quiet",
                    "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            self.log_signal.emit("ไม่พบ FFmpeg — กรุณาติดตั้ง (ดู README.md)", "error")
            self.status_signal.emit("error")
            return

        self.status_signal.emit("streaming")
        chunk_bytes = SAMPLE_RATE * 2 * CHUNK_DURATION
        previous_text = ""
        last_sent = None
        chunk_num = 0

        while self._running:
            raw_audio = self._process.stdout.read(chunk_bytes)
            if not raw_audio or len(raw_audio) < chunk_bytes:
                if self._running:
                    self.log_signal.emit("Stream สิ้นสุดหรือขาดการเชื่อมต่อ", "warning")
                break

            chunk_num += 1

            wav_buffer = BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(raw_audio)
            wav_bytes = wav_buffer.getvalue()

            wav_bytes = wav_buffer.getvalue()

            # ป้องกัน AI มโนคำพูด (Hallucination) เวลาได้ยินเสียงเงียบ
            import struct
            import math
            try:
                samples = struct.unpack(f"<{len(raw_audio)//2}h", raw_audio)
                rms = math.sqrt(sum(s*s for s in samples) / len(samples))
            except Exception:
                rms = 0
                
            if rms < 300: # ถ้าเสียงเบามากให้ข้ามไปเลย
                self.log_signal.emit(f"[Chunk {chunk_num}] เงียบ (สแตนด์บาย) ...", "dim")
                continue

            self.log_signal.emit(f"[Chunk {chunk_num}] กำลังถอดเสียง ...", "dim")
            try:
                text = transcribe_audio(wav_bytes)
            except Exception as e:
                self.log_signal.emit(f"Whisper error: {e}", "error")
                continue

            if not text:
                self.log_signal.emit(f"[Chunk {chunk_num}] ไม่พบเสียงพูด", "dim")
                continue

            self.log_signal.emit(f"{text}", "speech")

            combined = f"{previous_text} {text}".strip()
            if len(combined) > MAX_BUFFER_LEN:
                combined = combined[-MAX_BUFFER_LEN:]

            try:
                data = extract_data(combined)
            except Exception as e:
                self.log_signal.emit(f"LLM error: {e}", "error")
                previous_text = combined
                continue

            if data:
                if data == last_sent:
                    self.log_signal.emit("รายการซ้ำ — ข้าม", "dim")
                else:
                    self.order_signal.emit(data)
                    last_sent = data
                previous_text = ""
            else:
                self.log_signal.emit("ข้อมูลไม่ครบ — เก็บไว้ทบรอบถัดไป", "dim")
                previous_text = combined

        if self._process:
            self._process.terminate()
        self.status_signal.emit("idle")
        self.log_signal.emit("Audio Worker หยุดทำงาน", "info")


# ══════════════════════════════════════════════════════════════
#  Pulsing Dot Widget (status indicator)
# ══════════════════════════════════════════════════════════════

class PulsingDot(QWidget):
    """Animated status dot with color transitions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self._color = QColor("#555566")
        self._opacity = 1.0

        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse)
        self._pulse_dir = 1
        self._pulsing = False

    def set_status(self, status: str):
        colors = {
            "idle":       "#555566",
            "connecting": "#f39c12",
            "streaming":  "#2ecc71",
            "error":      "#e74c3c",
        }
        self._color = QColor(colors.get(status, "#555566"))
        if status == "streaming":
            self._pulsing = True
            self._pulse_timer.start(60)
        else:
            self._pulsing = False
            self._pulse_timer.stop()
            self._opacity = 1.0
        self.update()

    def _pulse(self):
        self._opacity += 0.04 * self._pulse_dir
        if self._opacity >= 1.0:
            self._opacity = 1.0
            self._pulse_dir = -1
        elif self._opacity <= 0.35:
            self._opacity = 0.35
            self._pulse_dir = 1
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        color = QColor(self._color)
        color.setAlphaF(self._opacity)
        # glow
        glow = QColor(self._color)
        glow.setAlphaF(self._opacity * 0.3)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(glow))
        p.drawEllipse(0, 0, 14, 14)
        # dot
        p.setBrush(QBrush(color))
        p.drawEllipse(3, 3, 8, 8)
        p.end()


# ══════════════════════════════════════════════════════════════
#  Stat Card Widget
# ══════════════════════════════════════════════════════════════

class StatCard(QFrame):
    def __init__(self, icon: str, label: str, value: str = "0", accent: str = "#6c5ce7"):
        super().__init__()
        self.setObjectName("statCard")
        self._accent = accent

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)

        # icon + label row
        top = QHBoxLayout()
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet(f"font-size: 16px; color: {accent};")
        top.addWidget(icon_lbl)
        text_lbl = QLabel(label)
        text_lbl.setStyleSheet("font-size: 11px; color: #8888aa; font-weight: 600; letter-spacing: 1px;")
        top.addWidget(text_lbl)
        top.addStretch()
        layout.addLayout(top)

        # value
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"""
            font-size: 28px;
            font-weight: bold;
            color: {accent};
            padding-top: 2px;
        """)
        layout.addWidget(self.value_label)

    def set_value(self, val):
        self.value_label.setText(str(val))


# ══════════════════════════════════════════════════════════════
#  Dark Theme QSS
# ══════════════════════════════════════════════════════════════

DARK_STYLE = """
/* ── Main ── */
QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #0a0a12, stop:1 #0f0f1a);
}
QWidget#centralWidget {
    background: transparent;
}

/* ── Header ── */
QLabel#headerTitle {
    color: #ffffff;
    font-size: 24px;
    font-weight: 800;
}
QLabel#headerSub {
    color: #6b6b88;
    font-size: 12px;
    font-weight: 500;
}

/* ── Cards ── */
QFrame#card, QFrame#statCard {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(22, 22, 35, 240), stop:1 rgba(18, 18, 28, 250));
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
}

/* ── URL Input ── */
QLineEdit#urlInput {
    background-color: rgba(20, 20, 32, 220);
    color: #d0d0e0;
    border: 1px solid rgba(108, 92, 231, 0.25);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    font-weight: 500;
    selection-background-color: #6c5ce7;
}
QLineEdit#urlInput:focus {
    border: 1px solid rgba(108, 92, 231, 0.7);
    background-color: rgba(25, 25, 40, 240);
}
QLineEdit#urlInput::placeholder {
    color: #4a4a66;
}

/* ── Start Button ── */
QPushButton#startBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #6c5ce7, stop:0.5 #7c6cf7, stop:1 #a855f7);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 12px 32px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.5px;
    min-width: 120px;
}
QPushButton#startBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #7c6cf7, stop:0.5 #8c7cff, stop:1 #b865ff);
}
QPushButton#startBtn:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #5b4bd6, stop:1 #9845e7);
    padding-top: 13px;
    padding-bottom: 11px;
}

/* ── Stop Button ── */
QPushButton#stopBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #e74c3c, stop:1 #ff6b81);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 12px 32px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.5px;
    min-width: 120px;
}
QPushButton#stopBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #ff5e4f, stop:1 #ff7b91);
}

/* ── Section Labels ── */
QLabel#sectionLabel {
    color: #7878a0;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    padding: 8px 4px 6px 6px;
    text-transform: uppercase;
}

/* ── Log Area ── */
QTextEdit#logArea {
    background-color: rgba(10, 10, 18, 200);
    color: #b0b0cc;
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 14px;
    font-family: 'Cascadia Code', 'JetBrains Mono', 'Consolas', monospace;
    font-size: 12px;
    line-height: 1.6;
    selection-background-color: rgba(108, 92, 231, 0.5);
}

/* ── Status Bar ── */
QFrame#statusBar {
    background: rgba(14, 14, 22, 220);
    border-top: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 0px;
    padding: 0px;
}
QLabel#statusText {
    color: #5a5a78;
    font-size: 11px;
    font-weight: 500;
}
QLabel#statusTime {
    color: #3a3a55;
    font-size: 11px;
}

/* ── Splitter ── */
QSplitter::handle:vertical {
    background: rgba(108, 92, 231, 0.15);
    height: 3px;
    margin: 2px 40px;
    border-radius: 1px;
}
QSplitter::handle:vertical:hover {
    background: rgba(108, 92, 231, 0.4);
}
QSplitter::handle:horizontal {
    background: rgba(108, 92, 231, 0.15);
    width: 3px;
    margin: 40px 2px;
    border-radius: 1px;
}
QSplitter::handle:horizontal:hover {
    background: rgba(108, 92, 231, 0.4);
}

/* ── Scrollbar ── */
QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 4px 0px;
}
QScrollBar::handle:vertical {
    background: rgba(108, 92, 231, 0.3);
    border-radius: 3px;
    min-height: 40px;
}
QScrollBar::handle:vertical:hover {
    background: rgba(108, 92, 231, 0.6);
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
}
"""


# ══════════════════════════════════════════════════════════════
#  Main Window
# ══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ManowBot")
        self.setWindowIcon(QIcon("icon.png"))
        self.setMinimumSize(1000, 740)
        self.resize(1100, 800)

        self.worker = None
        self.order_count = 0
        self.chunk_count = 0
        self.total_revenue = 0

        self._build_ui()
        self._start_clock()

    # ── Build UI ────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 18, 24, 0)
        root.setSpacing(12)

        # ── Header Row ──
        header = QHBoxLayout()
        header.setSpacing(10)

        # Title group
        title_group = QVBoxLayout()
        title_group.setSpacing(0)
        title = QLabel("ManowBot")
        title.setObjectName("headerTitle")
        title_group.addWidget(title)
        subtitle = QLabel("ระบบ AI ดึงออเดอร์ไลฟ์สด")
        subtitle.setObjectName("headerSub")
        title_group.addWidget(subtitle)
        header.addLayout(title_group)
        header.addStretch()

        # Status indicator
        status_container = QHBoxLayout()
        status_container.setSpacing(6)
        self.status_dot = PulsingDot()
        status_container.addWidget(self.status_dot)
        self.status_label = QLabel("รอทำงาน")
        self.status_label.setStyleSheet("color: #555566; font-size: 12px; font-weight: 600;")
        status_container.addWidget(self.status_label)
        header.addLayout(status_container)

        root.addLayout(header)

        # ── URL Input Card ──
        url_card = QFrame()
        url_card.setObjectName("card")
        url_inner = QHBoxLayout(url_card)
        url_inner.setContentsMargins(6, 6, 6, 6)
        url_inner.setSpacing(8)

        self.url_input = QLineEdit()
        self.url_input.setObjectName("urlInput")
        self.url_input.setPlaceholderText("วางลิงก์ไลฟ์สด YouTube ที่นี่ ...")
        self.url_input.returnPressed.connect(self.toggle_stream)
        url_inner.addWidget(self.url_input)

        self.start_btn = QPushButton("▶  เริ่มทำงาน")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self.toggle_stream)
        url_inner.addWidget(self.start_btn)

        # glow effect on card
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(30)
        glow.setColor(QColor(108, 92, 231, 40))
        glow.setOffset(0, 4)
        url_card.setGraphicsEffect(glow)

        root.addWidget(url_card)

        # ── Stats Row ──
        stats_row = QHBoxLayout()
        stats_row.setSpacing(10)

        self.stat_orders = StatCard("📦", "ออเดอร์", "0", "#6c5ce7")
        self.stat_chunks = StatCard("🎙", "รอบเสียง", "0", "#00cec9")
        self.stat_revenue = StatCard("💰", "ยอดขาย", "฿0", "#fdcb6e")

        for card in [self.stat_orders, self.stat_chunks, self.stat_revenue]:
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(20)
            shadow.setColor(QColor(0, 0, 0, 50))
            shadow.setOffset(0, 4)
            card.setGraphicsEffect(shadow)
            stats_row.addWidget(card)

        root.addLayout(stats_row)

        # ── Main Content: Splitter (WebView + Log) ──
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(6)

        # YouTube Card
        web_card = QFrame()
        web_card.setObjectName("card")
        web_lay = QVBoxLayout(web_card)
        web_lay.setContentsMargins(2, 2, 2, 2)
        web_lay.setSpacing(0)

        web_header = QLabel("  แสดงภาพไลฟ์สด")
        web_header.setObjectName("sectionLabel")
        web_lay.addWidget(web_header)

        self.web_view = QWebEngineView()
        self.web_view.setMinimumHeight(250)
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.PlaybackRequiresUserGesture, False)
        # blank page with dark bg
        self.web_view.setHtml("""
            <html><body style="margin:0;background:#0d0d16;display:flex;
            align-items:center;justify-content:center;height:100vh;
            font-family:sans-serif;color:#3a3a55;">
            <div style="text-align:center;">
                <div style="font-size:48px;margin-bottom:12px;">📺</div>
                <div style="font-size:13px;letter-spacing:1px;">
                    วางลิงก์ YouTube Live แล้วกดปุ่มเริ่มทำงาน
                </div>
            </div></body></html>
        """)
        web_lay.addWidget(self.web_view)
        web_card.setMaximumWidth(420)
        splitter.addWidget(web_card)

        # Log Card
        log_card = QFrame()
        log_card.setObjectName("card")
        log_lay = QVBoxLayout(log_card)
        log_lay.setContentsMargins(2, 2, 2, 6)
        log_lay.setSpacing(0)

        log_header_row = QHBoxLayout()
        log_lbl = QLabel("  บันทึกการทำงาน")
        log_lbl.setObjectName("sectionLabel")
        log_header_row.addWidget(log_lbl)
        log_header_row.addStretch()

        self.clear_btn = QPushButton("ล้าง")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #4a4a66;
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 6px;
                padding: 4px 14px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                color: #8888aa;
                border-color: rgba(255,255,255,0.12);
            }
        """)
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.clicked.connect(lambda: self.log_area.clear())
        log_header_row.addWidget(self.clear_btn)

        log_lay.addLayout(log_header_row)

        self.log_area = QTextEdit()
        self.log_area.setObjectName("logArea")
        self.log_area.setReadOnly(True)
        self.log_area.setMinimumHeight(140)
        log_lay.addWidget(self.log_area)
        splitter.addWidget(log_card)

        splitter.setSizes([360, 740])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

        # ── Status Bar ──
        status_bar = QFrame()
        status_bar.setObjectName("statusBar")
        status_bar.setFixedHeight(32)
        sb_layout = QHBoxLayout(status_bar)
        sb_layout.setContentsMargins(16, 0, 16, 0)

        self.sb_text = QLabel("พร้อมทำงาน")
        self.sb_text.setObjectName("statusText")
        sb_layout.addWidget(self.sb_text)
        sb_layout.addStretch()

        self.sb_time = QLabel("")
        self.sb_time.setObjectName("statusTime")
        sb_layout.addWidget(self.sb_time)

        root.addWidget(status_bar)

    # ── Clock ───────────────────────────────────────────────
    def _start_clock(self):
        self._clock = QTimer(self)
        self._clock.timeout.connect(
            lambda: self.sb_time.setText(datetime.now().strftime("%H:%M:%S"))
        )
        self._clock.start(1000)

    # ── Toggle Start / Stop ─────────────────────────────────
    def toggle_stream(self):
        if self.worker and self.worker.isRunning():
            self._stop_worker()
        else:
            self._start_worker()

    def _start_worker(self):
        url = self.url_input.text().strip()
        if not url:
            self.log("กรุณาใส่ URL ไลฟ์สด YouTube", "warning")
            return

        video_id = extract_video_id(url)
        if video_id:
            # ใช้ youtube-nocookie ควบคู่กับ HTTP origin localhost เพื่อบายพาสข้อจำกัด 152-4
            embed_url = f"https://www.youtube-nocookie.com/embed/{video_id}?autoplay=1&mute=1"
            html_content = f"<html><body style='margin:0;background:#0d0d16;overflow:hidden;'><iframe width='100%' height='100%' src='{embed_url}' frameborder='0' allow='autoplay; encrypted-media' allowfullscreen></iframe></body></html>"
            
            # ตั้งค่า User-Agent ใหม่ให้เหมือน Chrome ปกติเพื่อป้องกันการบล็อคจาก YouTube
            self.web_view.page().profile().setHttpUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            self.web_view.setHtml(html_content, QUrl("http://localhost"))
            self.log(f"โหลด YouTube Player — ID: {video_id}", "info")
        else:
            self.log("ไม่พบ Video ID จาก URL", "error")
            return

        # Reset stats
        self.order_count = 0
        self.chunk_count = 0
        self.total_revenue = 0
        self._update_stats()

        self.worker = AudioWorker(url)
        self.worker.log_signal.connect(self.log)
        self.worker.order_signal.connect(self.on_order)
        self.worker.status_signal.connect(self._set_status)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

        self.start_btn.setText("⏹  หยุดทำงาน")
        self.start_btn.setObjectName("stopBtn")
        self.start_btn.setStyle(self.start_btn.style())
        self.url_input.setEnabled(False)
        self.sb_text.setText("กำลังดึงข้อมูล ...")

    def _stop_worker(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(5000)
        self._reset_ui()
        self.log("หยุดทำงานแล้ว", "info")

    def _on_worker_finished(self):
        self._reset_ui()

    def _reset_ui(self):
        self.start_btn.setText("▶  เริ่มทำงาน")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setStyle(self.start_btn.style())
        self.url_input.setEnabled(True)
        self._set_status("idle")
        self.sb_text.setText("พร้อมทำงาน")
        
        # รีเซ็ตหน้าวีดีโอกลับเป็นค่าเริ่มต้น (หยุดเล่นเสียงคลิป)
        self.web_view.setUrl(QUrl("about:blank"))
        self.web_view.setHtml("""
            <html><body style="margin:0;background:#0d0d16;display:flex;
            align-items:center;justify-content:center;height:100vh;
            font-family:sans-serif;color:#3a3a55;">
            <div style="text-align:center;">
                <div style="font-size:48px;margin-bottom:12px;">📺</div>
                <div style="font-size:13px;letter-spacing:1px;">
                    วางลิงก์ YouTube Live แล้วกดปุ่มเริ่มทำงาน
                </div>
            </div></body></html>
        """)

    def _set_status(self, status: str):
        self.status_dot.set_status(status)
        labels = {
            "idle": ("รอทำงาน", "#555566"),
            "connecting": ("กำลังเชื่อมต่อ ...", "#f39c12"),
            "streaming": ("กำลังวิเคราะห์เสียง", "#2ecc71"),
            "error": ("เกิดข้อผิดพลาด", "#e74c3c"),
        }
        text, color = labels.get(status, ("รอทำงาน", "#555566"))
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 600;")

    def _update_stats(self):
        self.stat_orders.set_value(str(self.order_count))
        self.stat_chunks.set_value(str(self.chunk_count))
        self.stat_revenue.set_value(f"฿{self.total_revenue:,}")

    # ── Log ─────────────────────────────────────────────────
    def log(self, msg: str, level: str = "info"):
        time_str = now_str()
        styles = {
            "info":    ("💬", "#8888bb"),
            "speech":  ("🗣", "#74b9ff"),
            "dim":     ("·", "#4a4a66"),
            "warning": ("⚠", "#ffc048"),
            "error":   ("✖", "#ff6b6b"),
            "order":   ("✦", "#6c5ce7"),
            "success": ("✔", "#2ecc71"),
        }
        icon, color = styles.get(level, ("·", "#8888bb"))

        html = (
            f'<div style="margin:2px 0; line-height:1.5;">'
            f'<span style="color:#3a3a55; font-size:11px;">{time_str}</span> '
            f'<span style="color:{color};">{icon}</span> '
            f'<span style="color:{color};">{msg}</span>'
            f'</div>'
        )
        self.log_area.append(html)

        # auto-scroll
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

        # chunk counter
        if "กำลังถอดเสียง" in msg or "สแตนด์บาย" in msg:
            self.chunk_count += 1
            self.stat_chunks.set_value(str(self.chunk_count))

    # ── Order received ──────────────────────────────────────
    def on_order(self, data: dict):
        self.order_count += 1
        price = data.get("price", 0)
        self.total_revenue += price
        self._update_stats()

        self.log(
            f"ORDER #{self.order_count}  ·  "
            f"Item {data['item']}  ·  ฿{price:,}",
            "order"
        )

        item_id = data.get("item")
        size = data.get("size", "")

        url_text = self.url_input.text().strip()
        video_id = extract_video_id(url_text)

        if not video_id:
            self.log("ไม่พบ Video ID สำหรับส่ง Firebase", "error")
            return

        if not FIREBASE_URL or not FIREBASE_AUTH:
            self.log("ข้อมูล Firebase ใน .env ไม่ครบถ้วน", "warning")
            return

        clean_fb_url = FIREBASE_URL.replace("https://", "").rstrip("/")

        # Update stock price via PATCH
        stock_url = f"https://{clean_fb_url}/stock/{video_id}/{item_id}.json?auth={FIREBASE_AUTH}"
        stock_payload = {"price": price}
        if size:
            stock_payload["size"] = size

        # Update live overlay via PUT
        overlay_url = f"https://{clean_fb_url}/overlay/{video_id}/current_item.json?auth={FIREBASE_AUTH}"
        overlay_payload = {
            "id": item_id,
            "price": price,
            "size": size if size else None,
            "updatedAt": {".sv": "timestamp"}
        }

        # Send to Firebase
        try:
            # Update Stock
            r_stock = requests.patch(stock_url, json=stock_payload, timeout=5)
            r_stock.raise_for_status()

            # Update Overlay
            r_overlay = requests.put(overlay_url, json=overlay_payload, timeout=5)
            r_overlay.raise_for_status()

            self.log(f"อัปเดตรายการ {item_id} (Stock & Overlay) สำเร็จ", "success")
        except requests.RequestException as e:
            # Handle possible response object in exception for status code logging
            status_code = e.response.status_code if e.response is not None else "N/A"
            self.log(f"อัปเดตรายการ {item_id} ไม่สำเร็จ (Status: {status_code}) - {e}", "error")

    # ── Close event ─────────────────────────────────────────
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()


# ══════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    app.setFont(QFont("Segoe UI", 10))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
