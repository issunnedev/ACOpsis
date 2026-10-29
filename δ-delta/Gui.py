#!/usr/bin/env python3
"""
ACOpsis DELTA · control panel (PyQt6)

    pip install PyQt6
    python gui.py            # or:  python train.py   /   python train.py gui

The GUI never imports torch. Every job runs `train.py` in a child process and talks to it
through machine-readable events (ACOPSIS_EVENTS=1). That keeps the window responsive, frees
GPU memory the moment a job ends, avoids the Windows torch/Qt DLL clash, and lets "Stop"
shut a run down gracefully (the trainer finishes its step and saves before exiting).
"""

from __future__ import annotations

import html
import json
import os
import random
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PyQt6.QtCore import QObject, QPointF, QProcess, QProcessEnvironment, QSettings, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QDesktopServices, QFont, QPainter, QPainterPath, QPalette, QPen, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "train.py"
DATA_DIR = HERE / "data"
CORPUS = DATA_DIR / "acopsis_corpus.jsonl"
PREFIX = "@@ACO@@"

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #

BG, PANEL, PANEL2 = "#12121a", "#1a1a25", "#232334"
TEXT, DIM, ACCENT = "#e6e6f0", "#8a8aa3", "#b072ff"
GOOD, WARN, BAD, INFO = "#58d68d", "#f4d03f", "#ff6b6b", "#5dade2"

QSS = f"""
QWidget {{ font-size: 13px; }}
QGroupBox {{ border: 1px solid #2c2c3f; border-radius: 8px; margin-top: 14px; padding: 12px 8px 8px 8px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {ACCENT}; }}
QPushButton {{ background: #2a2a3e; border: 1px solid #3a3a54; border-radius: 6px; padding: 6px 14px; }}
QPushButton:hover {{ background: #34344e; }}
QPushButton:disabled {{ color: #666682; background: #1e1e2b; border-color: #2a2a3a; }}
QPushButton#primary {{ background: #7c4dff; border-color: #9b72ff; color: white; font-weight: 600; }}
QPushButton#primary:hover {{ background: #8f66ff; }}
QPushButton#primary:disabled {{ background: #3b2d70; border-color: #4a3d80; color: #8a80b0; }}
QPushButton#danger {{ background: #7d2b39; border-color: #a64455; color: white; }}
QPushButton#danger:hover {{ background: #96384a; }}
QPushButton#danger:disabled {{ background: #2a1e24; border-color: #3a2a30; color: #6a5a60; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit, QTableWidget {{
    background: #161620; border: 1px solid #2f2f45; border-radius: 6px; padding: 4px 6px;
    selection-background-color: #7c4dff; }}
QTableWidget {{ gridline-color: #262638; }}
QTabWidget::pane {{ border: 1px solid #2c2c3f; border-radius: 8px; top: -1px; }}
QTabBar::tab {{ background: {PANEL}; padding: 9px 22px; border-top-left-radius: 8px; border-top-right-radius: 8px;
    margin-right: 2px; color: #9a9ab6; }}
QTabBar::tab:selected {{ background: {PANEL2}; color: white; }}
QProgressBar {{ background: #161620; border: 1px solid #2f2f45; border-radius: 6px; text-align: center; min-height: 18px; }}
QProgressBar::chunk {{ background: #7c4dff; border-radius: 5px; }}
QHeaderView::section {{ background: {PANEL2}; border: none; padding: 4px 8px; }}
QScrollBar:vertical {{ background: transparent; width: 12px; }}
QScrollBar::handle:vertical {{ background: #34344e; border-radius: 5px; min-height: 28px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QToolTip {{ background: {PANEL2}; color: {TEXT}; border: 1px solid #3a3a54; }}
QSplitter::handle {{ background: #1f1f2d; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    roles = {
        QPalette.ColorRole.Window: BG, QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: "#161620", QPalette.ColorRole.AlternateBase: PANEL,
        QPalette.ColorRole.ToolTipBase: PANEL2, QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.Text: TEXT, QPalette.ColorRole.Button: "#2a2a3e",
        QPalette.ColorRole.ButtonText: TEXT, QPalette.ColorRole.BrightText: "#ffffff",
        QPalette.ColorRole.Highlight: "#7c4dff", QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.PlaceholderText: DIM,
    }
    for role, color in roles.items():
        pal.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor("#666682"))
    app.setPalette(pal)
    app.setStyleSheet(QSS)


# --------------------------------------------------------------------------- #
# Pure helpers (no Qt): protocol parsing and CLI argument building
# --------------------------------------------------------------------------- #

def parse_line(line: str):
    """('event', dict) for protocol lines, otherwise ('raw', text)."""
    if line.startswith(PREFIX):
        try:
            data = json.loads(line[len(PREFIX):])
            if isinstance(data, dict) and "t" in data:
                return "event", data
        except ValueError:
            pass
    return "raw", line


def fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_count(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return str(n)
    for unit, size in (("B", 1e9), ("M", 1e6), ("k", 1e4)):
        if abs(n) >= size:
            return f"{n / size:.2f}{unit}"
    return f"{int(n):,}"


def split_boards(text: str) -> list[str]:
    boards = []
    for part in text.replace(",", " ").split():
        board = part.strip("/ ").lower()
        if board and board not in boards:
            boards.append(board)
    return boards


def collect_args(v: dict) -> list[str]:
    boards = split_boards(str(v.get("boards", ""))) or ["g"]
    return ["collect", "--boards", *boards,
            "--max-threads", str(int(v["max_threads"])),
            "--min-length", str(int(v["min_length"])),
            "--delay", f"{float(v['delay']):g}"]


def train_args(v: dict, dry: bool = False) -> list[str]:
    args = ["train", "--model", str(v["model"]).strip()]
    if v.get("trust"):
        args.append("--trust-remote-code")
    eos = str(v.get("eos", "")).strip()
    if eos:
        args.append(f"--eos-token={eos}")
    overrides = shlex.split(str(v.get("overrides", "")))
    if overrides:
        args += ["--set", *overrides]
    init_std = str(v.get("init_std", "")).strip()
    if init_std:
        args += ["--init-std", init_std]
    args += [
        "--base-corpus", str(v["base_corpus"]).strip() or "none",
        "--base-tokens", str(int(v["base_tokens"])),
        "--forum-share", f"{float(v['forum_share']):g}",
        "--forum-repeat", str(int(v["forum_repeat"])),
    ]
    if v.get("no_curriculum"):
        args.append("--no-curriculum")
    args += [
        "--epochs", str(int(v["epochs"])),
        "--max-steps", str(int(v["max_steps"])),
        "--lr", f"{float(v['lr']):.8g}",
        "--batch-size", str(int(v["batch"])),
        "--grad-accum", str(int(v["accum"])),
        "--block-size", str(int(v["block"])),
        "--precision", str(v["precision"]),
        "--seed", str(int(v["seed"])),
        "--patience", str(int(v["patience"])),
        "--output", str(v["output"]).strip() or "checkpoints/acopsis-160m",
    ]
    if v.get("sample") and str(v.get("sample_prompt", "")):
        args.append(f"--sample-prompt={v['sample_prompt']}")
    if v.get("resume"):
        args.append("--resume")
    if v.get("anomaly"):
        args.append("--detect-anomaly")
    if dry:
        args.append("--dry-run")
    return args


def server_args(v: dict) -> list[str]:
    args = ["generate", "--serve", "--output", str(v["model_dir"]).strip()]
    if v.get("trust"):
        args.append("--trust-remote-code")
    return args


def cli_preview(args: list[str]) -> str:
    return "python train.py " + subprocess.list2cmdline(args)


def sample_posts(path: Path, k: int = 8) -> list[dict]:
    """Reservoir-sample k posts from the corpus without loading the whole file."""
    picked: list[dict] = []
    if not path.exists():
        return picked
    seen = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict) or not isinstance(rec.get("text"), str):
                continue
            seen += 1
            if len(picked) < k:
                picked.append(rec)
            else:
                j = random.randrange(seen)
                if j < k:
                    picked[j] = rec
    return picked


# --------------------------------------------------------------------------- #
# Child-process runner
# --------------------------------------------------------------------------- #

class Runner(QObject):
    """Runs train.py as a child process and turns its output into signals."""

    message = pyqtSignal(dict)    # structured event
    rawline = pyqtSignal(str)     # anything else (library warnings, tracebacks)
    done = pyqtSignal(int, bool)  # exit code, stopped-by-user

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_output)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_error)
        self._buf = b""
        self._stop_file: Path | None = None
        self._stopping = False
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self.kill)

    def running(self) -> bool:
        return self.proc.state() != QProcess.ProcessState.NotRunning

    def start(self, args: list[str]) -> bool:
        if self.running():
            return False
        self._stop_file = Path(tempfile.gettempdir()) / f"acopsis_stop_{os.getpid()}_{int(time.time() * 1000)}"
        try:
            self._stop_file.unlink()
        except OSError:
            pass
        env = QProcessEnvironment.systemEnvironment()
        for key, value in {
            "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
            "ACOPSIS_EVENTS": "1", "ACOPSIS_STOP_FILE": str(self._stop_file),
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TQDM_DISABLE": "1",
        }.items():
            env.insert(key, value)
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(HERE))
        self._buf, self._stopping = b"", False
        self.proc.start(sys.executable, ["-u", str(SCRIPT), *args])
        return True

    def send(self, payload: dict) -> None:
        if self.running():
            self.proc.write((json.dumps(payload) + "\n").encode("utf-8"))

    def request_stop(self, grace_ms: int = 120_000) -> None:
        """First call: ask nicely (the job saves and exits). Second call: kill."""
        if not self.running():
            return
        if self._stopping:
            self.kill()
            return
        self._stopping = True
        if self._stop_file is not None:
            try:
                self._stop_file.touch()
            except OSError:
                pass
        self._kill_timer.start(grace_ms)

    def quit_server(self) -> None:
        if self.running():
            self.send({"cmd": "quit"})
            self._stopping = True
            self._kill_timer.start(5000)

    def kill(self) -> None:
        if self.running():
            self.proc.kill()

    # -- internals
    def _dispatch(self, line: str) -> None:
        kind, payload = parse_line(line)
        if kind == "event":
            self.message.emit(payload)
        elif line.strip():
            self.rawline.emit(line)

    def _on_output(self) -> None:
        # tqdm-style progress bars redraw with "\r"; treat it as a line break.
        self._buf += bytes(self.proc.readAllStandardOutput().data()).replace(b"\r", b"\n")
        *lines, self._buf = self._buf.split(b"\n")
        for raw in lines:
            self._dispatch(raw.decode("utf-8", "replace").rstrip("\r"))

    def _on_finished(self, code, _status) -> None:
        self._kill_timer.stop()
        if self._buf:
            self._dispatch(self._buf.decode("utf-8", "replace").rstrip("\r"))
            self._buf = b""
        if self._stop_file is not None:
            try:
                self._stop_file.unlink()
            except OSError:
                pass
        self.done.emit(int(code), self._stopping)

    def _on_error(self, err) -> None:
        if err == QProcess.ProcessError.FailedToStart:
            self.rawline.emit(f"Could not start {sys.executable}: {self.proc.errorString()}")
            self.done.emit(-1, False)


# --------------------------------------------------------------------------- #
# Reusable widgets
# --------------------------------------------------------------------------- #

class LogView(QPlainTextEdit):
    COLORS = {"info": INFO, "ok": GOOD, "warn": WARN, "error": BAD, "raw": DIM,
              "banner": ACCENT, "section": ACCENT, "panel": TEXT, "cmd": "#9aa5ff"}
    PREFIX = {"info": "• ", "ok": "✔ ", "warn": "! ", "error": "✘ "}

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(8000)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(10)
        self.setFont(font)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)

    def add(self, level: str, text: str) -> None:
        bar = self.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(self.COLORS.get(level, TEXT)))
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(f"{self.PREFIX.get(level, '')}{text}\n", fmt)
        if at_bottom:
            bar.setValue(bar.maximum())


class LossChart(QWidget):
    """Tiny dependency-free line chart: raw + smoothed training loss, plus eval series."""

    PALETTE = ["#58d68d", "#f4d03f", "#ff9f43", "#ff6b9d"]

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(230)
        self.train: list[tuple[float, float]] = []
        self.evals: dict[str, list[tuple[float, float]]] = {}

    def clear(self) -> None:
        self.train, self.evals = [], {}
        self.update()

    def add_train(self, step, loss) -> None:
        self.train.append((float(step), float(loss)))
        self.update()

    def add_eval(self, step, name, loss) -> None:
        self.evals.setdefault(name, []).append((float(step), float(loss)))
        self.update()

    @staticmethod
    def _smooth(points, alpha=0.12):
        out, ema = [], None
        for x, y in points:
            ema = y if ema is None else ema + alpha * (y - ema)
            out.append((x, ema))
        return out

    @staticmethod
    def _path(points, mx, my) -> QPainterPath:
        path = QPainterPath()
        for i, (x, y) in enumerate(points):
            pt = QPointF(mx(x), my(y))
            if i == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        return path

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        try:
            self._paint(p)
        finally:
            p.end()

    def _paint(self, p: QPainter) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(PANEL))
        fm = p.fontMetrics()

        ys = sorted([y for _, y in self.train] + [y for pts in self.evals.values() for _, y in pts])
        if len(ys) < 2:
            msg = "Loss curves appear here once training starts"
            p.setPen(QColor(DIM))
            p.drawText(QPointF((w - fm.horizontalAdvance(msg)) / 2, h / 2), msg)
            return

        xs = [x for x, _ in self.train] + [x for pts in self.evals.values() for x, _ in pts]
        x0, x1 = min(xs), max(xs)
        if x1 - x0 < 1e-9:
            x1 = x0 + 1
        lo, hi = ys[0], ys[min(len(ys) - 1, int(0.97 * len(ys)))]  # ignore the huge first steps
        if hi - lo < 1e-9:
            hi = lo + 1
        pad = (hi - lo) * 0.08
        y0, y1 = lo - pad, hi + pad

        left, right, top, bottom = 58, 14, 14, 28
        pw, ph = max(10, w - left - right), max(10, h - top - bottom)

        def mx(x):
            return left + (x - x0) / (x1 - x0) * pw

        def my(y):
            return top + ph - (min(max(y, y0), y1) - y0) / (y1 - y0) * ph

        # grid + y labels
        for i in range(5):
            val = y0 + (y1 - y0) * i / 4
            yy = my(val)
            p.setPen(QPen(QColor("#2a2a3d"), 1))
            p.drawLine(QPointF(left, yy), QPointF(left + pw, yy))
            p.setPen(QColor(DIM))
            label = f"{val:.2f}"
            p.drawText(QPointF(left - 8 - fm.horizontalAdvance(label), yy + fm.ascent() / 2 - 1), label)
        p.setPen(QColor(DIM))
        p.drawText(QPointF(left, h - 8), f"step {int(x0)}")
        end_label = f"step {int(x1)}"
        p.drawText(QPointF(left + pw - fm.horizontalAdvance(end_label), h - 8), end_label)

        p.setBrush(Qt.BrushStyle.NoBrush)
        if len(self.train) >= 2:
            faint = QColor(INFO)
            faint.setAlpha(70)
            p.setPen(QPen(faint, 1))
            p.drawPath(self._path(self.train, mx, my))
            p.setPen(QPen(QColor(INFO), 2))
            p.drawPath(self._path(self._smooth(self.train), mx, my))

        legend = [("train", INFO)]
        for idx, (name, pts) in enumerate(self.evals.items()):
            color = QColor(self.PALETTE[idx % len(self.PALETTE)])
            p.setPen(QPen(color, 2))
            if len(pts) >= 2:
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(self._path(pts, mx, my))
            p.setBrush(color)
            for x, y in pts:
                p.drawEllipse(QPointF(mx(x), my(y)), 3.5, 3.5)
            p.setBrush(Qt.BrushStyle.NoBrush)
            legend.append((f"eval · {name}", color.name()))

        lx = left + pw
        for text, color in reversed(legend):
            lx -= fm.horizontalAdvance(text) + 18
            p.setPen(QColor(color))
            p.drawText(QPointF(lx, top + fm.ascent()), "● " + text)


class ProgressPanel(QFrame):
    stop_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel("Idle")
        self.label.setMinimumWidth(200)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setFormat("")
        self.detail = QLabel("")
        self.detail.setMinimumWidth(300)
        self.detail.setStyleSheet(f"color: {DIM};")
        self.stop = QPushButton("Stop")
        self.stop.setObjectName("danger")
        self.stop.setEnabled(False)
        self.stop.clicked.connect(lambda: self.stop_clicked.emit())
        row.addWidget(self.label)
        row.addWidget(self.bar, 1)
        row.addWidget(self.detail)
        row.addWidget(self.stop)
        self._t0: dict = {}

    def reset(self) -> None:
        self._t0.clear()
        self.label.setText("Starting…")
        self.bar.setValue(0)
        self.bar.setFormat("")
        self.detail.setText("")
        self.stop.setText("Stop")

    def idle(self, text: str = "Idle") -> None:
        self.label.setText(text)
        self.detail.setText("")

    def on_progress(self, ev: dict) -> None:
        pid, total, n = ev.get("id"), ev.get("total") or 0, ev.get("n") or 0
        started = self._t0.setdefault(pid, time.monotonic())
        frac = min(1.0, n / total) if total else 0.0
        self.bar.setValue(int(frac * 1000))
        self.bar.setFormat(f"{frac * 100:.1f}%")
        eta = (time.monotonic() - started) * (total - n) / n if n > 0 and total > n else 0
        self.label.setText(str(ev.get("desc", "")))
        status = ev.get("status") or ""
        self.detail.setText(f"{fmt_count(n)} / {fmt_count(total)} · ETA {fmt_duration(eta)}"
                            + (f" · {status}" if status else ""))


def get_value(w):
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, (QSpinBox, QDoubleSpinBox)):
        return w.value()
    if isinstance(w, QComboBox):
        return w.currentText()
    if isinstance(w, QLineEdit):
        return w.text()
    raise TypeError(type(w))


def set_value(w, v) -> None:
    if isinstance(w, QCheckBox):
        w.setChecked(bool(v))
    elif isinstance(w, QSpinBox):
        w.setValue(int(v))
    elif isinstance(w, QDoubleSpinBox):
        w.setValue(float(v))
    elif isinstance(w, QComboBox):
        idx = w.findText(str(v))
        if idx >= 0:
            w.setCurrentIndex(idx)
        elif w.isEditable():
            w.setCurrentText(str(v))
    elif isinstance(w, QLineEdit):
        w.setText(str(v))


def connect_changed(w, slot) -> None:
    if isinstance(w, QCheckBox):
        w.toggled.connect(slot)
    elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
        w.valueChanged.connect(slot)
    elif isinstance(w, QComboBox):
        w.currentTextChanged.connect(slot)
    elif isinstance(w, QLineEdit):
        w.textChanged.connect(slot)


def spin(lo, hi, value, step=1, group=False) -> QSpinBox:
    box = QSpinBox()
    box.setRange(lo, hi)
    box.setValue(value)
    box.setSingleStep(step)
    box.setGroupSeparatorShown(group)
    return box


def dspin(lo, hi, value, decimals=2, step=0.1) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(lo, hi)
    box.setValue(value)
    box.setSingleStep(step)
    return box


def primary(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("primary")
    return button


def scroll_wrap(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(widget)
    return area


class FieldsMixin:
    """Keeps a name -> widget registry so settings can be saved, restored and previewed."""

    def _init_fields(self) -> None:
        self.fields: dict[str, QWidget] = {}
        self.on_change = lambda *_: None

    def field(self, form: QFormLayout, key: str, label: str, widget: QWidget, tip: str = "") -> QWidget:
        self.fields[key] = widget
        if tip:
            widget.setToolTip(tip)
        form.addRow(label, widget)
        connect_changed(widget, lambda *_: self.on_change())
        return widget

    def values(self) -> dict:
        return {k: get_value(w) for k, w in self.fields.items()}

    def set_values(self, data: dict) -> None:
        for key, value in (data or {}).items():
            if key in self.fields:
                set_value(self.fields[key], value)


def new_form() -> QFormLayout:
    form = QFormLayout()
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setHorizontalSpacing(14)
    form.setVerticalSpacing(8)
    return form


def group(title: str, form: QFormLayout) -> QGroupBox:
    box = QGroupBox(title)
    box.setLayout(form)
    return box


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

class CollectTab(QWidget, FieldsMixin):
    def __init__(self, main: "MainWindow"):
        super().__init__()
        self._init_fields()
        self.main = main
        outer = QVBoxLayout(self)

        form = new_form()
        self.field(form, "boards", "Boards", QLineEdit("g sci"),
                   "Space- or comma-separated board names, e.g. g sci biz. Only collect from boards you have reviewed.")
        self.field(form, "max_threads", "New threads per board", spin(1, 5000, 20),
                   "Threads already archived are skipped, so you can run this repeatedly to grow the corpus.")
        self.field(form, "min_length", "Min post length (chars)", spin(0, 2000, 20))
        self.field(form, "delay", "Delay between requests (s)", dspin(1.0, 30.0, 1.1, 1, 0.1),
                   "The API is rate-limited; the script enforces at least 1 second.")
        outer.addWidget(group("Collect forum threads", form))

        note = QLabel("Posts are cleaned (quote links, URLs and emails removed), de-duplicated, and appended to "
                      "data/acopsis_corpus.jsonl. Collection is resumable and stops cleanly with the Stop button.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {DIM};")
        outer.addWidget(note)

        row = QHBoxLayout()
        self.start = primary("Start collecting")
        self.start.clicked.connect(lambda: main.start_job("collect", collect_args(self.values())))
        open_data = QPushButton("Open data folder")
        open_data.clicked.connect(main.open_data_folder)
        row.addWidget(self.start)
        row.addWidget(open_data)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)

    def set_busy(self, busy: bool) -> None:
        self.start.setEnabled(not busy)


TRAIN_DEFAULTS = dict(
    model="EleutherAI/pythia-160m", trust=False, eos="", overrides="", precision="auto", init_std="", block=256,
    base_corpus="fineweb-edu", base_tokens=20_000_000, forum_share=0.7, forum_repeat=3, no_curriculum=False,
    epochs=1, max_steps=0, lr=3e-4, batch=4, accum=8, patience=0, seed=1234,
    output="checkpoints/acopsis-160m", sample=True, sample_prompt="The ", resume=False, anomaly=False,
)

PRESETS = {
    "Pythia-160M · default (fits 6 GB)": {},
    "Smoke test · 5 steps, tiny data": dict(base_corpus="tinystories", base_tokens=300_000, max_steps=5,
                                            batch=1, accum=1, sample=False),
    "KLM-1.5 · experiment": dict(model="MinimaLabs/KLM-1.5", trust=True, batch=1, accum=32, base_tokens=2_000_000),
    "Baguettotron · shrunk to 24 layers": dict(model="PleIAs/Baguettotron", eos="<|end_of_text|>",
                                               overrides="num_hidden_layers=24", batch=1, accum=32,
                                               base_tokens=10_000_000, output="checkpoints/acopsis-baguette"),
}


class TrainTab(QWidget, FieldsMixin):
    def __init__(self, main: "MainWindow"):
        super().__init__()
        self._init_fields()
        self.main = main
        outer = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(split)
        split.addWidget(self._build_form())
        split.addWidget(self._build_monitor())
        split.setSizes([480, 760])
        self.set_values(TRAIN_DEFAULTS)
        self.on_change = self.refresh_command
        self.refresh_command()

    # -- left: settings
    def _build_form(self) -> QWidget:
        host = QWidget()
        lay = QVBoxLayout(host)

        top = QHBoxLayout()
        self.preset = QComboBox()
        self.preset.addItems(list(PRESETS))
        apply_btn = QPushButton("Apply preset")
        apply_btn.clicked.connect(self.apply_preset)
        top.addWidget(self.preset, 1)
        top.addWidget(apply_btn)
        lay.addLayout(top)

        row = QHBoxLayout()
        self.dry = QPushButton("Dry run")
        self.dry.setToolTip("Load the tokenizer and config, build the data mix, print the memory estimate. Trains nothing.")
        self.dry.clicked.connect(lambda: self.start(dry=True))
        self.start_btn = primary("Start training")
        self.start_btn.clicked.connect(lambda: self.start(dry=False))
        row.addWidget(self.dry)
        row.addWidget(self.start_btn, 1)
        lay.addLayout(row)

        f = new_form()
        model = QComboBox()
        model.setEditable(True)
        model.addItems(["EleutherAI/pythia-160m", "MinimaLabs/KLM-1.5", "PleIAs/Baguettotron", "LiquidAI/LFM3-1.7B"])
        self.field(f, "model", "Architecture", model,
                   "Hugging Face repo to copy the architecture and tokenizer from. Weights are NOT loaded: training starts from random.")
        self.field(f, "trust", "", QCheckBox("Trust remote code (runs the repo's Python; review it first)"))
        self.field(f, "eos", "EOS token", QLineEdit(),
                   "Document separator. Leave empty to use the tokenizer's own, e.g. <|end_of_text|> for Baguettotron.")
        self.fields["eos"].setPlaceholderText("auto")
        self.field(f, "overrides", "Config overrides", QLineEdit(),
                   "Shrink the model, e.g. hidden_size=512 num_hidden_layers=8. Check the result with Dry run.")
        self.fields["overrides"].setPlaceholderText("KEY=VALUE KEY=VALUE")
        prec = QComboBox()
        prec.addItems(["auto", "fp32", "bf16", "fp16"])
        self.field(f, "precision", "Precision", prec, "auto = bf16 on Ampere+ GPUs (RTX 30-series and newer), otherwise fp32.")
        self.field(f, "init_std", "Init std", QLineEdit(),
                   "Re-initialize Linear/Embedding weights with N(0, std). Empty = 0.02 for remote-code models, else the model's own.")
        self.fields["init_std"].setPlaceholderText("auto")
        self.field(f, "block", "Block size", spin(16, 32768, 256, 64), "Tokens per training sequence.")
        lay.addWidget(group("Model", f))

        f = new_form()
        corpus = QComboBox()
        corpus.setEditable(True)
        corpus.addItems(["fineweb-edu", "tinystories", "wikitext", "none"])
        self.field(f, "base_corpus", "Base corpus", corpus,
                   "General English that teaches fluency first. Any Hugging Face dataset id with a 'text' column also works.")
        self.field(f, "base_tokens", "Base tokens", spin(0, 2_000_000_000, 20_000_000, 1_000_000, True),
                   "More = more coherent text, slower training.")
        self.field(f, "forum_share", "Forum share (final phase)", dspin(0.05, 1.0, 0.7, 2, 0.05),
                   "Fraction of the last phase that is forum text; the rest is base-corpus replay.")
        self.field(f, "forum_repeat", "Forum passes", spin(1, 50, 3))
        self.field(f, "no_curriculum", "", QCheckBox("Shuffle base + forum together (no curriculum)"))
        lay.addWidget(group("Data", f))

        f = new_form()
        self.field(f, "epochs", "Epochs", spin(1, 100, 1), "Passes over the whole curriculum. Forum repetition is separate (Forum passes).")
        self.field(f, "max_steps", "Max steps (0 = all)", spin(0, 10_000_000, 0, 1, True))
        lr = dspin(0.000001, 1.0, 0.0003, 6, 0.0001)
        self.field(f, "lr", "Learning rate", lr)
        self.field(f, "batch", "Batch size", spin(1, 512, 4), "Per-step micro-batch. Lower this first if you run out of GPU memory.")
        self.field(f, "accum", "Grad accumulation", spin(1, 4096, 8), "Effective batch = batch size × accumulation.")
        self.field(f, "patience", "Early stop patience", spin(0, 100, 0), "Stop after N evaluations without improvement (0 = off).")
        self.field(f, "seed", "Seed", spin(0, 2_000_000_000, 1234))
        lay.addWidget(group("Optimization", f))

        f = new_form()
        out_row = QHBoxLayout()
        out = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse_output)
        out_row.addWidget(out, 1)
        out_row.addWidget(browse)
        self.fields["output"] = out
        connect_changed(out, lambda *_: self.on_change())
        f.addRow("Output folder", out_row)
        self.field(f, "sample", "", QCheckBox("Show a sample after every evaluation"))
        self.field(f, "sample_prompt", "Sample prompt", QLineEdit())
        self.field(f, "resume", "", QCheckBox("Resume from the latest checkpoint"))
        self.field(f, "anomaly", "", QCheckBox("Autograd anomaly detection (slow; for debugging)"))
        lay.addWidget(group("Output", f))
        lay.addStretch(1)
        return scroll_wrap(host)

    # -- right: monitor
    def _build_monitor(self) -> QWidget:
        host = QWidget()
        lay = QVBoxLayout(host)

        stats = QHBoxLayout()
        self.lbl_step, self.lbl_loss, self.lbl_lr, self.lbl_epoch = (QLabel("—") for _ in range(4))
        for title, label in (("Step", self.lbl_step), ("Loss", self.lbl_loss), ("LR", self.lbl_lr), ("Epoch", self.lbl_epoch)):
            box = QGroupBox(title)
            inner = QVBoxLayout(box)
            label.setStyleSheet("font-size: 18px; font-weight: 600;")
            inner.addWidget(label)
            stats.addWidget(box)
        lay.addLayout(stats)

        self.chart = LossChart()
        lay.addWidget(self.chart, 3)

        self.sub = QTabWidget()
        self.table = QTableWidget(0, 2)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.samples = QPlainTextEdit()
        self.samples.setReadOnly(True)
        cmd_host = QWidget()
        cmd_lay = QVBoxLayout(cmd_host)
        self.cmd = QPlainTextEdit()
        self.cmd.setReadOnly(True)
        copy = QPushButton("Copy command")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.cmd.toPlainText()))
        cmd_lay.addWidget(self.cmd)
        cmd_lay.addWidget(copy)
        self.sub.addTab(self.table, "Run summary")
        self.sub.addTab(self.samples, "Samples")
        self.sub.addTab(cmd_host, "CLI command")
        lay.addWidget(self.sub, 2)
        return host

    # -- actions
    def apply_preset(self) -> None:
        self.set_values({**TRAIN_DEFAULTS, **PRESETS.get(self.preset.currentText(), {})})
        self.main.status(f"Preset applied: {self.preset.currentText()}")

    def browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output folder", str(HERE))
        if path:
            self.fields["output"].setText(path)

    def command_args(self, dry: bool = False) -> list[str]:
        return train_args(self.values(), dry)

    def refresh_command(self) -> None:
        try:
            self.cmd.setPlainText(cli_preview(self.command_args()))
        except (ValueError, KeyError) as exc:
            self.cmd.setPlainText(f"(invalid settings: {exc})")

    def start(self, dry: bool) -> None:
        if not CORPUS.exists():
            QMessageBox.information(self, "No forum data yet",
                                    "Collect some threads first (Collect tab). Training needs the forum corpus.")
            return
        if not str(self.fields["model"].currentText()).strip():
            QMessageBox.warning(self, "Missing model", "Enter a Hugging Face repo id for the architecture.")
            return
        try:
            args = self.command_args(dry)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return
        self.main.start_job("dry" if dry else "train", args)

    def set_busy(self, busy: bool) -> None:
        self.dry.setEnabled(not busy)
        self.start_btn.setEnabled(not busy)

    def reset_run(self) -> None:
        self.chart.clear()
        self.table.setRowCount(0)
        self.samples.clear()
        for label in (self.lbl_step, self.lbl_loss, self.lbl_lr, self.lbl_epoch):
            label.setText("—")

    # -- events
    def on_metric(self, ev: dict) -> None:
        step, max_steps, loss = ev.get("step"), ev.get("max_steps"), ev.get("loss")
        if step is None or loss is None:
            return
        self.chart.add_train(step, loss)
        self.lbl_step.setText(f"{step:,} / {max_steps:,}" if max_steps else f"{step:,}")
        self.lbl_loss.setText(f"{loss:.3f}")
        if ev.get("lr") is not None:
            self.lbl_lr.setText(f"{ev['lr']:.2e}")
        if ev.get("epoch") is not None:
            self.lbl_epoch.setText(f"{ev['epoch']:.2f}")

    def on_eval(self, ev: dict) -> None:
        for name, loss in (ev.get("losses") or {}).items():
            self.chart.add_eval(ev.get("step", 0), name, loss)

    def on_table(self, title: str, rows: list) -> None:
        r = self.table.rowCount()
        self.table.setRowCount(r + 1 + len(rows))
        head = QTableWidgetItem(title)
        head.setForeground(QColor(ACCENT))
        font = head.font()
        font.setBold(True)
        head.setFont(font)
        self.table.setItem(r, 0, head)
        self.table.setSpan(r, 0, 1, 2)
        for i, (key, value) in enumerate(rows):
            self.table.setItem(r + 1 + i, 0, QTableWidgetItem(str(key)))
            self.table.setItem(r + 1 + i, 1, QTableWidgetItem(str(value)))
        self.table.scrollToBottom()

    def add_sample(self, title: str, text: str) -> None:
        self.samples.appendPlainText(f"── {title} ──\n{text}\n")


class GenerateTab(QWidget, FieldsMixin):
    def __init__(self, main: "MainWindow"):
        super().__init__()
        self._init_fields()
        self.main = main
        self.server = Runner(self)
        self.server.message.connect(self._on_message)
        self.server.rawline.connect(lambda t: main.log.add("raw", t))
        self.server.done.connect(self._on_done)
        self.ready = False
        self.loading = False
        self.waiting = False
        self.req_id = 0

        outer = QVBoxLayout(self)
        form = new_form()
        row = QHBoxLayout()
        model_dir = QLineEdit("checkpoints/acopsis-160m")
        self.fields["model_dir"] = model_dir
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse)
        row.addWidget(model_dir, 1)
        row.addWidget(browse)
        form.addRow("Model folder", row)
        self.field(form, "trust", "", QCheckBox("Trust remote code"))
        outer.addWidget(group("Model", form))

        ctl = QHBoxLayout()
        self.load_btn = primary("Load model")
        self.load_btn.clicked.connect(self.toggle_load)
        self.state = QLabel("Not loaded")
        self.state.setStyleSheet(f"color: {DIM};")
        ctl.addWidget(self.load_btn)
        ctl.addWidget(self.state, 1)
        outer.addLayout(ctl)

        params = new_form()
        self.field(params, "temperature", "Temperature", dspin(0.05, 2.0, 0.8, 2, 0.05), "Lower = safer, higher = wilder.")
        self.field(params, "top_p", "Top-p", dspin(0.1, 1.0, 0.9, 2, 0.05))
        self.field(params, "max_new_tokens", "Max new tokens", spin(1, 2048, 120, 10))
        outer.addWidget(group("Sampling", params))

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        outer.addWidget(self.chat, 1)

        send_row = QHBoxLayout()
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText("Type a prompt and press Enter — the model continues it")
        self.prompt.returnPressed.connect(self.send)
        self.send_btn = primary("Generate")
        self.send_btn.clicked.connect(self.send)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.chat.clear)
        send_row.addWidget(self.prompt, 1)
        send_row.addWidget(self.send_btn)
        send_row.addWidget(clear)
        outer.addLayout(send_row)
        self._refresh()

    def browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Model folder", str(HERE))
        if path:
            self.fields["model_dir"].setText(path)

    def _refresh(self) -> None:
        self.load_btn.setText("Unload model" if (self.ready or self.loading) else "Load model")
        can_send = self.ready and not self.waiting
        self.send_btn.setEnabled(can_send)
        self.prompt.setEnabled(self.ready)

    def toggle_load(self) -> None:
        if self.ready or self.loading:
            self.unload()
            return
        if self.main.runner.running():
            QMessageBox.information(self, "Busy", "A job is running; loading a model now would compete for GPU memory.")
            return
        model_dir = Path(str(self.fields["model_dir"].text()).strip())
        if not (model_dir if model_dir.is_absolute() else HERE / model_dir).exists():
            QMessageBox.information(self, "No model", f"Nothing found at {model_dir}. Train a model first.")
            return
        self.loading = True
        self.state.setText("Loading… (first load can take a minute)")
        self.server.start(server_args(self.values()))
        self._refresh()

    def unload(self) -> None:
        if self.server.running():
            self.server.quit_server()
        self.ready = self.loading = self.waiting = False
        self.state.setText("Not loaded")
        self._refresh()

    def shutdown(self) -> None:
        if self.server.running():
            self.server.kill()

    def send(self) -> None:
        text = self.prompt.text()
        if not (self.ready and not self.waiting) or not text.strip():
            return
        self.req_id += 1
        self.waiting = True
        self.chat.append(f'<p style="color:{ACCENT}"><b>▶ {html.escape(text)}</b></p>')
        v = self.values()
        self.server.send({"id": self.req_id, "prompt": text, "temperature": v["temperature"],
                          "top_p": v["top_p"], "max_new_tokens": v["max_new_tokens"]})
        self.state.setText("Generating…")
        self.prompt.clear()
        self._refresh()

    def _on_message(self, ev: dict) -> None:
        kind = ev.get("t")
        if kind == "ready":
            self.ready, self.loading = True, False
            self.state.setText(f"Ready · {ev.get('device', '')}")
            self.main.log.add("ok", "Generation server ready")
            self.prompt.setFocus()
        elif kind == "gen":
            body = html.escape(ev.get("text", "")).replace("\n", "<br>")
            self.chat.append(f'<p style="color:{TEXT}">{body}</p><hr>')
            self.waiting = False
            self.state.setText(f"Ready")
        elif kind == "gen_error":
            self.chat.append(f'<p style="color:{BAD}">✘ {html.escape(str(ev.get("msg", "")))}</p>')
            self.waiting = False
            self.state.setText("Ready")
        else:
            self.main.log_event(ev)
        self._refresh()

    def _on_done(self, code: int, _stopped: bool) -> None:
        if self.loading and code != 0:
            self.main.log.add("error", f"Generation server exited with code {code} while loading (see log above).")
        self.ready = self.loading = self.waiting = False
        self.state.setText("Not loaded")
        self._refresh()


class CorpusTab(QWidget):
    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        outer = QVBoxLayout(self)

        row = QHBoxLayout()
        self.stats_btn = primary("Refresh stats")
        self.stats_btn.clicked.connect(lambda: main.start_job("stats", ["stats"]))
        self.doctor_btn = QPushButton("Check environment")
        self.doctor_btn.clicked.connect(lambda: main.start_job("doctor", ["doctor"]))
        shuffle = QPushButton("Show random posts")
        shuffle.clicked.connect(self.show_posts)
        row.addWidget(self.stats_btn)
        row.addWidget(self.doctor_btn)
        row.addWidget(shuffle)
        row.addStretch(1)
        outer.addLayout(row)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.tables_host = QWidget()
        self.tables_lay = QVBoxLayout(self.tables_host)
        self.tables_lay.addStretch(1)
        split.addWidget(scroll_wrap(self.tables_host))
        self.posts = QTextEdit()
        self.posts.setReadOnly(True)
        split.addWidget(self.posts)
        split.setSizes([520, 700])
        outer.addWidget(split, 1)
        self.show_posts()

    def set_busy(self, busy: bool) -> None:
        self.stats_btn.setEnabled(not busy)
        self.doctor_btn.setEnabled(not busy)

    def clear_tables(self) -> None:
        while self.tables_lay.count() > 1:
            item = self.tables_lay.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

    def on_table(self, title: str, rows: list) -> None:
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        table = QTableWidget(len(rows), 2)
        table.horizontalHeader().setVisible(False)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(26)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for i, (key, value) in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(str(key)))
            table.setItem(i, 1, QTableWidgetItem(str(value)))
        table.setFixedHeight(26 * len(rows) + 8)
        lay.addWidget(table)
        self.tables_lay.insertWidget(self.tables_lay.count() - 1, box)

    def show_posts(self) -> None:
        posts = sample_posts(CORPUS, 8)
        if not posts:
            self.posts.setHtml(f'<p style="color:{DIM}">No forum posts yet. Use the Collect tab first.</p>')
            return
        parts = []
        for rec in posts:
            tag = f"/{rec.get('board', '?')}/ #{rec.get('post_id', '')}"
            body = html.escape(rec["text"][:900]).replace("\n", "<br>")
            parts.append(f'<p><span style="color:{ACCENT}">{html.escape(tag)}</span><br>{body}</p><hr>')
        self.posts.setHtml("".join(parts))


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ACOpsis DELTA")
        self.resize(1360, 900)
        self.settings = QSettings()
        self.runner = Runner(self)
        self.label: str | None = None
        self.t0 = 0.0

        self.log = LogView()
        self.progress = ProgressPanel()
        self.progress.stop_clicked.connect(self.stop_job)

        self.collect_tab = CollectTab(self)
        self.train_tab = TrainTab(self)
        self.gen_tab = GenerateTab(self)
        self.corpus_tab = CorpusTab(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.collect_tab, "1 · Collect")
        self.tabs.addTab(self.train_tab, "2 · Train")
        self.tabs.addTab(self.gen_tab, "3 · Generate")
        self.tabs.addTab(self.corpus_tab, "Corpus & environment")

        bottom = QWidget()
        blay = QVBoxLayout(bottom)
        blay.setContentsMargins(0, 6, 0, 0)
        blay.addWidget(self.progress)
        tools = QHBoxLayout()
        title = QLabel("Log")
        title.setStyleSheet(f"color: {DIM}; font-weight: 600;")
        clear = QPushButton("Clear")
        clear.clicked.connect(self.log.clear)
        save = QPushButton("Save…")
        save.clicked.connect(self.save_log)
        tools.addWidget(title)
        tools.addStretch(1)
        tools.addWidget(clear)
        tools.addWidget(save)
        blay.addLayout(tools)
        blay.addWidget(self.log, 1)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.tabs)
        split.addWidget(bottom)
        split.setSizes([640, 240])
        self.setCentralWidget(split)

        self.runner.message.connect(self.on_event)
        self.runner.rawline.connect(self.on_raw)
        self.runner.done.connect(self.on_done)

        self.clock = QTimer(self)
        self.clock.setInterval(1000)
        self.clock.timeout.connect(self._tick)

        self._build_menu()
        self.statusBar().showMessage("Ready")
        self.load_settings()
        if not SCRIPT.exists():
            QMessageBox.critical(self, "train.py not found", f"Expected train.py next to gui.py:\n{SCRIPT}")

    # -- menu
    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for text, slot, shortcut in (
            ("Save settings…", self.save_settings_file, None),
            ("Load settings…", self.load_settings_file, None),
            ("Open data folder", self.open_data_folder, None),
            ("Quit", self.close, "Ctrl+Q"),
        ):
            action = QAction(text, self)
            if shortcut:
                action.setShortcut(shortcut)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        tools_menu = self.menuBar().addMenu("&Tools")
        for text, slot in (
            ("Check environment", lambda: self.start_job("doctor", ["doctor"])),
            ("Refresh corpus stats", lambda: self.start_job("stats", ["stats"])),
        ):
            action = QAction(text, self)
            action.triggered.connect(slot)
            tools_menu.addAction(action)
        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("About", self)
        about.triggered.connect(lambda: QMessageBox.about(
            self, "ACOpsis DELTA",
            "A control panel for train.py.\n\nEvery job runs in a separate process; Stop asks the job to save "
            "and exit, pressing it again force-kills it."))
        help_menu.addAction(about)

    # -- job control
    def status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def set_busy(self, busy: bool) -> None:
        for tab in (self.collect_tab, self.train_tab, self.corpus_tab):
            tab.set_busy(busy)
        self.progress.stop.setEnabled(busy)

    def start_job(self, label: str, args: list[str]) -> None:
        if self.runner.running():
            QMessageBox.information(self, "Busy", "Another job is running. Stop it first.")
            return
        if label in ("train", "dry"):
            self.gen_tab.unload()  # free GPU memory for training
        if label in ("train", "dry"):
            self.train_tab.reset_run()
            self.tabs.setCurrentWidget(self.train_tab)
        elif label in ("stats", "doctor"):
            self.corpus_tab.clear_tables()
            self.tabs.setCurrentWidget(self.corpus_tab)
        self.label, self.t0 = label, time.monotonic()
        self.progress.reset()
        self.log.add("cmd", "$ " + cli_preview(args))
        if not self.runner.start(args):
            return
        self.set_busy(True)
        self.clock.start()
        self.status(f"Running: {label}")

    def stop_job(self) -> None:
        if not self.runner.running():
            return
        self.runner.request_stop()
        if self.progress.stop.text() == "Stop":
            self.progress.stop.setText("Force kill")
            self.log.add("warn", "Stop requested. The job will save and exit; press again to force-kill.")
        else:
            self.log.add("warn", "Force-killed.")

    def _tick(self) -> None:
        self.status(f"Running: {self.label} · {fmt_duration(time.monotonic() - self.t0)}")

    def on_done(self, code: int, stopped: bool) -> None:
        self.clock.stop()
        elapsed = fmt_duration(time.monotonic() - self.t0)
        if stopped:
            self.log.add("warn", f"{self.label}: stopped after {elapsed}.")
            self.status(f"Stopped ({self.label})")
        elif code == 0:
            self.log.add("ok", f"{self.label}: finished in {elapsed}.")
            self.status(f"Finished: {self.label} in {elapsed}")
        else:
            self.log.add("error", f"{self.label}: exited with code {code} after {elapsed}. See the log above.")
            self.status(f"Failed: {self.label} (code {code})")
        self.progress.idle("Idle")
        self.set_busy(False)

    # -- events
    def log_event(self, ev: dict) -> None:
        kind = ev.get("t")
        if kind == "log":
            self.log.add(ev.get("level", "info"), str(ev.get("msg", "")))
        elif kind == "banner":
            sub = f"  {ev['subtitle']}" if ev.get("subtitle") else ""
            self.log.add("banner", f"═══ {ev.get('title', '')} ═══{sub}")
        elif kind == "section":
            self.log.add("section", f"── {ev.get('title', '')} ──")
        elif kind == "panel":
            self.log.add("panel", f"[{ev.get('title', '')}]\n{ev.get('text', '')}")
        elif kind == "table":
            rows = ev.get("rows") or []
            width = max((len(str(k)) for k, _ in rows), default=0)
            body = "\n".join(f"  {str(k):<{width}}  {v}" for k, v in rows)
            self.log.add("info", f"{ev.get('title', '')}\n{body}")

    def on_event(self, ev: dict) -> None:
        kind = ev.get("t")
        if kind == "progress":
            self.progress.on_progress(ev)
        elif kind == "metric":
            self.train_tab.on_metric(ev)
        elif kind == "eval":
            self.train_tab.on_eval(ev)
        elif kind == "table":
            self.log_event(ev)
            if self.label in ("train", "dry"):
                self.train_tab.on_table(ev.get("title", ""), ev.get("rows") or [])
            else:
                self.corpus_tab.on_table(ev.get("title", ""), ev.get("rows") or [])
        elif kind == "panel" and str(ev.get("title", "")).startswith("sample"):
            self.train_tab.add_sample(ev.get("title", ""), ev.get("text", ""))
            self.log_event(ev)
        else:
            self.log_event(ev)

    def on_raw(self, line: str) -> None:
        stripped = line.strip()
        bad = stripped.startswith("Traceback") or stripped.split(":")[0].endswith(("Error", "Exception"))
        self.log.add("error" if bad else "raw", line)

    # -- misc
    def open_data_folder(self) -> None:
        DATA_DIR.mkdir(exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(DATA_DIR)))

    def save_log(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save log", str(HERE / "acopsis-log.txt"), "Text (*.txt)")
        if path:
            Path(path).write_text(self.log.toPlainText(), encoding="utf-8")

    def _all_settings(self) -> dict:
        return {"collect": self.collect_tab.values(), "train": self.train_tab.values(), "generate": self.gen_tab.values()}

    def _apply_settings(self, data: dict) -> None:
        self.collect_tab.set_values(data.get("collect", {}))
        self.train_tab.set_values(data.get("train", {}))
        self.gen_tab.set_values(data.get("generate", {}))

    def save_settings_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", str(HERE / "acopsis-settings.json"), "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(self._all_settings(), indent=2), encoding="utf-8")
            self.status(f"Saved settings to {path}")

    def load_settings_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", str(HERE), "JSON (*.json)")
        if path:
            try:
                self._apply_settings(json.loads(Path(path).read_text(encoding="utf-8")))
                self.status(f"Loaded settings from {path}")
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "Could not load settings", str(exc))

    def load_settings(self) -> None:
        raw = self.settings.value("state", "")
        if raw:
            try:
                self._apply_settings(json.loads(str(raw)))
            except ValueError:
                pass
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def closeEvent(self, event) -> None:
        if self.runner.running():
            answer = QMessageBox.question(
                self, "Job running", "A job is still running. Quit and kill it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.runner.kill()
        self.gen_tab.shutdown()
        try:
            self.settings.setValue("state", json.dumps(self._all_settings()))
            self.settings.setValue("geometry", self.saveGeometry())
        except (TypeError, ValueError):
            pass
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("ACOpsis DELTA")
    app.setOrganizationName("ACOpsis")
    apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())