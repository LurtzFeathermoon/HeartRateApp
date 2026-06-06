import sys
import cv2
import csv,os,ctypes
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout,QLineEdit,
    QProgressBar, QMessageBox, QRubberBand,QCheckBox,
)
from PySide6.QtCore import (
    Qt, QRect, QPoint, QSize,
    QObject, QThread, Signal
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QIcon, QFont
)
from processor import process_video

# ========================= ROI SELECTOR

class ROISelector(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)

        self.setScaledContents(False)
        self._current_frame = None

        self.display_rect = QRect()
        self.origin = QPoint()
        self.rubber = QRubberBand(QRubberBand.Rectangle, self)
        self.roi_rect = None
        self.image_size = None

    def set_image(self, pixmap, image_size):
        self.setPixmap(pixmap)
        self.update_display_rect()
        self.image_size = image_size
        self.roi_rect = None
        self.rubber.hide()

    def mousePressEvent(self, event):
        if self.pixmap() is None:
            return
        if not self.display_rect.contains(event.pos()):
            return
        self.origin = event.pos()
        self.rubber.setGeometry(QRect(self.origin, QSize()))
        self.rubber.show()

    def mouseMoveEvent(self, event):
        pos = event.pos()
        pos.setX(max(self.display_rect.left(), min(pos.x(), self.display_rect.right())))
        pos.setY(max(self.display_rect.top(), min(pos.y(), self.display_rect.bottom())))

        self.rubber.setGeometry(QRect(self.origin, pos).normalized())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display_rect()

    def update_display_rect(self):
        if not self.pixmap():
            return

        pm = self.pixmap()
        pm_size = pm.size()
        label_size = self.size()

        scaled = pm_size.scaled(label_size, Qt.KeepAspectRatio)

        x = (label_size.width() - scaled.width()) // 2
        y = (label_size.height() - scaled.height()) // 2

        self.display_rect = QRect(
            x, y,
            scaled.width(),
            scaled.height()
        )

    def mouseReleaseEvent(self, event):
        self.roi_rect = self.rubber.geometry()
        self.rubber.hide()

    def get_roi_cv(self):
        if not self.roi_rect or self.display_rect.isNull():
            return None

        img_w, img_h = self.image_size
        disp = self.display_rect

        scale_x = img_w / disp.width()
        scale_y = img_h / disp.height()

        x = int((self.roi_rect.x() - disp.x()) * scale_x)
        y = int((self.roi_rect.y() - disp.y()) * scale_y)
        w = int(self.roi_rect.width() * scale_x)
        h = int(self.roi_rect.height() * scale_y)

        return x, y, w, h

    def show_frame(self, frame_bgr):
        self._current_frame = frame_bgr

        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg)

        scaled = pix.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.setPixmap(scaled)

        # calcular display_rect REAL
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self.display_rect = QRect(
            x, y,
            scaled.width(),
            scaled.height()
        )

        self.update()

# ========================= SIGNAL PLOT

class SignalPlot(QWidget):
    def __init__(self):
        super().__init__()

        # ===== CONFIGURACIÓN CRÍTICA =====
        self.setMinimumHeight(220)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(True)

        self.signal = None
        self.peaks = []

        # Cuadrícula
        self.grid_x = 10   # divisiones en tiempo
        self.grid_y = 5    # divisiones en amplitud

    def set_data(self, signal, peaks):
        self.signal = signal
        self.peaks = peaks
        self.update()      # FORZAR REPINTADO

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self._draw_plot(painter, self.rect())

        rect = self.rect()
        w = rect.width()
        h = rect.height()

        # ===== FONDO =====
        painter.fillRect(rect, Qt.white)

        if self.signal is None or len(self.signal) < 2:
            painter.end()
            return

        # ===== CUADRÍCULA =====
        grid_pen = QPen(Qt.lightGray, 1, Qt.DashLine)
        painter.setPen(grid_pen)

        for i in range(1, self.grid_x):
            x = int(i * w / self.grid_x)
            painter.drawLine(x, 0, x, h)

        for j in range(1, self.grid_y):
            y = int(j * h / self.grid_y)
            painter.drawLine(0, y, w, y)

        # ===== EJES =====
        axis_pen = QPen(Qt.black, 2)
        painter.setPen(axis_pen)
        painter.drawLine(0, h - 1, w, h - 1)  # eje X
        painter.drawLine(0, 0, 0, h)          # eje Y

        # ===== ETIQUETAS =====
        painter.setFont(QFont("Arial", 9))
        painter.drawText(5, 15, "Intensidad")
        painter.drawText(w - 40, h - 5, "Tiempo")

        # ===== SEÑAL =====
        n = len(self.signal)
        signal_pen = QPen(Qt.black, 2)
        painter.setPen(signal_pen)

        for i in range(n - 1):
            x1 = int(i * w / (n - 1))
            y1 = int((1 - self.signal[i]) * h)
            x2 = int((i + 1) * w / (n - 1))
            y2 = int((1 - self.signal[i + 1]) * h)
            painter.drawLine(x1, y1, x2, y2)

        # ===== PICOS =====
        peak_pen = QPen(Qt.red, 3)
        painter.setPen(peak_pen)

        for p in self.peaks:
            x = int(p * w / (n - 1))
            y = int((1 - self.signal[p]) * h)
            painter.drawPoint(x, y)

        painter.end()

    def save_png(self, filename, width=1920, height=1080):
        image = QImage(width, height, QImage.Format_ARGB32)
        image.fill(Qt.white)

        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        self._draw_plot(painter,QRect(0,0,width,height))

        painter.end()
        image.save(filename)

    def _draw_plot(self, painter, rect):
        if self.signal is None or len(self.signal) == 0:
            return

        w = rect.width()
        h = rect.height()
        margin = 60

        plot_w = w - 2 * margin
        plot_h = h - 2 * margin

        # Fondo
        painter.fillRect(rect, Qt.white)

        # ===== CUADRÍCULA =====
        grid_pen = QPen(Qt.lightGray, 1, Qt.DashLine)
        painter.setPen(grid_pen)

        for i in range(1, self.grid_x):
            x = int(i * w / self.grid_x)
            painter.drawLine(x, 0, x, h)

        for j in range(1, self.grid_y):
            y = int(j * h / self.grid_y)
            painter.drawLine(0, y, w, y)

        # ===== EJES =====
        axis_pen = QPen(Qt.black, 2)
        painter.setPen(axis_pen)
        painter.drawLine(0, h - 1, w, h - 1)  # eje X
        painter.drawLine(0, 0, 0, h)          # eje Y

        # ===== ETIQUETAS =====
        painter.setFont(QFont("Arial", 9))
        painter.drawText(5, 15, "Intensidad")
        painter.drawText(w - 40, h - 5, "Tiempo")

        # ===== SEÑAL =====
        n = len(self.signal)
        signal_pen = QPen(Qt.black, 2)
        painter.setPen(signal_pen)

        for i in range(n - 1):
            x1 = int(i * w / (n - 1))
            y1 = int((1 - self.signal[i]) * h)
            x2 = int((i + 1) * w / (n - 1))
            y2 = int((1 - self.signal[i + 1]) * h)
            painter.drawLine(x1, y1, x2, y2)

        # ===== PICOS =====
        peak_pen = QPen(Qt.red, 6)
        painter.setPen(peak_pen)

        for p in self.peaks:
            x = int(p * w / (n - 1))
            y = int((1 - self.signal[p]) * h)
            painter.drawPoint(x, y)


# ========================= WORKER THREAD

class VideoWorker(QObject):
    finished    = Signal(object)
    progress    = Signal(int)
    error       = Signal(str)
    frame_ready = Signal(object, object)


    def __init__(self, video_path, roi, accelerated):
        super().__init__()
        self.video_path = video_path
        self.roi = roi
        self.accelerated = accelerated
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            result = process_video(
                self.video_path,
                self.roi,
                self.accelerated,
                progress_cb=self.progress.emit,
                frame_cb=self.frame_ready.emit,
                cancel_flag=lambda: self._cancel
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

# ========================= MAIN APPLICATION

class HeartRateApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Análisis Cardíaco")
        self.resize(400, 300)

        self.video_path = None
        self.roi = None

        self.video_label  = ROISelector()
        self.plot         = SignalPlot()
        self.result_label = QLabel("BPM: --")
        self.progress     = QProgressBar()

        self.btn_select   = QPushButton("Seleccionar video")
        self.btn_run      = QPushButton("Analizar")
        self.btn_cancel   = QPushButton("Cancelar")
        self.btn_cancel.setEnabled(False)
        self.chk_realtime = QCheckBox("Análisis en tiempo real")
        self.chk_realtime.setChecked(False)
        self.output_name  = QLineEdit()
        self.output_name.setPlaceholderText("Nombre de la muestra")
        self.btn_export   = QPushButton("Exportar resultados")
        self.btn_export.setEnabled(False)


        layout = QVBoxLayout(self)
        layout.addWidget(self.video_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.chk_realtime)
        layout.addWidget(self.plot)
        layout.addWidget(self.progress)
        layout.addWidget(self.output_name)

        btns = QHBoxLayout()
        btns.addWidget(self.btn_select)
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_export)
        layout.addLayout(btns)

        self.btn_select.clicked.connect(self.select_video)
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_export.clicked.connect(self.export_results)

    def select_video(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar video", "",
            "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if not file:
            return

        cap = cv2.VideoCapture(file)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            QMessageBox.critical(self, "Error", "No se pudo leer el video")
            return

        self.video_label.show_frame(frame)
        self.video_label.image_size = (frame.shape[1], frame.shape[0])
        self.video_path = file

    def show_realtime_frame(self, frame, roi):
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        self.video_label.show_frame(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_, w_, _ = rgb.shape
        qimg = QImage(rgb.data, w_, h_, 3*w_, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def run_analysis(self):
        self.roi = self.video_label.get_roi_cv()
        if not self.video_path or not self.roi:
            QMessageBox.warning(self, "Error", "Seleccione video y ROI")
            return

        accelerated = not self.chk_realtime.isChecked()

        self.thread = QThread()
        self.worker = VideoWorker(
            self.video_path,
            self.roi,
            accelerated
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.show_realtime_frame) 
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)

        self.btn_cancel.setEnabled(True)
        self.thread.start()

    def export_results(self):
        if not hasattr(self, "last_result"):
            return

        name = self.output_name.text().strip()
        if not name:
            QMessageBox.warning(
                self,
                "Nombre requerido",
                "Debe ingresar un nombre para los archivos de salida"
            )
            return

        folder = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar carpeta de destino"
        )
        if not folder:
            return

        csv_path = os.path.join(folder, f"{name}.csv")
        png_path = os.path.join(folder, f"{name}.png")

        bpm, signal, peaks, fps = self.last_result

        self.save_csv(csv_path, bpm, fps)
        self.plot.save_png(png_path)

        QMessageBox.information(
            self,
            "Exportación completa",
            f"Archivos guardados:\n{name}.csv\n{name}.png"
        )

    def save_csv(self, path, bpm, fps):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "video",
                "bpm",
                "fps",
                "roi_x",
                "roi_y",
                "roi_w",
                "roi_h"
            ])
            writer.writerow([
                datetime.now().isoformat(),
                self.video_path,
                f"{bpm:.2f}",
                fps,
                *self.roi
            ])

    def cancel(self):
        if self.worker:
            self.worker.cancel()

    def on_finished(self, result):
        self.thread.quit()
        self.thread.wait()
        self.btn_cancel.setEnabled(False)

        if result is None:
            return

        bpm, signal, peaks, fps = result
        self.result_label.setText(f"BPM estimado: {bpm:.1f}")
        self.plot.set_data(signal, peaks)
        self.last_result = (bpm, signal, peaks, fps)
        self.btn_export.setEnabled(True)

    def on_error(self, msg):
        self.thread.quit()
        self.thread.wait()
        self.btn_cancel.setEnabled(False)
        QMessageBox.critical(self, "Error", msg)


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "HeartRateApp"
        )
    app = QApplication(sys.argv)
    icon_path = Path(__file__).parent / "assets" / "icon.ico"
    app.setWindowIcon(QIcon(str(icon_path)))
    win = HeartRateApp()
    win.show()
    sys.exit(app.exec())
