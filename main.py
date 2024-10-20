import sys
import numpy as np
import cv2
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal
import pyqtgraph as pg
import time


# Класс для вычисления углов и передачи данных в интерфейс
class CameraAngleWorker(QThread):
    angle_changed = pyqtSignal(float)
    frame_changed = pyqtSignal(np.ndarray)
    edges_changed = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.prev_angle = 0  # Предыдущее значение угла для фильтрации
        self.alpha = 0.5  # Коэффициент сглаживания

    def run(self):
        cap = cv2.VideoCapture(0)  # Подключение к камере
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Вычисление угла и получение изображения границ
            angle, edges, lines = self.calculate_angle_and_edges(frame)
            # Применение экспоненциального сглаживания
            smoothed_angle = self.alpha * angle + (1 - self.alpha) * self.prev_angle
            self.prev_angle = smoothed_angle

            self.angle_changed.emit(smoothed_angle)  # Передача сглаженного значения угла в интерфейс

            # Отображение линий на исходном кадре
            self.draw_lines(frame, lines)  # Добавление линий на исходное изображение

            self.frame_changed.emit(frame)  # Передача исходного кадра с линиями
            self.edges_changed.emit(edges)  # Передача изображения с границами
            time.sleep(0.1)  # Задержка между обновлениями

        cap.release()

    def calculate_angle_and_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (13, 13), 0)  # Применение размытия для уменьшения шума
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100,
                                maxLineGap=30)  # Увеличены значения

        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)

            median_angle = np.median(angles)
            return median_angle, edges, lines
        else:
            return 0, edges, None

    def draw_lines(self, frame, lines):
        # Если линии найдены, рисуем их на исходном кадре
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Красные линии на исходном видео

    def stop(self):
        self.running = False
        self.wait()


class RealTimePlot(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Plot of Angles")
        self.setGeometry(100, 100, 1200, 600)

        # Установка центрального виджета
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Горизонтальный компоновщик
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Инициализация графика
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Данные для графика
        self.x_data = []
        self.y_data = []
        self.curve = self.plot_widget.plot(pen='b')  # Линия графика
        self.plot_widget.addLine(y=0, pen='r')  # Красная линия на уровне нуля

        # Горизонтальный компоновщик для видео
        video_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(video_layout)  # Добавляем горизонтальный компоновщик в основной вертикальный

        # QLabel для отображения видео
        self.video_label = QtWidgets.QLabel()
        video_layout.addWidget(self.video_label)  # Добавляем обычное видео

        # QLabel для отображения границ
        self.edges_label = QtWidgets.QLabel()
        video_layout.addWidget(self.edges_label)  # Добавляем контурное видео справа

        # Запуск потока для получения углов
        self.worker = CameraAngleWorker()
        self.worker.angle_changed.connect(self.add_angle)
        self.worker.frame_changed.connect(self.update_frame)
        self.worker.edges_changed.connect(self.update_edges)
        self.worker.start()

        # Начальное значение времени
        self.time_step = 0

        # Установка размеров графика
        self.plot_widget.setYRange(-180, 180)  # Устанавливаем пределы оси Y
        self.plot_widget.setXRange(0, 100)  # Устанавливаем пределы оси X

    def add_angle(self, angle):
        self.x_data.append(self.time_step)  # Добавляем текущее время
        self.y_data.append(angle)  # Добавляем угол в график
        self.time_step += 1  # Увеличиваем шаг времени на 1

        # Ограничиваем количество точек на графике (например, 100)
        if len(self.x_data) > 100:
            self.x_data.pop(0)
            self.y_data.pop(0)

        # Обновляем оси Y
        self.plot_widget.setYRange(-180, 180)  # Устанавливаем пределы оси Y

        # Динамическое обновление оси X
        if len(self.x_data) > 0:
            self.plot_widget.setXRange(self.x_data[0],
                                       self.x_data[-1])  # Устанавливаем пределы оси X от первого до последнего значения
        self.curve.setData(self.x_data, self.y_data)  # Обновляем данные графика

    def update_frame(self, frame):
        # Конвертируем BGR в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Преобразуем в QImage
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Отображаем изображение на QLabel (видео)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def update_edges(self, edges):
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # Преобразуем в QImage
        h, w, ch = edges_colored.shape
        bytes_per_line = ch * w
        q_img = QImage(edges_colored.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Отображаем изображение границ на QLabel (границы)
        self.edges_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


# Основная функция запуска приложения
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    plot = RealTimePlot()
    plot.show()
    sys.exit(app.exec_())
