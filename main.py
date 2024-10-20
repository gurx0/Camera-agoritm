import sys
import numpy as np
import cv2
from PyQt5 import QtWidgets
import pyqtgraph as pg
from PyQt5.QtCore import QThread, pyqtSignal
import time

# Класс для вычисления углов и передачи данных в интерфейс
class CameraAngleWorker(QThread):
    angle_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(1)  # Подключение к камере
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Вычисление угла
            angle, edges, lines = self.calculate_angle_and_edges(frame)
            self.angle_changed.emit(angle)  # Передача значения угла в интерфейс
            time.sleep(0.1)  # Задержка между обновлениями

        cap.release()

    def calculate_angle_and_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=20)

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

    def stop(self):
        self.running = False
        self.wait()

class RealTimePlot(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Plot of Angles")
        self.setGeometry(100, 100, 800, 400)

        # Инициализация графика
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        # Данные для графика
        self.x_data = []
        self.y_data = []
        self.curve = self.plot_widget.plot(pen='b')  # Линия графика
        self.plot_widget.addLine(y=0, pen='r')  # Красная пунктирная линия на уровне нуля


        # Запуск обновления графика
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)  # Обновляем график каждые 100 мс

        # Начальное значение времени
        self.time_step = 0

        # Запуск потока для получения углов
        self.worker = CameraAngleWorker()
        self.worker.angle_changed.connect(self.add_angle)
        self.worker.start()

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
        self.curve.setData(self.x_data, self.y_data)  # Обновляем данные графика

    def update_plot(self):
        # Этот метод можно оставить пустым, если обновление происходит только в add_angle.
        pass

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

# Основная функция запуска приложения
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    plot = RealTimePlot()
    plot.show()
    sys.exit(app.exec_())
