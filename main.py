import cv2
import numpy as np


# Функция для определения угла отклонения камеры и получения линий
def calculate_angle_and_edges(frame):
    # Преобразуем изображение в серый цвет
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Применяем детектор границ (Canny) с умеренной чувствительностью
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Находим линии в изображении с помощью HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=20)

    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)

        # Рассчитываем медианный угол
        median_angle = np.median(angles)
        return median_angle, edges, lines
    else:
        return 0, edges, None


# Инициализация видео-потока
cap = cv2.VideoCapture(1)  # Используем камеру по умолчанию

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Рассчитываем угол и получаем изображение границ
    angle, edges, lines = calculate_angle_and_edges(frame)
    print(f"Угол отклонения: {angle:.2f} градусов")

    # Отображаем угол на кадре
    cv2.putText(frame, f"Угол отклонения: {angle:.2f} градусов",
                (10, 30),  # Позиция текста
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Конвертируем границы в трехканальное изображение для совмещения с исходным
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Отображаем линии на исходном кадре
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Рисуем линии на исходном кадре
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Красные линии на исходном видео

            # Рисуем линии на изображении с границами
            cv2.line(edges_colored, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Красные линии на границах

    # Совмещаем исходное видео и визуализацию границ (по горизонтали)
    combined_frame = np.hstack((frame, edges_colored))

    # Отображаем кадр с границами и углом
    cv2.imshow('Video Stream with Edges', combined_frame)

    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
