import cv2
import numpy as np

# Функция для определения угла отклонения камеры и получения линий
def calculate_angle_and_edges(frame):
    # Преобразуем изображение в серый цвет
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Применяем детектор границ (например, Canny)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Находим линии в изображении
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta * 180) / np.pi - 90  # Преобразуем радианы в градусы
            angles.append(angle)

        # Рассчитываем средний угол
        mean_angle = np.mean(angles)
        return mean_angle, edges, lines
    else:
        return 0, edges, None

# Инициализация видео-потока
cap = cv2.VideoCapture(0)  # Используем камеру по умолчанию

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Получаем размеры кадра
    height, width = frame.shape[:2]

    # Рассчитываем центр кадра
    center_x, center_y = width // 2, height // 2

    # Рассчитываем угол и получаем изображение границ
    angle, edges, lines = calculate_angle_and_edges(frame)
    print(f"Угол отклонения: {angle:.2f} градусов")

    # Отображаем угол на кадре
    cv2.putText(frame, f"Угол отклонения: {angle:.2f} градусов",
                (10, 30),  # Позиция текста
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Конвертируем границы в трехканальное изображение для совмещения с исходным
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Отображаем линии на кадре с учетом сдвига координат в центр
    if lines is not None:
        for rho, theta in lines[:, 0]:
            # Преобразуем полярные координаты в декартовы
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # Найдем две точки на линии, чтобы нарисовать её
            x1 = int(x0 + 1000 * (-b)) + center_x  # смещаем координаты относительно центра
            y1 = int(y0 + 1000 * (a)) + center_y
            x2 = int(x0 - 1000 * (-b)) + center_x
            y2 = int(y0 - 1000 * (a)) + center_y

            # Рисуем линии на исходном кадре
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Совмещаем исходное видео и визуализацию границ (по горизонтали)
    combined_frame = np.hstack((frame, edges_colored))

    # Отображаем кадр с границами и углом
    cv2.imshow('Video Stream with Edges', combined_frame)

    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
