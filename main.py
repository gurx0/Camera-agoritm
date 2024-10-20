import cv2
import numpy as np

# Функция для определения угла отклонения камеры
def calculate_angle(frame):
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
        return mean_angle
    else:
        return 0

# Инициализация видео-потока
cap = cv2.VideoCapture(1)  # Используем камеру по умолчанию

while True:
    ret, frame = cap.read()
    if not ret:
        break

    angle = calculate_angle(frame)
    print(f"Угол отклонения: {angle:.2f} градусов")

    # Отображаем угол на кадре
    cv2.putText(frame, f"Угол отклонения: {angle:.2f} градусов",
                (10, 30),  # Позиция текста
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Отображаем кадр
    cv2.imshow('Video Stream', frame)

    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
