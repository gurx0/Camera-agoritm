import cv2
import numpy as np

# Захват видео с камеры
cap = cv2.VideoCapture(0)

# Пороговое значение для определения движения
threshold = 0.2  # Уменьшили порог для большей чувствительности
fixed_length = 100  # Длина вектора, когда движение фиксируется

# Параметры для сглаживания
alpha = 0.3  # Уменьшили сглаживание для большей чувствительности
smooth_dx, smooth_dy = 0, 0  # Начальные значения сглаженных компонент вектора
smooth_angle = 0  # Начальное значение сглаженного угла

# Захват первого кадра
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Создаем массив для HSV-отображения потока
hsv = np.zeros_like(prev_frame)
hsv[..., 1] = 255  # Устанавливаем максимальную насыщенность

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось захватить кадр")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Рассчет оптического потока с помощью метода Фарнебека
    flow = cv2.calcOpticalFlowFarneback(prev_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)

    # Разделение потока на составляющие по осям
    flow_x, flow_y = flow[..., 0], flow[..., 1]

    # Рассчет угла и величины смещения
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y)

    # Угол используется для определения цветового тона (hue)
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Величина смещения нормализована для использования в HSV (0–255)
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Преобразуем HSV в BGR для отображения
    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Рассчет среднего смещения
    avg_dx = np.mean(flow_x)
    avg_dy = np.mean(flow_y)

    # Сглаживание значений с использованием экспоненциального скользящего среднего (EMA)
    smooth_dx = alpha * avg_dx + (1 - alpha) * smooth_dx
    smooth_dy = alpha * avg_dy + (1 - alpha) * smooth_dy

    # Рассчитываем величину смещения
    motion_magnitude = np.sqrt(smooth_dx ** 2 + smooth_dy ** 2)

    # Определяем центр изображения для отображения вектора
    h, w, _ = frame.shape
    center = (w // 2, h // 2)

    if motion_magnitude > threshold:
        # Если есть движение, нормализуем вектор до постоянной длины
        norm_dx = (smooth_dx / motion_magnitude) * fixed_length
        norm_dy = (smooth_dy / motion_magnitude) * fixed_length

        # Конечная точка для вектора движения
        end_point = (int(center[0] + norm_dx), int(center[1] + norm_dy))

        # Рисуем вектор движения (стрелку)
        frame = cv2.arrowedLine(frame, center, end_point, (0, 0, 255), 3, tipLength=0.5)

        # Корректировка угла через арктангенс с учётом параллельности осям
        if abs(norm_dx) < 1e-5:  # Если смещение по оси X очень маленькое (≈ 0), угол равен 0
            angle_deg = 0
        elif abs(norm_dy) < 1e-5:  # Если смещение по оси Y очень маленькое (≈ 0), угол равен 0
            angle_deg = 0
        else:
            # Рассчитываем угол отклонения вектора через арктангенс относительно вертикальной оси (оси Y)
            angle_radians = np.arctan2(norm_dx, norm_dy)  # Угол в радианах
            angle_deg = np.degrees(angle_radians)  # Угол в градусах

        # Нивелирование угла на ±10 градусов
        if -10 <= angle_deg <= 10:
            angle_deg = 0

        # Сглаживание угла
        smooth_angle = alpha * angle_deg + (1 - alpha) * smooth_angle

        # Выводим угол на изображение
        angle_text = f"Angle: {smooth_angle:.2f} degrees"
        cv2.putText(frame, angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Если нет значимого движения, просто обновляем текст
        cv2.putText(frame, "No significant motion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Объединение исходного кадра и визуализации потока
    combined = np.hstack((frame, flow_bgr))
    cv2.imshow('Optical Flow and Motion Vector', combined)

    # Нажмите "q" для выхода
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Обновляем предыдущий кадр
    prev_gray = frame_gray.copy()

# Освобождение камеры и закрытие всех окон
cap.release()
cv2.destroyAllWindows()
