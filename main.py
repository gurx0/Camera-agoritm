import cv2
import numpy as np

# Захват видео с камеры (если камера по умолчанию, то 0)
cap = cv2.VideoCapture(0)

# Определение параметров для функции ShiTomasi
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Параметры для оптического потока Лукаса-Канаде
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Генерация случайных цветов для линий
color = np.random.randint(0, 255, (100, 3))

# Захват первого кадра и нахождение начальных углов
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Создание маски для рисования линий
mask = np.zeros_like(prev_frame)

# Пороговое значение для определения движения
threshold = 2.0
fixed_length = 50  # Длина вектора, когда движение фиксируется

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось захватить кадр")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Рассчет оптического потока
    next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_points, None, **lk_params)

    # Отбор только тех точек, которые были найдены
    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]

    # Инициализируем переменные для накопления смещений
    total_dx, total_dy = 0, 0
    count = len(good_new)

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()  # Преобразуем в одномерный массив
        c, d = old.ravel()  # Преобразуем в одномерный массив

        # Проверяем, что значения существуют и являются целыми числами
        if a is not None and b is not None and c is not None and d is not None:
            a, b = int(a), int(b)
            c, d = int(c), int(d)
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)

            # Рисуем кружки в новой позиции
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

            # Накопление смещения
            total_dx += a - c
            total_dy += b - d

    # Рассчитываем средний вектор движения
    if count > 0:
        avg_dx = total_dx / count
        avg_dy = total_dy / count

        # Рассчитываем величину смещения
        motion_magnitude = np.sqrt(avg_dx ** 2 + avg_dy ** 2)

        # Определяем центр изображения для отображения вектора
        h, w, _ = frame.shape
        center = (w // 2, h // 2)

        if motion_magnitude > threshold:
            # Если есть движение, нормализуем вектор до постоянной длины
            norm_dx = (avg_dx / motion_magnitude) * fixed_length
            norm_dy = (avg_dy / motion_magnitude) * fixed_length

            # Конечная точка для вектора движения
            end_point = (int(center[0] + norm_dx), int(center[1] + norm_dy))

            # Рисуем вектор движения (стрелку)
            frame = cv2.arrowedLine(frame, center, end_point, (0, 255, 0), 3, tipLength=0.5)  # Зелёная стрелка
        else:
            # Если движения нет, не рисуем вектор
            pass

    # Наложение маски с линиями на текущее видео
    img = cv2.add(frame, mask)

    # Отображение изображения
    cv2.imshow('Optical Flow', img)

    # Нажмите "q" для выхода
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Обновляем предыдущий кадр и точки
    prev_gray = frame_gray.copy()
    prev_points = good_new.reshape(-1, 1, 2)

# Освобождение камеры и закрытие всех окон
cap.release()
cv2.destroyAllWindows()
