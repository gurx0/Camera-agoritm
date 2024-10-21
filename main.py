import cv2
import numpy as np

# Захват видео с камеры
cap = cv2.VideoCapture(0)

# Параметры для функции ShiTomasi для поиска хороших углов
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

# Параметры для сглаживания
alpha = 0.1  # Коэффициент сглаживания для EMA
smooth_dx, smooth_dy = 0, 0  # Начальные значения сглаженных компонент вектора
smooth_yaw = 0  # Инициализация сглаженного угла yaw
smooth_angle = 0  # Начальное значение сглаженного угла

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

    # Если количество точек стало меньше 10, заново инициализируем точки
    if len(good_new) < 10:
        good_new = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        prev_points = good_new
        mask = np.zeros_like(frame)  # Очистим маску для рисования

    # Инициализируем переменные для накопления смещений
    total_dx, total_dy = 0, 0
    count = len(good_new)

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()  # Преобразуем в одномерный массив
        c, d = old.ravel()  # Преобразуем в одномерный массив

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

        # Сглаживание значений с использованием экспоненциального скользящего среднего (EMA)
        smooth_dx = alpha * avg_dx + (1 - alpha) * smooth_dx
        smooth_dy = alpha * avg_dy + (1 - alpha) * smooth_dy

        # Определяем центр изображения
        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2

        # Инициализация переменных для расчета угла поворота
        delta_x = 0
        delta_y = 0

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()  # Новая позиция
            c, d = old.ravel()  # Старая позиция

            # Вычисляем смещение относительно центра
            delta_x += (a - center_x) - (c - center_x)
            delta_y += (b - center_y) - (d - center_y)

        # Рассчитываем угол поворота
        if delta_x != 0 or delta_y != 0:  # Убедитесь, что вектор не нулевой
            rotation_angle = np.degrees(np.arctan2(delta_y, delta_x))
        else:
            rotation_angle = 0  # Если нет движения, угол равен 0

        # Выводим угол на изображение
        rotation_text = f"Rotation Angle: {rotation_angle:.2f} degrees"
        cv2.putText(frame, rotation_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Далее рассчитайте yaw
        yaw = np.degrees(np.arctan2(smooth_dy, smooth_dx))
        smooth_yaw = alpha * yaw + (1 - alpha) * smooth_yaw

        # Проверяем направление yaw для изменения коэффициента сглаживания
        if abs(yaw) < 10 or abs(yaw) > 170:  # Если yaw почти параллелен одной из осей
            alpha_yaw = 0.05  # Увеличиваем сглаживание
        else:
            alpha_yaw = 0.1  # Обычное сглаживание

        smooth_yaw = alpha_yaw * yaw + (1 - alpha_yaw) * smooth_yaw

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
