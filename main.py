import os
import cv2
from keras.models import load_model
from scipy.ndimage.measurements import center_of_mass
from skimage.restoration import denoise_tv_chambolle
import numpy as np
import matplotlib.pyplot as plt


def getBestShift(img):
    cy, cx = center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def find_best_rotation(image_path, model):
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Обернення кольорів (за умови, що цифри білі, а фон чорний)
    image = 255 - original_image

    # Порогова обробка для перетворення зображення на бінарне
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Видалення зайвих відступів навколо зображення
    while np.sum(image[0]) == 0:
        image = image[1:]
    while np.sum(image[:, 0]) == 0:
        image = np.delete(image, 0, 1)
    while np.sum(image[-1]) == 0:
        image = image[:-1]
    while np.sum(image[:, -1]) == 0:
        image = np.delete(image, -1, 1)

    # Зміна розміру зображення до 20x20 пікселів, зберігаючи пропорції
    rows, cols = image.shape
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
    image = cv2.resize(image, (cols, rows))

    # Додавання відступів до розміру 28x28 пікселів
    colsPadding = (int(np.ceil((28 - cols) / 2.0)), int(np.floor((28 - cols) / 2.0)))
    rowsPadding = (int(np.ceil((28 - rows) / 2.0)), int(np.floor((28 - rows) / 2.0)))
    image = np.lib.pad(image, (rowsPadding, colsPadding), 'constant')

    # Центрування зображення
    shiftx, shifty = getBestShift(image)
    image = shift(image, shiftx, shifty)

    rotations_info = []

    for angle in range(0, 360, 5):  # Тестуємо кути від 0 до 360 градусів з кроком 5 градусів
        rotated_image = rotate_image(image, angle)

        # Нормалізація та підготовка зображення для передбачення моделлю
        rotated_image = rotated_image / 255.0
        rotated_image = rotated_image.reshape(1, 28, 28, 1)

        # Обезшумлення по алгоритму ROF
        denoised_image = denoise_tv_chambolle(rotated_image, weight=0.1)

        # Після попередньої обробки передбачаємо клас зображення
        prediction = model.predict(rotated_image)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        # Додаємо інформацію про оберт до списку
        rotations_info.append({
            "photo_path": image_path,
            "angle": angle,
            "predicted_class": predicted_class,
            "confidence": confidence*100,
            "rotated_image": rotated_image,
            "original_image": original_image,
            "denoised_image": denoised_image
        })

    # Відсортовуємо список за впевненістю в оберненому порядку (від найбільшої до найменшої)
    rotations_info_sorted = sorted(rotations_info, key=lambda x: x["confidence"], reverse=True)

    # Вибираємо три найкращі оберти
    top_3_rotations = rotations_info_sorted[:3]

    return top_3_rotations


def display_images(original, rotated, denoised):
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original.squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(rotated[0, :, :, 0], cmap='gray')
    plt.title('Rotated Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(denoised[0, :, :, 0], cmap='gray')
    plt.title('Denoised Image')
    plt.axis('off')

    plt.show()


model = load_model('model.h5')

# Шлях до папки з фотографіями
photos_path = 'Photos'
photos_files = [os.path.join(photos_path, f) for f in os.listdir(photos_path) if os.path.isfile(os.path.join(photos_path, f))]

all_photos_top_3_rotations_info = []  # Створюємо порожній список для зберігання результатів

# Обробка кожної фотографії
for photo_path in photos_files:
    top_3_rotations = find_best_rotation(photo_path, model)
    all_photos_top_3_rotations_info.append(top_3_rotations)  # Додаємо результати до загального списку

# Аналізуємо результати в кінці
for index, top_3_rotations in enumerate(all_photos_top_3_rotations_info):
    print(f"\n\nФотографія: {photos_files[index]}")
    for rotation in top_3_rotations:
        print(f"Angle: {rotation['angle']}°, Predicted Class: {rotation['predicted_class']}, Confidence: {rotation['confidence']:.0f}%")
        display_images(rotation['original_image'], rotation['rotated_image'], rotation['denoised_image'])
