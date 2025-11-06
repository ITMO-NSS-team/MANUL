import tensorflow as tf
import numpy as np


def explore_real_structure(filepath):
    """Исследуем реальную структуру файла"""
    print("=== ИССЛЕДОВАНИЕ РЕАЛЬНОЙ СТРУКТУРЫ DICHASUS ===")

    # Сначала прочитаем без reshape чтобы увидеть реальные размеры
    raw_dataset = tf.data.TFRecordDataset(filepath)

    feature_description = {
        "csi": tf.io.FixedLenFeature([], tf.string),
        "pos-tachy": tf.io.FixedLenFeature([], tf.string),
    }

    def parse_without_reshape(proto):
        record = tf.io.parse_single_example(proto, feature_description)

        # Парсим без reshape чтобы увидеть реальный размер
        csi_raw = tf.io.parse_tensor(record["csi"], out_type=tf.float32)
        pos_raw = tf.io.parse_tensor(record["pos-tachy"], out_type=tf.float64)

        return csi_raw, pos_raw

    dataset = raw_dataset.map(parse_without_reshape)

    # Смотрим первые несколько samples
    for i, (csi, pos) in enumerate(dataset.take(3)):
        print(f"\nSample {i}:")
        print(f"  CSI raw shape: {csi.shape}, total elements: {tf.size(csi)}")
        print(f"  CSI dtype: {csi.dtype}")
        print(f"  Position shape: {pos.shape}")
        print(f"  Position: {pos.numpy()}")

        # Пробуем разные варианты reshape
        total_elements = tf.size(csi).numpy()
        print(f"  Возможные формы CSI:")
        print(f"    - [64, 1024, 2] = {64 * 1024 * 2} элементов")
        print(f"    - [32, 1024, 2] = {32 * 1024 * 2} элементов")
        print(f"    - [32, 2048, 2] = {32 * 2048 * 2} элементов")
        print(f"    - [16, 1024, 2] = {16 * 1024 * 2} элементов")


def parse_dichasus_correct(filepath):
    """Парсинг с правильной формой на основе исследования"""
    raw_dataset = tf.data.TFRecordDataset(filepath)

    feature_description = {
        "csi": tf.io.FixedLenFeature([], tf.string),
        "pos-tachy": tf.io.FixedLenFeature([], tf.string),
    }

    def parse_function(proto):
        record = tf.io.parse_single_example(proto, feature_description)

        # Парсим тензор
        csi = tf.io.parse_tensor(record["csi"], out_type=tf.float32)

        # АВТОМАТИЧЕСКИЙ подбор формы на основе количества элементов
        total_elements = tf.size(csi)

        # Пробуем разные варианты форм
        if total_elements == 64 * 1024 * 2:  # 131072 элементов
            csi = tf.reshape(csi, [64, 1024, 2])
        elif total_elements == 32 * 1024 * 2:  # 65536 элементов
            csi = tf.reshape(csi, [32, 1024, 2])
        elif total_elements == 32 * 2048 * 2:  # 131072 элементов
            csi = tf.reshape(csi, [32, 2048, 2])
        else:
            # Используем форму по умолчанию или оставляем как есть
            print(f"Неизвестная форма: {total_elements} элементов")
            # csi = tf.reshape(csi, [-1])  # или оставляем плоским

        # Позиция всегда [3]
        pos = tf.io.parse_tensor(record["pos-tachy"], out_type=tf.float64)
        pos = tf.reshape(pos, [3])

        return csi, pos

    return raw_dataset.map(parse_function)


def convert_adaptive(filepath):
    """Адаптивная конвертация"""
    print("Загрузка данных...")
    dataset = parse_dichasus_correct(filepath)

    csi_samples = []
    positions = []

    for i, (csi_tensor, pos_tensor) in enumerate(dataset):
        if i % 1000 == 0:
            print(f"Обработано {i} samples...")

        csi_np = csi_tensor.numpy()
        pos_np = pos_tensor.numpy()

        # Преобразуем в комплексные числа
        csi_complex = csi_np[..., 0] + 1j * csi_np[..., 1]

        csi_samples.append(csi_complex)
        positions.append(pos_np)

    csi_array = np.array(csi_samples)  # [N_samples, N_antennas, N_subcarriers]

    print(f"\nРеальная форма CSI: {csi_array.shape}")

    # Автоматически определяем конфигурацию
    N_samples, N_antennas, N_subcarriers = csi_array.shape

    # Создаем формат [N_antennas, 1, N_subcarriers, N_samples]
    csi_final = csi_array.transpose(1, 2, 0)  # [N_ant, N_sub, N_samples]
    csi_final = np.expand_dims(csi_final, axis=1)  # [N_ant, 1, N_sub, N_samples]

    positions_array = np.array(positions)  # [N_samples, 3]

    print(f"Итоговая форма: {csi_final.shape}")
    print(f"Позиции: {positions_array.shape}")

    return csi_final, positions_array


def load_data(name):
    print("=== ШАГ 1: ИССЛЕДОВАНИЕ ===")
    explore_real_structure("dichasus-ad01.tfrecords")
    print("\n=== ШАГ 2: КОНВЕРТАЦИЯ ===")
    csi_data, positions = convert_adaptive(name)
    return csi_data

