import os
import numpy as np
from hamming_shapes import make_network_from_csv


def excecute_tests():
    train_path = "dataset_extra_feature"        # contiene ejemplos para calcular
    test_path = "dataset_test_extra_feature"    # contiene ejemplos para evaluación

    print("Evaluando modelo base con invariancia rotacional y escalado")
    print("=" * 60)

    # Cargar modelo base con invariancia
    net, labels, _ = make_network_from_csv(train_path)

    # Función de preprocesamiento con Sobel
    def preprocess_with_sobel(img, size=28):
        """Preprocesamiento con detección de bordes Sobel"""
        from scipy import ndimage
        import cv2

        # Centrar (simplificado)
        centered = img.copy()
        if centered.shape != (size, size):
            centered = cv2.resize(centered, (size, size), interpolation=cv2.INTER_NEAREST)

        # Aplicar Sobel
        img_float = centered.astype(np.float64)
        sobel_x = ndimage.sobel(img_float, axis=0)
        sobel_y = ndimage.sobel(img_float, axis=1)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)

        return (edges / 255).astype(np.uint8)  # Normalizar a 0-1

    # Evaluar con invariancia + Sobel
    correct = 0
    total = 0
    # shapes_names = ["circulo", "cuadrado", "estrella", "triangulo", "corazon","rombo"]
    shapes_names = ["circulo", "cuadrado", "estrella", "triangulo", "corazon"]
    for shape_name in shapes_names:
        shape_path = f"{test_path}/{shape_name}"
        if not os.path.isdir(shape_path):
            continue

        for file in os.listdir(shape_path):
            if file.endswith(".csv"):
                file_path = os.path.join(shape_path, file)
                ejemplo = np.loadtxt(file_path, delimiter=",")
                binario = (ejemplo >= 0.24).astype(int)

                total += 1

                # Preprocesar con Sobel + usar invariancia rotacional y escalado
                procesado_sobel = preprocess_with_sobel(binario)
                pred = net.predict(procesado_sobel.flatten(), use_rotation_invariance=True, use_scaling_invariance=True)
                if pred == shape_name:
                    correct += 1

    acc = correct / total if total > 0 else 0
    print(f"Accuracy: {(acc*100):.2f}%")
    print(f"Correctas: {correct}/{total}")

    # Calcular y mostrar estadísticas detalladas
    from collections import defaultdict
    from sklearn.metrics import confusion_matrix, classification_report

    errores_por_clase = defaultdict(int)
    total_por_clase = defaultdict(int)
    y_true = []
    y_pred = []

    # Recalcular para obtener estadísticas detalladas
    for shape_name in shapes_names:
        shape_path = f"{test_path}/{shape_name}"
        if not os.path.isdir(shape_path):
            continue

        for file in os.listdir(shape_path):
            if file.endswith(".csv"):
                file_path = os.path.join(shape_path, file)
                ejemplo = np.loadtxt(file_path, delimiter=",")
                binario = (ejemplo >= 0.24).astype(int)

                total_por_clase[shape_name] += 1
                y_true.append(shape_name)

                # Preprocesar con Sobel + usar invariancia
                procesado_sobel = preprocess_with_sobel(binario)
                pred = net.predict(procesado_sobel.flatten(), use_rotation_invariance=True, use_scaling_invariance=True)
                y_pred.append(pred)

                if pred != shape_name:
                    errores_por_clase[shape_name] += 1

    # Ratio de error por clase
    print("\n📊 Ratio de error por clase:")
    for clase in sorted(total_por_clase.keys()):
        errores = errores_por_clase[clase]
        total_c = total_por_clase[clase]
        ratio = errores / total_c if total_c > 0 else 0
        print(f" - {clase}: {ratio*100:.2f}%  ({errores}/{total_c})")

    # Matriz de confusión
    labels_sorted = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    print("\n📌 Matriz de confusión:")
    print("Etiquetas:", labels_sorted)
    print(cm)

    # Reporte detallado
    print("\n📑 Reporte de clasificación:")
    print(classification_report(y_true, y_pred, labels=labels_sorted))

    # Solo evaluar el modelo base con todas las mejoras
    print("\n✅ Modelo base con invariancia rotacional y escalado evaluado exitosamente")


if __name__ == "__main__":
    excecute_tests()
