import os
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

from hamming_shapes import (
    make_network_from_csv,
    make_network_from_csv_kmedoids,
    make_network_from_csv_topk,
    make_network_from_csv_kmeans
)

def evaluate_network(train_path, test_path, type="normal"):
    # ---------------------------
    # Selección del modelo
    # ---------------------------
    if type == "kmedoids":
        net, etiquetas, _ = make_network_from_csv_kmedoids(train_path)
    elif type == "topk":
        net, etiquetas, _ = make_network_from_csv_topk(train_path)
    elif type == "kmeans":
        net, etiquetas, _ = make_network_from_csv_kmeans(train_path)
    else:   
        net, etiquetas, _ = make_network_from_csv(train_path)
    
    correct = 0
    total = 0
    
    # Guardar resultados
    y_true = []
    y_pred = []
    errores_por_clase = defaultdict(int)
    total_por_clase = defaultdict(int)

    # ---------------------------
    # Evaluación sobre el set
    # ---------------------------
    for shape_name in os.listdir(test_path):
        shape_path = os.path.join(test_path, shape_name)
        if not os.path.isdir(shape_path):
            continue
        
        for file in os.listdir(shape_path):
            if file.endswith(".csv"):
                file_path = os.path.join(shape_path, file)
                ejemplo = np.loadtxt(file_path, delimiter=",")
                # binarizar igual que en entrenamiento
                binario = (ejemplo >= 0.24).astype(int)
                
                pred = net.predict(binario.flatten())
                
                total += 1
                total_por_clase[shape_name] += 1
                y_true.append(shape_name)
                y_pred.append(pred)

                if pred == shape_name:
                    correct += 1
                else:
                    errores_por_clase[shape_name] += 1
                    print(f"❌ Error: esperado={shape_name}, predicho={pred}, archivo={file}")

    # ---------------------------
    # Métricas
    # ---------------------------
    acc = correct / total if total > 0 else 0
    print(f"\n✅ Accuracy: {acc*100:.2f}%  ({correct}/{total})")

    # Ratio de error por clase
    print("\n📊 Ratio de error por clase:")
    for clase in sorted(total_por_clase.keys()):
        errores = errores_por_clase[clase]
        total_c = total_por_clase[clase]
        ratio = errores / total_c if total_c > 0 else 0
        print(f" - {clase}: {ratio*100:.2f}%  ({errores}/{total_c})")

    # ---------------------------
    # Matriz de confusión
    # ---------------------------
    labels_sorted = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    print("\n📌 Matriz de confusión:")
    print("Etiquetas:", labels_sorted)
    print(cm)

    # Reporte detallado (precisión, recall, f1)
    print("\n📑 Reporte de clasificación:")
    print(classification_report(y_true, y_pred, labels=labels_sorted))
