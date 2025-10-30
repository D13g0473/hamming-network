import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from hamming_shapes import make_network_from_csv

def analyze_threshold_impact():
    """Analiza el impacto de diferentes thresholds en la calidad de los prototipos"""

    base_path = "dataset_centered"
    test_path = "dataset_test_centered"

    # Thresholds a probar
    thresholds = np.arange(0.1, 0.6, 0.05)
    accuracies = []
    pixel_counts = []

    print("Analizando impacto del threshold en prototipos...")
    print("=" * 60)

    for threshold in thresholds:
        print(f"Probando threshold: {threshold:.2f}")

        # Crear prototipos con este threshold
        patrones = {}
        etiquetas = []

        for shape_name in sorted(os.listdir(base_path)):
            shape_path = os.path.join(base_path, shape_name)
            if not os.path.isdir(shape_path):
                continue

            ejemplos = []
            for file in os.listdir(shape_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(shape_path, file)
                    ejemplo = np.loadtxt(file_path, delimiter=",")
                    ejemplos.append(ejemplo)

            if ejemplos:
                ejemplos = np.array(ejemplos)
                promedio = np.mean(ejemplos, axis=0)
                binario = (promedio >= threshold).astype(int)
                bipolar = np.where(binario == 0, -1.0, 1.0).astype(np.float32)

                patrones[shape_name] = bipolar
                etiquetas.append(shape_name)

        # Contar píxeles activos por prototipo
        total_pixels = 0
        for patron in patrones.values():
            total_pixels += np.sum(patron > 0)  # píxeles positivos
        avg_pixels = total_pixels / len(patrones) if patrones else 0
        pixel_counts.append(avg_pixels)

        # Crear red temporal y evaluar
        if patrones:
            protos_list = [patrones[label].flatten() for label in etiquetas]  # Aplanar cada prototipo
            protos = np.stack(protos_list, axis=0)  # Esto asegura que sea 2D (M, D)
            from hamming_shapes import HammingNetwork
            net = HammingNetwork(protos, labels=etiquetas)

            # Evaluar en test set
            correct = 0
            total = 0

            for shape_name in etiquetas:
                shape_path = os.path.join(test_path, shape_name)
                if not os.path.isdir(shape_path):
                    continue

                for file in os.listdir(shape_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(shape_path, file)
                        ejemplo = np.loadtxt(file_path, delimiter=",")
                        binario = (ejemplo >= 0.24).astype(int)

                        pred = net.predict(binario.flatten(), use_rotation_invariance=True, use_scaling_invariance=True)
                        if pred == shape_name:
                            correct += 1
                        total += 1

            acc = correct / total if total > 0 else 0
            accuracies.append(acc)
            print(".2f")
        else:
            accuracies.append(0.0)
            print(f"  Threshold {threshold:.2f}: No se generaron prototipos")

    # Crear gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico 1: Accuracy vs Threshold
    ax1.plot(thresholds, accuracies, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=0.30, color='red', linestyle='--', linewidth=2, label='Threshold óptimo (0.30)')
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy vs Threshold en Prototipos', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Gráfico 2: Píxeles activos vs Threshold
    ax2.plot(thresholds, pixel_counts, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=0.30, color='red', linestyle='--', linewidth=2, label='Threshold óptimo (0.30)')
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Píxeles activos promedio por prototipo', fontsize=12)
    ax2.set_title('Complejidad del Prototipo vs Threshold', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Encontrar threshold óptimo
    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    best_accuracy = accuracies[best_idx]

    print("\n📊 ANÁLISIS COMPLETADO")
    print("=" * 60)
    print(f"Threshold óptimo encontrado: {best_threshold:.2f}")
    print(f"Accuracy máxima obtenida: {best_accuracy:.2f}")
    print(f"Píxeles activos promedio: {pixel_counts[best_idx]:.1f}")
    # Mostrar tabla resumen
    print("\n📋 TABLA RESUMEN:")
    print("Threshold | Accuracy | Píxeles")
    print("-" * 30)
    for t, a, p in zip(thresholds, accuracies, pixel_counts):
        marker = " ← ÓPTIMO" if abs(t - 0.30) < 0.01 else ""
        print(f"{t:.2f}      | {a:.2f}     | {p:.1f}{marker}")

    return best_threshold, best_accuracy

def visualize_prototypes_at_thresholds():
    """Visualiza cómo cambian los prototipos con diferentes thresholds"""

    base_path = "dataset_centered"
    thresholds_to_show = [0.1, 0.2, 0.3, 0.4, 0.5]

    fig, axes = plt.subplots(len(thresholds_to_show), 5, figsize=(20, 12))
    fig.suptitle('Evolución de Prototipos con Diferentes Thresholds', fontsize=16, fontweight='bold')

    for i, threshold in enumerate(thresholds_to_show):
        patrones = {}

        for shape_name in sorted(os.listdir(base_path)):
            shape_path = os.path.join(base_path, shape_name)
            if not os.path.isdir(shape_path):
                continue

            ejemplos = []
            for file in os.listdir(shape_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(shape_path, file)
                    ejemplo = np.loadtxt(file_path, delimiter=",")
                    ejemplos.append(ejemplo)

            if ejemplos:
                ejemplos = np.array(ejemplos)
                promedio = np.mean(ejemplos, axis=0)
                binario = (promedio >= threshold).astype(int)
                patrones[shape_name] = binario

        # Mostrar prototipos
        shape_names = ['circulo', 'corazon', 'cuadrado', 'estrella', 'triangulo']
        for j, shape_name in enumerate(shape_names):
            if shape_name in patrones:
                axes[i, j].imshow(patrones[shape_name].reshape(28, 28), cmap='binary')
                axes[i, j].set_title(f'{shape_name}\n(th={threshold})')
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('prototypes_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Análisis principal
    best_threshold, best_accuracy = analyze_threshold_impact()

    # Visualización de prototipos
    print("\nGenerando visualización de prototipos...")
    visualize_prototypes_at_thresholds()

    print("\n✅ Análisis completado. Gráficos guardados como:")
    print("   - threshold_analysis.png")
    print("   - prototypes_evolution.png")