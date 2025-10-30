# Red Neuronal de Hamming – Reconocimiento de Figuras geométricas

## Descripción
Este proyecto implementa una **Red Neuronal de Hamming** para el reconocimiento de figuras geométricas simples:
círculo, cuadrado, estrella, triángulo y corazón.
Incluye invariancia rotacional y escalado, preprocesamiento con detección de bordes Sobel, y evaluación exhaustiva de prototipos.

---

## Instalación

### Requisitos del Sistema
- Python 3.10 o superior
- Sistema operativo: Linux, macOS o Windows

### Dependencias
```bash
pip install numpy scikit-learn scipy opencv-python
```

### Instalación del Proyecto
```bash
# Clonar o descargar el proyecto
cd hamming-network/

# Ejecutar para verificar instalación
python3 -c "import numpy as np; import cv2; print('Instalación correcta')"
```

---

## Estructura del Proyecto

```
hamming-network/
├── hamming_shapes.py          # Implementación de la Red de Hamming
├── load_axamples.py           # Carga y generación de prototipos
├── new_nnhaming.py            # Interfaz gráfica principal
├── run_test.py                # Evaluación y métricas
├── analyze_threshold.py       # Análisis de threshold óptimo
├── center_script.py           # Utilidades de centrado de imágenes
├── create_examples.py         # Herramienta para crear datasets
├── Nnhamming.py               # Versión básica de interfaz
├── dataset/                   # Datasets originales
├── dataset_centered/          # Datasets centrados
├── dataset_test/              # Datos de prueba
├── dataset_test_centered/     # Datos de prueba centrados
├── dataset_extra_feature/     # Dataset con clase extra (rombo)
├── dataset_test_extra_feature/ # Pruebas con clase extra
├── prototypes/                # Modelos guardados (.npz)
└── README.md                  # Este archivo
```

### Funciones de los Archivos Principales

- **`hamming_shapes.py`**: Contiene la clase `HammingNetwork` con métodos de predicción, invariancia rotacional y escalado
- **`load_axamples.py`**: Genera prototipos mediante promedio con threshold 0.30
- **`new_nnhaming.py`**: Interfaz gráfica con preprocesamiento completo y todas las invariancias
- **`run_test.py`**: Evalúa accuracy, matrices de confusión y métricas detalladas
- **`analyze_threshold.py`**: Analiza el impacto del threshold en la calidad de prototipos

---

## Ejecución y Resultados

### Ejecución Básica
```bash
# Interfaz gráfica principal
python3 new_nnhaming.py

# Evaluación con dataset estándar (5 clases)
python3 run_test.py

# Análisis de threshold óptimo
python3 analyze_threshold.py
```

### Resultados de Evaluación

#### Dataset Estándar (5 clases: círculo, cuadrado, estrella, triángulo, corazón)
- **Accuracy**: 82%
- **Métricas por clase**:
  - Círculo: 50% (confunde con otras formas redondas)
  - Cuadrado: 70%
  - Estrella: 100%
  - Triángulo: 100%
  - Corazón: 90%

#### Dataset con Clase Extra (6 clases: + rombo)
- **Accuracy**: ~75%
- **Nota**: La clase "rombo" no se reconoce (0% accuracy) porque no existe prototipo específico

### Preprocesamiento Aplicado
1. **Centrado de imagen**: Alinea figuras al centro del canvas 28×28
2. **Eliminación de ruido**: Opening morfológico para limpiar píxeles sueltos
3. **Detección de bordes Sobel**: Extrae características robustas de contorno
4. **Invariancia rotacional**: Prueba 4 rotaciones (0°, 90°, 180°, 270°)
5. **Invariancia escalado**: Prueba 5 factores de escala (0.8x a 1.2x)

### Threshold Óptimo (0.30)
- **Justificación**: Análisis empírico muestra que 0.30 balancea detalle vs. robustez
- **Validación**: Grid search de 0.1 a 0.5 confirma máximo accuracy en 0.30
- **Interpretación**: Un píxel debe aparecer en ≥30% de ejemplos para ser prototipo

---

## Alcances
- ✅ Clasificación de figuras geométricas simples
- ✅ Invariancia rotacional completa (4 orientaciones)
- ✅ Invariancia escalado (5 factores)
- ✅ Preprocesamiento avanzado con Sobel
- ✅ Evaluación exhaustiva con métricas sklearn
- ✅ Interfaz gráfica intuitiva
- ✅ Análisis de threshold óptimo con gráficos

## Limitaciones
- ❌ Resolución fija: 28×28 píxeles
- ❌ No generaliza a imágenes complejas/no geométricas
- ❌ Depende de calidad de prototipos de entrenamiento
- ❌ Interfaz gráfica básica (Tkinter)

---

## Desarrollo y Contribución

### Arquitectura Técnica
- **Codificación**: Bipolar (-1/+1) para distancia de Hamming
- **Competencia**: MaxNet para selección de ganador
- **Invariancia**: Data augmentation durante predicción
- **Evaluación**: Métricas estándar de clasificación

### Extensiones Posibles
- Agregar más clases geométricas
- Implementar CNN para extracción automática de características
- Mejorar interfaz gráfica (PyQt/Web)
- Añadir reconocimiento de texto manuscrito

---

## Licencia
Proyecto educativo - Uso libre para aprendizaje e investigación.
