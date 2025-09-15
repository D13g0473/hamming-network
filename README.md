# Red Neuronal de Hamming – Reconocimiento de Figuras geométricas

## Descripción
Este proyecto implementa una **Red Neuronal de Hamming** para el reconocimiento de figuras geométricas simples:  
círculo, cuadrado, estrella, triángulo y corazón.  
Permite evaluar la tolerancia al ruido, comparar distintos prototipos y probar figuras desde CSV o dibujo manual en pantalla.

---

## Alcances
- Clasificación de 5 figuras básicas.
- Entrada desde archivos CSV o dibujo manual.
- Comparación de prototipos: Media, K-Means, K-Medoids, Top-K.
- Tolerancia al ruido (0 a 1).
- Orientado a demostraciones y aprendizaje de conceptos de redes neuronales.

## Limitaciones
- Resolución fija: **28x28 píxeles**.
- No generaliza a imágenes complejas.
- Depende de los prototipos generados.
- Interfaz gráfica básica (Tkinter).

---

## Requisitos
- Python 3.10 o superior
- Librerías:
```bash
pip install numpy
```

---

## Ejecución

```bash
python Nnhamming.py
```
