import tkinter as tk
import numpy as np
import cv2
from scipy.ndimage import binary_opening
from center_script import center_image
import hamming_shapes as hs
from scipy import ndimage

# --------------------------
# Configuración inicial
# --------------------------
modelo = "prototypes/base_media.npz"

print(f"Usando modelo base con threshold optimizado: {modelo}")
net, labels, protos = hs.load_network_from_file(modelo)

print('Modo: dibujo en pantalla con todas las mejoras activadas')
print('- Preprocesamiento con detección de bordes Sobel')
print('- Invariancia rotacional (4 rotaciones)')
print('- Invariancia escalado (5 factores)')

# modo 2: dibujo en pantalla

# --------------------------
# Parámetros
# --------------------------
N = 28          # tamaño lógico de la pizarra
cell_size = 20  # tamaño de cada celda en píxeles
canvas_size = N * cell_size

_, _, protos_p = hs.make_network(N)

# --------------------------
# Tkinter UI
# --------------------------
root = tk.Tk()
root.title("Reconocimiento con Red de Hamming (28x28)")

frame = tk.Frame(root)
frame.pack()

# Canvas de dibujo (izquierda)
canvas_input = tk.Canvas(frame, width=canvas_size, height=canvas_size, bg="white")
canvas_input.grid(row=0, column=0, padx=10, pady=10)

# Canvas de salida (derecha)
canvas_output = tk.Canvas(frame, width=canvas_size, height=canvas_size, bg="white")
canvas_output.grid(row=0, column=1, padx=10, pady=10)

# Matriz de dibujo
data = np.zeros((N, N), dtype=np.uint8)

# --------------------------
# Preprocesamiento
# --------------------------
def preprocess(img, size=28, use_sobel=False):
    """Centrar, limpiar ruido, detectar bordes y escalar figura"""
    # Centrar
    centered = center_image(img, size=size).astype(np.uint8)

    # Limpiar ruido (puntos sueltos)
    clean = binary_opening(centered, structure=np.ones((2,2))).astype(np.uint8)

    if use_sobel:
        # Aplicar detección de bordes Sobel
        # Convertir a float para cálculo de gradientes
        img_float = clean.astype(np.float64)

        # Filtros Sobel
        sobel_x = ndimage.sobel(img_float, axis=0)
        sobel_y = ndimage.sobel(img_float, axis=1)

        # Magnitud del gradiente
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Normalizar y umbralizar
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        # Umbralizar para obtener bordes binarios
        _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
        clean = edges.astype(np.uint8)

    # Bounding box para detectar área activa
    rows = np.any(clean, axis=1)
    cols = np.any(clean, axis=0)
    if not rows.any() or not cols.any():
        return clean  # figura vacía

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    cropped = clean[ymin:ymax+1, xmin:xmax+1]

    # Escalar figura para que ocupe ~20x20 dentro del 28x28
    resized = cv2.resize(cropped, (20, 20), interpolation=cv2.INTER_NEAREST)

    # Colocar en canvas vacío de 28x28
    new_img = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - resized.shape[0]) // 2
    x_offset = (size - resized.shape[1]) // 2
    new_img[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized

    return new_img

# --------------------------
# Funciones auxiliares
# --------------------------
def paint(event):
    """Pinta un cuadrado negro en la celda correspondiente"""
    x, y = event.x // cell_size, event.y // cell_size
    if 0 <= x < N and 0 <= y < N:
        data[y, x] = 1
        canvas_input.create_rectangle(
            x * cell_size, y * cell_size,
            (x + 1) * cell_size, (y + 1) * cell_size,
            fill="black", outline="black"
        )

def clear_canvas():
    """Limpia las pizarras"""
    global data
    data.fill(0)
    canvas_input.delete("all")
    canvas_output.delete("all")
    result_label.config(text="Dibuja una figura y presiona Predecir")

def predict_shape():
    """Convierte la pizarra en vector, preprocesa y predice con la red"""
    # Preprocesar (centrar + limpiar + detectar bordes + escalar)
    processed = preprocess(data, size=28, use_sobel=True)

    # Aplanar para la red
    bin_arr = processed.ravel()

    # Predicción con invariancia rotacional y escalado
    pred, scores = net.predict(bin_arr, return_scores=True, use_rotation_invariance=True, use_scaling_invariance=True)

    # Mostrar predicción textual
    result_label.config(text=f"Predicción: {pred}")

    # Dibujar prototipo correspondiente al modelo seleccionado en canvas_output
    canvas_output.delete("all")
    try:
        idx = labels.index(pred)
        # Usar los prototipos del modelo cargado, no los de make_network
        if protos.ndim == 3:  # Si está en formato (N_shapes, H, W)
            proto = protos[idx]
        else:  # Si está en formato (N_shapes, H*W), reshape
            proto = protos[idx].reshape(N, N)
    except Exception as e:
        print(f"Error al acceder al prototipo: {e}")
        return

    for y in range(N):
        for x in range(N):
            if proto[y, x] > 0:
                canvas_output.create_rectangle(
                    x * cell_size, y * cell_size,
                    (x + 1) * cell_size, (y + 1) * cell_size,
                    fill="black", outline="black"
                )

# --------------------------
# Eventos UI
# --------------------------
canvas_input.bind("<B1-Motion>", paint)

btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

btn_clear = tk.Button(btn_frame, text="Limpiar", command=clear_canvas)
btn_clear.grid(row=0, column=0, padx=10)

btn_predict = tk.Button(btn_frame, text="Predecir", command=predict_shape)
btn_predict.grid(row=0, column=1, padx=10)

result_label = tk.Label(root, text="Dibuja una figura y presiona Predecir", font=("Arial", 14))
result_label.pack(pady=10)

# --------------------------
root.mainloop()
