import tkinter as tk
import numpy as np
import hamming_shapes as hs

# --------------------------
# Configuración inicial
# --------------------------

# Modelos disponibles
modelos = [
    "prototypes/base_media.npz",
    "prototypes/base_kmeans.npz",
    "prototypes/base_kmedoids.npz",
    "prototypes/base_topk.npz"
]

print("Seleccione el modelo:")
for i, m in enumerate(modelos):
    print(f"{i}: {m}")

idx = int(input("Modelo a usar (0-3): "))
net, labels, protos = hs.load_network_from_file(modelos[idx])

# --------------------------
# Parámetros
# --------------------------
N = 28          # tamaño lógico de la pizarra
cell_size = 20  # tamaño de cada celda en píxeles
canvas_size = N * cell_size

# --------------------------
# Tkinter UI
# --------------------------
root = tk.Tk()
root.title("Reconocimiento con Red de Hamming (28x28)")

canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="white")
canvas.pack()

# Matriz de dibujo
data = np.zeros((N, N), dtype=np.uint8)

# --------------------------
# Funciones auxiliares
# --------------------------

def paint(event):
    """Pinta un cuadrado negro en la celda correspondiente"""
    x, y = event.x // cell_size, event.y // cell_size
    if 0 <= x < N and 0 <= y < N:
        data[y, x] = 1
        canvas.create_rectangle(
            x * cell_size, y * cell_size,
            (x + 1) * cell_size, (y + 1) * cell_size,
            fill="black", outline="black"
        )

def clear_canvas():
    """Limpia la pizarra"""
    global data
    data.fill(0)
    canvas.delete("all")
    result_label.config(text="Dibuja una figura y presiona Predecir")

def predict_shape():
    """Convierte la pizarra en vector y predice con la red"""
    bin_arr = data.ravel().astype(np.uint8)  # 784 bits
    pred = net.predict(bin_arr)
    result_label.config(text=f"Predicción: {pred}")

# --------------------------
# Eventos UI
# --------------------------
canvas.bind("<B1-Motion>", paint)

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
