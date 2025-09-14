import tkinter as tk
import numpy as np
import hamming_shapes as hs

# --------------------------
# Configuración inicial
# --------------------------
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
    """Convierte la pizarra en vector y predice con la red"""
    bin_arr = data.ravel().astype(np.uint8)  # 784 bits
    pred, scores = net.predict(bin_arr, return_scores=True)

    # Mostrar predicción textual
    result_label.config(text=f"Predicción: {pred}")

    # Dibujar prototipo perfecto en canvas_output
    canvas_output.delete("all")
    try:
        idx = labels.index(pred)
        proto = protos_p[idx].reshape(N, N)  # recuperar figura prototipo
    except Exception:
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
