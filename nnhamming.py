import tkinter as tk
import numpy as np
from tkinter import filedialog, messagebox
from center_script import center_image
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

try:
    print("Seleccione el modelo:")
    for i, m in enumerate(modelos):
        print(f"{i}: {m}")

    idx = int(input("Modelo a usar (0-3): "))
    net, labels, protos = hs.load_network_from_file(modelos[idx])
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    exit(1)

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
    try:
        x, y = event.x // cell_size, event.y // cell_size
        if 0 <= x < N and 0 <= y < N:
            data[y, x] = 1
            canvas_input.create_rectangle(
                x * cell_size, y * cell_size,
                (x + 1) * cell_size, (y + 1) * cell_size,
                fill="black", outline="black"
            )
    except Exception as e:
        messagebox.showerror("Error", f"Error en paint(): {e}")

def clear_canvas():
    """Limpia las pizarras"""
    try:
        global data
        data.fill(0)
        canvas_input.delete("all")
        canvas_output.delete("all")
        result_label.config(text="Dibuja una figura o carga un CSV")
    except Exception as e:
        messagebox.showerror("Error", f"Error en clear_canvas(): {e}")

def draw_matrix_on_canvas(matrix, canvas):
    """Dibuja una matriz binaria 28x28 en el canvas"""
    canvas.delete("all")
    for y in range(N):
        for x in range(N):
            if matrix[y, x] > 0:
                canvas.create_rectangle(
                    x * cell_size, y * cell_size,
                    (x + 1) * cell_size, (y + 1) * cell_size,
                    fill="black", outline="black"
                )

def predict_shape():
    """Convierte la pizarra en vector, centra la figura y predice con la red"""
    try:
        centered = center_image(data, size=28).astype(np.uint8)
        bin_arr = centered.ravel()
        pred, scores = net.predict(bin_arr, return_scores=True)
        result_label.config(text=f"Predicción: {pred}")

        # Dibujar prototipo perfecto
        canvas_output.delete("all")
        try:
            idx = labels.index(pred)
            proto = protos_p[idx].reshape(N, N)
            draw_matrix_on_canvas(proto, canvas_output)
        except Exception:
            pass
    except Exception as e:
        messagebox.showerror("Error", f"Error en predict_shape(): {e}")

def load_csv_and_predict():
    """Carga un CSV de 28x28 y lo predice"""
    try:
        file_path = filedialog.askopenfilename(
            title="Seleccionar CSV",
            filetypes=[("CSV Files", "*.csv")]
        )
        if not file_path:
            return

        # Leer CSV
        ejemplo = np.loadtxt(file_path, delimiter=",")
        if ejemplo.shape != (28, 28):
            raise ValueError("El CSV debe ser una matriz 28x28")

        if not np.isin(ejemplo, [0, 1]).all():
            raise ValueError("El CSV debe contener solo valores 0 y 1")

        # Centrar
        centered = center_image(ejemplo.astype(np.uint8), size=28)

        # Mostrar figura cargada en canvas_input
        draw_matrix_on_canvas(centered, canvas_input)

        # Predicción
        pred, scores = net.predict(centered.ravel(), return_scores=True)
        result_label.config(text=f"Predicción desde CSV: {pred}")

        # Dibujar prototipo en salida
        idx = labels.index(pred)
        proto = protos_p[idx].reshape(N, N)
        draw_matrix_on_canvas(proto, canvas_output)

    except Exception as e:
        messagebox.showerror("Error", f"No se pudo cargar el CSV: {e}")

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

btn_load_csv = tk.Button(btn_frame, text="Cargar CSV", command=load_csv_and_predict)
btn_load_csv.grid(row=0, column=2, padx=10)

result_label = tk.Label(root, text="Dibuja una figura o carga un CSV", font=("Arial", 14))
result_label.pack(pady=10)

# --------------------------
root.mainloop()
