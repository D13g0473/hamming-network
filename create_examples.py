import tkinter as tk
import numpy as np
import csv
import os

class DrawingApp:
    def __init__(self, root, size=28, cell_size=20, save_dir="dataset", shape_name="figure"):
        self.size = size
        self.cell_size = cell_size
        self.canvas_size = size * cell_size
        self.data = np.zeros((size, size), dtype=int)

        # Carpeta donde se guardan los CSV
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Estado de la figura
        self.figure_name = shape_name
        self.example_index = 1

        # Interfaz
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)

        self.btn_save = tk.Button(root, text="Guardar Ejemplo", command=self.save_csv, state="disabled")
        self.btn_save.pack()
        self.btn_save.config(state="normal")

        self.btn_new = tk.Button(root, text="Nueva Figura", command=self.new_figure)
        self.btn_new.pack()

    def paint(self, event):
        x, y = event.x // self.cell_size, event.y // self.cell_size
        if 0 <= x < self.size and 0 <= y < self.size:
            self.data[y, x] = 1
            self.canvas.create_rectangle(
                x * self.cell_size, y * self.cell_size,
                (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                fill="black"
            )

    def new_figure(self):
        """Pide un nuevo nombre de figura y reinicia el índice"""
        name = input("Nombre de la figura (ej: circulo, cuadrado): ").strip()
        if name:
            self.figure_name = name
            self.example_index = 1
            print(f"Figura seleccionada: {self.figure_name}")
            self.btn_save.config(state="normal")
            # Crear carpeta para la figura
            os.makedirs(os.path.join(self.save_dir, self.figure_name), exist_ok=True)
        else:
            print("⚠ Debes ingresar un nombre válido.")

    def save_csv(self):
        """Guarda el dibujo actual como CSV con nombre automático"""
        if not self.figure_name:
            print("⚠ Primero selecciona una figura con 'Nueva Figura'.")
            return

        # Crear carpeta si no existe
        folder = os.path.join(self.save_dir, self.figure_name)
        os.makedirs(folder, exist_ok=True)

        filename = f"{self.figure_name}{self.example_index}.csv"
        filepath = os.path.join(folder, filename)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.data)

        print(f"Guardado: {filepath}")
        self.example_index += 1

        # limpiar lienzo
        self.data.fill(0)
        self.canvas.delete("all")


if __name__ == "__main__":
    shape = input("Nombre de la figura (ej: circulo, cuadrado): ").strip()
    root = tk.Tk()
    root.title("Dibujar Figura → CSV")
    app = DrawingApp(root, size=28, cell_size=20, save_dir="dataset", shape_name = shape )  # Guarda en dataset/
    root.mainloop()
