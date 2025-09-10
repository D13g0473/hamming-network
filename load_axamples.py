import os
import numpy as np

def load_patterns_from_csv(base_path):
    patrones = {}
    etiquetas = []
    
    # Recorremos las carpetas
    for shape_name in os.listdir(base_path):
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
            # promedio
            promedio = np.mean(ejemplos, axis=0)
            # umbralizar a binario
            binario = (promedio >= 0.5).astype(int)
            # convertir a bipolar
            bipolar = np.where(binario == 0, -1, 1)
            
            patrones[shape_name] = bipolar.flatten()
            etiquetas.append(shape_name)
    
    return patrones, etiquetas

# Ejemplo de uso
base_path = "dataset"
patrones, etiquetas = load_patterns_from_csv(base_path)

for nombre, patron in patrones.items():
    print(f"Patrón para {nombre}: {patron}")
