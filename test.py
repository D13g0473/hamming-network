import os
import numpy as np
from hamming_shapes import make_network_from_csv,make_network_from_csv_kmedoids,make_network_from_csv_topk,make_network_from_csv_kmeans

def evaluate_network(train_path, test_path, type="normal"):
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
                if pred == shape_name:
                    correct += 1
                else:
                    print(f"❌ Error: esperado={shape_name}, predicho={pred}, archivo={file}")
    
    acc = correct / total if total > 0 else 0
    print(f"\n✅ Accuracy: {acc*100:.2f}%  ({correct}/{total})")
