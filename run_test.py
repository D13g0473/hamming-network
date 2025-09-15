from test import evaluate_network
from hamming_shapes import HammingNetwork
import extraer_prototipos as ep


def excecute_tests():
    train_path = "dataset_centered"        # contiene ejemplos para calcular
    test_path = "dataset_test_centered"    # contiene ejemplos para evaluación
    
    # evaluate_network(train_path, test_path)
    # evaluar con distintos métodos
    for method in ["kmedoids", "topk", "kmeans", "normal"]:
        print(f"\nEvaluando con método: {method}")
        evaluate_network(train_path, test_path, type=method)    


if __name__ == "__main__":
    excecute_tests()
