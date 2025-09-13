"""
extraer_prototipos.py
- Lee dataset organizado como: dataset/<clase>/exampleX.csv
- Por cada clase extrae k_prototypes (por defecto usando k-medoids con Hamming)
- Devuelve:
    prototypes_bip: np.array shape (total_K, D) con valores -1.0 / +1.0 (float32)
    prototype_labels: list de etiquetas (len == total_K), p. ej. ["circulo","circulo","circulo", ...]
    prototypes_bin (opcional): prototipos en 0/1
"""

import os
import numpy as np
import hamming_shapes as hs

# --------------------------
# utilidades
# --------------------------
def load_class_samples(class_folder):
    """Carga todos los CSV de class_folder como arrays 0/1 flatten.
       Devuelve X (M, D) dtype=int (0/1)
    """
    files = sorted([f for f in os.listdir(class_folder) if f.lower().endswith(".csv")])
    mats = []
    shape_ref = None
    for f in files:
        p = os.path.join(class_folder, f)
        arr = np.loadtxt(p, delimiter=",")
        # aceptar si es entero o float; convertir a 0/1 por threshold 0.5
        arr_bin = (arr >= 0.5).astype(np.uint8)
        if shape_ref is None:
            shape_ref = arr_bin.shape
        else:
            if arr_bin.shape != shape_ref:
                raise ValueError(f"Archivo {p} tiene forma {arr_bin.shape} distinta a {shape_ref}")
        mats.append(arr_bin.ravel())
    if len(mats) == 0:
        return np.zeros((0, 0), dtype=np.uint8), None
    X = np.stack(mats, axis=0)
    return X, shape_ref

def pairwise_hamming(X):
    """X: (M,D) binary 0/1 -> devuelve matriz (M,M) de distancias de Hamming"""
    # broadcasting XOR and sum
    return (X[:, None, :] != X[None, :, :]).sum(axis=2)

# --------------------------
# k-medoids (PAM) básico
# --------------------------
def k_medoids_pam(dist_matrix, k, random_state=None, max_iter=100):
    """
    PAM greedy swap algorithm.
    dist_matrix: (M,M) symmetric nonnegative
    devuelve indices de medoids (k indices)
    """
    rng = np.random.default_rng(random_state)
    M = dist_matrix.shape[0]
    if k >= M:
        return np.arange(M)

    medoids = rng.choice(M, size=k, replace=False)
    current_cost = np.sum(np.min(dist_matrix[:, medoids], axis=1))
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        # para cada medoid m y cada no-medoid h probar swap
        for Ni in range(M):
            if Ni in medoids:
                continue
            for j_idx, med in enumerate(medoids):
                candidate = medoids.copy()
                candidate[j_idx] = Ni
                cost = np.sum(np.min(dist_matrix[:, candidate], axis=1))
                if cost < current_cost - 1e-9:
                    medoids = candidate
                    current_cost = cost
                    improved = True
                    break
            if improved:
                break
    return np.array(medoids)

# --------------------------
# otras estrategias
# --------------------------
def topk_by_centroid_similarity(X, k):
    """Centroid = mean(X); threshold 0.5 -> binary centroid.
       Luego seleccionar top-k muestras con mayor igualdad (D - Hamming).
    """
    M, D = X.shape
    if k >= M:
        return np.arange(M)
    centroid = X.mean(axis=0)
    centroid_bin = (centroid >= 0.5).astype(np.uint8)
    equal_counts = (X == centroid_bin).sum(axis=1)  # mayor es mejor
    idx_sorted = np.argsort(-equal_counts)
    return idx_sorted[:k]

def kmeans_threshold(X, k, random_state=None, max_iter=50):
    """Kmeans-like on binary data: centroids = mean then threshold, then pick medoid per cluster."""
    rng = np.random.default_rng(random_state)
    M, D = X.shape
    if k >= M:
        return np.arange(M)
    init_idx = rng.choice(M, size=k, replace=False)
    centroids = X[init_idx].astype(float)
    for _ in range(max_iter):
        # calc hamming distances to centroids (centroids are float; compare via rounding)
        cent_bin = (centroids >= 0.5).astype(np.uint8)
        dists = (X[:, None, :] != cent_bin[None, :, :]).sum(axis=2)  # (M,k)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.zeros((k, D), dtype=float)
        changed = False
        for j in range(k):
            members = X[labels == j]
            if len(members) == 0:
                # reinit empty centroid
                new_centroids[j] = X[rng.integers(0, M)]
            else:
                new_centroids[j] = members.mean(axis=0)
            if not np.allclose(new_centroids[j], centroids[j]):
                changed = True
        centroids = new_centroids
        if not changed:
            break
    # para cada centro, elegir medoid (muestra más cercana)
    cent_bin = (centroids >= 0.5).astype(np.uint8)
    dists = (X[:, None, :] != cent_bin[None, :, :]).sum(axis=2)  # (M,k)
    medoids = []
    for j in range(k):
        col = dists[:, j]
        med_idx = int(np.argmin(col))
        medoids.append(med_idx)
    return np.array(medoids)

# --------------------------
# función principal
# --------------------------
def extract_representative_prototypes(dataset_dir, k_per_class=3, method="kmedoids", random_state=None):
    """
    dataset_dir/<class_name>/*.csv
    method: "kmedoids" | "topk" | "kmeans"
    devuelve:
        prototypes_bip (np.float32) shape (total_protos, D)  con -1.0/+1.0
        prototype_labels: list length total_protos (string)
        prototypes_bin: np.uint8 0/1 shape (total_protos, D)
        sample_shape: (H,W)
    """
    all_protos = []
    all_labels = []
    sample_shape = None

    for class_name in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        X, sh = load_class_samples(class_path)
        if X.size == 0:
            print(f"[WARN] clase {class_name} no tiene CSV - se omite.")
            continue
        if sample_shape is None:
            sample_shape = sh
        else:
            if sh != sample_shape:
                raise ValueError("Todas las muestras deben tener la misma forma!")

        M, D = X.shape
        k = min(k_per_class, M)
        if M == 0:
            continue

        if method == "kmedoids":
            dist = pairwise_hamming(X)
            medoids = k_medoids_pam(dist, k, random_state=random_state)
            protos_bin = X[medoids]
        elif method == "topk":
            idx = topk_by_centroid_similarity(X, k)
            protos_bin = X[idx]
        elif method == "kmeans":
            idx = kmeans_threshold(X, k, random_state=random_state)
            protos_bin = X[idx]
        else:
            raise ValueError("method must be one of 'kmedoids','topk','kmeans'")

        # opcional: si preferís prototipo = promedio umbralizado:
        # avg = X.mean(axis=0); proto_avg = (avg >= 0.5).astype(np.uint8)
        # luego convertir proto_avg a bipolar, etc.

        for p in protos_bin:
            all_protos.append(p)
            all_labels.append(class_name)

    if len(all_protos) == 0:
        return np.zeros((0,0), dtype=np.float32), [], np.zeros((0,0), dtype=np.uint8), sample_shape

    protos_bin = np.stack(all_protos, axis=0).astype(np.uint8)  # (P,D)
    # convertir a bipolar -1/+1 float32
    protos_bip = np.where(protos_bin == 0, -1.0, 1.0).astype(np.float32)

    return protos_bip, all_labels, protos_bin, sample_shape

# --------------------------
# integración con HammingNetwork (ejemplo)
# --------------------------

def ascii_show(arr, shape=(28,28)):
    arr = arr.reshape(shape)                 # lo paso a matriz
    chars = np.where(arr > 0, "█", "·")      # █ = 1, · = 0
    lines = ["".join(row) for row in chars]
    return "\n".join(lines)

def extract_prototypes_as_dict(dataset_dir, k_per_class=3, method="kmedoids", random_state=None):
    """
    Versión adaptada que devuelve como load_patterns_from_csv:
      - patrones: dict {clase: vector_bipolar}
      - etiquetas: list de clases
    """
    protos_bip, labels, protos_bin, sample_shape = extract_representative_prototypes(
        dataset_dir, k_per_class=k_per_class, method=method, random_state=random_state
    )

    patrones = {}
    clases_vistas = set()
    etiquetas = []

    for proto, label in zip(protos_bip, labels):
        if label not in clases_vistas:
            patrones[label] = proto.flatten()
            etiquetas.append(label)
            clases_vistas.add(label)

    return patrones, etiquetas

if __name__ == "__main__":
    # ejemplo de uso
    dataset_dir = "dataset"   # estructura: dataset/<clase>/*.csv
    k_per_class = 3
    protos_bip, labels, protos_bin, shape = extract_representative_prototypes(
        dataset_dir, k_per_class=k_per_class, method="kmeans", random_state=0
    )
    print("Prototipos extraidos:", protos_bip.shape, "labels:", len(labels))
    
    for i, proto in enumerate(protos_bip):
        print(f"\nPatrón {i}:")
        print(ascii_show(proto, shape=(28,28)))
    # ahora podes hacer:
    # from hamming_shapes import HammingNetwork
    # net = HammingNetwork(protos_bip, labels=labels)
    # y usar net.predict(...) en entradas binarias transformadas a bipolar
