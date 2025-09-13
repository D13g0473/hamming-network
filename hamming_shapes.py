# hamming_shapes.py
from math import cos, sin, pi, sqrt
import numpy as np
from load_axamples import load_patterns_from_csv  
from load_prototypes_kmeans import extract_prototypes_as_dict

def make_network_from_csv(base_path):
    patrones, etiquetas = load_patterns_from_csv(base_path)
    # Convertimos a array de prototipos
    protos = np.stack([patrones[e] for e in etiquetas], axis=0).astype(np.float32)
    net = HammingNetwork(protos, labels=etiquetas)
    return net, etiquetas, protos

def make_network_from_csv_kmedoids(base_path):
    patrones, etiquetas = extract_prototypes_as_dict(base_path, k_per_class=3, method="kmedoids", random_state=0)
    protos = np.stack([patrones[e] for e in etiquetas], axis=0).astype(np.float32)
    net = HammingNetwork(protos, labels=etiquetas)
    return net, etiquetas, protos

def make_network_from_csv_topk(base_path):
    patrones, etiquetas = extract_prototypes_as_dict(base_path, k_per_class=3, method="topk", random_state=0)
    protos = np.stack([patrones[e] for e in etiquetas], axis=0).astype(np.float32)
    net = HammingNetwork(protos, labels=etiquetas)
    return net, etiquetas, protos

def make_network_from_csv_kmeans(base_path):
    patrones, etiquetas = extract_prototypes_as_dict(base_path, k_per_class=3, method="kmeans", random_state=0)
    protos = np.stack([patrones[e] for e in etiquetas], axis=0).astype(np.float32)
    net = HammingNetwork(protos, labels=etiquetas)
    return net, etiquetas, protos   

def _grid_coords(N):
    lin = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(lin, -lin)
    return X, Y

def _polygon_mask(N, vertices):
    X, Y = _grid_coords(N)
    x = X.ravel()
    y = Y.ravel()
    verts = np.array(vertices, dtype=float)
    n = len(verts)
    inside = np.zeros_like(x, dtype=bool)
    x0 = verts[:, 0]; y0 = verts[:, 1]
    x1 = np.roll(x0, -1); y1 = np.roll(y0, -1)

    for i in range(n):
        y_min = np.minimum(y0[i], y1[i])
        y_max = np.maximum(y0[i], y1[i])
        cond = (y >= y_min) & (y < y_max)
        denom = (y1[i] - y0[i])
        denom = denom if abs(denom) > 1e-12 else 1e-12
        x_int = x0[i] + (x1[i] - x0[i]) * (y - y0[i]) / denom
        inside ^= cond & (x <= x_int)
    return inside.reshape(N, N)

def shape_circle(N, radius=0.7, filled=True, thickness=0.12):
    X, Y = _grid_coords(N)
    R = np.sqrt(X**2 + Y**2)
    if filled:
        mask = R <= radius
    else:
        mask = (R <= radius) & (R >= max(0.0, radius - thickness))
    return mask.astype(np.uint8)

def shape_square(N, size=1.2, filled=True, thickness=0.12):
    size = np.clip(size, 0.0, 2.0)
    half = size / 2.0
    X, Y = _grid_coords(N)
    if filled:
        mask = (np.abs(X) <= half) & (np.abs(Y) <= half)
    else:
        inner = half - thickness
        mask = (np.abs(X) <= half) & (np.abs(Y) <= half) & ~((np.abs(X) < inner) & (np.abs(Y) < inner))
    return mask.astype(np.uint8)

def shape_triangle(N, filled=True, thickness=0.12):
    angles = np.array([-pi/2, -pi/2 + 2*pi/3, -pi/2 + 4*pi/3])
    r = 0.95
    verts = [(r*np.cos(a), r*np.sin(a)) for a in angles]
    mask = _polygon_mask(N, verts)
    if not filled:
        shrink = 0.12
        verts_in = [(v[0]*(1-shrink), v[1]*(1-shrink)) for v in verts]
        inner = _polygon_mask(N, verts_in)
        mask = mask & ~inner
    return mask.astype(np.uint8)

def shape_star(N, points=5, filled=True, inner_ratio=0.45, outer_ratio=0.95, rotation=-pi/2, thickness=0.1):
    verts = []
    for k in range(points*2):
        angle = rotation + k * pi / points
        r = outer_ratio if k % 2 == 0 else inner_ratio
        verts.append((r*np.cos(angle), r*np.sin(angle)))
    mask = _polygon_mask(N, verts)
    if not filled:
        inner_ratio2 = max(0.0, inner_ratio - thickness)
        outer_ratio2 = max(0.0, outer_ratio - thickness)
        verts_in = []
        for k in range(points*2):
            angle = rotation + k * pi / points
            r = outer_ratio2 if k % 2 == 0 else inner_ratio2
            verts_in.append((r*np.cos(angle), r*np.sin(angle)))
        inner = _polygon_mask(N, verts_in)
        mask = mask & ~inner
    return mask.astype(np.uint8)

def shape_heart(N, filled=True, scale=1.0, thickness=0.12):
    X, Y = _grid_coords(N)
    Xs = X * 1.2 * scale
    Ys = Y * 1.2 * scale
    val = (Xs**2 + Ys**2 - 1)**3 - (Xs**2) * (Ys**3)
    mask = val <= 0
    if not filled:
        g = np.gradient(val)
        grad_mag = np.sqrt(g[0]**2 + g[1]**2) + 1e-9
        dist_like = np.abs(val) / grad_mag
        mask = (dist_like <= thickness)
    return mask.astype(np.uint8)

def to_bipolar(arr):
    return np.where(arr > 0, 1.0, -1.0).astype(np.float32)

def from_bipolar(arr):
    return (arr >= 0).astype(np.uint8)

class HammingNetwork:
    def __init__(self, prototypes, labels=None, epsilon=None):
        P = np.asarray(prototypes, dtype=np.float32)
        if P.ndim != 2:
            raise ValueError("prototypes must be 2D [M, D]")
        self.P = P
        self.M, self.D = P.shape
        self.labels = labels if labels is not None else [str(i) for i in range(self.M)]
        if len(self.labels) != self.M:
            raise ValueError("labels length must match number of prototypes")
        self.epsilon = epsilon if epsilon is not None else 1.0 / (10.0 * self.M)
    # calculate hamming similarity
    def _similarity(self, x_bipolar):
        return self.P @ x_bipolar
    # maxnet implementation to find the winner
    def _maxnet(self, s, max_iters=1000, tol=1e-6):
        y = np.maximum(s.copy(), 0.0)
        eps = self.epsilon
        for _ in range(max_iters):
            total = np.sum(y)
            if total <= 0:
                break
            y_new = (1 - eps) * y - eps * (total - y)
            y_new = np.maximum(y_new, 0.0)
            if np.allclose(y_new, y, atol=tol, rtol=0):
                y = y_new
                break
            y = y_new
            if np.count_nonzero(y > 0) == 1:
                break
        return y

    def predict(self, x_binary, use_maxnet=True, return_scores=False):
        x = np.asarray(x_binary)
        if x.ndim == 2:
            x = x.ravel()
        if x.ndim != 1:
            raise ValueError("x_binary must be 1D or 2D array")
        if x.size != self.D:
            raise ValueError(f"Input dimension {x.size} does not match prototype dimension {self.D}")
        x_b = to_bipolar(x.astype(np.uint8))
        s = self._similarity(x_b)
        if use_maxnet:
            y = self._maxnet(s)
            idx = int(np.argmax(y))
        else:
            idx = int(np.argmax(s))
        if return_scores:
            return self.labels[idx], s
        return self.labels[idx]

def build_prototypes(N):
    shapes = {
        "circulo": shape_circle(N, radius=0.72, filled=True),
        "cuadrado": shape_square(N, size=1.35, filled=True),
        "estrella": shape_star(N, points=5, filled=True, inner_ratio=0.38, outer_ratio=0.92),
        "triangulo": shape_triangle(N, filled=True),
        "corazon": shape_heart(N, filled=True, scale=1.0),
    }
    labels = list(shapes.keys())
    protos = np.stack([shapes[k].astype(np.uint8).ravel() for k in labels], axis=0)
    return labels, protos

def ascii_show(arr):
    chars = np.where(arr > 0, "█", "·")
    lines = ["".join(row) for row in chars]
    return "\n".join(lines)

def make_network(N):
    labels, protos_bin = build_prototypes(N)
    protos_bip = to_bipolar(protos_bin)
    net = HammingNetwork(protos_bip, labels=labels)
    return net, labels, protos_bin.reshape(len(labels), N, N)

def add_noise(x_bin, flip_prob=0.05, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    noise = rng.random(x_bin.shape) < flip_prob
    return (x_bin ^ noise.astype(np.uint8)).astype(np.uint8)

if __name__ == "__main__":
    # Demo
    N = 25
    net, labels, protos = make_network(N)
    print("Clases:", labels)
    for name, proto in zip(labels, protos):
        pred, scores = net.predict(proto, return_scores=True)
        print(f"Proto {name:9s} -> pred={pred:9s} | scores={np.round(scores,2)}")
    # Noisy test
    import numpy as np
    rng = np.random.default_rng(0)
    for name, proto in zip(labels, protos):
        noisy = add_noise(proto, flip_prob=0.08, rng=rng)
        pred, scores = net.predict(noisy, return_scores=True)
        print(f"Noisy {name:9s} -> pred={pred:9s} | max score={np.max(scores):.2f}")