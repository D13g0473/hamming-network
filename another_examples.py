# We'll create a self-contained Python module for a Hamming neural network that
# recognizes shapes (circle, square, star, triangle, heart) on an N x N binary grid.
# The module includes:
# - Shape generators for a given N
# - A HammingNetwork class (bipolar coding + Maxnet competition)
# - A simple demo when run as a script
#
# The file will be saved at /mnt/data/hamming_shapes.py

from math import cos, sin, pi, sqrt
import numpy as np

def _grid_coords(N):
    """
    Returns two NxN arrays X, Y with coordinates normalized to [-1, 1] range,
    where (0,0) is at the center of the grid and each entry represents the
    pixel center coordinates.
    """
    # Centers from -1 to 1
    lin = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(lin, -lin)  # invert Y to have +Y upwards visually
    return X, Y

def _polygon_mask(N, vertices):
    """
    Ray-casting point-in-polygon for all grid points.
    vertices: list of (x, y) pairs in normalized coords [-1,1], closed polygon implied.
    Returns NxN boolean mask of points inside the polygon.
    """
    X, Y = _grid_coords(N)
    x = X.ravel()
    y = Y.ravel()
    verts = np.array(vertices, dtype=float)
    n = len(verts)
    inside = np.zeros_like(x, dtype=bool)

    # Ray casting: count crossings for each point
    x0 = verts[:, 0]
    y0 = verts[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)

    for i in range(n):
        # Check edges where y is between y0 and y1 (half-open to avoid double counting)
        y_min = np.minimum(y0[i], y1[i])
        y_max = np.maximum(y0[i], y1[i])
        cond = (y >= y_min) & (y < y_max)
        # X position where the horizontal ray at 'y' intersects edge (x0,y0)->(x1,y1)
        # Avoid division by zero with small epsilon
        denom = (y1[i] - y0[i])
        denom = denom if abs(denom) > 1e-12 else 1e-12
        x_int = x0[i] + (x1[i] - x0[i]) * (y - y0[i]) / denom
        # Count if the intersection is to the right of the point (x <= x_int)
        inside ^= cond & (x <= x_int)

    return inside.reshape(N, N)

def shape_circle(N, radius=0.7, filled=True, thickness=0.12):
    """
    Returns NxN binary image of a circle centered at (0,0).
    If filled=True, fill the disk. If filled=False, draw a ring with given thickness (normalized).
    """
    X, Y = _grid_coords(N)
    R = np.sqrt(X**2 + Y**2)
    if filled:
        mask = R <= radius
    else:
        mask = (R <= radius) & (R >= max(0.0, radius - thickness))
    return mask.astype(np.uint8)

def shape_square(N, size=1.2, filled=True, thickness=0.12):
    """
    Returns NxN binary image of an axis-aligned square centered at (0,0).
    'size' is the full side length in normalized units (clipped to 2.0). Default fills most of grid.
    """
    size = np.clip(size, 0.0, 2.0)
    half = size / 2.0
    X, Y = _grid_coords(N)
    if filled:
        mask = (np.abs(X) <= half) & (np.abs(Y) <= half)
    else:
        # Outline as border band of given thickness
        inner = half - thickness
        mask = (np.abs(X) <= half) & (np.abs(Y) <= half) & ~((np.abs(X) < inner) & (np.abs(Y) < inner))
    return mask.astype(np.uint8)

def shape_triangle(N, filled=True, thickness=0.12):
    """
    Equilateral triangle pointing up, centered roughly at (0,0), inscribed in the unit circle.
    """
    # Equilateral triangle: 3 vertices on a circle
    angles = np.array([-pi/2, -pi/2 + 2*pi/3, -pi/2 + 4*pi/3])
    r = 0.95  # slightly inset
    verts = [(r*cos(a), r*sin(a)) for a in angles]
    mask = _polygon_mask(N, verts)
    if not filled:
        # For outline, subtract an eroded (shrunk) version
        shrink = 0.12
        verts_in = [(v[0]*(1-shrink), v[1]*(1-shrink)) for v in verts]
        inner = _polygon_mask(N, verts_in)
        mask = mask & ~inner
    return mask.astype(np.uint8)

def shape_star(N, points=5, filled=True, inner_ratio=0.45, outer_ratio=0.95, rotation=-pi/2, thickness=0.1):
    """
    Regular star polygon with given number of points (default 5). Constructed by alternating
    outer and inner vertices on concentric circles. Rotation is in radians.
    """
    verts = []
    for k in range(points*2):
        angle = rotation + k * pi / points
        r = outer_ratio if k % 2 == 0 else inner_ratio
        verts.append((r*cos(angle), r*sin(angle)))
    mask = _polygon_mask(N, verts)
    if not filled:
        inner_ratio2 = max(0.0, inner_ratio - thickness)
        outer_ratio2 = max(0.0, outer_ratio - thickness)
        verts_in = []
        for k in range(points*2):
            angle = rotation + k * pi / points
            r = outer_ratio2 if k % 2 == 0 else inner_ratio2
            verts_in.append((r*cos(angle), r*sin(angle)))
        inner = _polygon_mask(N, verts_in)
        mask = mask & ~inner
    return mask.astype(np.uint8)

def shape_heart(N, filled=True, scale=1.0, thickness=0.12):
    """
    Uses the classic implicit heart curve: (x^2 + y^2 - 1)^3 - x^2 y^3 <= 0
    scaled to fit the grid nicely.
    """
    X, Y = _grid_coords(N)
    Xs = X * 1.2 * scale
    Ys = Y * 1.2 * scale
    val = (Xs**2 + Ys**2 - 1)**3 - (Xs**2) * (Ys**3)
    mask = val <= 0
    if not filled:
        # outline band approximation by thickness around zero level set
        # Normalize 'val' to roughly comparable scale
        g = np.gradient(val)
        grad_mag = np.sqrt(g[0]**2 + g[1]**2) + 1e-9
        # distance-like quantity
        dist_like = np.abs(val) / grad_mag
        mask = (dist_like <= thickness)
    return mask.astype(np.uint8)

def to_bipolar(arr):
    """Convert {0,1} to {-1,+1}."""
    return np.where(arr > 0, 1.0, -1.0).astype(np.float32)

def from_bipolar(arr):
    """Convert {-1,+1} to {0,1} (>=0 -> 1)."""
    return (arr >= 0).astype(np.uint8)

class HammingNetwork:
    """
    Two-layer Hamming network for pattern classification.
    Layer 1: correlation (dot product) with stored bipolar prototypes.
    Layer 2: MAXNET competition (lateral inhibition) to pick the winner.
    """
    def __init__(self, prototypes, labels=None, epsilon=None):
        """
        prototypes: array of shape [M, D] in bipolar {-1,+1}
        labels: list of length M with class names. If None, labels = [0..M-1]
        epsilon: inhibition factor for MAXNET (default: 1/(10*M))
        """
        P = np.asarray(prototypes, dtype=np.float32)
        if P.ndim != 2:
            raise ValueError("prototypes must be 2D [M, D]")
        self.P = P
        self.M, self.D = P.shape
        self.labels = labels if labels is not None else [str(i) for i in range(self.M)]
        if len(self.labels) != self.M:
            raise ValueError("labels length must match number of prototypes")
        self.epsilon = epsilon if epsilon is not None else 1.0 / (10.0 * self.M)

    def _similarity(self, x_bipolar):
        # Dot product similarity; higher means closer in Hamming sense
        return self.P @ x_bipolar  # shape [M]

    def _maxnet(self, s, max_iters=1000, tol=1e-6):
        """
        Classic MAXNET: iteratively apply lateral inhibition until only one positive remains
        or convergence is reached.
        s: initial scores (copy will be made)
        """
        y = np.maximum(s.copy(), 0.0)
        eps = self.epsilon
        for _ in range(max_iters):
            total = np.sum(y)
            if total <= 0:
                break
            # y_i <- y_i - eps * (sum_j y_j - y_i) = (1 - eps) y_i - eps * sum_{j!=i} y_j
            y_new = (1 - eps) * y - eps * (total - y)
            y_new = np.maximum(y_new, 0.0)
            if np.allclose(y_new, y, atol=tol, rtol=0):
                y = y_new
                break
            y = y_new
            # Early exit if single survivor
            if np.count_nonzero(y > 0) == 1:
                break
        return y

    def predict(self, x_binary, use_maxnet=True, return_scores=False):
        """
        x_binary: 1D or 2D binary array (0/1). If 2D (N x N), it will be flattened.
        Returns: label (and optionally raw scores)
        """
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
    """
    Generates filled prototypes for the 5 target shapes on an NxN grid.
    Returns (labels, prototypes_binary[N_shapes, N*N]).
    """
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
    """
    Quick ASCII visualization for debugging. 1 -> '█', 0 -> '·'
    """
    chars = np.where(arr > 0, "█", "·")
    lines = ["".join(row) for row in chars]
    return "\n".join(lines)

def make_network(N):
    labels, protos_bin = build_prototypes(N)
    protos_bip = to_bipolar(protos_bin)
    net = HammingNetwork(protos_bip, labels=labels)
    return net, labels, protos_bin.reshape(len(labels), N, N)

def add_noise(x_bin, flip_prob=0.05, rng=None):
    """
    Flip bits with independent probability flip_prob.
    """
    rng = np.random.default_rng(None if rng is None else rng)
    noise = rng.random(x_bin.shape) < flip_prob
    return (x_bin ^ noise.astype(np.uint8)).astype(np.uint8)

# Save this module to a file
code_contents = r'''# hamming_shapes.py
from math import cos, sin, pi, sqrt
import numpy as np

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

    def _similarity(self, x_bipolar):
        return self.P @ x_bipolar

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
        print(f"Noisy {name:9s} -> pred={pred:9s} | max score={np.max(scores):.2f}")'''

with open('/mnt/data/hamming_shapes.py', 'w', encoding='utf-8') as f:
    f.write(code_contents)

# Quick smoke test: build a net at N=25 and run a single prediction
from importlib import import_module, reload
spec = import_module('importlib.util')
import importlib.util, sys, types

# Load the module we just wrote
spec = importlib.util.spec_from_file_location("hamming_shapes", "/mnt/data/hamming_shapes.py")
hamming_shapes = importlib.util.module_from_spec(spec)
sys.modules["hamming_shapes"] = hamming_shapes
spec.loader.exec_module(hamming_shapes)

# Build network and test with the exact prototypes (should classify perfectly)
net, labels, protos = hamming_shapes.make_network(25)
results = [(name, net.predict(proto)) for name, proto in zip(labels, protos)]

results
