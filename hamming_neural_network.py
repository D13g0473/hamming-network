import numpy as np
import hamming_shapes as hs

# 1) Crear red para tamaño N
N = 25
net, labels, protos = hs.make_network(N)  # protos: [5, N, N] en binario (0/1)

# 2) Clasificar una matriz binaria NxN (0/1)
x = protos[0]                               # por ejemplo, el círculo “limpio”
print(net.predict(x))                       # -> 'circulo'

# 3) Probar con ruido
x_noisy = hs.add_noise(x, flip_prob=0.08)
pred, scores = net.predict(x_noisy, return_scores=True)
print(pred)           # clase predicha
print(scores)         # similitudes (dot product) vs cada prototipo

# 4) Ver un patrón en ASCII
print(hs.ascii_show(x))
