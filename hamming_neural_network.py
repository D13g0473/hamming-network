import numpy as np
import hamming_shapes as hs
import sys

# 1) Crear red para tamaño N
N = int(sys.argv[2]) 
net, labels, protos = hs.make_network(N)  # protos: [5, N, N] en binario (0/1)

# 2) Clasificar una matriz binaria NxN (0/1)
x = protos[int(sys.argv[1])]                     # por ejemplo, el círculo “limpio”
# 3) Probar con ruido
x_noisy = hs.add_noise(x, flip_prob=0.10)
# prediccion normal
print("prediccion con ruido:") 
print(net.predict(x_noisy))                       # -> 'circulo'
print("prediccion con ruido y rotacion:")
# prediccion con rotacion
rotada = np.rot90(x_noisy, k=4)
print(net.predict(rotada))              # -> 'circulo' (si no, probar con --no-maxnet)

# pred, scores = net.predict(x_noisy, return_scores=True)


# print(pred)           # clase predicha
# print(scores)         # similitudes (dot product) vs cada prototipo

# 4) Ver un patrón en ASCII
print(hs.ascii_show(x_noisy))
print(hs.ascii_show(rotada))