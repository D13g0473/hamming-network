import numpy as np
import hamming_shapes as hs
import sys



# 1) Crear red para tamaño N

N                   = int(input("ingrese dimension de la tabla:"))
noise               = float(input("ingrese nivel de ruido (0 a 1): "))
figura              = int(input("ingrese figura prototipo:"))
net, labels, protos = hs.make_network(N)  # protos: [5, N, N] en binario (0/1)

x = protos[figura]
# 3) Probar con ruido
x_noisy = hs.add_noise(x, noise)
# prediccion normal
print("prediccion con ruido:") 
print(net.predict(x_noisy))                       # -> 'circulo'
print("prediccion con ruido y rotacion:")
# prediccion con rotacion
rotada = np.rot90(x_noisy, k=2)
print(net.predict(rotada))              # -> 'circulo' (si no, probar con --no-maxnet)

# pred, scores = net.predict(x_noisy, return_scores=True)


# print(pred)           # clase predicha
# print(scores)         # similitudes (dot product) vs cada prototipo

# 4) Ver un patrón en ASCII
print(hs.ascii_show(x_noisy))
print(hs.ascii_show(rotada))