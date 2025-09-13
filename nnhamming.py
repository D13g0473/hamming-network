import numpy as np
import hamming_shapes as hs
import sys



# 1) Crear red para tamaño N

modelos = [
    "prototypes/base_media.npz",
    "prototypes/base_kmeans.npz",
    "prototypes/base_kmedoids.npz",
    "prototypes/base_topk.npz"
]

N                   = int(input("Modelo a con el que quiere trabajar valores admitidos del 0 - 3: "))
noise               = float(input("ingrese nivel de ruido (0 a 1): "))
figura              = int(input("ingrese figura prototipo : circulo -> 0, cuadrado -> 1 , estrella -> 2,  triangulo -> 3, corazon -> 4 : "))
rotations           = int(input("ingrese cantidad de rotaciones (0 a 3): "))
net, labels, protos = hs.make_network(N)

net2, labels2, protos2 = hs.load_network_from_file("prototypes/base_math.npz")
x = protos[figura]
# 3) Probar con ruido
x_noisy = hs.add_noise(x, noise)
# prediccion normal
print("prediccion con ruido:") 
print(net.predict(x_noisy))
print("prediccion con ruido y rotacion:")
# prediccion con rotacion
rotada = np.rot90(x_noisy, k=2)
print(net.predict(rotada))              # -> 'circulo' (si no, probar con --no-maxnet)

# pred, scores = net.predict(x_noisy, return_scores=True)


# print(pred)           # clase predicha
# print(scores)         # similitudes (dot product) vs cada prototipo

# 4) Ver un patrón en ASCII
# print(hs.ascii_show(x_noisy))
# print(hs.ascii_show(rotada))