import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()   # Create a figure containing a single Axes.
ax.plot([1,2,3,4], [1,4,6,2]) # Plot some data on the Axes.
#primeiro eh o eixo y
ax.set_title("Função teste") 
ax.set_xlabel("Eixo X")
ax.set_ylabel("Eixo Y")

plt.savefig("MedMNIST/BreastMNISTResNet18/Clara_teste/plot_apagar.png")
#plt.savefig("MedMNIST/BreastMNISTResNet18/Clara_teste/plot_apagar2.png")

#plt.show()
a=[1,2,3,4]
b=[1,2,1,1]
c=[x**2 for x in a] #lsit comprehension

fig, constx = plt.subplots()
constx.plot(c, a)
plt.savefig("MedMNIST/BreastMNISTResNet18/Clara_teste/constx.png")

x = np.linspace(0, 2) # (star, stop)
fig, funcao =  plt.subplots(figsize=(5, 5)) 
funcao.plot (x, x, color = 'blue', label = 'linear')
funcao.plot (x, x**2, color = 'green', label = 'quadratic')
funcao.plot (x, np.cos(x)*2, color = 'yellow', label = 'cosseno')
funcao.legend()
funcao.grid(True) 
funcao.annotate('funções', xy=(.85, 2.75))

plt.savefig("MedMNIST/BreastMNISTResNet18/Clara_teste/funcao.png")

fig, bs = plt.subplots(1, 3)
bs[0].plot (a, a)
bs[0].set_title("Função Linear a")
bs[1].plot (a, b)
bs[1].set_title("Função ab")
bs[2].plot (b, b)
bs[2].set_title("Função Linear b")
plt.savefig("MedMNIST/BreastMNISTResNet18/Clara_teste/3_gráficos.png")