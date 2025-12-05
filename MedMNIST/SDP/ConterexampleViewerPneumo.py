from PIL import Image
import numpy as np
import re
import glob
import os
# Caminho para o arquivo
caminho_central = "./safety_benchmarks/counterexamples/PneumoniaMNIST"
lista_caminhos = glob.glob(os.path.join(caminho_central, "*.txt"))
#print('tamanho', len(dataset))
for caminho_arquivo in lista_caminhos:
# Lista onde os valores de pixel serão armazenados
    valores_pixels = []

# Leitura linha por linha e extração dos valores
    with open(caminho_arquivo, 'r') as f:
        for linha in f:
            linha = linha.strip()
            valor = re.search(r"\d\.\d*", linha).group()
            valores_pixels.append(float(valor.strip()))
    valores_pixels.pop()
    i = re.search(r"(?<=\D)\d+(?=\D)", caminho_arquivo).group()
# Converte os valores para imagem
    imagem_array = np.array(valores_pixels).reshape(28, 28)
    imagem_uint8 = (imagem_array * 255).astype(np.uint8)

# Cria e salva a imagem
    imagem = Image.fromarray(imagem_uint8, mode='L')
    imagem.save(f'example_images/PneumoniaMNIST/Counterexemples/imagem_reconstruida_{i}.png')