from PIL import Image
import numpy as np
import re
# Executar gerador de propriedades com epsilon=0 antes
for i in range(390):
    # Caminho para o arquivo
    caminho_arquivo = f'safety_benchmarks/benchmarks/FC_Net/vnnlib/OCTMNIST/Property_{i}.vnnlib'
    
    # Lista onde os valores de pixel serão armazenados
    valores_pixels = []
    
    # Leitura linha por linha e extração dos valores
    with open(caminho_arquivo, 'r') as f:
        for linha in f:
            if '>=' in linha:
               linha = linha.strip()
               valor = re.search(r"\d\.\d*", linha).group()
               valores_pixels.append(float(valor.strip()))
    # Converte os valores para imagem
    imagem_array = np.array(valores_pixels).reshape(28, 28)
    imagem_uint8 = (imagem_array * 255).astype(np.uint8)
    
    # Cria e salva a imagem
    imagem = Image.fromarray(imagem_uint8, mode='L')
    imagem.save(f'example_images/OCTMNIST/imagem_reconstruida_original_{i}.png')