import matplotlib.pyplot as plt
import argparse

def draw_graph(mode, path, output_path, k, p):
    
    if (mode == 'rel'):
        with open(path) as f:
            rel = [float(line.strip()[:-1]) for line in f]
        epsilon = [ i / 1000 for i in range(len(rel))]
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(epsilon, rel)
        ax.set_title ("Robustez com variação do epsilon relativa")
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Porcentagem de propriedades seguras")

    if (mode == 'abs'):
        with open(path) as f:
            abs = [float(line.strip()[:-1]) for line in f]
        epsilon = [ i / 1000 for i in range(len(abs))]       
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(epsilon, rel)
        ax.set_title ("Robustez com variação do epsilon absoluta")
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Porcentagem de propriedades seguras")

    if (mode == 'SnP'):
        with open(path) as f:
            snp = [float(line.strip()[:-1]) for line in f]

        x = [i for i in range(k)]

        fig, ax = plt.subplots(figsize=(10,5)) 
        ax.plot(x, snp)
        ax.set_title ("Robustez aplicando 'Salt and Pepper' - 4°Quadrante")
        ax.set_xlabel(f"Quantidade de pixels perturbados com proporção {p}%")
        ax.set_ylabel("Porcentagem de propriedades seguras")


    fig.savefig(output_path)

def main():
    parser = argparse.ArgumentParser(description='VNN spec generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file_path', type=str, default=None,
                        help='Caminho do arquivo para leitura dos resultados')
    parser.add_argument('--output_file_path', type=str, default=None,
                        help='Caminho do arquivo para salvar o gráfico .png')
    parser.add_argument('--mode', type=str, default=None,
                        help='Modo de operação')
    parser.add_argument('--k', type=int, default=10,
                        help='Quatidade de pixels perturbados')
    parser.add_argument('--p', type=str, default=50,
                        help='Proporção de pixels com valor 1')
                        
    args = parser.parse_args()

    draw_graph(args.mode, args.file_path, args.output_file_path, args.k, args.p)
 
if __name__ == "__main__":
    main()