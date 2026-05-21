import matplotlib.pyplot as plt
import argparse

def draw_graph(mode, path, output_path, k, p, angle):
    
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
        print(path)
        with open(path) as f:
            snp = [float(line.strip()[:-1]) for line in f]

        x = [i for i in range(k)]

        fig, ax = plt.subplots(figsize=(10,5)) 
        ax.plot(x, snp)
        ax.set_title ("Robustez aplicando 'Salt and Pepper'")
        ax.set_xlabel(f"Quantidade de pixels perturbados com proporção {p}%")
        ax.set_ylabel("Porcentagem de propriedades seguras")

    if (mode == 'Rot'):
        with open(path) as f:
            rot = [float(line.strip()[:-1]) for line in f]

        #x = [0.5*i for i in range(2*int(angle))]
        x = [i for i in range (len(rot))] 
        print (x)
        print(len(rot))
        fig, ax = plt.subplots(figsize=(10,5)) 
        ax.plot(x, rot)
        ax.set_title ("Robustez Rotacionando a Imagem")
        ax.set_xlabel(f"Ângulo de rotação")
        ax.set_ylabel("Porcentagem de propriedades seguras")
    
    if (mode == '3Rot'):
        with open("results/outputs/BreastMNIST/resultadosbreastRot.txt") as f:
            rot1 = [float(line.strip()[:-1]) for line in f]
        with open("results/outputs/PneumoniaMNIST/resultadospneumomnist_Rot_05.txt") as f:
            rot2 = [float(line.strip()[:-1]) for line in f]
        with open("results/outputs/OCTMNIST/resultadosoctmnist_Rot.txt") as f:
            rot3 = [float(line.strip()[:-1]) for line in f]
        x = [0.5*i for i in range(2*int(angle))]
        fig, ax = plt.subplots(figsize=(10,5)) 
        ax.plot(x, rot1)
        ax.plot(x, rot2)
        ax.plot(x, rot3)
        ax.set_title ("Robustez Rotacionando a Imagem")
        ax.set_xlabel(f"Ângulo de rotação")
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
    parser.add_argument('--angle', type=float, default=45,
                        help='Angulo máximo da rotação')
                        
    args = parser.parse_args()

    draw_graph(args.mode, args.file_path, args.output_file_path, args.k, args.p, args.angle)
 
if __name__ == "__main__":
    main()