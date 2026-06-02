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

    if (mode == 'abs_rel_pneumo_oct'):
        with open("/home/stephano/snap/snapd-desktop-integration/current/Ana/NN-ECG-classification-Verifier/results/outputs/PneumoniaMNIST/resultadospneumomnist_abs.txt") as f:
            abs1 = [float(line.strip()[:-1]) for line in f]
        with open("/home/stephano/snap/snapd-desktop-integration/current/Ana/NN-ECG-classification-Verifier/results/outputs/OCTMNIST/resultadosoctmnist_abs.txt") as f:
            abs2 = [float(line.strip()[:-1]) for line in f]
        with open("/home/stephano/snap/snapd-desktop-integration/current/Ana/NN-ECG-classification-Verifier/results/outputs/PneumoniaMNIST/resultadospneumomnist_rel.txt") as f:
            rel1 = [float(line.strip()[:-1]) for line in f]
        with open("/home/stephano/snap/snapd-desktop-integration/current/Ana/NN-ECG-classification-Verifier/results/outputs/OCTMNIST/resultadosoctmnist_rel.txt") as f:
            rel2 = [float(line.strip()[:-1]) for line in f]
        abs1.extend([0.0] * (len(rel1) - len(abs1)))
        abs2.extend([0.0] * (len(rel1) - len(abs2)))
        rel2.extend([0.0] * (len(rel1) - len(rel2)))
        epsilon = [ i / 1000 for i in range(len(rel1))]
        fig, ax = plt.subplots(figsize=(10,5))
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Liberation Serif']
        ax.plot(epsilon, abs1, label="Robustez Absoluta - PneumoniaMNIST", ls="-", lw="2")
        ax.plot(epsilon, abs2, label="Robustez Absoluta - OCTMNIST", ls=":", lw="3")
        ax.plot(epsilon, rel1, label="Robustez Relativa - PneumoniaMNIST", ls="--", lw="2")
        ax.plot(epsilon, rel2, label="Robustez Relativa - OCTMNIST", ls="-.", lw="2")
        ax.set_title ("Robustez em Relação a Perturbações Locais", fontsize="16")
        ax.set_xlabel("Epsilon", fontsize="14")
        ax.set_ylabel("Porcentagem de Propriedades Seguras", fontsize="14")
        ax.legend(fontsize="16")

    if (mode == 'abs_rel'):
        with open("/home/stephano/snap/snapd-desktop-integration/current/Ana/NN-ECG-classification-Verifier/results/outputs/BreastMNIST/resultadosbreastabsupto60.txt") as f:
            abs = [float(line.strip()[:-1]) for line in f]
        with open("/home/stephano/snap/snapd-desktop-integration/current/Ana/NN-ECG-classification-Verifier/results/outputs/BreastMNIST/resultadosbreastRELall.txt") as f:
            rel = [float(line.strip()[:-1]) for line in f]
        abs.extend([0.0] * (len(rel) - len(abs)))
        epsilon = [ i / 1000 for i in range(len(rel))]
        fig, ax = plt.subplots(figsize=(10,5))
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Liberation Serif']
        ax.plot(epsilon, abs, label="Robustez Absoluta - BreastMNIST", ls="-", lw="2")
        ax.plot(epsilon, rel, label="Robustez Relativa - BreastMNIST", ls="--", lw="2")
        ax.set_title ("Robustez em Relação a Perturbações Locais - BreastMNIST", fontsize="16")
        ax.set_xlabel("Epsilon", fontsize="14")
        ax.set_ylabel("Porcentagem de Propriedades Seguras", fontsize="14")
        ax.legend(fontsize="12")

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
        ax.set_title ("Robustez Rotacionando a Imagem", fontsize="16")
        ax.set_xlabel(f"Ângulo de rotação", fontsize="14")
        ax.set_ylabel("Porcentagem de propriedades seguras", fontsize="14")
    
    if (mode == '3Rot'):
        with open("results/outputs/BreastMNIST/resultadosbreastRot.txt") as f:
            rot1 = [float(line.strip()[:-1]) for line in f]
        with open("results/outputs/PneumoniaMNIST/resultadospneumomnist_Rot_05.txt") as f:
            rot2 = [float(line.strip()[:-1]) for line in f]
        with open("results/outputs/OCTMNIST/resultadosoctmnist_Rot.txt") as f:
            rot3 = [float(line.strip()[:-1]) for line in f]
        x = [0.5*i for i in range(2*int(angle))]
        fig, ax = plt.subplots(figsize=(10,5)) 
        ax.plot(x, rot1, label="BreastMNIST", ls=":", lw="2")
        ax.plot(x, rot2, label="PneumoniaMNIST", ls="--", lw="2")
        ax.plot(x, rot3, label="OCTMNIST", ls="-", lw="2")
        ax.set_title ("Robustez Rotacionando a Imagem")
        ax.set_xlabel(f"Ângulo de rotação")
        ax.set_ylabel("Porcentagem de propriedades seguras")
        ax.legend(fontsize="12")
    print(output_path)
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