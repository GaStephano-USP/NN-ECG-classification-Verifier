import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.ndimage import uniform_filter1d

def draw_graph(mode, path, output_path, k, p, angle, mm):
    
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
        with open("/home/stephano/snap/snapd-desktop-integration/current/Ana/NN-ECG-classification-Verifier/results/outputs/OCTMNIST/FC/REFEITO_resultadosoctmnist_Abs.txt") as f:
            abs_oct = [float(line.strip()[:-1]) for line in f]
        with open("/home/stephano/snap/snapd-desktop-integration/current/Ana/NN-ECG-classification-Verifier/results/outputs/PneumoniaMNIST/resultadospneumomnist_abs.txt") as f:
            abs_pneumo = [float(line.strip()[:-1]) for line in f]
        with open("/home/stephano/snap/snapd-desktop-integration/current/Ana/NN-ECG-classification-Verifier/results/outputs/OCTMNIST/FC/REFEITO_resultadosoctmnist_Rel.txt") as f:
            rel_oct = [float(line.strip()[:-1]) for line in f]
        with open("/home/stephano/snap/snapd-desktop-integration/current/Ana/NN-ECG-classification-Verifier/results/outputs/PneumoniaMNIST/resultadospneumomnist_rel.txt") as f:
            rel_pneumo = [float(line.strip()[:-1]) for line in f]
        abs_oct.extend([0.0] * (len(rel_pneumo) - len(abs_oct)))
        abs_pneumo.extend([0.0] * (len(rel_pneumo) - len(abs_pneumo)))
        rel_oct.extend([0.0] * (len(rel_pneumo) - len(rel_oct)))
        epsilon = [ i / 1000 for i in range(len(rel_pneumo))]
        fig, ax = plt.subplots(figsize=(10,5))
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Liberation Serif']
        ax.plot(epsilon, abs_pneumo, label="Robustez Absoluta - PneumoniaMNIST", ls="-", lw="2")
        ax.plot(epsilon, abs_oct, label="Robustez Absoluta - OCTMNIST", ls=":", lw="3")
        ax.plot(epsilon, rel_pneumo, label="Robustez Relativa - PneumoniaMNIST", ls="--", lw="2")
        ax.plot(epsilon, rel_oct, label="Robustez Relativa - OCTMNIST", ls="-.", lw="2")
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
        ax.legend(fontsize="12", loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))

    if (mode == 'SnP'):
        print(path)
        with open(path) as f:
            snp = [float(line.strip()[:-1]) for line in f]
        fig, ax = plt.subplots(figsize=(10,5)) 
        if mm != None:
            snp_arr = np.array(snp)
            snp_smooth = uniform_filter1d(snp_arr, size=mm, mode='nearest')
            print(snp_smooth)
            x = [i for i in range(len(snp_smooth))]
            ax.plot(x, snp_smooth)
            ax.set_ylabel(f"Porcentagem de propriedades seguras com média móvel {mm}")
        else:
            x = [i for i in range(k)]
            ax.plot(x, snp)
            ax.set_ylabel("Porcentagem de propriedades seguras")   
        ax.set_title ("Robustez aplicando 'Salt and Pepper'")
        ax.set_xlabel(f"Quantidade de pixels perturbados com proporção {p}%")

    if (mode == 'SnP_proporções'):
        print(path)
        with open("results/outputs/PneumoniaMNIST/CNN/resultadospneumomnist_SnP_max_0.txt") as f:
            snp_0 = [float(line.strip()[:-1]) for line in f]
        with open("results/outputs/PneumoniaMNIST/CNN/resultadospneumomnist_SnP_max_50.txt") as f:
            snp_50 = [float(line.strip()[:-1]) for line in f]
        with open("results/outputs/PneumoniaMNIST/CNN/resultadospneumomnist_SnP_max_100.txt") as f:
            snp_100 = [float(line.strip()[:-1]) for line in f]
        fig, ax = plt.subplots(figsize=(10,5)) 
        if mm != None:
            snp_arr1 = np.array(snp_0)
            snp_arr2 = np.array(snp_50)
            snp_arr3 = np.array(snp_100)
            snp_smooth_0 = uniform_filter1d(snp_arr1, size=mm, mode='nearest')
            snp_smooth_50 = uniform_filter1d(snp_arr2, size=mm, mode='nearest')
            snp_smooth_100 = uniform_filter1d(snp_arr3, size=mm, mode='nearest')            
            x = [i for i in range(len(snp_smooth_0))]
            ax.plot(x, snp_smooth_0, label="Proporção 0%", ls="--", lw="2")
            ax.plot(x, snp_smooth_50, label="Proporção 50%", ls="-", lw="2")
            ax.plot(x, snp_smooth_100, label="Proporção 100%", ls=":", lw="2")       
            ax.set_ylabel(f"Porcentagem de propriedades seguras com média móvel {mm}")
            ax.legend(fontsize="12", loc='best', bbox_to_anchor=(0.5, 0.5, 0.47, 0.45))
        else:
            x = [i for i in range(k)]
            ax.plot(x, snp)
            ax.set_ylabel("Porcentagem de propriedades seguras")   
        ax.set_title ("Robustez aplicando 'Salt and Pepper' PneumoniaMNIST CNN")
        ax.set_xlabel(f"Quantidade de pixels perturbados com diferentes proporções")
        
    if (mode == 'SnP_3'):
        print(path)
        with open("results/outputs/PneumoniaMNIST/FC/resultadospneumomnist_SnP_AllP_seed1.txt") as f:
            snp_pneumo = [float(line.strip()[:-1]) for line in f]
        with open("results/outputs/OCTMNIST/FC/REFEITO_resultadosoctmnist_SnP0.txt") as f:
            snp_oct = [float(line.strip()[:-1]) for line in f]
        with open("results/outputs/BreastMNIST/FC/resultadosbreastSnP_allP.txt") as f:
            snp_breast = [float(line.strip()[:-1]) for line in f]
        fig, ax = plt.subplots(figsize=(10,5)) 
        if mm != None:
            snp_arr1 = np.array(snp_pneumo)
            snp_arr2 = np.array(snp_oct)
            snp_arr3 = np.array(snp_breast)
            snp_smooth_pneumo = uniform_filter1d(snp_arr1, size=mm, mode='nearest')
            snp_smooth_oct = uniform_filter1d(snp_arr2, size=mm, mode='nearest')
            snp_smooth_breast = uniform_filter1d(snp_arr3, size=mm, mode='nearest')            
            print(snp_smooth_pneumo)
            print(snp_smooth_oct)
            print(snp_smooth_breast)
            x = [i for i in range(len(snp_smooth_pneumo))]
            ax.plot(x, snp_smooth_pneumo, label="PneumoniaMNIST", ls="--", lw="2")
            ax.plot(x, snp_smooth_oct, label="OCTMNIST", ls="-", lw="2")
            ax.plot(x, snp_smooth_breast, label="BreastMNIST", ls=":", lw="2")       
            ax.set_ylabel(f"Porcentagem de propriedades seguras com média móvel {mm}")
            ax.legend(fontsize="12", loc='best', bbox_to_anchor=(0.5, 0.5, 0.47, 0.45))
        else:
            x = [i for i in range(k)]
            ax.plot(x, snp)
            ax.set_ylabel("Porcentagem de propriedades seguras")   
        ax.set_title ("Robustez aplicando 'Salt and Pepper' FC")
        ax.set_xlabel(f"Quantidade de pixels perturbados com proporção 0%")
        

    if (mode == 'SnP_seeds'):
        print(path)
        with open("results/outputs/PneumoniaMNIST/CNN/resultadospneumomnist_SnP_max_0.txt") as f:
            snp_1 = [float(line.strip()[:-1]) for line in f]
        with open("results/outputs/PneumoniaMNIST/CNN/resultadospneumomnistSnP0_seed2.txt") as f:
            snp_2 = [float(line.strip()[:-1]) for line in f]
       
        fig, ax = plt.subplots(figsize=(10,5)) 
        if mm != None:
            snp_arr1 = np.array(snp_1)
            snp_arr2 = np.array(snp_2)
            snp_1 = uniform_filter1d(snp_arr1, size=mm, mode='nearest')
            snp_2 = uniform_filter1d(snp_arr2, size=mm, mode='nearest')
            x = [i for i in range(len(snp_1))]
            ax.plot(x, snp_1, label="Seed 1", ls="--", lw="2")
            ax.plot(x, snp_2, label="Seed 2", ls="-", lw="2")

            ax.set_ylabel(f"Porcentagem de propriedades seguras com média móvel {mm}")
            ax.legend(fontsize="12", loc='best', bbox_to_anchor=(0.5, 0.5, 0.47, 0.45))
        else:
            x = [i for i in range(k)]
            ax.plot(x, snp)
            ax.set_ylabel("Porcentagem de propriedades seguras")   
        ax.set_title ("Robustez aplicando 'Salt and Pepper' com diferentes Seeds - BreastMNIST")
        ax.set_xlabel(f"Quantidade de pixels perturbados com proporção 0%")



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
        with open("results/outputs/BreastMNIST/CNN/resultadosbreast_Rot.txt") as f:
            rot1 = [float(line.strip()[:-1]) for line in f]
        with open("results/outputs/PneumoniaMNIST/CNN/resultadospneumomnist_Rot_max.txt") as f:
            rot2 = [float(line.strip()[:-1]) for line in f]
        with open("results/outputs/OCTMNIST/CNN/REFEITO_resultadosoctmnist_Rot.txt") as f:
            rot3 = [float(line.strip()[:-1]) for line in f]
        x1 = [0.5*i for i in range(2*int(angle))]
        x2 = [4*i for i in range (int(angle/4))]
        x = [i for i in range(len(rot1))]
        x = np.array(x)
        x = 4*x
        fig, ax = plt.subplots(figsize=(10,5)) 
        ax.plot(x, rot1, label="BreastMNIST", ls=":", lw="2")
        ax.plot(x, rot2, label="PneumoniaMNIST", ls="--", lw="2")
        ax.plot(x, rot3, label="OCTMNIST", ls="-", lw="2")
        ax.set_title ("Robustez Rotacionando a Imagem")
        ax.set_xlabel(f"Ângulo de rotação")
        ax.set_ylabel("Porcentagem de propriedades seguras")
        ax.legend(fontsize="12", loc='best', bbox_to_anchor=(0.5, 0.5, 0.5, 0.))
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
    parser.add_argument('--mm', type=int, default=None,
                        help='Média Móvel')
                        
    args = parser.parse_args()

    draw_graph(args.mode, args.file_path, args.output_file_path, args.k, args.p, args.angle, args.mm)
 
if __name__ == "__main__":
    main()