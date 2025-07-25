# Inspired by https://github.com/wu-haoze/dist-shift-vnn-comp/blob/main/generate_properties.py
import onnxruntime
import numpy as np
import random
import argparse
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import os
from medmnist import PneumoniaMNIST
from medmnist import INFO

def getMuStd(priorN, image):
    session = onnxruntime.InferenceSession(priorN, None)
    input_name = session.get_inputs()[0].name
    output_name1 = session.get_outputs()[0].name
    output_name2 = session.get_outputs()[1].name
    mu, logVar = session.run([output_name1, output_name2], {input_name: image})
    std = np.exp(0.5*logVar)
    return mu[0], std[0]

def getPrediction(concatN, image):
    session = onnxruntime.InferenceSession(concatN, None)
    imageInput = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {imageInput: image})[0][0]
    top = np.argmax(output)
    output[top] = -10000.0
    second = np.argmax(output)
    return top, second

def runClassifier(classifier, image):
    session = onnxruntime.InferenceSession(classifier, None)
    imageInput = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    output = session.run([output_name], {imageInput: np.expand_dims(image, axis=0)})[0][0]
    top = np.argmax(output)
    output[top] = -10000.0
    second = np.argmax(output)
    return top, second

def runGenerator(generator, image, noise):
    session = onnxruntime.InferenceSession(generator, None)
    imageInput = session.get_inputs()[0].name
    noiseInput = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {imageInput: image,
                                         noiseInput: [noise]})[0][0]
    plt.imshow(output[0], cmap='gray')
    plt.show()
    return output


def main(args):
    priorN = "./onnx/mnist_prior.onnx"
    concatN = "./onnx/mnist_concat.onnx"
    generator = "./onnx/mnist_generator.onnx"
    delta = args.delta
    showImg = args.show_img
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = PneumoniaMNIST(split="test", download=True, transform=transform)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    IMAGE_SIZE = (1,1,28,28)
    TIMEOUT = 300
    NUM_BENCHMARKS = 3600 * 6 / TIMEOUT
    random.shuffle(indices)

    if not os.path.isdir("./vnnlib"):
        os.mkdir("./vnnlib")
    open('instances.csv', 'w')

    added_benchmarks = 0
    for index in indices:
        if added_benchmarks == NUM_BENCHMARKS:
            break

        image = (np.array(dataset[index][0]).reshape(IMAGE_SIZE) / 255).astype(np.float32)
        label = dataset[index][1]
        mu, std = getMuStd(priorN, image)

        if showImg:
            mu += delta * std
            runGenerator(generator, image, mu)
            exit(0)

        image = image.flatten()
        net_input = np.array([list(image.flatten()) + list(mu)])
        topLabel, secondLabel = getPrediction(concatN, net_input)
        #print("Index: {}, correct label: {}, top label: {}, runner-up label: {}, delta: {}".format(index,
        #                                                                                           label,
        #                                                                                           topLabel,
        #                                                                                           secondLabel,
        #                                                                                           delta))
        if topLabel != label:
            continue
        else:
            vnnlib_filename = f'./vnnlib/index{index}_delta{delta}.vnnlib'
            with open(vnnlib_filename,'w') as f:
                for i in range(792):
                    f.write(f'(declare-const X_{i} Real)\n')
                for i in range(1):
                    f.write(f'(declare-const Y_{i} Real)\n')
                for i in range(784):
                    f.write(f'(assert (<= X_{i} {image[i]}))\n')
                    f.write(f'(assert (>= X_{i} {image[i]}))\n')
                for i in range(8):
                    f.write(f'(assert (<= X_{i + 784} {mu[i] + delta * std[i]}))\n')
                    f.write(f'(assert (>= X_{i + 784} {mu[i] - delta * std[i]}))\n')
                f.write(f'(assert (>= Y_{secondLabel} Y_{topLabel}))\n')

            with open('instances.csv','a') as f:
                f.write(f'{concatN},{vnnlib_filename},{TIMEOUT}\n')

            added_benchmarks += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Dump a single verification query')
    parser.add_argument('seed', type=int, help="random seed")
    parser.add_argument('--delta', type=float, default=0.13, help="Perturbation bound")
    parser.add_argument('--show-img', action='store_true')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)
