import matplotlib.pyplot as plt

#dados com mode = 'rel'
with open("../../resultados_rel.txt") as f:
    rel = [int(line.strip()[:-1]) for line in f]
print (rel)

#dados com mode = 'abs'
with open("../../resultados_abs.txt") as f:
    abs = [int(line.strip()[:-1]) for line in f]
print (abs)

epsilon = [ i / 1000 for i in range(len(list))]


fig, ax = plt.subplots(1, 2, figsize=(10,5))

    