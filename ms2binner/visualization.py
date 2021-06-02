import bin
import numpy as np
import nimfa
import sys
import matplotlib
import matplotlib.patches as mpatches
# matplotlib.use('Agg') #for plotting w/out GUI on server
import matplotlib.pyplot as plt

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

mgf_data = sys.path[0] + "/agp3k.mgf"
bin_size = 0.01
rank = 10

input_data, bins, scan_names = bin.bin_mgf(mgf_data, verbose = True, bin_size = bin_size)
nmf_model = nimfa.Nmf(input_data, rank=rank)
model = nmf_model()

# W = model.fit.W
# W_norm = []
# for x in W:
#         W_norm.append(softmax(x.toarray()[0]))

# W_norm = np.array(W_norm)

H = model.fit.H
H_norm = []
for x in H:
        H_norm.append(softmax(x.toarray()[0]))

H_norm = np.array(H_norm)

colors = ['#ffd700', '#ff8c00', '#e81123', '#ec008c', '#68217a', "#00188f", '#00bcf2', '#04e912', '#d7a36a', '#000']
labels = ['Component 1', 'Component 2', 'Component 3', 'Component 4', 'Component 5', 'Component 6', 'Component 7', 'Component 8', 'Component 9', 'Component 10']
components = np.arange(1,11)
# repeated_colors = np.tile(colors, H_norm.shape[1])
repeated_colors = np.repeat(colors, H_norm.shape[1])
patches = []
for i in range(0, np.shape(colors)[0]):
    patches.append(mpatches.Patch(color=colors[i], label=labels[i]))

# plt.xlim((50,1200))
# plt.scatter(np.repeat(bins, W_norm.shape[1]), W_norm.flat, s=1, c=repeated_colors)
plt.scatter(np.repeat(components, H_norm.shape[1]), H_norm.flat, s=1, c=repeated_colors)
plt.xticks(np.arange(0,18))
plt.legend(handles=patches, loc='upper right')

plt.show()
