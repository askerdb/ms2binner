import bin
import numpy as np
import nimfa
import sys
import matplotlib
# matplotlib.use('Agg') #for plotting w/out GUI on server
import matplotlib.pyplot as plt

mgf_data = sys.path[0] + "agp3k.mgf"
bin_size = 0.01
rank = 30

input_data, bins, scan_names = bin.bin_mgf(mgf_data, verbose = True, bin_size = bin_size)
nmf_model = nimfa.Nmf(input_data, rank=rank)
model = nmf_model()

W = model.fit.W
W_norm = (W/W.max()).toarray()

plt.scatter(np.repeat(bins, W_norm.shape[1]), W_norm.flat)
plt.show()