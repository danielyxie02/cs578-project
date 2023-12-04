import matplotlib.pyplot as plt
import numpy as np

# This file is for plotting results
# (after we get them, we can just kind of hard-code them here.)
# I was thinking of testing sparsities 1, 0.5, 0.25, 0.125, ... 1/(2^7)
# since the sparsities only really change (get worse) for extreme sparsities
sparsities = [1.0 / (2 ** i) for i in range(7)]
num_sparsities = len(sparsities)

# Currently planning on doing the results in like a layered dictionary, basically 
# results[model name][dataset name][pruning method] will give you a list of results.
# Then, we each graph corresponds to some (model, dataset) pair, and will have a bunch of
# lines corresponding to each pruning method (so we can compare them.)

# Define your results here
results = {}
results['LeNet_5'] = {}
results['LeNet_5']['MNIST'] = {}
results['LeNet_5']['MNIST']['None']   = [0.9912] * num_sparsities  # no pruning => duplicate the 100% sparsity result as a reference horizontal line
results['LeNet_5']['MNIST']['Random'] = [0.9906, 0.9906, 0.9878, 0.9793, 0.9545, 0.8288, 0.2164,]
results['LeNet_5']['MNIST']['SNIP']   = [0.9912, 0.9904, 0.9898, 0.9892, 0.9840 ,0.9768, 0.8706,]

# Generic loop to generate plots of every result
# Change num_models, num_datasets as you will, and more subplots will be created
num_models = 1
num_datasets = 1
fig, axs = plt.subplots(num_models, num_datasets, figsize=(10,6))  # all subplots should have the sparsity as x-axis
axs = np.array(axs)  # small hack when num_models = num_datasets = 1
axs = axs.reshape(-1)

# Let's just define which pruning method maps to which color. 
model_color_map = {
	"None": "black",
	"Random": "purple",
	"SNIP": "blue",
}

i = 0  # tracks which plot we're currently doing
for model in results.keys():
	for dataset in results[model].keys():
		ax = axs[i]
		for method in results[model][dataset].keys():
			ax.title.set_text(f"{model} on {dataset}")
			ax.plot(np.arange(num_sparsities), results[model][dataset][method], color=model_color_map[method], label=method)
			ax.set_xticks(np.arange(num_sparsities), [f"{s:.2f}" for s in sparsities])  # admittedly, this is kind of hacky.
			ax.set_xlim(0, len(sparsities) - 1)
			ax.set_xlabel("Sparsity (fraction of weights remaining)")
			ax.set_ylim(0.85, 1)  # only really care about high-ish accuracies
			ax.set_ylabel("Test accuracy")
		i += 1
plt.figlegend()
plt.savefig("results.png") 