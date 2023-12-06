import matplotlib.pyplot as plt
import numpy as np

# This file is for plotting results
# (after we get them, we can just kind of hard-code them here.)
# I was thinking of testing sparsities 1, 0.5, 0.25, 0.125, ... 1/(2^7)
# since the sparsities only really change (get worse) for extreme sparsities
sparsities = [1.0 / (2 ** i) for i in range(9)]
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
results['LeNet_5']['MNIST']['Random'] = [0.9906, 0.9906, 0.9878, 0.9793, 0.9545, 0.8288, 0.2164, 0.1135, 0.1135]
results['LeNet_5']['MNIST']['SNIP']   = [0.9912, 0.9904, 0.9898, 0.9892, 0.9840 ,0.9768, 0.8706, 0.1135, 0.1135]

results['LeNet_300_100'] = {}
results['LeNet_300_100']['MNIST'] = {}
results['LeNet_300_100']['MNIST']['None']   = [0.9835] * num_sparsities
results['LeNet_300_100']['MNIST']['Random'] = [0.9824, 0.9827, 0.9803, 0.9766, 0.9632, 0.9383, 0.5814, 0.2952, 0.114]
results['LeNet_300_100']['MNIST']['SNIP']   = [0.9835, 0.9831, 0.9817, 0.9822, 0.9749, 0.9707, 0.9574, 0.9197, 0.7129]

results['ResNet18'] = {}
results['ResNet18']['MNIST'] = {} 
results['ResNet18']['MNIST']['None']   = [0.9942] * num_sparsities
results['ResNet18']['MNIST']['Random'] = [0.9945, 0.9939, 0.9934, 0.9921, 0.9902, 0.9865, 0.9737, 0.9452, 0.8542]
results['ResNet18']['MNIST']['SNIP']   = [0.9942, 0.993, 0.9931, 0.9939, 0.9933, 0.9942, 0.9926, 0.9937, 0.8946]

results_names = [
	('LeNet_5', 'MNIST'),
	('LeNet_300_100', 'MNIST'),
	('ResNet18', 'MNIST'),
]

# Generate plots
plots_rows = 3
plots_cols = 1
fig, axs = plt.subplots(plots_rows, plots_cols, figsize=(10,8))  # all subplots should have the sparsity as x-axis
axs = np.array(axs)  # small hack when num_models = num_datasets = 1
axs = axs.reshape(-1)

# Let's just define which pruning method maps to which color. 
model_color_map = {
	"None": "black",
	"Random": "purple",
	"SNIP": "blue",
}

i = 0  # tracks which plot we're currently doing
ax = None
for (model, dataset) in results_names:
	ax = axs[i]
	for method in results[model][dataset].keys():
		ax.title.set_text(f"{model} on {dataset}")
		if (method == "None"):
			ax.plot(np.arange(num_sparsities), results[model][dataset][method], color=model_color_map[method], label=method)
		else:
			ax.plot(np.arange(num_sparsities), results[model][dataset][method], marker='o', color=model_color_map[method], label=method)
		ax.set_xticks(np.arange(num_sparsities), [f"{s*100:.2f}" for s in sparsities])  # admittedly, this is kind of hacky.
		ax.set_xlim(0, len(sparsities) - 1)
		ax.set_xlabel("Sparsity (percent of weights remaining)")
		ax.set_ylim(0.85, 1)  # only really care about high-ish accuracies
		ax.set_ylabel("Test accuracy")
	i += 1
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
lgd = dict(zip(labels, lines))
fig.legend(lgd.values(), lgd.keys(),loc="lower center", ncol=len(lgd.values()), bbox_to_anchor=(0.5, -0.05))
fig.tight_layout()
plt.savefig("results.png", bbox_inches='tight') 