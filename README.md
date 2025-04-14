# cnn_pruning_cifar10

gpt prompt
I’m working on a CNN pruning comparison project using the CIFAR-10 dataset. I have a custom ResNet-56 model implemented in Keras and I’m comparing five pruning techniques on it. I want to train each pruned model from scratch and compare their performance with the baseline. I also have limited Colab units, so I need optimized code.

Here’s what I have done so far and need:

✅ Dataset:
	•	CIFAR-10 (already normalized and preprocessed)

✅ Baseline:
	•	ResNet-56 from scratch (custom 6n+2 Keras model)

✅ Goal:
	•	Train the baseline and 5 pruned variants of it on CIFAR-10
	•	Compare accuracy, training time, parameter sparsity, etc.

🧪 Pruning Methods:
	1.	Magnitude-Based Pruning (Global Unstructured) – DONE ✅
	2.	SNIP (Single-shot pruning using saliency scores)
	3.	Lottery Ticket Hypothesis (LTH)
	4.	L1-Norm Filter Pruning (Structured Pruning)
	5.	Random Pruning (as control baseline)

🔧 Requirements:
	•	Efficient training setup (early stopping, checkpoints)
	•	Save model weights, training logs, and evaluation metrics
	•	Code should be Colab- and RTX 4060–friendly
	•	Show side-by-side comparison plots (accuracy, loss, params)

Start from method 2 (SNIP) if resuming progress.
