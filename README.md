# DeepLearning_LogiGAN

🧩 Step 1. Understanding What Each Phase Does

Phase 1 — Baseline Logical Classifier
	Goal: Train a baseline DistilBERT classifier on logical vs non-logical sentences.
Pipeline:
		1. Load pre-trained distilbert-base-uncased.
		2. Fine-tune it on a dataset of logical statements identified via heuristics (like “if”, “because”, etc.).
	•	Train using a standard classification loss (CrossEntropyLoss).
	•	Evaluate using accuracy (and possibly loss) on validation set.
	•	Output: Baseline model accuracy = your reference point.

So Phase 1 = plain supervised fine-tuning.

⸻

Phase 2 — Adversarial Fine-tuning with Hard Negative Mining
	•	Goal: Make the model more robust by exposing it to perturbed logical sentences.
	•	New components:
	•	AdversarialGenerator() → creates perturbed (slightly altered or misleading) versions of logical sentences.
	•	HardNegativeMiner() → identifies “difficult” examples the model is uncertain about, and adds them back into training.
	•	Pipeline:
	1.	Train initial model on dataset (same as Phase 1).
	2.	Use generator to produce adversarial versions.
	3.	Fine-tune the model again on original + adversarial examples.
	4.	Every few epochs, mine “hard negatives” and retrain.
	•	Output: Model learns logical robustness and improves generalization.

So Phase 2 = adversarial augmentation + iterative hard-negative mining to increase reasoning consistency.

⸻

Phase 3 — Contrastive Logical Pre-training
	•	Goal: Encourage the model to learn relational similarity between logically equivalent sentences.
	•	New components:
	•	LogicalPairConstructor() → constructs positive and negative sentence pairs:
	•	Positive = logically equivalent (e.g., “If A then B” ↔ “When A, B”).
	•	Negative = unrelated or adversarial pairs.
	•	ContrastiveWrapper() → wraps the Phase 2 model and outputs sentence embeddings.
	•	InfoNCELoss() → pushes similar pairs together and dissimilar ones apart in embedding space.
	•	Pipeline:
	1.	Build a dataset of sentence pairs (from Phase 2 outputs).
	2.	Train with contrastive loss (instead of cross-entropy).
	3.	Optionally mine hard negatives using uncertainty from Phase 2.
	•	Output: A model that encodes logical relationships more coherently (better semantic alignment).

So Phase 3 = contrastive fine-tuning on logic-consistent sentence pairs.
