# DeepLearning_LogiGAN

Perfect — here’s a clean, professional README.md for your GitHub repo that explains your three LogiGAN experiments clearly and follows the standard structure used in research code repositories. It’s written to make your project look polished, reproducible, and publication-ready.

⸻


# 🧠 LogiGAN: Progressive Fine-Tuning for Logical Reasoning in Language Models

This repository contains three sequential experiments exploring how large language models can be fine-tuned to **understand and reason over logical relationships** in natural language.  
We start from a baseline classifier and progressively add adversarial and contrastive learning objectives to improve logical robustness and semantic alignment.

---

## 📘 Overview

| Phase | Objective | Core Idea |
|--------|------------|------------|
| **Phase 1** | Baseline Fine-Tuning | Train a DistilBERT classifier on logical vs. non-logical sentences. |
| **Phase 2** | Adversarial Enhancement | Introduce adversarial examples and hard-negative mining to increase robustness. |
| **Phase 3** | Contrastive Alignment | Train on logical sentence pairs using contrastive loss to improve logical equivalence understanding. |

Each phase builds on the previous one, forming a curriculum that moves from *keyword-based logic detection* → *robust reasoning under perturbation* → *semantic understanding of logical relationships*.

---

## 🧩 Project Structure

📂 LogiGAN/
├── 764_LOGIGAN_phase1.ipynb        # Baseline DistilBERT classifier
├── 764_LogiGAN_phase2.ipynb        # Adversarial fine-tuning + hard negative mining
├── 764_LogiGAN_phase3.ipynb        # Contrastive logical fine-tuning
├── data/                           # Dataset of logical/non-logical sentences
├── models/                         # Saved model checkpoints
├── results/                        # Evaluation outputs, logs, and plots
└── README.md                       # This file

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/LogiGAN.git
cd LogiGAN

2. Install Dependencies

pip install -r requirements.txt

Required packages include:
	•	torch, transformers, scikit-learn, numpy, matplotlib, tqdm

3. Dataset

The dataset consists of labeled logical and non-logical sentences.
You can use any binary-labeled dataset with fields like:

{
  "text": "If it rains, the ground gets wet.",
  "label": 1
}

For Phase 3, additional pairs are constructed for contrastive training:

{
  "sent1": "If A then B.",
  "sent2": "When A, B.",
  "label": 1
}


⸻

🚀 Experiments

🧮 Phase 1 — Baseline Logical Classifier

Goal: Establish a benchmark for logical reasoning using supervised fine-tuning.
	•	Model: DistilBERT-base-uncased
	•	Objective: Binary classification (CrossEntropyLoss)
	•	Output: Accuracy and loss on logical sentence detection.

Run 764_LOGIGAN_phase1.ipynb


⸻

⚔️ Phase 2 — Adversarial Fine-Tuning

Goal: Improve logical robustness using adversarial training.
	•	Introduces AdversarialGenerator() to produce perturbed sentences.
	•	Uses HardNegativeMiner() to retrain on difficult examples.
	•	Objective: Enhance model’s resilience to paraphrased or misleading logic.

Run 764_LogiGAN_phase2.ipynb


⸻

🧭 Phase 3 — Contrastive Logical Learning

Goal: Encode logical equivalence relationships in embedding space.
	•	Constructs sentence pairs with positive (equivalent) and negative (non-equivalent) relations.
	•	Uses InfoNCE loss for contrastive learning.
	•	Produces logically consistent embeddings with high similarity for equivalent pairs.

Run 764_LogiGAN_phase3.ipynb


⸻

📊 Evaluation

Each model is evaluated on the same held-out validation set for comparability.

Phase	Model	Validation Accuracy	Validation Loss
1	DistilBERT (baseline)	↑	↓
2	+ Adversarial Training	↑↑	↓↓
3	+ Contrastive Learning	↑↑↑	↓↓↓

1. Quantitative Evaluation
	•	Accuracy, precision, recall, F1-score
	•	Loss curves across epochs

2. Qualitative Evaluation
	•	Logical equivalence similarity (cosine similarity on sentence pairs)
	•	Adversarial robustness checks
	•	Embedding visualization via t-SNE or PCA

Example Evaluation Script

# Load trained models
phase1_model = torch.load('models/best_model_phase1.pt')
phase2_model = torch.load('models/best_logic_classifier.pt')
phase3_model = torch.load('models/contrastive_model_exp3.pt')

# Evaluate on the same test set
evaluate(phase1_model, val_loader)
evaluate(phase2_model, val_loader)
evaluate_contrastive(phase3_model, val_loader)


⸻

📈 Results & Interpretation

Phase 1: Understands logic keywords but struggles with paraphrases.
Phase 2: Learns robustness via adversarial exposure; handles perturbations better.
Phase 3: Embeddings capture logical equivalence; performs best on contrastive and similarity tasks.

⸻

🧠 Key Insights
	•	Logical reasoning in language models improves when fine-tuning is progressive and multi-objective.
	•	Adversarial and contrastive methods significantly enhance robustness and semantic coherence.
	•	Combining cross-entropy and contrastive learning can yield logic-aware embeddings useful for downstream reasoning tasks.

⸻

📜 Citation

If you use this code or build upon this work, please cite:

@misc{logigan2025,
  author = {Ipsita Singh},
  title = {LogiGAN: Progressive Fine-Tuning for Logical Reasoning in Language Models},
  year = {2025},
  url = {https://github.com/<your-username>/LogiGAN}
}


⸻

🧩 License

This project is released under the MIT License — free for research and educational use.

⸻

🙌 Acknowledgements

Inspired by advances in:
	•	Adversarial Fine-tuning for robust NLP models
	•	Contrastive Representation Learning for semantic alignment
	•	BERTology studies on model interpretability and reasoning

⸻


---

Would you like me to tailor this README’s **tone** (e.g. make it more “academic paper style” vs “developer-friendly”) before you commit it to GitHub? I can polish it either way.
