# Physics constraints for molecular machine learning on QM7

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThomasAko/physics-constraints-qm7-qm9/blob/main/Physics_Constraints_QM7_to_QM9.ipynb)

**Can building physics into a neural network's loss make it learn better?** This project tests that question on QM7, 7,165 small organic molecules with DFT atomization energies. Along the way it became a study of something more basic: how to tell whether a result can be trusted at all.

## Notebooks

| Notebook | What it covers |
|---|---|
| [`Physics_Constraints_QM7_to_QM9.ipynb`](Physics_Constraints_QM7_to_QM9.ipynb) | The main study: data check, classification, regression baselines, a virial-theorem penalty and why it cannot help, conclusions and the QM9 outlook |
| [`QM7_DeepChem_Diagnosis.ipynb`](QM7_DeepChem_Diagnosis.ipynb) | The earlier version, kept as a record: how a sanity check exposed a broken data pipeline |

## Findings

**1. A broken data pipeline.** The first version loaded QM7 through DeepChem's `load_qm7`. Every model stalled near 60% accuracy on a high/low energy classification. A composition-only baseline exposed the cause: atomization energy is close to extensive, so atom counts should predict it almost perfectly, yet they explained only 19% of the variance in DeepChem's labels (R² = 0.992 on the original file), and some "molecules" contained nothing but hydrogen.

**2. Fixing the data makes classification trivial.** Loading the original `qm7.mat`, the same models with the same settings jump from about 60% to 97–98% accuracy:

| Model | Accuracy, DeepChem data | Accuracy, original data |
|---|---|---|
| Random forest, Coulomb matrix | 0.632 | 0.978 |
| Neural network, tuned | 0.605 | 0.971 |
| Neural network, funnel | 0.621 | 0.969 |
| Logistic regression, atom counts | 0.632 | 0.974 |
| Random forest, atom counts | 0.655 | 0.974 |

The remaining errors sit a median of 5.8 kcal/mol from the threshold, against 162 kcal/mol for a typical molecule, so the study switches to predicting the energy itself.

**3. Regression: training settings and representation matter.** On 717 test molecules:

| Model | MAE (kcal/mol) | RMSE (kcal/mol) |
|---|---|---|
| Mean predictor | 180.6 | 222.5 |
| Linear model, atom counts | 16.3 | 21.1 |
| Random forest, atom counts | 15.3 | 20.7 |
| Random forest, Coulomb matrix | 11.9 | 18.3 |
| Neural network, Coulomb matrix (dropout, fixed learning rate) | 21.5 | 29.3 |
| Neural network, Coulomb matrix (no dropout, LR schedule) | 15.2 | 19.7 |
| Random forest, sorted Coulomb matrix | **10.5** | **15.6** |
| Neural network, sorted Coulomb matrix | 12.7 | 16.6 |
| Neural network + virial penalty, sorted Coulomb matrix | 13.9 | 18.3 |

Settings carried over from classification left the network worse than a five-coefficient linear model. Removing dropout, adding a learning-rate schedule and sorting the Coulomb matrices brought it about 2.7 kcal/mol below the best atom-count model.

**4. The virial-theorem penalty** (2⟨T⟩ + ⟨V⟩ = 0) on two auxiliary output heads gives no measurable gain; across four runs it never beat the plain network (0.2–1.5 kcal/mol worse). It cannot help: QM7 has no labels for ⟨T⟩ or ⟨V⟩, so the penalty can be satisfied exactly by the output heads alone. The notebook demonstrates this by rewriting one head, which drives the residual to float32 rounding without touching the rest of the network.

**5. Next step: QM9.** A useful physics term must be anchored to data and impossible to satisfy trivially. The energy loss of *Energy Loss Functions for Physical Systems* (NeurIPS 2025, [arXiv:2511.02087](https://arxiv.org/abs/2511.02087)) meets both, and testing it on QM9 geometry relaxation is the planned follow-up.

## Running it

Open the main notebook in Colab with a T4 GPU and run all cells (about 30 minutes). QM7 is downloaded from [quantum-machine.org](http://quantum-machine.org/datasets/) at run time and is not stored in this repository. Random forests and data splits are seeded; neural-network results vary by about 1 kcal/mol between GPU runs, so the saved outputs are the reference run.

## Author

Thomas Ako, PhD · [thomasako.github.io](https://thomasako.github.io)
