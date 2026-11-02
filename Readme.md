# Physics constraints for molecular machine learning: QM7 → QM9

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThomasAko/physics-constraints-qm7-qm9/blob/main/Physics_Constraints_QM7_to_QM9.ipynb)

**Can building physics into a neural network's loss make it learn better?** This project follows that question through two datasets. Along the way it became a study of something more basic: how to tell whether a result can be trusted at all.

## Notebooks

| Notebook | What it covers |
|---|---|
| [`Physics_Constraints_QM7_to_QM9.ipynb`](Physics_Constraints_QM7_to_QM9.ipynb) | The main study: QM7 classification and regression, a virial-theorem penalty, diagnostic checks, then QM9 geometry relaxation with a NeurIPS 2025 energy loss |
| [`QM7_DeepChem_Diagnosis.ipynb`](QM7_DeepChem_Diagnosis.ipynb) | The earlier version, kept as a record: how a sanity check exposed a broken data pipeline |

## The story so far

**1. A broken data pipeline.** The first version loaded QM7 through DeepChem's `load_qm7`. Every model, from random forests to tuned neural networks, stalled near 60% accuracy on a high/low energy classification. A composition-only baseline exposed the cause. Atomization energy is close to extensive, so atom counts should predict it almost perfectly, yet they explained only 19% of the variance in DeepChem's labels, and some "molecules" contained nothing but hydrogen. The features and labels did not match.

**2. Fixing the data makes the task trivial.** Loading the original `qm7.mat` instead, the same models with the same settings jump from about 60% to 97% accuracy:

| Model | Accuracy, DeepChem data | Accuracy, original data |
|---|---|---|
| Random forest, Coulomb matrix | 0.632 | 0.978 |
| Neural network, tuned | 0.605 | 0.972 |
| Neural network, funnel | 0.621 | 0.971 |
| Logistic regression, atom counts | 0.632 | 0.974 |
| Random forest, atom counts | 0.655 | 0.974 |

Composition alone solves 97.4% of the task, and every remaining error sits a median of 5.8 kcal/mol from the threshold, against 162 kcal/mol for a typical molecule. Classification can no longer tell models apart, so the study switches to predicting the energy itself.

**3. Regression baselines.** On 717 test molecules:

| Model | MAE (kcal/mol) | RMSE (kcal/mol) |
|---|---|---|
| Mean predictor | 180.6 | 222.5 |
| Linear model, atom counts | 16.3 | 21.1 |
| Random forest, atom counts | 15.3 | 20.7 |
| Random forest, Coulomb matrix | 11.9 | 18.3 |
| Neural network, Coulomb matrix | *in progress* | |

Geometry helps: the random forest on the Coulomb matrix cuts the error of the atom-count model by 27%. A first neural network did worse than a five-coefficient linear model; the diagnosis (training settings carried over from classification) and its fix are in the notebook.

**4. The virial-theorem penalty** (2⟨T⟩ + ⟨V⟩ = 0) on two auxiliary output heads: *in progress.* The structural result is already clear. QM7 has no labels for ⟨T⟩ or ⟨V⟩, so the penalty can be satisfied exactly by the output heads alone, without the network learning any physics.

**5. QM9.** The energy loss of *Energy Loss Functions for Physical Systems* (NeurIPS 2025, [arXiv:2511.02087](https://arxiv.org/abs/2511.02087)) avoids that weakness: it is anchored to the data and cannot be met trivially. Part 2 tests it, plus two physics-motivated variations of my own, on QM9 geometry relaxation with an E(n)-equivariant graph network: *in progress.*

## Running it

Open the main notebook in Colab with a T4 GPU and run all cells. The first cell installs the dependencies in `requirements.txt`. QM7 is downloaded from [quantum-machine.org](http://quantum-machine.org/datasets/) at run time and is not stored in this repository.

* Part 1 (QM7) runs on CPU in under an hour.
* Part 2 (QM9) needs a GPU. The full study takes a few hours; it saves progress and resumes if the session drops.

## Author

Thomas Ako, PhD · [thomasako.github.io](https://thomasako.github.io)
