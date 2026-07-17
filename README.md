# Final Project 1 · Computer Vision — Trash Classification

> **Your task:** build an image classifier that looks at a photo of a piece of waste and sorts it into the correct recycling category.
>
> **Your metric:** classification **accuracy** on the held-out test images.

This README is your complete guide for the project. Read it top to bottom before writing any code.

> ### 🏆 The Golden Rule
>
> Hit a weird error? Not sure which route to pick? Need to brainstorm ideas for your angle? **Why not ask your DeepSeek / ChatGPT / Claude?** Treat an AI assistant like a tireless senior classmate: paste in the full error message, describe what you expected vs. what happened, ask it to explain concepts, or bounce project ideas off it. Asking early beats being stuck for hours.
>
> One catch: you must understand and be able to explain every line you submit — we will ask. Use AI to get unstuck and to learn, not to outsource the thinking.

---

## Table of Contents

1. [What Is This Project About?](#1-what-is-this-project-about)
2. [Set Up Your Environment First (Please Use Conda!)](#2-set-up-your-environment-first-please-use-conda)
3. [The Dataset](#3-the-dataset)
4. [The Task, Explained](#4-the-task-explained)
5. [Finding Your Own Angle](#5-finding-your-own-angle)

---

## 1. What Is This Project About?

Recycling only works if waste is sorted correctly — and in the real world, sorting is still mostly done by hand (or not at all). Modern recycling facilities are starting to use **computer vision** to automate this: a camera looks at an item on a conveyor belt, and a model decides whether it's cardboard, glass, metal, paper, plastic, or general trash.

In this project you will build a small version of that system. Along the way you will need to figure out:

- How to load and explore an **image dataset**
- How to build an **image classifier** — there is more than one way to do this, and choosing your approach is part of the project
- How to fight **overfitting** on a small dataset
- How to **evaluate** a model honestly with a held-out test set
- How to **analyze errors** and iterate — which is what real ML work actually looks like

---

## 2. Set Up Your Environment First (Please Use Conda!)

**First decide where you will run your code — it changes what you need to install:**

- **On your own computer** (local development): you need conda. If you don't have it yet, install **[Miniconda](https://www.anaconda.com/docs/getting-started/installation)** (a lightweight Anaconda — all you need for this project), then follow the steps below. This section is **mandatory** for you.
- **Entirely in cloud notebooks** (Kaggle / Colab — see the [cloud GPU guide](#no-gpu-on-your-laptop-use-a-free-cloud-gpu) below): you can **skip conda**.

⚠️ **If you work locally: do not skip this section, and do not install packages into your `base` environment.**

### Why you must create a new conda environment

When you install Anaconda/Miniconda, you get a default environment called `base`. It is tempting to just `pip install` everything there. **Don't.** Here's what happens if you do:

- **You will break `base` sooner or later.** ML libraries (PyTorch, TensorFlow, numpy, scikit-learn…) have strict and sometimes conflicting version requirements. One bad install can leave `base` in a state where `conda` itself stops working correctly — and fixing a broken `base` usually means reinstalling Anaconda from scratch.
- **Your projects will fight each other.** Next semester's project might need `numpy 2.x` while this one needs `numpy 1.x`. If everything lives in `base`, upgrading for one project silently breaks the other.
- **Your work won't be reproducible.** When you hand in your project, we should be able to recreate your exact setup. A clean, dedicated environment (with an `environment.yml`) makes that possible. A messy `base` with two years of random packages does not.

Think of environments like lab benches: you get a clean bench per experiment, and if you spill something, you throw away that bench — not the whole lab.

### Setup steps

```bash
# 1. Create a fresh environment for this project (Python 3.10)
conda create -n trash-cls python=3.10

# 2. Activate it — do this EVERY time you work on the project
conda activate trash-cls

# 3. Install the packages you need (PyTorch route shown here)
pip install torch torchvision scikit-learn matplotlib numpy jupyter
```

### Sanity check

Your terminal prompt should show `(trash-cls)`, **not** `(base)`. Verify with:

```bash
conda env list        # the * should be next to trash-cls
python -c "import torch; print(torch.__version__)"
```

If you ever see `(base)` in your prompt while working on this project, stop and run `conda activate trash-cls`.

### No GPU on your laptop? Use a free cloud GPU

Neural-network training is much more pleasant with a GPU. Two free options, depending on your network situation:

- **If you can't access Google services (no VPN):** use **[Kaggle Notebooks](https://www.kaggle.com/code)** — upload your code as a notebook, turn on a free GPU under *Settings → Accelerator*, and train there. Big bonus: our dataset is already hosted on Kaggle, so you can attach it directly via *Add Input* without downloading or uploading anything. See the [Kaggle Notebooks docs](https://www.kaggle.com/docs/notebooks) for details.
- **If you have a VPN / can access Google:** use **[Google Colab](https://colab.research.google.com)** — free GPU under *Runtime → Change runtime type*. Colab notebooks can be **shared like a Google Doc**, which makes it easy to get help: send your share link to a teammate or PM and they can see your exact code, outputs, and error messages.

Either way, treat the cloud notebook as your *training machine*, not your storage: download the `.ipynb` regularly into your project folder — cloud sessions expire, and the project folder is what you submit.

---

## 3. The Dataset

**Source:** [Garbage Classification on Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)

This is the classic **6-class garbage dataset**. It contains roughly **2,500 labeled photos**, each showing a single piece of waste on a plain, light background, at **512 × 384** pixels.

⚠️ **Note:** the download is just six class folders — there is **no ready-made train/test split**. How to create one is something you'll need to think about yourself (see [Section 4](#4-the-task-explained)).

### The six classes

| Class | Approx. # images | Examples |
|-----------|:---:|----------|
| 📦 Cardboard | ~400 | boxes, packaging |
| 🍾 Glass | ~500 | bottles, jars |
| 🥫 Metal | ~410 | cans, foil |
| 📄 Paper | ~590 | newspapers, documents |
| 🧴 Plastic | ~480 | bottles, containers, bags |
| 🚮 Trash | ~140 | non-recyclable misc. items |

Exact counts may differ slightly — verify them yourself after downloading. That check is part of exploring your data.

### Explore before you build

Before writing any model code, spend real time exploring the structure of the dataset.

### Download

1. Create a free Kaggle account and download from the link above.
2. Unzip into `data/` inside your project folder.
3. **Do not include the images in your submission** — the `data/` folder stays on your machine.

> 🏆 **A Golden-Rule moment:** not sure how to download the dataset, or how to load a folder of images with their class labels in Python? Prompt your coding agent — e.g. *"How do I download the garbage-classification dataset from Kaggle and load the images with labels in PyTorch / Keras / scikit-learn?"* — and it will walk you through it, for your exact setup, faster than any tutorial.

---

## 4. The Task, Explained

Formally, this is **single-label image classification**:

- **Input:** one RGB photo containing one piece of waste.
- **Output:** exactly one label out of six — `cardboard`, `glass`, `metal`, `paper`, `plastic`, or `trash`.
- **Metric:** **accuracy** = (correct predictions) / (total predictions), measured on **test images your model has never seen during training**.

Whatever approach you choose, your pipeline has the same shape:

```
photo (512×384 RGB)
   → preprocess
   → model
   → 6 scores, one per class
   → predicted class = highest score
```

### The rules of honest evaluation

- **Split your data** into train / validation / test. The dataset comes as plain class folders with no split — deciding how to divide it is part of the project.
- **Train** only on the training set.
- **Tune** everything — model choice, hyperparameters, when to stop — using the **validation** set only.
- **Touch the test set exactly once**, at the very end, to report your final accuracy.

Peeking at the test set while developing — even "just to check" — is the ML equivalent of reading the answer key before the exam. Your final number becomes meaningless.

---

## 5. Finding Your Own Angle

A working classifier is the baseline. What makes your project *yours* is the question you choose to dig into beyond that — and no, we won't hand out a list of suggested directions. Picking your own is part of the work, and the best ones can't be assigned anyway.

Here's the secret: **you don't need to invent your angle — you need to notice it.** Build a first, unremarkable model, then pay attention:

- What *surprised* you when you explored the data?
- What does your model get *wrong* — and is there a pattern to it?
- Which of your design decisions felt *arbitrary*? Each one is an experiment waiting to happen.
- What would break if your model left the clean dataset and met the messy real world?

Anything that surprises, bothers, or confuses you is a candidate. Pick **one** and go deep.

Whatever you choose, a good angle:

1. **starts from something you actually observed** — in your data or your results, not in a tutorial;
2. **can be phrased as a question with a measurable answer**, not just "I'll try adding X";
3. **compares something** — with vs. without, before vs. after, method A vs. method B;
4. **ends with you explaining *why* the result came out the way it did**, not just reporting a number.

Depth beats breadth: one question pursued carefully is worth more than three touched superficially.

> 🏆 **A Golden-Rule footnote:** brainstorming with your AI assistant is encouraged — but feed it *your* observations ("my model keeps confusing these two classes, here's my confusion matrix — help me reason about why") rather than asking "give me a project idea." The first conversation sharpens **your** angle; the second produces the same project as everyone else who typed that prompt.

Good luck with developing your own project! 🚮✨