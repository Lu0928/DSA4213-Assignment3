# DSA4213-Assignment3
The code and outcome of DSA4213 Assignment3 is in the .ipnyfiles.If you want to check it,please download the Assignment3.ipny



# Assignment 3: Fine-tuning DistilBERT for Sentiment Analysis

This repository contains the code and documentation for Assignment 3, which explores fine-tuning pretrained Transformer models for a downstream task. The project focuses on sentiment analysis using the IMDb dataset.

## 📝 Table of Contents
* [Project Overview](#-project-overview)
* [Dataset](#-dataset)
* [Models and Fine-Tuning Strategies](#-models-and-fine-tuning-strategies)
* [Installation](#-installation)
* [Running the Experiments](#-running-the-experiments)
* [Experimental Setup](#-experimental-setup)
* [Results](#-results)
* [Key Takeaways and Limitations](#-key-takeaways-and-limitations)
* [Interactive Sentiment Prediction](#-interactive-sentiment-prediction)

## 📖 Project Overview

This project implements and compares two fine-tuning strategies for adapting a pretrained Transformer model to a text classification task. The chosen model is **DistilBERT**, and the task is **sentiment analysis** on the IMDb movie review dataset. The goal is to evaluate the performance and efficiency trade-offs between full fine-tuning and a parameter-efficient fine-tuning (PEFT) method, specifically **LoRA**.

## 📊 Dataset

The project uses the **IMDb dataset**, a well-known benchmark for binary sentiment classification[cite: 8]. It consists of 50,000 highly polarized movie reviews (25,000 for training and 25,000 for testing), which are labeled as either positive or negative.

**Motivation for Choice**:
* **Standard Benchmark**: It is a classic and widely used dataset for sentiment analysis, which makes it easy to compare results against established baselines.
* **Sufficient Size**: The dataset is large enough to demonstrate the effectiveness of fine-tuning without being computationally prohibitive.
* **Clear Task**: The binary classification task is straightforward, allowing the focus to remain on the fine-tuning techniques themselves.

## 🤖 Models and Fine-Tuning Strategies

The following model and strategies were implemented:

* **Pretrained Model**: `distilbert-base-uncased` was chosen as the base model. It's a smaller, faster, and lighter version of BERT, making it suitable for experiments with limited computing resources.

* **Fine-Tuning Strategies**:
    1. **Full Fine-Tuning**: This approach updates all the parameters of the pretrained model during training. It is resource-intensive but often yields the highest performance.
    2.  **LoRA (Low-Rank Adaptation)**: A parameter-efficient fine-tuning (PEFT) technique that freezes the pretrained model weights and injects trainable low-rank decomposition matrices into the layers of the Transformer. This dramatically reduces the number of trainable parameters, leading to faster training and lower memory usage.

## ⚙️ Installation

To set up the environment and install the required dependencies, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Lu0928/DSA4213-Assignment3.git]
    cd DSA4213-Assignment3
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Running the Experiments

You can reproduce all experiments with a single command. The script will automatically:
1.  Evaluate the baseline (zero-shot) performance of the pretrained model.
2.  Perform full fine-tuning for 3 epochs.
3.  Perform LoRA fine-tuning for 3 epochs.
4.  Print a summary of the results.
5.  Launch an interactive prediction prompt.

To run the script, execute:
```bash
python main.py
```
**Note**: The script requires a GPU (`cuda` device) for efficient training. The dataset will be downloaded and cached automatically by the `datasets` library.

## 🛠️ Experimental Setup

The key hyperparameters used for the training process are detailed below:

| Hyperparameter              | Full Fine-Tuning | LoRA Fine-Tuning |
| --------------------------- | ---------------- | ---------------- |
| `learning_rate`             | 2e-5             | 2e-5             |
| `per_device_train_batch_size`| 8                | 8                |
| `per_device_eval_batch_size` | 8                | 8                |
| `num_train_epochs`          | 3                | 3                |
| `weight_decay`              | 0.01             | 0.01             |
| **LoRA `r` (rank)** | N/A              | 4                |
| **LoRA `alpha`** | N/A              | 32               |

Evaluation is performed after each epoch, and the model with the best performance on the evaluation set is saved and used for the final report. The metrics used are **Accuracy** and weighted **F1 Score**.

## 📈 Results

Here is a summary of the final evaluation results on the test set.

**Note:** Please replace these numbers with the actual results from running your script.

| Experiment                      | Trainable Parameters | Accuracy | F1 Score |
| ------------------------------- | -------------------- | -------- | -------- |
| **Baseline (No Fine-Tuning)** | 0                    | 0.5305   | 0.5077   |
| **Full Fine-Tuning** | ~66.9 Million        | 0.9279   | 0.9279   |
| **LoRA Fine-Tuning** | ~7.0 Million (1.03%)  | 0.9054   | 0.9054   |

## 💡 Key Takeaways and Limitations

* **Effectiveness of Fine-Tuning**: Both fine-tuning methods dramatically outperformed the zero-shot baseline, which performed at chance level. This highlights the necessity of adapting pretrained models to specific downstream tasks.
* **Performance vs. Efficiency**: Full fine-tuning achieved the highest accuracy. However, LoRA achieved a highly competitive score while only training **~1.03%** of the total parameters. This demonstrates the power of PEFT methods in achieving excellent results with a fraction of the computational cost and memory, making them ideal for scenarios with limited resources.
* **Limitations**: The hyperparameter search was minimal. Further tuning of parameters like the learning rate, LoRA rank (`r`), and alpha could potentially close the small performance gap between LoRA and full fine-tuning. The experiment was also limited to a single dataset and model architecture.

## 💬 Interactive Sentiment Prediction

After training completes, the script loads the best-performing LoRA model and allows for interactive sentiment prediction on custom sentences. This provides a qualitative way to test the model's capabilities.

Example usage from the script:
```
- Sentence: "This movie was a complete waste of my time. The acting was terrible and the plot was boring."
  Prediction: Negative Sentiment

- Sentence: "I absolutely loved this film! The visuals were stunning and the story was heartwarming."
  Prediction: Positive Sentiment
```
