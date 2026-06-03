# Romanized to Hindi Script Transliteration (Seq2Seq & Attention)

This repository contains the code, models, and Weights & Biases (W&B) training reports for a Sequence-to-Sequence (Seq2Seq) Deep Learning project. 

**Project Objective:** To build and fine-tune a recurrent neural network model capable of character-level transliteration, converting Romanized Hindi words (e.g., "daakhil") into their corresponding native Devanagari script (e.g., "दाखिल")[cite: 4].

## 📁 Repository Contents

* **[`hindi_transliteration.ipynb`](./hindi_transliteration.ipynb)**
  The core Google Colab notebook containing data preprocessing, model definitions (Vanilla & Attention Seq2Seq), W&B sweep configurations, model training, evaluation, and visualizations[cite: 4].
* **[`wandb_report.pdf`](./wandb_report.pdf)**
  The exported Weights & Biases report detailing the hyperparameter sweep, validation accuracy, and training loss curves.
* **[`wandb_report.tex`](./wandb_report.tex)**
  The LaTeX source file for the Weights & Biases report.

## 📊 Dataset
The model is trained and evaluated on the **Dakshina Hindi Dataset** (`dakshina_dataset_v1.0/hi`)[cite: 4]. 
* **Input (Source):** Romanized Hindi words (Vocab size: 30 characters)[cite: 4].
* **Output (Target):** Devanagari script words (Vocab size: 67 characters)[cite: 4].
* **Tokenization:** Character-level tokenization utilizing `<SOS>`, `<EOS>`, `<PAD>`, and `<UNK>` special tokens[cite: 4].

## 🧠 Model Architectures
This project implements and compares two distinct Encoder-Decoder architectures using PyTorch[cite: 4]:

1. **Vanilla Seq2Seq:**
   * A flexible architecture allowing the selection of **RNN, GRU, or LSTM** cells[cite: 4].
   * Includes embedding layers and dropout for regularization[cite: 4].
2. **Attention-Based Seq2Seq:**
   * An upgraded model incorporating an **Attention Mechanism** to calculate alignment weights between the encoder hidden states and the decoder, allowing the model to focus on specific characters during transliteration[cite: 4].

## 🎛️ Hyperparameter Tuning (W&B Sweeps)
A Bayesian sweep was executed via Weights & Biases to maximize `val_accuracy`[cite: 4]. The following hyperparameters were explored:
* **Cell Type:** RNN, GRU, LSTM[cite: 4]
* **Embedding Size:** 32, 64, 128[cite: 4]
* **Hidden Size:** 64, 128, 256[cite: 4]
* **Number of Layers:** 1, 2, 3[cite: 4]
* **Dropout:** 0.0, 0.2, 0.3[cite: 4]
* **Learning Rate:** 1e-3, 5e-4[cite: 4]

🏆 **Best Champion Model Configuration:**
* **Cell Type:** LSTM[cite: 4]
* **Embedding/Hidden Size:** 128 / 256[cite: 4]
* **Layers:** 3[cite: 4]
* **Dropout:** 0.3[cite: 4]
* **Learning Rate:** 0.001[cite: 4]

## 📈 Results & Analysis

### Performance
* **Test Set Exact Match Accuracy:** **28.14%**[cite: 4]
* Results were saved to `predictions_vanilla/test_predictions.csv`[cite: 4].

### Visualizations & Error Analysis
The notebook includes advanced evaluation outputs:
* **Character Confusion Analysis:** Tracked the most frequent character translation errors (e.g., predicting 'ि' instead of 'ी', or 'त' instead of 'ट')[cite: 4].
* **Attention Heatmaps:** A 3x3 Seaborn grid visualizing the attention weights across 9 test samples[cite: 4].
* **Connectivity Graphs:** Custom Matplotlib visualizations mapping the exact connections between the English source characters and the predicted Hindi script (rendered natively using the `NotoSansDevanagari-Regular.ttf` font)[cite: 4].

### Bonus Task
* **Text Generation:** The notebook also includes a brief demonstration of loading the Hugging Face `gpt2` model for basic text generation[cite: 4].

## 🛠️ Tools & Technologies Used
* **Framework:** PyTorch (`torch`, `torch.nn`)[cite: 4]
* **Data Processing:** Pandas, Numpy[cite: 4]
* **Experiment Tracking:** Weights & Biases (`wandb`)[cite: 4]
* **Visualization:** Matplotlib, Seaborn[cite: 4]
* **Environment:** Google Colab (CUDA GPU)[cite: 4]

## 🚀 How to Run
1. Clone this repository and open the `.ipynb` file in Google Colab or a local Jupyter environment.
2. Ensure you have a GPU enabled.
3. Run the installation cell to install `transformers`, `datasets`, `accelerate`, and `wandb`[cite: 4].
4. Provide your Weights & Biases API key when prompted to track your own training sweeps[cite: 4].
5. Run the notebook sequentially to download the Dakshina dataset, train the models, and generate the attention visualizations[cite: 4].

