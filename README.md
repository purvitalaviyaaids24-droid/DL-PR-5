# 🎬 IMDB Movie Review Sentiment Analysis Using Deep Learning

## 📌 Project Overview

This project focuses on **Sentiment Analysis of IMDB Movie Reviews using Deep Learning techniques**. The objective is to classify movie reviews as either positive or negative based on their textual content.

The project explores different neural network architectures, including Artificial Neural Networks (ANN), Simple Recurrent Neural Networks (SimpleRNN), Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU).

It also demonstrates how recurrent neural networks process sequential data, how they learn contextual information, and how different architectures can be compared using performance metrics.

## 🎯 Objectives

- Perform text preprocessing and cleaning on movie reviews.
- Convert textual data into numerical sequences using tokenization.
- Understand the difference between sequence-based models and traditional neural networks.
- Implement and train ANN, SimpleRNN, LSTM, and GRU models.
- Understand Recurrent Neural Network (RNN) architecture and its working mechanism.
- Explore Backpropagation Through Time (BPTT).
- Understand vanishing and exploding gradient problems.
- Compare LSTM and GRU architectures.
- Evaluate model performance using classification metrics.
- Visualize training performance and compare model behavior.

## 📂 Dataset

**Dataset:** IMDB Movie Reviews Dataset

The dataset contains movie reviews labeled according to their sentiment.

| Feature | Description |
|---|---|
| `review` | Textual content of a movie review |
| `sentiment` | Sentiment label: positive or negative |

### Target Labels

- **Positive:** The review expresses a positive opinion.
- **Negative:** The review expresses a negative opinion.

The sentiment labels are converted into numerical values:

- Positive → `1`
- Negative → `0`

**Dataset Source:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Download the dataset and place `IMDB Dataset.csv` in the appropriate project directory.

## 🛠️ Technologies and Libraries

The project uses the following technologies and Python libraries:

- **Python** – Programming language
- **Pandas** – Data loading and manipulation
- **NumPy** – Numerical computations
- **Matplotlib** – Data visualization and training plots
- **Scikit-learn** – Dataset splitting and model evaluation
- **TensorFlow** – Deep learning framework
- **Keras** – Neural network implementation and training
- **Jupyter Notebook** – Interactive development environment

## 🧹 Data Preprocessing

The following preprocessing steps are performed before model training:

1. **Data Loading:** Load the IMDB dataset using Pandas.
2. **Duplicate Removal:** Remove duplicate reviews to reduce the risk of data leakage.
3. **Text Cleaning:** Convert reviews to lowercase, remove HTML tags, remove non-alphabetic characters, and normalize whitespace.
4. **Sentiment Encoding:** Convert positive and negative sentiment labels into numerical values.
5. **Train-Test Split:** Split the dataset into training and testing sets using an 80:20 ratio.
6. **Tokenization:** Convert words into numerical tokens using the Keras Tokenizer.
7. **Sequence Conversion:** Represent each review as a sequence of integer token IDs.
8. **Padding and Truncation:** Standardize sequence lengths to prepare the data for neural network models.

The vocabulary size is configured to use up to 10,000 words, with an out-of-vocabulary token for unseen words.

The maximum sequence length is selected using the 95th percentile of review lengths.

Stratified splitting helps preserve the original proportion of positive and negative reviews in both datasets.

## 🧠 Deep Learning Models

### 1. Artificial Neural Network (ANN)

An ANN model is implemented using an embedding layer, a flattening layer, a fully connected hidden layer, and a sigmoid output layer.

**Architecture:**
- Embedding Layer
- Flatten Layer
- Dense Layer with ReLU activation
- Output Layer with Sigmoid activation

The ANN serves as a baseline model for binary sentiment classification.

### 2. Simple Recurrent Neural Network (SimpleRNN)

A SimpleRNN processes sequential inputs while maintaining a hidden state that carries information from previous time steps.

The project demonstrates:

- Recurrent hidden-state calculations.
- Shared weights across time steps.
- Manual RNN calculations compared with Keras outputs.
- Word-sequence processing for sentiment classification.
- Training and validation performance.

### 3. Long Short-Term Memory (LSTM)

LSTM is a specialized recurrent neural network designed to learn long-term dependencies and reduce the vanishing gradient problem.

The project explores the four main LSTM components:

- **Forget Gate:** Determines which information should be removed from the cell state.
- **Input Gate:** Controls the information to be stored.
- **Candidate Cell State:** Generates potential new memory.
- **Output Gate:** Controls the information passed to the hidden state.

Additional experiments include:

- Manual LSTM gate calculations.
- LSTM parameter-count calculations.
- Gradient analysis.
- Stacked LSTM architecture.
- Sentiment classification and model evaluation.

### 4. Gated Recurrent Unit (GRU)

GRU is a recurrent neural network architecture that uses gating mechanisms to control information flow.

Its two main gates are:

- **Update Gate:** Controls how much previous information is retained.
- **Reset Gate:** Controls how much previous information is used when calculating new information.

The project explores GRU training, parameter counts, training time, and classification performance.

GRU and LSTM are compared to understand their differences in architecture, computational requirements, and predictive performance.

## 📚 Additional Concepts Covered

The notebook also covers several important deep learning concepts.

### RNN Input-Output Architectures

- One-to-One
- One-to-Many
- Many-to-One
- Many-to-Many

### Backpropagation Through Time (BPTT)

BPTT is explained as the method used to propagate gradients backward through the time steps of a recurrent neural network during training.

### Vanishing and Exploding Gradients

The project explores how gradients can become too small or too large during recurrent training.

It also demonstrates gradient clipping as a technique for controlling exploding gradients.

### Embedding Layer

The embedding layer converts integer word IDs into dense numerical vectors that can be learned during model training.

### Model Comparison

The project compares recurrent architectures using model performance, parameter counts, and training time.

## 📊 Model Evaluation

The models are evaluated using several performance metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correct predictions |
| Precision | Measures the proportion of positive predictions that are correct |
| Recall | Measures how many actual positive examples are identified |
| F1-Score | Balances precision and recall |
| Confusion Matrix | Shows correct and incorrect classifications |
| ROC-AUC | Measures the model's ability to distinguish between classes |
| Training Time | Measures the time required to train a model |
| Parameter Count | Indicates the number of trainable model parameters |

The notebook also includes visualizations of training accuracy, validation accuracy, training loss, validation loss, gradient magnitudes, and ROC curves.

These evaluations help identify the strengths and limitations of different neural network architectures.

## ⚙️ Installation and Setup

### Prerequisites

Ensure that Python and Jupyter Notebook are installed on your system.

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

Replace the placeholders with your actual GitHub repository URL and project directory name.

### Step 2: Install Dependencies

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow jupyter
```

### Step 3: Add the Dataset

Download the IMDB dataset and place the CSV file in your project directory.

Your dataset filename should be:

```text
IMDB Dataset.csv
```

Update the file path in the notebook if your dataset is stored in another location.

### Step 4: Launch Jupyter Notebook

Run the following command:

```bash
jupyter notebook
```

Open the project notebook and execute its cells in the appropriate order.

## 📁 Project Structure

A recommended repository structure is:

```text
IMDB-Sentiment-Analysis/
│
├── Code-5(1).ipynb
├── IMDB Dataset.csv
├── README.md
└── requirements.txt
```

**Note:** The dataset is shown in this structure for convenience. If it is not included in your repository, download it separately using the dataset source provided above.

## 🔍 Expected Outcomes

After running the notebook, the project is designed to demonstrate:

- How raw movie reviews can be transformed into numerical sequences.
- How neural networks can classify text into positive and negative sentiment categories.
- How SimpleRNN, LSTM, and GRU process sequential information.
- How recurrent architectures differ in their gating mechanisms and memory capabilities.
- How training performance and classification metrics can be used to compare models.
- How gradient behavior affects recurrent neural network training.

Actual accuracy, ROC-AUC scores, training times, and other results depend on the dataset, preprocessing, model configuration, and training environment. These values should be reported after executing the notebook successfully.

## ⚠️ Limitations

- Sentiment classification depends on the quality and diversity of the training data.
- Basic text cleaning may remove useful information, including punctuation and other non-alphabetic characters.
- Models may struggle with sarcasm, complex negation, and subtle emotional expressions.
- SimpleRNN may have difficulty learning dependencies across long sequences.
- LSTM and GRU can require significant computational resources for long reviews.
- Model performance may vary with vocabulary size, sequence length, hyperparameters, and training duration.

## 🚀 Future Improvements

Potential improvements include:

- Using pretrained word embeddings.
- Comparing additional architectures, such as Bidirectional LSTM and Bidirectional GRU.
- Applying hyperparameter tuning.
- Implementing dropout and other regularization techniques.
- Saving the best-performing trained model for future predictions.
- Developing a simple web application for interactive sentiment prediction.
- Adding more comprehensive model comparisons using precision, recall, F1-score, and ROC-AUC.

## 🏁 Conclusion

This project provides a practical introduction to **sentiment analysis using deep learning**.

By implementing ANN, SimpleRNN, LSTM, and GRU architectures, it explores how neural networks learn from textual data and how recurrent models process sequential information.

The experiments involving gradient behavior, gating mechanisms, parameter counts, and evaluation metrics provide a foundation for understanding and comparing different deep learning approaches to natural language processing.

The project can serve as a starting point for further experiments in sentiment analysis and other sequence-based machine learning applications.

---

**Project Type:** Deep Learning / Natural Language Processing (NLP)  
**Task:** Binary Sentiment Classification  
**Dataset:** IMDB Movie Reviews  
**Framework:** TensorFlow / Keras  
**Language:** Python

Author 
PURVI TALAVIYA
