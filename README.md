# 🎬 Sentiment Analysis Using Bidirectional GRU (BiGRU)
BiGRU-Sentiment-Analysis/
│
├── train_model.py                 # Model training script
├── predict_sentiment.py           # Prediction script
│
├── IMDB Dataset.csv               # Dataset
│
├── sentiment_bigru.keras          # Trained BiGRU model
├── tokenizer.pkl                  # Saved tokenizer
├── label_encoder.pkl              # Saved label encoder
│
├── positive.png                   # Positive image
├── negative.png                   # Negative image
│
├── requirements.txt
├── README.md
│
└── screenshots/
      │
      ├── training.png
      ├── positive_prediction.png
      └── negative_prediction.png


## 📌 Project Overview

This project implements a **Sentiment Analysis** system using a **Bidirectional Gated Recurrent Unit (BiGRU)** neural network built with **TensorFlow/Keras**. The model is trained on the **IMDb 50K Movie Reviews** dataset to classify movie reviews as either **Positive** or **Negative**.

Unlike traditional machine learning approaches, the BiGRU processes text in both forward and backward directions, enabling it to capture context from both preceding and succeeding words. This improves the model's understanding of sentence semantics and results in more accurate sentiment classification.

The project includes a complete deep learning pipeline—from text preprocessing and tokenization to model training, evaluation, and real-time prediction using a saved model.

---

## 🚀 Features

* Text preprocessing using Regular Expressions (Regex)
* HTML tag and URL removal
* Lowercasing and text normalization
* Tokenization with Out-of-Vocabulary (OOV) handling
* Sequence padding for fixed-length inputs
* Label encoding of sentiment classes
* Train-test data splitting
* Bidirectional GRU (BiGRU) architecture
* Dropout regularization to reduce overfitting
* Early Stopping during training
* Model Checkpointing to save the best-performing model
* Model, tokenizer, and label encoder serialization
* Training and validation accuracy/loss visualization
* Real-time sentiment prediction using user input

---

## 🧠 Model Architecture

* Embedding Layer
* Bidirectional GRU Layer
* Dense Layer (ReLU Activation)
* Dropout Layer
* Output Layer (Sigmoid Activation)

---

## 📂 Dataset

**IMDb 50K Movie Reviews Dataset**

* 50,000 labeled movie reviews
* Binary sentiment classification
* Balanced dataset containing positive and negative reviews

Dataset Link:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Pickle
* Regular Expressions (re)

---

## 📁 Project Structure

```text
BiGRU-Sentiment-Analysis/
│
├── train_model.py
├── predict_sentiment.py
├── IMDB Dataset.csv
├── sentiment_bigru.keras
├── tokenizer.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md
```

---

## ⚙️ Training Pipeline

1. Load the IMDb dataset.
2. Clean and preprocess the text.
3. Tokenize reviews using a Keras Tokenizer.
4. Convert text into integer sequences.
5. Pad sequences to a fixed length.
6. Encode sentiment labels.
7. Split the dataset into training and testing sets.
8. Train a Bidirectional GRU model.
9. Monitor validation performance using EarlyStopping and ModelCheckpoint.
10. Save the trained model, tokenizer, and label encoder.

---

## 📊 Evaluation

The model is evaluated using:

* Training Accuracy
* Validation Accuracy
* Test Accuracy
* Training Loss
* Validation Loss

Accuracy and loss curves are generated to visualize the learning process and monitor overfitting.

---

## 💻 Prediction Pipeline

The prediction script:

* Loads the trained BiGRU model.
* Loads the tokenizer and label encoder.
* Accepts a movie review from the user.
* Applies the same preprocessing used during training.
* Converts the review into a padded sequence.
* Predicts the sentiment using the trained model.
* Displays the predicted sentiment along with the prediction probability.

---

## 📚 Deep Learning Concepts Demonstrated

* Natural Language Processing (NLP)
* Text Preprocessing
* Tokenization
* Out-of-Vocabulary (OOV) Tokens
* Sequence Padding
* Word Embeddings
* Recurrent Neural Networks (RNNs)
* Bidirectional Recurrent Neural Networks
* Gated Recurrent Units (GRUs)
* Binary Classification
* Dropout Regularization
* Early Stopping
* Model Checkpointing
* Model Serialization
* Neural Network Inference

---

## 🎯 Future Improvements

* Add a desktop GUI using Tkinter.
* Integrate Attention Mechanisms with BiGRU.
* Support batch prediction from text files.
* Add confusion matrix and classification report.
* Visualize review word clouds.
* Deploy the model as a web application using Flask or FastAPI.
* Containerize the project with Docker.

---

## 📖 Learning Outcomes

Through this project, I gained practical experience in:

* Building an end-to-end NLP pipeline.
* Designing and training Bidirectional GRU models.
* Working with sequence data in TensorFlow/Keras.
* Saving and loading trained deep learning models.
* Performing sentiment classification on real-world text data.
* Developing reusable prediction pipelines for deployment.

---

## ⭐ Acknowledgements

* IMDb 50K Movie Reviews Dataset
* TensorFlow/Keras
* Scikit-learn
* Kaggle
