# **Sentiment Analysis of Audit Documents**

## **Project Overview**
The objective of this project is to predict the sentiment of audit documents using Natural Language Processing (NLP). Sentiment classification is divided into three categories:
- **Negative Sentiment (Label: 0)**
- **Neutral Sentiment (Label: 1)**
- **Positive Sentiment (Label: 2)**

---

## **Table of Contents**
1. [Introduction to the Data](#introduction-to-the-data)  
2. [Approach](#approach)  
   - [Data Ingestion](#data-ingestion)  
   - [Data Transformation](#data-transformation)  
   - [Model Training](#model-training)  
   - [Prediction Pipeline](#prediction-pipeline)  
   - [Flask Application](#flask-application)  
3. [GCP Deployment Link](#gcp-deployment-link)  

---

## **Introduction to the Data**
- **Independent Variable**: `sentence`  
  Represents the text data from audit documents.
- **Target Variable**: `label`  
  Indicates the sentiment of the document:
  - `0`: Negative Sentiment  
  - `1`: Neutral Sentiment  
  - `2`: Positive Sentiment  

---

## **Approach**

### **1. Data Ingestion**
- The dataset is loaded using the `load_dataset` function from Hugging Face Transformers.

---

### **2. Data Transformation**
- **Tokenization**:  
  The `BertTokenizer` is used to transform the `sentence` column into token sequences of length 512.
- **Input Tensors**:  
  The tokens are converted into a tensor tuple containing:
  - `input_ids`  
  - `attention_mask`  
- **Label Transformation**:  
  The labels are also converted into a tensor format to feed the data into the BERT model.
- **Shape Transformation**:  
  The data shape is transformed from a 3D tensor array to a 2D array with dimensions `(2, 1)`.

---

### **3. Model Training**
- **Model Initialization**:  
  A BERT model is initialized for the sentiment classification task.
- **Input Layers**:  
  Two input layers are added to handle `input_ids` and `attention_mask`.
- **Dense Layers**:  
  - A dense layer with ReLU activation is added to process BERT embeddings.  
  - The dense layer is connected to an output layer of 3 neurons with a softmax activation function for the three sentiment labels.

---

### **4. Prediction Pipeline**
- This pipeline processes new data for prediction by:
  - Tokenizing and transforming inputs.  
  - Loading the necessary pickle files for the trained model and tokenizer.  
  - Predicting the sentiment based on the input data.

---

### **5. Flask Application**
- A Flask-based web application is created to provide a user-friendly interface for sentiment prediction.  
- Users can input audit document text through the web application, and the model predicts the sentiment.

---

## **GCP Deployment Link**
The model is deployed on Google Cloud Platform (GCP) and can be accessed at:  
**[Add the deployment link here]**

