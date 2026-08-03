# Heart-Disease-Prediction-Using-ANN

## 📌 Project Overview

This project uses an **Artificial Neural Network (ANN)** to predict the presence of heart disease based on patient medical information.

The model is trained on clinical features such as age, sex, chest pain type, cholesterol level, resting blood pressure, maximum heart rate, and other health indicators.

The trained ANN model is deployed using **Streamlit**, allowing users to enter patient information and receive a real-time heart disease prediction.

---

# 🎯 Objective

The main objective of this project is to:

* Build a binary classification model using Deep Learning (ANN)
* Predict whether a patient is likely to have heart disease
* Apply preprocessing techniques for numerical and categorical features
* Evaluate model performance using Accuracy and ROC-AUC Score
* Deploy the trained model using Streamlit

---

# 📂 Dataset Information

The dataset contains **918 patient records** with the following features:

| Feature        | Description                                   |
| -------------- | --------------------------------------------- |
| Age            | Patient age                                   |
| Sex            | Gender of patient                             |
| ChestPainType  | Type of chest pain                            |
| RestingBP      | Resting blood pressure                        |
| Cholesterol    | Cholesterol level                             |
| FastingBS      | Fasting blood sugar                           |
| RestingECG     | Resting electrocardiogram results             |
| MaxHR          | Maximum heart rate achieved                   |
| ExerciseAngina | Exercise induced angina                       |
| Oldpeak        | ST depression value                           |
| ST_Slope       | Slope of peak exercise ST segment             |
| HeartDisease   | Target variable (0 = No Disease, 1 = Disease) |

---

# 🧠 Model Architecture

The Artificial Neural Network architecture used:

```
Input Layer
     |
Dense Layer (16 neurons)
Activation: ReLU
     |
Dense Layer (8 neurons)
Activation: ReLU
     |
Output Layer (1 neuron)
Activation: Sigmoid
```

---

# ⚙️ Data Preprocessing

The following preprocessing steps were performed:

### Numerical Features

* Standard Scaling using `StandardScaler`

### Categorical Features

* One Hot Encoding using `OneHotEncoder`

### Pipeline

Implemented using:

```python
ColumnTransformer
```

The preprocessing pipeline was saved using:

```python
joblib.dump(preprocessor, "preprocessor.pkl")
```

---

# 🔥 ANN Model Training

The model was developed using:

* TensorFlow
* Keras

Compilation:

```python
optimizer = Adam

loss = Binary Crossentropy

metric = Accuracy
```

Training configuration:

```
Epochs: 100
Batch Size: 32
Validation Split: 20%
```

---

# 📊 Model Performance

The model achieved:

| Metric        | Score |
| ------------- | ----- |
| Accuracy      | 88%   |
| ROC-AUC Score | 0.937 |

### Why ROC-AUC?

ROC-AUC measures how well the model separates:

* Positive class (Heart Disease)
* Negative class (No Heart Disease)

A higher ROC-AUC indicates better classification ability.

---

# 🚀 Streamlit Deployment

The trained model is deployed using Streamlit.

Application workflow:

```
User Input
     |
     ↓
Preprocessing Pipeline
     |
     ↓
ANN Model
     |
     ↓
Prediction Probability
     |
     ↓
Heart Disease Result
```

---

# 🛠️ Technologies Used

## Programming Language

* Python

## Machine Learning / Deep Learning

* TensorFlow
* Keras
* Scikit-Learn

## Data Processing

* Pandas
* NumPy

## Model Deployment

* Streamlit

## Model Saving

* Joblib
* Keras Model (.keras)

---

# 📁 Project Structure

```
Heart_Disease_Prediction_Using_ANN/

│
├── app.py
│
├── heart_ann_model.keras
│
├── preprocessor.pkl
│
├── Heart_Disease_Prediction_ANN.ipynb
│
├── requirements.txt
│
└── README.md
```

---


# 📦 Required Libraries

```
tensorflow
keras
numpy
pandas
scikit-learn
streamlit
joblib
```

---

# 🔮 Future Improvements

* Increase dataset size
* Perform hyperparameter optimization
* Compare ANN with XGBoost, Random Forest, and Gradient Boosting
* Add Explainable AI (SHAP/LIME)
* Deploy using Docker
* Add cloud deployment

---

# 👨‍💻 Author

**Shekhar Mandal**

Data Scientist | Machine Learning Engineer | AI Enthusiast

GitHub:
https://github.com/Shekhar-Mandal08

LinkedIn:
https://linkedin.com/in/shekhar-mandal-aa02542b2/

---

# ⭐ If you find this project useful

Give this repository a ⭐ on GitHub!
