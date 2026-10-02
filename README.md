# 🛡️ AI-Based Online Fraud Detection System

An AI-powered online transaction fraud detection system that uses **Machine Learning, Python, and a graphical user interface** to analyze transaction data and identify potentially fraudulent activities.

The system evaluates transaction characteristics and provides a fraud detection result to help identify suspicious financial transactions.

---

## 🎯 Project Overview

Online financial transactions are increasing rapidly, making automated fraud detection an important application of machine learning.

This project implements a machine-learning-based fraud detection system that analyzes transaction-related parameters and predicts whether a transaction is potentially:

- ✅ Genuine
- ⚠️ Suspicious
- 🚨 Fraudulent

The project combines **Machine Learning with an interactive Python GUI** to provide an accessible fraud detection interface.

---

## ✨ Key Features

### 🤖 Machine Learning-Based Detection

The system uses a trained machine learning model to analyze transaction characteristics and generate a fraud prediction.

The model is trained using transaction-related features and can be used to classify new transactions.

---

### 💳 Transaction Analysis

The system can analyze transaction information such as:

- Transaction amount
- Transaction time
- Transaction frequency
- Location-related information

These features are processed before being passed to the machine learning model.

---

### 🚨 Fraud Classification

The system provides different transaction classifications:

| Result | Meaning |
|--------|---------|
| ✅ Genuine | Transaction appears normal |
| ⚠️ Suspicious | Transaction contains potentially unusual characteristics |
| 🚨 Fraud | Transaction matches patterns associated with fraudulent activity |

---

### 📊 Risk Information

The system can provide additional information related to the detected transaction, such as:

- Fraud detection result
- Risk level
- Fraud probability
- Transaction characteristics

This allows users to understand the output rather than receiving only a binary prediction.

---

### 🖥️ Graphical User Interface

The project includes a Python-based graphical interface that allows users to enter transaction information and receive a fraud detection result.

The interface is designed to make the machine learning model easier to use without requiring users to interact directly with Python code.

---

## 🧠 Machine Learning Workflow

The overall workflow of the system is:

```text
                 START
                   │
                   ▼
          Enter Transaction Data
                   │
                   ▼
          Data Preprocessing
                   │
                   ▼
          Feature Engineering
                   │
                   ▼
          Machine Learning Model
                   │
                   ▼
          Fraud Prediction
                   │
                   ▼
        ┌─────────────────────┐
        │  Analyze Prediction │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Genuine / Suspicious│
        │       / Fraud       │
        └──────────┬──────────┘
                   │
                   ▼
           Display Result
                   │
                   ▼
                  END
```

---

## 🔬 Feature Engineering

Feature engineering is used to transform raw transaction information into features that can be processed by the machine learning model.

Important features include:

### Transaction Amount

Represents the monetary value of the transaction.

Large or unusual transaction amounts can be useful indicators when combined with other transaction characteristics.

### Transaction Time

Represents when the transaction occurs.

Certain transaction patterns may be associated with unusual activity.

### Transaction Frequency

Represents how frequently transactions occur.

A high number of transactions within a short period can be useful when analyzing potentially suspicious behavior.

### Location

Location-related information can be incorporated into the transaction analysis to identify unusual transaction patterns.

---

## 🤖 Machine Learning Model

The project uses a **Random Forest Classifier** for fraud detection.

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to produce a prediction.

Advantages of using Random Forest include:

- Handles multiple input features
- Can model nonlinear relationships
- Works well with structured data
- Provides robust classification performance
- Supports class imbalance handling

The trained model is stored as:

```text
fraud_model.pkl
```

---

## 🏋️ Model Training

The model can be trained using:

```text
new_train_model.py
```

The training process includes:

1. Loading the transaction dataset
2. Data preprocessing
3. Feature engineering
4. Preparing input and target variables
5. Splitting the dataset
6. Training the Random Forest model
7. Evaluating the model
8. Saving the trained model

The trained model can then be used by the application for prediction.

---

## 🖥️ Application Files

### `ui.py`

Main graphical interface of the fraud detection application.

It handles:

- User input
- Transaction data collection
- Model loading
- Prediction
- Result display
- Risk information

---

### `diagnose.py`

Used for testing and diagnosing the fraud detection system.

It can help verify:

- Model availability
- Prediction behavior
- Input processing
- Detection results

---

### `new_train_model.py`

Contains the machine learning model training process.

It handles:

- Dataset loading
- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Model saving

---

### `fraud_model.pkl`

Serialized trained machine learning model.

The file contains the trained Random Forest model used for prediction.

> **Note:** The trained model file is excluded from this repository using `.gitignore`. A model-training or model-download procedure can be added for reproducibility.

---

### `requirements-gui.txt`

Contains the Python libraries required to run the graphical application.

Example dependencies include:

```text
customtkinter
matplotlib
numpy
pandas
scikit-learn
```

---

## 📁 Project Structure

```text
ai-fraud-detection/
│
├── .gitignore
├── diagnose.py
├── new_train_model.py
├── requirements-gui.txt
├── ui.py
├── fraud_model.pkl
└── README.md
```

> `fraud_model.pkl` is used by the application but is excluded from GitHub through `.gitignore`.

---

## 🛠️ Technologies Used

- **Python**
- **Machine Learning**
- **Scikit-learn**
- **Random Forest**
- **Pandas**
- **NumPy**
- **CustomTkinter**
- **Matplotlib**
- **Pickle**
- **Git & GitHub**

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yashagrawal0727-debug/ai-fraud-detection.git
```

### 2. Enter the project directory

```bash
cd ai-fraud-detection
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 📦 Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements-gui.txt
```

If the requirements file does not contain all required packages, install them manually:

```bash
pip install customtkinter matplotlib numpy pandas scikit-learn
```

---

## ▶️ Running the Application

After installing the dependencies and making the trained model available, run:

```bash
python ui.py
```

The graphical interface should open.

Enter the transaction details and submit them to receive the fraud detection result.

---

## 🧪 Testing

The project can be tested using different transaction scenarios.

### Example Genuine Transaction

```text
Amount:        500
Time:          14
Frequency:     2
Location:      Normal
```

Expected classification:

```text
Genuine
```

---

### Example Suspicious Transaction

```text
Amount:        5000
Time:          3
Frequency:     6
Location:      Unusual
```

Expected classification:

```text
Suspicious
```

---

### Example High-Risk Transaction

```text
Amount:        15000
Time:          2
Frequency:     10
Location:      High-Risk
```

Expected classification:

```text
Fraud
```

> Actual predictions depend on the trained model, preprocessing, and configured detection rules.

---

## 📈 System Architecture

```text
┌─────────────────────────────┐
│        User Interface       │
│          (ui.py)            │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     Transaction Input       │
│ Amount / Time / Frequency   │
│ Location                    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Data Preprocessing &      │
│    Feature Engineering      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Random Forest Classifier  │
│     fraud_model.pkl         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      Fraud Prediction       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Genuine / Suspicious / Fraud│
│       + Risk Information    │
└─────────────────────────────┘
```

---

## 🔐 Why Fraud Detection Matters

Online fraud can result in:

- Financial losses
- Unauthorized transactions
- Identity-related risks
- Account compromise
- Loss of customer trust

Automated fraud detection systems can help analyze large numbers of transactions and identify potentially unusual patterns for further investigation.

---

## ⚠️ Limitations

The system is a machine-learning-based project intended for educational and demonstration purposes.

Its predictions depend on:

- Training dataset quality
- Feature selection
- Model training
- Data preprocessing
- Input accuracy
- Detection rules

Therefore, predictions should not be treated as definitive proof that a real-world transaction is fraudulent.

---

## 🚀 Future Improvements

Possible improvements include:

- Real-time transaction monitoring
- Larger and more diverse datasets
- Advanced feature engineering
- XGBoost and other ensemble models
- Deep learning-based fraud detection
- Real-time fraud alerts
- Geographic fraud heatmaps
- Transaction history analysis
- User behavior profiling
- Database integration
- REST API deployment
- Cloud deployment
- Model performance dashboards
- Continuous model retraining

---

## 🎓 Educational Purpose

This project demonstrates how **machine learning can be applied to financial transaction analysis and fraud detection**.

It combines:

**Machine Learning + Data Processing + Python + GUI Development + Fraud Detection**

to create an end-to-end prototype for detecting potentially fraudulent online transactions.

---

## 👨‍💻 Author

**Yash Agrawal**

Computer Science Engineering Student

**Vishwakarma Institute of Technology (VIT Pune)**

---

## 📜 License

This project is developed for educational and academic purposes.
