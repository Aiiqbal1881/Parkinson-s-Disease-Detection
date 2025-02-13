# Parkinson's Disease Detection

## Overview
This project focuses on detecting Parkinson's disease using machine learning techniques. The dataset is processed using Python and employs the Support Vector Machine (SVM) model for classification. StandardScaler from `sklearn.preprocessing` is used for feature scaling, and the dataset is split into training and testing sets using `train_test_split`. The model's performance is evaluated using `accuracy_score`.

## Dataset
The dataset consists of biomedical voice measurements from individuals, some of whom have been diagnosed with Parkinson's disease. It includes attributes such as frequency-based features, fundamental frequency variations, and noise-to-harmonics ratios.

## Technologies Used
- **Python**
- **Machine Learning**
- **Support Vector Machine (SVM)**
- **Scikit-Learn (`sklearn`)**
- **StandardScaler for Feature Scaling**
- **Train-Test Split for Data Splitting**
- **Accuracy Score for Model Evaluation**

## Installation
Ensure you have Python installed, then install the required libraries using:
```bash
pip install numpy pandas scikit-learn
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/parkinsons-disease-detection.git
   cd parkinsons-disease-detection
   ```
2. Run the script:
   ```bash
   python parkinsons_detection.py
   ```

## Model Implementation
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("parkinsons.csv")

# Selecting features and target
X = df.drop(columns=['name', 'status'])  # 'status' is the target variable
y = df['status']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluating Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
```

## Results
The model is trained using an SVM classifier and achieves high accuracy in detecting Parkinson's disease. The exact accuracy may vary depending on dataset variations.

## Contributing
Feel free to fork the repository, make improvements, and submit a pull request.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments
- Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Parkinsons)
- `scikit-learn` for machine learning algorithms

Ariz iqbal
