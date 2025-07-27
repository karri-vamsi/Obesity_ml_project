# 🧠 Obesity Level Prediction using Machine Learning and Deep Learning

This project predicts a person's **obesity level** using machine learning and deep learning models based on lifestyle and health-related features.

---

## 📂 Project Files

- `Obesity_ML_project.ipynb`: Complete Jupyter Notebook with EDA, preprocessing, training, evaluation
- `Obesity_Dataset.xlsx`: Original dataset (1610 samples, 15 features)
- `Requirement and explanation.pdf`: Project requirements and summary document

---

## 📌 Dataset Overview

- 📦 **Total Samples**: 1610
- 🔢 **Total Features**: 14 predictors + 1 target (Class)
- 🎯 **Target Variable**: `Class` (Obesity Level – 4 categories)
- 💡 **Goal**: Predict obesity level based on attributes like age, weight, height, food habits,     physical activity, etc.
- 🧪 **Task Type**: Multi-class Classification

---

## 🚀 Models Trained & Tuned

| Model               | Test Accuracy | Best CV Accuracy | Highlights                                    |
|--------------------|---------------|------------------|------------------------------------------------|
| ANN (Neural Net)   | 65.5%         | 67% (val acc)    | Used BatchNorm, Dropout, EarlyStopping         |
| KNN                | 83%           | 80.1%            | Tuned: `n_neighbors=4`, `metric='manhattan'`   |
| SVM (RBF Kernel)   | 82%           | 79.6%            | Tuned: `C=10`, `gamma='scale'`                 |
| Random Forest      | 87%           | 83.2%            | Tuned: `max_depth=15`, `n_estimators=150`      |

✅ **Best Model**: Random Forest Classifier

---

## 🔍 Evaluation Metrics

Each model was evaluated using:
- 10-fold Cross-Validation
- Accuracy on test set
- Precision, Recall, F1-score
- Confusion Matrix

📊 Class 2 (most frequent) was easiest to predict  
⚠️ Class 1 (least data) was harder for all models

---

## 🧠 Observations

- **Random Forest** outperformed all models with strong generalization
- **KNN** and **SVM** also achieved good accuracy with tuned hyperparameters
- **ANN** showed decent learning behavior but slightly lower accuracy (65.5%)
- No overfitting observed due to:
  - EarlyStopping in ANN
  - Cross-validation in ML models

---

## 🛠️ Tech Stack

- Python
- pandas, numpy, matplotlib, seaborn
- scikit-learn (KNN, SVM, Random Forest, scaling, metrics)
- tensorflow / keras (for ANN)

---

## 🧪 How to Run

1. Clone this repo or download the folder  
2. Open `Obesity_ML_project.ipynb` in Jupyter or VS Code  
3. Make sure the dataset file is in the same folder  
4. Run the notebook cells step-by-step

---

## 📚 References
  
- Project Summary: See `Requirement and explanation.pdf`

---

## 👨‍💻 Author

**Karri Vamsi**  
Feel free to connect or suggest improvements!

