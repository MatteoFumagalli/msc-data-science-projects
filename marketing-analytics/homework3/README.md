## Marketing Analytics Homework 3
Customer Churn Prediction - Spark API

---

### :bar_chart: Project Description
1. Load datasets
2. Dataset overview
3. Split features and target
4. Identify column types
5. Build preprocessing pipeline
6. Build model pipeline
7. Train models (Linear Regression, Random Forest, Gradient Boosting)
8. Generate predictions
9. Evaluate models
10. Confusion matrix
11. Feature importance analysis
12. Save training, test and predictions
* Multi models comparison
* ROC curve
* Precision-Recall curve

---

### :information_source: Dataset
The dataset used in this project was provided for academic purposes and is not included in this repository. 

---

### :file_folder: Repository Structure 
- `visualizations/`:  feature importance plots, ROC curve, Precision-Recall curve
- `environment.yml`:  reproducible Python environment
- `HW3_code.py`: complete pipeline
- `HW3_presentation.pdf`: presentation slides

---

### :dart: Objectives
- Load and inspect tabular datasets with Spark API
- Verify different datasets and define accordingly which training models to use with Spark API
- Define train and test dataset
- Handle missing values
- Identify categorical and numerical features
- Build preprocessing pipelines with Spark API
- Train machine learning classification models with Spark API
- Evaluate classification performance
- Interpret feature importance
- Save prediction outputs
- Create a 5 slides presentation explaining the approach to solution

---

### :arrow_forward: Example Execution

```bash
python HW3_code.py
```

---

### :link: External Sources
- [Installation guide for PySpark](https://spark.apache.org/docs/latest/api/python/getting_started/install.html)
- [Machine Learning Library (MLlib) Guide](https://spark.apache.org/docs/4.1.2/ml-guide.html)
- [Manual ROC/Precision-Recall curve (Kaggle)](https://www.kaggle.com/code/himelsarder/manual-roc-auc-approach-without-using-sklearn)