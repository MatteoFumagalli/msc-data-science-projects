# =========================================================
# HW 1 Marketing Analytics 2026
# Customer Churn Prediction
#
# Author: Matteo Fumagalli  
# Student ID: 894468
# University: University of Milano-Bicocca
# Degree Program: Master’s Degree in Data Science 
# Academic Year: 2025/2026
# =========================================================


# =========================================================
# STEP 0 - IMPORTS
# =========================================================

from typing import Tuple, List
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.linear_model import *
from sklearn.ensemble import *


# =========================================================
# STEP 1 - LOAD DATASETS
# =========================================================

def stratified_split(
    df: pd.DataFrame,
    target_column: str,
    train_size: float = 0.75
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Perform manual stratified split.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.

    target_column : str
        Target column name.

    train_size : float
        Percentage of training data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        train_df : Training dataset
        test_df  : Test dataset
    '''

    # Separate and shuffle each class to simulate a stratified sampling
    churn_df = df[df[target_column] == 1]
    churn_df = churn_df.sample(frac=1, random_state=3)

    no_churn_df = df[df[target_column] == 0]
    no_churn_df = no_churn_df.sample(frac=1, random_state=3)

    # Define training and test DataFrames
    churn_train_size = int(train_size * len(churn_df))
    churn_train_df = churn_df.iloc[:churn_train_size]
    churn_test_df = churn_df.iloc[churn_train_size:]

    no_churn_train_size = int(train_size * len(no_churn_df))
    no_churn_train_df = no_churn_df.iloc[:no_churn_train_size]
    no_churn_test_df = no_churn_df.iloc[no_churn_train_size:]

    train_df = pd.concat([churn_train_df, no_churn_train_df])
    test_df = pd.concat([churn_test_df, no_churn_test_df])

    # Shuffle final datasets
    train_df = train_df.sample(frac=1, random_state=3).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=3).reset_index(drop=True)

    # Return both DataFrames
    return train_df, test_df


def load_datasets(
    dataset_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Load training and test datasets.

    Parameters
    ----------
    dataset_path : str
        File path for training dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        train_df : Training dataset
        test_df  : Test dataset
    '''

    # Read dataset CSV file
    df = pd.read_csv(dataset_path)

    # Define training and test DataFrames
    train_df, test_df = stratified_split(
        df, target_column='churn', train_size=0.75)

    # Return both DataFrames
    return train_df, test_df


# =========================================================
# STEP 2 - DATASET OVERVIEW
# =========================================================

def dataset_overview(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    '''
    Display basic dataset information.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.

    test_df : pd.DataFrame
        Test dataset.

    Returns
    -------
    None
    '''

    # Analyse both DataFrames
    datasets = [(train_df, 'TRAINING'), (test_df, 'TEST')]

    for df, name in datasets:
        print(f'\n==== {name} DATASET ====')
    
        # Display dataset shape
        print(f'\n{name.capitalize()} Shape: {df.shape}')

        # Display missing values
        print('\nMissing Values:')
        missing = df.isna().sum()
        if missing.sum() > 0:
            print(f' -> Found {missing.sum()} missing values across columns:')
            print(missing[missing > 0])
        else:
            print(' -> No missing values found')

        # Display basic statistics for numerical variables
        print('\nNumerical Statistics:')
        numeric_cols = df.select_dtypes(include=['number'])
        if not numeric_cols.empty:
            print(numeric_cols.describe())
        else:
            print(' -> No numeric variables found')

        # Display basic statistics for categorical variables
        print('\nCategorical Statistics:')
        categorical_cols = df.select_dtypes(
            include=['object', 'category', 'str'])
        if not categorical_cols.empty:
            print(categorical_cols.describe())
        else:
            print(' -> No categorical variables found')

        # Display the distribution of the target column (churn)
        target_col = 'churn'
        print(f'\nTarget Distribution ({target_col}):')
        if target_col in df.columns:
            counts = df[target_col].value_counts()
            percentages = df[target_col].value_counts(normalize=True) * 100
            for val in counts.index:
                print(f'Class {val}: {counts[val]} ({percentages[val]:.2f}%)')
        else:
            print(f' -> Target column {target_col} not found')


# =========================================================
# STEP 3 - DEFINE FEATURES AND TARGET
# =========================================================

def split_features_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series
]:
    '''
    Separate features and target variable.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.

    test_df : pd.DataFrame
        Test dataset.

    target_column : str
        Name of target column.

    Returns
    -------
    Tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series
    ]
        X_train : Training features
        y_train : Training labels
        X_test  : Test features
        y_test  : Test labels
    '''

    # Split features and labels
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Return all split datasets
    return X_train, y_train, X_test, y_test


# =========================================================
# STEP 4 - IDENTIFY COLUMN TYPES
# =========================================================

def identify_column_types(
    X_train: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    '''
    Identify categorical and numerical columns.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature dataset.

    Returns
    -------
    Tuple[List[str], List[str]]
        categorical_cols : List of categorical columns
        numeric_cols     : List of numerical columns
    '''

    # Identify categorical columns
    categorical_cols = X_train.select_dtypes(
        include=['object', 'category', 'str']).columns.tolist()

    # Identify numerical columns
    numeric_cols = X_train.select_dtypes(
        include=['number']).columns.tolist()
    
    # Return both lists
    return categorical_cols, numeric_cols


# =========================================================
# STEP 5 - BUILD PREPROCESSOR
# =========================================================

def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    '''
    Create preprocessing pipeline.

    Parameters
    ----------
    numeric_cols : List[str]
        Numerical feature column names.

    categorical_cols : List[str]
        Categorical feature column names.

    Returns
    -------
    ColumnTransformer
        Preprocessing pipeline.
    '''

    # Create numerical pipeline
    numeric_pipeline = Pipeline(
        [('imputer', SimpleImputer(strategy='median')),
         ('scaler', StandardScaler())]
    )

    # Create categorical pipeline
    categorical_pipeline = Pipeline(
        [('imputer', SimpleImputer(strategy='most_frequent')),
         ('onehot_encoder', OneHotEncoder(handle_unknown='ignore'))]
    )

    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        [('numeric_preprocessor', numeric_pipeline, numeric_cols),
         ('categorical_preprocessor', categorical_pipeline, categorical_cols)]
    )

    # Return preprocessor
    return preprocessor


# =========================================================
# STEP 6 - BUILD MODEL PIPELINE
# =========================================================

def build_model(
    preprocessor: ColumnTransformer
) -> Pipeline:
    '''
    Create machine learning pipeline.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Data preprocessing pipeline.

    Returns
    -------
    Pipeline
        Full machine learning pipeline.
    '''

    # Create Random Forest classifier
    classifier = RandomForestClassifier(
        class_weight='balanced',
        random_state=3)

    # Combine preprocessing + classifier
    model = Pipeline(
        [('preprocessor', preprocessor),
         ('classifier', classifier)]
    )

    # Return pipeline
    return model


# =========================================================
# STEP 7 - TRAIN MODEL
# =========================================================

def train_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    '''
    Train machine learning model.

    Parameters
    ----------
    model : Pipeline
        Machine learning pipeline.

    X_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training labels.

    Returns
    -------
    Pipeline
        Trained model.
    '''

    # Fit model
    model.fit(X_train, y_train)

    # Return trained model
    return model


# =========================================================
# STEP 8 - MAKE PREDICTIONS
# =========================================================

def make_predictions(
    model: Pipeline,
    X_test: pd.DataFrame
) -> Tuple[pd.Series, pd.Series]:
    '''
    Generate predictions and probabilities.

    Parameters
    ----------
    model : Pipeline
        Trained machine learning model.

    X_test : pd.DataFrame
        Test features.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        y_pred : Predicted labels
        y_prob : Prediction probabilities
    '''

    # Predict class labels
    y_pred = pd.Series(model.predict(X_test), 
                       index=X_test.index)

    # Predict probabilities
    y_prob = pd.Series(model.predict_proba(X_test)[:, 1], 
                       index=X_test.index)

    # Return both outputs
    return y_pred, y_prob


# =========================================================
# STEP 9 - EVALUATE MODEL
# =========================================================

def evaluate_model(
    y_test: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.Series
) -> None:
    '''
    Evaluate classification performance.

    Parameters
    ----------
    y_test : pd.Series
        True labels.

    y_pred : pd.Series
        Predicted labels.

    y_prob : pd.Series
        Predicted probabilities.

    Returns
    -------
    None
    '''

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)

    # Print model performance
    print('Model performance:')
    print(f'Accuracy : {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall   : {recall:.2f}')
    print(f'F1 score : {f1:.2f}')
    print(f'ROC-AUC  : {roc:.2f}')


# =========================================================
# STEP 10 - CONFUSION MATRIX
# =========================================================

def display_confusion_matrix(
    y_test: pd.Series,
    y_pred: pd.Series
) -> None:
    '''
    Display confusion matrix.

    Parameters
    ----------
    y_test : pd.Series
        True labels.

    y_pred : pd.Series
        Predicted labels.

    Returns
    -------
    None
    '''

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Display results
    print('\nConfusion Matrix:')
    print('                                PREDICTED')
    print('                       0 (No Churn)     1 (Churn)')
    print('                     +--------------+--------------+')
    print(f'       0 (No Churn)  |  {tn:^10d}  |  {fp:^10d}  |')
    print('ACTUAL               +--------------+--------------+')
    print(f'       1 (Churn)     |  {fn:^10d}  |  {tp:^10d}  |')
    print('                     +--------------+--------------+')  


# =========================================================
# STEP 11 - FEATURE IMPORTANCE
# =========================================================

def feature_importance_analysis(
    model: Pipeline,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> pd.DataFrame:
    '''
    Analyze feature importance.

    Parameters
    ----------
    model : Pipeline
        Trained machine learning pipeline.

    numeric_cols : List[str]
        Numerical feature names.

    categorical_cols : List[str]
        Categorical feature names.

    Returns
    -------
    pd.DataFrame 
        DataFrame containing feature importances.
    '''

    # Extract classifier, preprocessor, and OneHotEncoder
    classifier = model.named_steps['classifier']
    raw_name = classifier.__class__.__name__
    models = {
        'LogisticRegression': 'Logistic Regression',
        'RandomForestClassifier': 'Random Forest',
        'GradientBoostingClassifier': 'Gradient Boosting'
    }
    model_name = models.get(raw_name, raw_name)

    preprocessor = model.named_steps['preprocessor']
    onehot_encoder = preprocessor.named_transformers_[
        'categorical_preprocessor'].named_steps['onehot_encoder']

    # Extract feature names
    encoded_categorical_variables = list(
        onehot_encoder.get_feature_names_out(categorical_cols))
    feature_names = numeric_cols + encoded_categorical_variables
    
    # Extract feature importance values
    print('\nFeature importance:')
    if hasattr(classifier, 'feature_importances_'):
        importance_values = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        importance_values = [abs(value) for value in classifier.coef_[0]]
    else:
        print(' -> Feature importance not available for this model')
        return pd.DataFrame()
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    # Aggregate importance of categorical variables
    feature_to_original = {col: col for col in numeric_cols}
    for original_col, categories in zip(
        categorical_cols, onehot_encoder.categories_):
        for category in categories:
            encoded_name = f'{original_col}_{category}'
            feature_to_original[encoded_name] = original_col

    importance_df['original_feature'] = importance_df[
        'feature'].map(feature_to_original).fillna(importance_df['feature'])
    aggregated_importance_df = importance_df.groupby('original_feature')[
            'importance'].sum().reset_index()
    
    importance_df.drop(columns=['original_feature'], inplace=True)
    aggregated_importance_df = aggregated_importance_df.rename(
        columns={'original_feature': 'feature'}).sort_values(
            by='importance', ascending=False).reset_index(drop=True)
    
    # Display and plot top 5 features
    top_features = aggregated_importance_df.head(5)
    print(' -> Top 5 most important original features:')
    print(top_features.to_string(index=False))

    plt.figure(figsize=(7, 5))
    plt.barh(y=top_features['feature'], width=top_features['importance'],
             color='#81191a', edgecolor='black')
    plt.gca().invert_yaxis()

    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top 5 Most Important Original Features ({model_name})', 
              fontweight='bold')
    plt.tight_layout()

    # Save the plot
    model_name_file = model_name.replace(' ', '_').lower()
    num = globals().get('CURRENT_DATASET_INDEX', 1)
    filename = (f'visualizations/feature_importance_'
                f'{model_name_file}_dataset{num}.png')
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f'\n -> Feature importance plot saved to: {filename}')
    except Exception:
        file_exc = (f'feature_importance_{model_name_file}_dataset{num}.png')
        plt.savefig(file_exc, dpi=300, bbox_inches='tight')
        print(f'\n -> Feature importance plot saved to: {file_exc}')
    plt.close()
    
    # Return DataFrame
    return importance_df


# =========================================================
# STEP 12 - SAVE PREDICTIONS
# =========================================================

def save_predictions(
    test_df: pd.DataFrame,
    y_pred: pd.Series,
    y_prob: pd.Series,
    output_file: str
) -> None:
    '''
    Save prediction results to CSV file.

    Parameters
    ----------
    test_df : pd.DataFrame
        Original test dataset.

    y_pred : pd.Series
        Predicted labels.

    y_prob : pd.Series
        Prediction probabilities.

    output_file : str
        Output CSV filename.

    Returns
    -------
    None
    '''

    # Create output DataFrame
    output_df = test_df.copy()

    # Add predicted labels and probabilities for the positive class
    output_df['churn_prediction'] = y_pred
    output_df['churn_probability'] = y_prob

    # Save to CSV file
    try:
        output_df.to_csv(output_file, index=False)
        print(f' -> Prediction file saved to: {output_file}')
    except:
        filename_exc = (
            output_file.split('/')[-1] if '/' in output_file else output_file)
        output_df.to_csv(filename_exc, index=False)
        print(f' -> Prediction file saved to: {filename_exc}')


# =========================================================
# EXTRA - MULTI MODELS COMPARISON
# =========================================================

def multi_models_comparison(
        models_results: dict
) -> pd.DataFrame:
    '''
    Compare models performance.

    Parameters
    ----------
    models_results : dict
        Dictionary including trained models.

    Returns
    ----------
    pd.DataFrame
        Comparison DataFrame.    
    '''

    comparison = []
    for model_name, values in models_results.items():
        # Extract predicted labels and probabilities
        y_test = values['y_test']
        y_pred = values['y_pred']
        y_prob = values['y_prob']

        # Compute evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = values['f1_score']
        roc = roc_auc_score(y_test, y_prob)

        comparison.append({
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc
        })

    # Define the comparison DataFrame
    comparison_df = pd.DataFrame(comparison)

    print('\n==== MULTI MODELS COMPARISON ====')
    print(comparison_df.round(2).to_string(index=False))


# =========================================================
# EXTRA - ROC CURVE
# =========================================================

def plot_roc_curve(
        models_results: dict
) -> None:
    '''
    Plot ROC curves for multiple models.

    Parameters
    ----------
    models_results : dict
        Dictionary including trained models.

    Returns
    ----------
    None   
    '''

    plt.figure(figsize=(7, 5))
    # Generate the list of thresholds from 1.0 to 0.0
    thresholds = [round(i / 500, 3) for i in range(500, -1, -1)]

    for model_name, values in models_results.items():
        # Extract correct labels and predicted probabilities
        y_test = values['y_test']
        y_prob = values['y_prob']

        tpr_values = []
        fpr_values = []
        for threshold in thresholds:
            # Define the prediction according to the threshold value
            y_pred_threshold = (y_prob >= threshold).astype(int)
            
            # Define the confusion matrix
            cm = confusion_matrix(y_test, y_pred_threshold)
            tn, fp, fn, tp = cm.ravel()

            # Compute the values of TPR and FPR for the threshold value
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
            tpr_values.append(tpr)
            fpr_values.append(fpr)

        # Compute the ROC-AUC measure
        roc_auc = roc_auc_score(y_test, y_prob)

        # Plot the ROC curve
        plt.plot(fpr_values, tpr_values,
                 label=f'{model_name} (ROC-AUC={roc_auc:.2f})')
        
    # Plot the random baseline
    plt.plot([0, 1], [0, 1],
             color='grey', linestyle='--', label='Random Model')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves', fontweight='bold')
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0.05))
    plt.tight_layout()

    # Save the plot
    num = globals().get('CURRENT_DATASET_INDEX', 1)
    filename = f'visualizations/roc_curves_dataset{num}.png'
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f'\n -> ROC curves plot saved to: {filename}')
    except Exception:
        file_exc = (f'roc_curves_dataset{num}.png')
        plt.savefig(file_exc, dpi=300, bbox_inches='tight')
        print(f'\n -> ROC curves plot saved to: {file_exc}')
    plt.close()


# =========================================================
# EXTRA - PRECISION-RECALL CURVE
# =========================================================

def plot_precision_recall_curve(
        models_results: dict
) -> None:
    '''
    Plot Precision-Recall curves for multiple models.

    Parameters
    ----------
    models_results : dict
        Dictionary including trained models.

    Returns
    ----------
    None  
    '''

    plt.figure(figsize=(7, 5))
    # Generate the list of thresholds from 1.0 to 0.0
    thresholds = [round(i / 500, 3) for i in range(500, -1, -1)]

    for model_name, values in models_results.items():
        # Extract correct labels and predicted probabilities
        y_test = values['y_test']
        y_prob = values['y_prob']

        precision_values = []
        recall_values = []
        for threshold in thresholds:
            # Define the prediction according to the threshold value
            y_pred_threshold = (y_prob >= threshold).astype(int)
            
            # Compute the values of precision and recall for the threshold value
            precision = precision_score(
                y_test, y_pred_threshold, zero_division=0)
            recall = recall_score(
                y_test, y_pred_threshold, zero_division=0)
            if recall == 0 and precision == 0:
                precision = 1.0

            precision_values.append(precision)
            recall_values.append(recall)

        # Compute the PR-AUC measure
        pr_auc = 0
        for i in range(1, len(recall_values)):
            pr_auc += abs(recall_values[i] - recall_values[i - 1]) * (
                precision_values[i] + precision_values[i - 1]
            ) / 2

        # Plot the PR curve
        plt.plot(recall_values, precision_values,
                 label=f'{model_name} (PR-AUC={pr_auc:.2f})')
        
    # Plot the random baseline
    positive_rate = y_test.mean()
    plt.plot([0, 1], [positive_rate, positive_rate], 
              color='grey', linestyle='--', label='Random Model')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves', fontweight='bold')
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0.05))
    plt.tight_layout()

    # Save the plot
    num = globals().get('CURRENT_DATASET_INDEX', 1)
    filename = f'visualizations/precision_recall_curves_dataset{num}.png'
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f'\n -> Precision-Recall curves plot saved to: {filename}')
    except Exception:
        file_exc = (f'precision_recall_curves_dataset{num}.png')
        plt.savefig(file_exc, dpi=300, bbox_inches='tight')
        print(f'\n -> Precision-Recall curves plot saved to: {file_exc}')
    plt.close()


# =========================================================
# CUSTOMER CHURN PREDICTION PIPELINE
# =========================================================

if __name__ == '__main__':
    # Define the dataset paths
    dataset_paths = ['data/dataset1_HW1.csv', 'data/dataset2_HW1.csv']

    for i, dataset_path in enumerate(dataset_paths):
        print(f'\n\n======== DATASET {i + 1} ========')
        CURRENT_DATASET_INDEX = i+1

        # Load training (75%) and test (25%) datasets
        train_df, test_df = load_datasets(dataset_path)

        # Display basic dataset information
        dataset_overview(train_df, test_df)

        # Split training set into full training (75%) and validation (25%)
        subtrain_df, validation_df = stratified_split(
            df=train_df, target_column='churn', train_size=0.75)

        # Separate features and target variable for full training and test
        X_train, y_train, X_test, y_test = split_features_target(
            train_df, test_df, target_column='churn')
        
        # Separate features and target for full training and validation
        X_subtrain, y_subtrain, X_validation, y_validation = (
            split_features_target(
                subtrain_df, validation_df, target_column='churn'))      
        
        # Identify categorical and numerical columns
        categorical_cols, numeric_cols = identify_column_types(X_train)

        # Create preprocessing pipeline
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)

        # Train machine learning models
        print('\n==== MODEL TRAINING WITH HYPERPARAMETER TUNING ====')
        best_models = {}

        # Simulate grid search for hyperparameter tuning
        hyperparameter_grid = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 0.5, 1.0]
            },

            'Random Forest': {
                'n_estimators': [50, 100, 150],
                'max_depth': [5, 10, 15]
            },

            'Gradient Boosting': {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.05, 0.1, 0.2]
            }
        }

        # Model 1: Logistic Regression
        best_f1_score = -1
        best_lr = None
        param_lr = hyperparameter_grid['Logistic Regression']

        print('\n -> Logistic Regression:')
        for c in param_lr['C']:
            model = build_model(preprocessor)
            classifier = LogisticRegression(
                C=c, class_weight='balanced', max_iter=1000, random_state=3)
            model.set_params(classifier=classifier)
            trained_model = train_model(model, X_subtrain, y_subtrain)
            
            y_pred, y_prob = make_predictions(trained_model, X_validation)
            score = f1_score(y_validation, y_pred, zero_division=0)
            
            if score > best_f1_score:
                best_f1_score = score
                best_lr = {
                    'classifier': classifier,
                    'config': f'C={c}, class_weight=balanced',
                    'f1_score': score
                }
        best_models['Logistic Regression'] = best_lr
        print(f"Best configuration: {best_lr['config']}")
        
        # Model 2: Random Forest
        best_f1_score = -1
        best_rf = {}
        param_rf = hyperparameter_grid['Random Forest']

        print('\n -> Random Forest:')
        for n_estimators in param_rf['n_estimators']:
            for max_depth in param_rf['max_depth']:
                model = build_model(preprocessor)
                classifier = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth,
                    class_weight='balanced', random_state=3)
                model.set_params(classifier=classifier)
                trained_model = train_model(model, X_subtrain, y_subtrain)

                y_pred, y_prob = make_predictions(trained_model, X_validation)
                score = f1_score(y_validation, y_pred, zero_division=0)
                
                if score > best_f1_score:
                    best_f1_score = score
                    best_rf = {
                        'classifier': classifier,
                        'config': f'n_estimators={n_estimators}, '
                                  f'max_depth={max_depth}, '
                                  'class_weight=balanced',
                        'f1_score': score
                    }
        best_models['Random Forest'] = best_rf
        print(f"Best configuration: {best_rf['config']}")
 
        # Model 3: Gradient Boosting
        best_f1_score = -1
        best_gb = {}
        param_gb = hyperparameter_grid['Gradient Boosting']

        print('\n -> Gradient Boosting')
        for n_estimators in param_gb['n_estimators']:
            for learning_rate in param_gb['learning_rate']:
                model = build_model(preprocessor)
                classifier = GradientBoostingClassifier(
                    n_estimators=n_estimators, 
                    learning_rate=learning_rate, 
                    random_state=3)
                model.set_params(classifier=classifier)
                trained_model = train_model(model, X_subtrain, y_subtrain)

                y_pred, y_prob = make_predictions(trained_model, X_validation)
                score = f1_score(y_validation, y_pred, zero_division=0)
                
                if score > best_f1_score:
                    best_f1_score = score
                    best_gb = {
                        'classifier': classifier,
                        'config': f'n_estimators={n_estimators}, '
                                  f'learning_rate={learning_rate}',
                        'f1_score': score
                    }
        best_models['Gradient Boosting'] = best_gb
        print(f"Best configuration: {best_gb['config']}")

        # Final retraining of each best configuration
        final_results = {}
        print('\n==== FINAL RETRAINING AND EVALUATION ====')
        for model_name, values in best_models.items():
            print(f'\n -> {model_name}:')

            classifier = values['classifier']
            model = build_model(preprocessor)
            model.set_params(classifier=classifier)
            trained_model = train_model(model, X_train, y_train)
            
            # Evaluate performance and display confusion matrix
            y_pred, y_prob = make_predictions(trained_model, X_test)
            evaluate_model(y_test, y_pred, y_prob)
            display_confusion_matrix(y_test, y_pred)
            
            # Analyze feature importance
            importance_df = feature_importance_analysis(
                trained_model, numeric_cols, categorical_cols)
            
            final_results[model_name] = {
                'model': trained_model,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'f1_score': f1_score(y_test, y_pred, zero_division=0)
            }

        # Select the best model considering the F1 score
        best_model_name = max(
            final_results, key=lambda name: final_results[name]['f1_score'])
        best_model = final_results[best_model_name]['model']
        print('\n==== BEST MODEL ====')
        print(f' -> {best_model_name}')

        # Save prediction for training, test, and full datasets, 
        # considering the best model results to CSV file
        print(f'\n==== SAVE PREDICTIONS ====')

        y_pred_train, y_prob_train = make_predictions(best_model, X_train)
        save_predictions(
            train_df, y_pred_train, y_prob_train,
            output_file=f'data/churn_predictions_dataset{i + 1}_train.csv')
        
        save_predictions(
            test_df, 
            final_results[best_model_name]['y_pred'], 
            final_results[best_model_name]['y_prob'],
            output_file=f'data/churn_predictions_dataset{i + 1}_test.csv')
        
        full_df = pd.concat([train_df, test_df]).reset_index(drop=True)
        X_full = full_df.drop(columns=['churn'])
        y_pred_full, y_prob_full = make_predictions(best_model, X_full)
        save_predictions(
            full_df, y_pred_full, y_prob_full,
            output_file=f'data/churn_predictions_dataset{i + 1}.csv')
        
        # Compare models performance
        comparison_df = multi_models_comparison(final_results)

        # Plot ROC curves for all models
        plot_roc_curve(final_results)

        # Plot Precision-Recall curves for all models
        plot_precision_recall_curve(final_results)