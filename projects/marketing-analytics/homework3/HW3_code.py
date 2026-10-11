# =========================================================
# HW 3 Marketing Analytics 2026
# Customer Churn Prediction
#
# Author: Matteo Fumagalli
# Student ID: 894468
# University: University of Milano-Bicocca
# Degree Program: Master’s Degree in Data Science
# Academic Year: 2025/2026
# =========================================================


# =========================================================
# STEP 0 - IMPORTS AND SPARK SESSION
# =========================================================

import matplotlib.pyplot as plt
import os
import sys
from typing import Tuple, List, Union

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType
from pyspark.sql.window import Window

from pyspark.ml import Pipeline, PipelineModel, Transformer, Estimator
from pyspark.ml.feature import (
    Imputer, StandardScaler, VectorAssembler, StringIndexer, OneHotEncoder)
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator, BinaryClassificationEvaluator)
from pyspark.ml.functions import vector_to_array


def initialize_spark_session(app_name: str) -> SparkSession:
    '''
    Initialize a local Spark session.

    Parameters:
    ----------
    app_name : str
        Name of the application.

    Returns
    -------
    SparkSession
        Instance of Spark Session    
    '''
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master('local[*]')
        .config('spark.sql.shuffle.partitions', '4')
        .config('spark.sql.debug.maxToStringFields', '1000')
        .config('spark.driver.memory', '6g')
        .config('spark.driver.maxResultSize', '2g')
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel('ERROR')
    return spark


# =========================================================
# STEP 1 - LOAD DATASETS
# =========================================================

def stratified_split(
        df: DataFrame, target_column: str, train_size: float = 0.75
        ) -> Tuple[DataFrame, DataFrame]:
    '''
    Perform manual stratified split.

    Parameters
    ----------
    df : DataFrame
        Original dataset.

    target_column : str
        Name of target column.
    
    train_size : float
        Percentage of size of training set.

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        train_df : Training dataset
        test_df  : Test dataset    
    '''

    # Group records by target class and randomly order records within each class
    partition_window = Window.partitionBy(target_column)
    random_window = partition_window.orderBy(F.rand(seed=3))

    # Assign a rank within the class and count the number of records in each group
    split_df = (
        df.withColumn('_row_number', F.row_number().over(random_window))
          .withColumn('_class_size', F.count('*').over(partition_window))
    )
    
    # Define the training set
    train_df = (
        split_df.filter(
            F.col('_row_number') <= F.floor(F.col('_class_size') * train_size))
                .drop('_row_number', '_class_size')
    )

    # Define the test set
    test_df = (
        split_df.filter(
            F.col('_row_number') > F.floor(F.col('_class_size') * train_size))
                .drop('_row_number', '_class_size')
    )

    return train_df, test_df


def load_datasets(
        spark: SparkSession, dataset_path: str) -> Tuple[DataFrame, DataFrame]:
    '''
    Load training and test datasets.
    
    Parameters
    ----------
    spark : SparkSession
        Instance of Spark Session.

    dataset_path : str
        Path of the original dataset.

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        train_df : Training dataset
        test_df  : Test dataset
    '''
    
    # Read dataset CSV file
    df = spark.read.csv(dataset_path, header=True, inferSchema=True)
    
    # Define training and test DataFrames
    train_df, test_df = stratified_split(df, target_column='churn')

    # Return both DataFrames
    return train_df, test_df


# =========================================================
# STEP 2 - DATASET OVERVIEW
# =========================================================

def dataset_overview(train_df: DataFrame, test_df: DataFrame) -> None:
    '''
    Display basic dataset information.
    
    Parameters
    ----------
    train_df : DataFrame
        Training dataset.

    test_df : DataFrame
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
        num_rows = df.count()
        num_columns = len(df.columns) 
        print(f'\n{name.capitalize()} Shape: ({num_rows}, {num_columns})')

        # Display missing values by counting the number of null for each column
        missing = df.select([
            F.count(F.when(F.col(column).isNull(), column)).alias(column) 
            for column in df.columns
        ]).first().asDict()
        missing_filtered = {
            column: num_missing for column, num_missing in missing.items() 
                                if num_missing > 0}

        print('\nMissing Values:')
        if sum(missing_filtered.values()) > 0:
            print(f' -> Found {sum(missing_filtered.values())} '
                  'missing values across columns:')
            for column, num_missing in missing_filtered.items():
                print(f'    {column}: {num_missing}')
        else:
            print(' -> No missing values found')

        # Display basic statistics for numerical variables
        print('\nNumerical Statistics:')
        numeric_cols = [field.name for field in df.schema.fields 
                        if isinstance(field.dataType, NumericType)]
        if numeric_cols:
            stats_df = df.select(numeric_cols).describe()
            for column in numeric_cols:
                stats_df = stats_df.withColumn(
                    column, 
                    F.when(F.col('summary') == 'count', F.col(column))
                     .otherwise(
                         F.round(F.col(column).cast('double'), 2).cast('string'))
                )
            stats_df.show()
        else:
            print(' -> No numeric variables found')

        # Display basic statistics for categorical variables
        print('\nCategorical Statistics:')
        categorical_cols = [column for column in df.columns 
                            if column not in numeric_cols]
        if categorical_cols:
            df.select(categorical_cols).describe().show()
        else:
            print(' -> No categorical variables found')        

        # Display the distribution of the target column (churn)
        target_col = 'churn'
        print(f'\nTarget Distribution ({target_col}):')
        if target_col in df.columns:
            counts = (
                df.groupBy(target_col).count().orderBy('count', ascending=False))
            for row in counts.collect():
                val = row[target_col]
                count = row['count']
                percentage = (count / df.count()) * 100
                print(f'Class {val}: {count} ({percentage:.2f}%)')
        else:
            print(f' -> Target column {target_col} not found')


def compute_class_weights(train_df: DataFrame) -> DataFrame:
    '''
    Compute a weight associated with each class.
    
    Parameters
    ----------
    train_df : DataFrame
        Training dataset.

    Returns
    -------
    DataFrame
        train_df : Weighted training dataset
    '''
    
    # Compute the number of instances for each class 
    counts = {
        row['churn']: row['count']
        for row in train_df.groupBy('churn').count().collect()
    }

    # Assign a weight to each class
    total = counts[0] + counts[1]
    weight_0 = total / (2 * counts[0])
    weight_1 = total / (2 * counts[1])

    return train_df.withColumn(
        'classWeight', F.when(F.col('churn') == 0, weight_0).otherwise(weight_1))


# =========================================================
# STEP 3 - DEFINE FEATURES AND TARGET
# =========================================================

# -> Spark pipeline does not require to separate the target 
#    variable from the explanatory variables, so the function
#    split_features_target will not be implemented 


# =========================================================
# STEP 4 - IDENTIFY COLUMN TYPES
# =========================================================

def identify_column_types(train_df: DataFrame) -> Tuple[List[str], List[str]]:
    '''
    Identify categorical and numerical columns.
    
    Parameters
    ----------
    train_df : DataFrame
        Training dataset.

    Returns
    -------
    Tuple[List[str], List[str]]
        categorical_cols : List of categorical columns
        numeric_cols     : List of numerical columns
    '''

    # Identify numerical columns
    numeric_cols = [field.name for field in train_df.schema.fields 
                        if isinstance(field.dataType, NumericType) 
                        and field.name not in ['churn', 'classWeight']]
    
    # Identify categorical columns
    categorical_cols = [column for column in train_df.columns 
                            if column not in 
                                numeric_cols + ['churn', 'classWeight']]
    
    # Return both lists
    return categorical_cols, numeric_cols


# =========================================================
# STEP 5 - BUILD PREPROCESSOR
# =========================================================

def build_preprocessor(
        numeric_cols: List[str], categorical_cols: List[str]
        ) -> List[Union[Transformer, Estimator]]:
    '''
    Create preprocessing pipeline.
    
    Parameters
    ----------
    numeric_cols : List[str]
        List of numerical columns.

    categorical_cols : List[str]
        List of categorical columns.
    
    Returns
    -------
    List[Union[Transformer, Estimator]]
        preprocessor_stages : List of stages of preprocessing pipeline
    '''

    preprocessor_stages = []
    feature_cols = []

    # Create numerical pipeline
    if numeric_cols:
        imputed_cols = [f'{column}_imputed' for column in numeric_cols]
        imputer = Imputer(
            inputCols=numeric_cols, outputCols=imputed_cols, strategy='median')
        preprocessor_stages.append(imputer)

        vector_assembler = VectorAssembler(
            inputCols=imputed_cols, outputCol='numeric_features')
        preprocessor_stages.append(vector_assembler)

        scaler = StandardScaler(
            inputCol='numeric_features', outputCol='scaled_numeric_features',
            withMean=True, withStd=True)
        preprocessor_stages.append(scaler)
        feature_cols.append('scaled_numeric_features')
    
    # Create categorical pipeline
    if categorical_cols:
        indexed_cols = [f'{column}_indexed' for column in categorical_cols]
        encoded_cols = [f'{column}_encoded' for column in categorical_cols]

        string_indexers = [StringIndexer(
            inputCol=column, outputCol=indexed_column, handleInvalid='keep') 
            for column, indexed_column in zip(categorical_cols, indexed_cols)]
        preprocessor_stages.extend(string_indexers)

        onehot_encoder = OneHotEncoder(
            inputCols=indexed_cols, outputCols=encoded_cols, 
            handleInvalid='keep')
        preprocessor_stages.append(onehot_encoder)

        feature_cols.extend(encoded_cols)   

    # Combine pipelines
    preprocessor_stages.append(
        VectorAssembler(inputCols=feature_cols, outputCol='features'))

    # Return preprocessor
    return preprocessor_stages


# =========================================================
# STEP 6 - BUILD MODEL PIPELINE
# =========================================================

def build_model(
        preprocessor_stages: List[Union[Transformer, Estimator]], 
        classifier: Union[
            LogisticRegression, RandomForestClassifier, GBTClassifier]
        ) -> Pipeline:
    '''
    Create machine learning pipeline.
       
    Parameters
    ----------
    preprocessor_stages : List[Union[Transformer, Estimator]]
        List of stages of preprocessing pipeline.

    classifier : Union[LogisticRegression, RandomForestClassifier, GBTClassifier]
        Machine Learning model.
    
    Returns
    -------
    Pipeline
        pipeline : Full Machine Learning pipeline
    '''

    # Combine preprocessing steps with classifier
    preprocessor = preprocessor_stages.copy()
    preprocessor.append(classifier)
    pipeline = Pipeline(stages=preprocessor)

    # Return pipeline
    return pipeline


# =========================================================
# STEP 7 - TRAIN MODEL
# =========================================================

def train_model(pipeline: Pipeline, train_df: DataFrame) -> PipelineModel:
    '''
    Train machine learning model.

    Parameters
    ----------
    pipeline : Pipeline
        Full Machine Learning pipeline.

    train_df : DataFrame
        Training dataset.
    
    Returns
    -------
    PipelineModel
        model : Trained Machine Learning model    
    '''
    
    # Fit the model
    model = pipeline.fit(train_df)

    # Return trained model
    return model


# =========================================================
# STEP 8 - MAKE PREDICTIONS
# =========================================================

def make_predictions(model: PipelineModel, test_df: DataFrame) -> DataFrame:
    '''
    Generate predictions and probabilities.
    
    Parameters
    ----------
    model : PipelineModel
        Trained Machine Learning model.

    test_df : DataFrame
        Test dataset.
    
    Returns
    -------
    DataFrame
        predictions_df : Predictions dataset
    '''

    # Generate predictions for test set
    predictions_df = model.transform(test_df)

    # Return predictions
    return predictions_df


# =========================================================
# STEP 9 - EVALUATE MODEL
# =========================================================

def evaluate_model(predictions_df: DataFrame) -> None:
    '''
    Evaluate classification performance.
    
    Parameters
    ----------
    predictions_df : DataFrame
        Predictions dataset.
    
    Returns
    -------
    None
    '''

    # Compute evaluation metrics
    accuracy = MulticlassClassificationEvaluator(
        labelCol='churn', predictionCol='prediction', metricName='accuracy'
    ).evaluate(predictions_df)

    precision = MulticlassClassificationEvaluator(
        labelCol='churn', predictionCol='prediction', 
        metricName='weightedPrecision'
    ).evaluate(predictions_df)

    recall = MulticlassClassificationEvaluator(
        labelCol='churn', predictionCol='prediction', 
        metricName='weightedRecall'
    ).evaluate(predictions_df)

    f1 = MulticlassClassificationEvaluator(
        labelCol='churn', predictionCol='prediction', metricName='f1'
    ).evaluate(predictions_df)    

    roc = BinaryClassificationEvaluator(
        labelCol='churn', rawPredictionCol='rawPrediction', 
        metricName='areaUnderROC'
    ).evaluate(predictions_df)

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

def display_confusion_matrix(predictions_df: DataFrame) -> None:
    '''
    Display confusion matrix.
    
    Parameters
    ----------
    predictions_df : DataFrame
        Predictions dataset.
    
    Returns
    -------
    None
    '''

    # Compute confusion matrix
    confusion_matrix = (
        predictions_df.agg(
            F.sum(F.when(
                (F.col('churn') == 1) & (F.col('prediction') == 1), 
                1).otherwise(0)
            ).alias('tp'),
            F.sum(F.when(
                (F.col('churn') == 0) & (F.col('prediction') == 1), 
                1).otherwise(0)
            ).alias('fp'),
            F.sum(F.when(
                (F.col('churn') == 1) & (F.col('prediction') == 0), 
                1).otherwise(0)
            ).alias('fn'),
            F.sum(F.when(
                (F.col('churn') == 0) & (F.col('prediction') == 0),
                1).otherwise(0)
            ).alias('tn'),
        ).first()
    )

    tp, fp = confusion_matrix['tp'], confusion_matrix['fp']
    fn, tn = confusion_matrix['fn'], confusion_matrix['tn']

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
    model: PipelineModel, numeric_cols: List[str], categorical_cols: List[str]
    ) -> List[Tuple[str, float]]:
    '''
    Analyze feature importance.
    
    Parameters
    ----------
    model : PipelineModel
        Trained Machine Learning model.

    numeric_cols : List[str]
        List of numerical columns.

    categorical_cols : List[str]
        List of categorical columns.
    
    Returns
    -------
    List[Tuple[str, float]]
        importance_list : List of feature importance
    '''

    # Extract classifier
    classifier = model.stages[-1]
    raw_name = classifier.__class__.__name__
    models = {
        'LogisticRegressionModel': 'Logistic Regression',
        'RandomForestClassificationModel': 'Random Forest',
        'GBTClassificationModel': 'Gradient Boosting'
    }
    model_name = models.get(raw_name, raw_name)

    # Reconstruct feature names
    feature_names = []
    feature_names.extend(numeric_cols)
    for column in categorical_cols:
        indexer = next(
            stage for stage in model.stages
            if hasattr(stage, 'getOutputCol')
            and stage.getOutputCol() == f'{column}_indexed'
        )
        
        labels = indexer.labels
        for label in labels[:-1]:
            feature_names.append(f'{column}_{label}')
    
    # Extract feature importance values
    print('\nFeature importance:')
    if hasattr(classifier, 'featureImportances'):
        importance_values = classifier.featureImportances.toArray()
    elif hasattr(classifier, 'coefficients'):
        importance_values = [abs(value) 
                             for value in classifier.coefficients.toArray()]
    else:
        print(' -> Feature importance not available for this model')
        return []
    
    # Create importance list
    importance_list = []
    for feature, importance in zip(feature_names, importance_values):
        importance_list.append((feature, float(importance)))
    importance_list.sort(key=lambda x: x[1], reverse=True)   
    
    # Aggregate importance of categorical variables
    aggregated_importance = {}
    for feature, importance in importance_list:
        original_feature = feature

        for categorical_col in categorical_cols:
            prefix = f'{categorical_col}_'
            if feature.startswith(prefix):
                original_feature = categorical_col
                break

        aggregated_importance[original_feature] = (
            aggregated_importance.get(original_feature, 0.0) + importance)
    
    aggregated_importance_list = sorted(
        aggregated_importance.items(), key=lambda x: x[1], reverse=True)

    # Display and plot top 5 features
    top_features = aggregated_importance_list[:5]
    print(' -> Top 5 most important original features:')
    for feature, importance in top_features:
        print(f'{feature} - {importance:.4f}')
    
    plt.figure(figsize=(7, 5))
    plt.barh(y=[feature for feature, _ in top_features], 
             width=[importance for _, importance in top_features],
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
    return importance_list


# =========================================================
# STEP 12 - SAVE PREDICTIONS
# =========================================================

def save_predictions(
    predictions_df: DataFrame, output_file: str) -> None:
    '''
    Save prediction results to CSV file.
    
    Parameters
    ----------
    predictions_df : DataFrame
        Predictions dataset.

    output_file : str
        Path of the output file.
    
    Returns
    -------
    None
    '''  

    # Add predicted labels and probabilities for the positive class
    output_df = predictions_df.withColumn(
        'churn_prediction', F.col('prediction').cast('int'))
    output_df = output_df.withColumn(
        'churn_probability', 
        vector_to_array(F.col('probability'))[1].cast('double'))
    
    columns_to_drop = [
        column for column in output_df.columns
        if column.endswith('_imputed') or
           column.endswith('_indexed') or
           column.endswith('_encoded') or
           column in ['numeric_features', 'scaled_numeric_features',
                      'features', 'rawPrediction', 'probability', 'prediction']]
    output_df = output_df.drop(*columns_to_drop)
    
    # Save to CSV file
    try:
        output_df.toPandas().to_csv(output_file, index=False)
        print(f' -> Prediction file saved to: {output_file}')
    except:
        filename_exc = (
            output_file.split('/')[-1] if '/' in output_file else output_file)
        output_df.toPandas().to_csv(filename_exc, index=False)
        print(f' -> Prediction file saved to: {filename_exc}')
    

# =========================================================
# EXTRA - MULTI MODELS COMPARISON
# =========================================================

def multi_models_comparison(models_results: dict) -> None:
    '''
    Compare models performance.
    
    Parameters
    ----------
    models_results : dict
        Dictionary of results for each model.
    
    Returns
    -------
    None
    '''

    comparison = []
    for model_name, values in models_results.items():
        # Extract prediction
        predictions_df = values['predictions']

        # Compute evaluation metrics
        accuracy = MulticlassClassificationEvaluator(
            labelCol='churn', predictionCol='prediction', metricName='accuracy'
        ).evaluate(predictions_df)

        precision = MulticlassClassificationEvaluator(
            labelCol='churn', predictionCol='prediction', 
            metricName='weightedPrecision'
        ).evaluate(predictions_df)

        recall = MulticlassClassificationEvaluator(
            labelCol='churn', predictionCol='prediction', 
            metricName='weightedRecall'
        ).evaluate(predictions_df)

        f1 = values['f1_score']   
        roc = values['roc']
    
        comparison.append({
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc
        })

    # Display comparison results
    print('\n==== MULTI MODELS COMPARISON ====')
    for row in comparison:
        print(f"{row['model']} | Accuracy={row['accuracy']:.2f} | "
              f"Precision={row['precision']:.2f} | Recall={row['recall']:.2f} | "
              f"F1={row['f1_score']:.2f} | ROC-AUC={row['roc_auc']:.2f}")


# =========================================================
# EXTRA - ROC CURVE
# =========================================================

def plot_roc_curve(models_results: dict) -> None:
    '''
    Plot ROC curves for multiple models.
    
    Parameters
    ----------
    models_results : dict
        Dictionary of results for each model.
    
    Returns
    -------
    None
    '''

    plt.figure(figsize=(7, 5))
    # Generate the list of thresholds from 1.0 to 0.0
    thresholds = [round(i / 500, 3) for i in range(500, -1, -1)]

    for model_name, values in models_results.items():
        # Extract correct labels and predicted probabilities
        predictions_df = values['predictions']
        rows = predictions_df.select(
            F.col('churn').cast('int'),
            vector_to_array(F.col('probability'))[1].cast(
                'double').alias('probability')).collect()
        y_test = [row['churn'] for row in rows]
        y_prob = [row['probability'] for row in rows]

        tpr_values = []
        fpr_values = []
        for threshold in thresholds:
            tp = fp = fn = tn = 0
            
            for actual, probability in zip(y_test, y_prob):
                # Define the prediction according to the threshold value
                prediction = 1 if probability >= threshold else 0

                if actual == 1 and prediction == 1:
                    tp += 1
                elif actual == 0 and prediction == 1:
                    fp += 1
                elif actual == 1 and prediction == 0:
                    fn += 1
                else:
                    tn += 1   
            
            # Compute the values of TPR and FPR for the threshold value
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
            tpr_values.append(tpr)
            fpr_values.append(fpr)

        # Plot the ROC curve
        plt.plot(fpr_values, tpr_values,
                 label=f'{model_name} (ROC-AUC={values['roc']:.2f})')
        
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

def plot_precision_recall_curve(models_results: dict) -> None:
    '''
    Plot Precision-Recall curves for multiple models.
    
    Parameters
    ----------
    models_results : dict
        Dictionary of results for each model.
    
    Returns
    -------
    None
    '''

    plt.figure(figsize=(7, 5))
    # Generate the list of thresholds from 1.0 to 0.0
    thresholds = [round(i / 500, 3) for i in range(500, -1, -1)]

    for model_name, values in models_results.items():
        # Extract correct labels and predicted probabilities
        predictions_df = values['predictions']
        rows = predictions_df.select(
            F.col('churn').cast('int'),
            vector_to_array(F.col('probability'))[1].cast(
                'double').alias('probability')).collect()
        y_test = [row['churn'] for row in rows]
        y_prob = [row['probability'] for row in rows]      

        precision_values = []
        recall_values = []
        for threshold in thresholds:
            tp = fp = fn = 0
            
            for actual, probability in zip(y_test, y_prob):
                # Define the prediction according to the threshold value
                prediction = 1 if probability >= threshold else 0

                if actual == 1 and prediction == 1:
                    tp += 1
                elif actual == 0 and prediction == 1:
                    fp += 1
                elif actual == 1 and prediction == 0:
                    fn += 1
                        
            # Compute the values of precision and recall for the threshold value
            precision = tp / (tp + fp) if tp + fp != 0 else 1
            recall = tp / (tp + fn) if tp + fn != 0 else 1
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
    positive_rate = sum(y_test) / len(y_test)
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
    spark = None

    try: 
        # Initialize the Spark session
        spark = initialize_spark_session(app_name='CustomerChurn')

        # Define the dataset paths
        dataset_paths = ['data/dataset1_HW3.csv', 'data/dataset2_HW3.csv']

        for i, dataset_path in enumerate(dataset_paths):
            print(f'\n\n======== DATASET {i + 1} ========')
            CURRENT_DATASET_INDEX = i + 1
            
            # Load training (75%) and test (25%) datasets
            train_df, test_df = load_datasets(spark, dataset_path)

            # Display basic dataset information
            dataset_overview(train_df, test_df)

            # Add class weights to the train set
            train_df = compute_class_weights(train_df)

            # Split training set into full training (75%) and validation (25%)
            subtrain_df, validation_df = stratified_split(
                df=train_df, target_column='churn')
            
            # Identify categorical and numerical columns
            categorical_cols, numeric_cols = identify_column_types(train_df)

            # Create preprocessing pipeline
            preprocessor_stages = build_preprocessor(
                numeric_cols, categorical_cols)

            # Train machine learning models
            print('\n==== MODEL TRAINING WITH HYPERPARAMETER TUNING ====')
            best_models = {}

            # Simulate grid search for hyperparameter tuning
            hyperparameter_grid = {
                'Logistic Regression': {
                    'reg_param': [1, 2, 10, 100]
                },

                'Random Forest': {
                    'num_trees': [50, 100, 150],
                    'max_depth': [5, 10, 15]
                },

                'Gradient Boosting': {
                    'max_iter': [50, 100, 150],
                    'step_size': [0.05, 0.1, 0.2]
                }
            }

            # Model 1: Logistic Regression
            best_f1_score = -1
            best_lr = None
            param_lr = hyperparameter_grid['Logistic Regression']

            print('\n -> Logistic Regression:')
            for reg_param in param_lr['reg_param']:
                # Define the classifier
                classifier = LogisticRegression(
                    labelCol='churn', featuresCol='features',
                    weightCol='classWeight', regParam=reg_param, maxIter=1000)
                
                # Create machine learning pipeline
                pipeline = build_model(preprocessor_stages, classifier)
                
                # Train the model and generate predictions
                trained_model = train_model(pipeline, subtrain_df)
                predictions_df = make_predictions(trained_model, validation_df)
                score = MulticlassClassificationEvaluator(
                    labelCol='churn', predictionCol='prediction', metricName='f1'
                ).evaluate(predictions_df) 
                
                # Save the best configuration so far
                if score > best_f1_score:
                    best_f1_score = score
                    best_lr = {
                        'classifier': classifier,
                        'config': f'reg_param={reg_param}',
                        'f1_score': score
                    }
            best_models['Logistic Regression'] = best_lr
            print(f"Best configuration: {best_lr['config']}")

            # Model 2: Random Forest
            best_f1_score = -1
            best_rf = None
            param_rf = hyperparameter_grid['Random Forest']

            print('\n -> Random Forest:')
            for num_trees in param_rf['num_trees']:
                for max_depth in param_rf['max_depth']:
                    # Define the classifier
                    classifier = RandomForestClassifier(
                        labelCol='churn', featuresCol='features',
                        weightCol='classWeight', numTrees=num_trees, 
                        maxDepth=max_depth, seed=3)
                    
                    # Create machine learning pipeline
                    pipeline = build_model(preprocessor_stages, classifier)

                    # Train the model and generate predictions
                    trained_model = train_model(pipeline, subtrain_df)
                    predictions_df = make_predictions(
                        trained_model, validation_df)
                    score = MulticlassClassificationEvaluator(
                        labelCol='churn', predictionCol='prediction', 
                        metricName='f1'
                    ).evaluate(predictions_df) 
                    
                    # Save the best configuration so far
                    if score > best_f1_score:
                        best_f1_score = score
                        best_rf = {
                            'classifier': classifier,
                            'config': f'num_trees={num_trees}, '
                                    f'max_depth={max_depth}',
                            'f1_score': score
                        }
            best_models['Random Forest'] = best_rf
            print(f"Best configuration: {best_rf['config']}")

            # Model 3: Gradient Boosting
            best_f1_score = -1
            best_gb = {}
            param_gb = hyperparameter_grid['Gradient Boosting']

            print('\n -> Gradient Boosting')
            for max_iter in param_gb['max_iter']:
                for step_size in param_gb['step_size']:
                    # Define the classifier
                    classifier = GBTClassifier(
                        labelCol='churn', featuresCol='features',
                        weightCol='classWeight', maxIter=max_iter, 
                        stepSize=step_size, seed=3)
                    
                    # Create machine learning pipeline
                    pipeline = build_model(preprocessor_stages, classifier)

                    # Train the model and generate predictions
                    trained_model = train_model(pipeline, subtrain_df)
                    predictions_df = make_predictions(
                        trained_model, validation_df)
                    score = MulticlassClassificationEvaluator(
                        labelCol='churn', predictionCol='prediction', 
                        metricName='f1'
                    ).evaluate(predictions_df) 
                    
                    # Save the best configuration so far
                    if score > best_f1_score:
                        best_f1_score = score
                        best_gb = {
                            'classifier': classifier,
                            'config': f'max_iter={max_iter}, '
                                    f'step_size={step_size}',
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
                pipeline = build_model(preprocessor_stages, classifier)
                trained_model = train_model(pipeline, train_df)
                
                # Evaluate performance and display confusion matrix
                predictions_df = make_predictions(trained_model, test_df)
                evaluate_model(predictions_df)
                display_confusion_matrix(predictions_df)
                
                # Analyze feature importance
                importance_list = feature_importance_analysis(
                    trained_model, numeric_cols, categorical_cols)
                
                final_results[model_name] = {
                    'model': trained_model,
                    'predictions': predictions_df,
                    'f1_score': MulticlassClassificationEvaluator(
                        labelCol='churn', predictionCol='prediction', 
                        metricName='f1'
                    ).evaluate(predictions_df),
                    'roc': BinaryClassificationEvaluator(
                        labelCol='churn', rawPredictionCol='rawPrediction', 
                        metricName='areaUnderROC'
                    ).evaluate(predictions_df)
                }
            
            # Select the best model considering the F1 score and the ROC-AUC
            best_model_name = max(
                final_results, 
                key=lambda name: (round(final_results[name]['f1_score'], 2), 
                                  final_results[name]['roc']))
            best_model = final_results[best_model_name]['model']
            print('\n==== BEST MODEL ====')
            print(f' -> {best_model_name}')
                
            # Save prediction for training, test, and full datasets, 
            # considering the best model results to CSV file
            print(f'\n==== SAVE PREDICTIONS ====')

            train_predictions_df = make_predictions(trained_model, train_df)
            save_predictions(
                train_predictions_df, 
                f'data/churn_predictions_dataset{i + 1}_train.csv')
            
            save_predictions(
                predictions_df, 
                f'data/churn_predictions_dataset{i + 1}_test.csv')
            
            full_predictions_df = (
                train_predictions_df.unionByName(train_predictions_df))
            save_predictions(
                full_predictions_df, 
                f'data/churn_predictions_dataset{i + 1}.csv')
            
            # Compare models performance
            multi_models_comparison(final_results)

            # Plot ROC curves for all models
            plot_roc_curve(final_results)

            # Plot Precision-Recall curves for all models
            plot_precision_recall_curve(final_results)

    except Exception as e:
        print(f' -> Error during execution: {e}')
    
    finally:
        if spark:
            # Close the Spark session
            spark.stop()
            print('\n -> Successfully closed the Spark session')
        sys.exit(0)