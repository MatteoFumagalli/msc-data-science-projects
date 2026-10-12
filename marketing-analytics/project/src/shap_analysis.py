# Import libraries
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns

import shap
from shap import Explanation

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from meridian.analysis import analyzer
from meridian.model.model import Meridian

# Define paths
OUTPUTS_CHARTS_DIR = '../outputs/charts/shap_analysis'
OUTPUTS_TABLES_DIR = '../outputs/tables'

# Define plot theme and parameters
sns.set_theme(style='white')
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})


def save_plot(fig: Figure, file_name: str) -> None:
    '''Save the plot in the visualizations folder or in the current folder'''
    target_path = f'{OUTPUTS_CHARTS_DIR}/{file_name}'
    try:
        fig.savefig(target_path, dpi=300, bbox_inches='tight')
    except:
        fig.savefig(file_name, dpi=300, bbox_inches='tight')


def build_train_test_sets(
        mmm_df: pd.DataFrame, 
        mmm: Meridian) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                pd.Series, pd.Series]:
    '''Perform the dataset split for the ML model'''
    
    # Extract the expected daily revenue from the MMM
    analyzer_instance = analyzer.Analyzer(mmm)
    expected_outcome = analyzer_instance.expected_outcome(
        use_posterior=True, aggregate_geos=True, aggregate_times=False)
    
    # Compute the posterior mean prediction for each day
    y = pd.Series(
        np.asarray(expected_outcome).mean(axis=(0, 1)), index=mmm_df.index)
    
    # Construct the set of explanatory variables
    channels = [
        'GOOGLE_PAID_SEARCH', 'GOOGLE_SHOPPING', 'GOOGLE_PMAX', 
        'GOOGLE_VIDEO', 'META_FACEBOOK', 'META_INSTAGRAM', 'TIKTOK']
    media_spend_variables = [f'{channel}_SPEND' for channel in channels]
    media_exposure_variables = [
        'GOOGLE_PAID_SEARCH_CLICKS', 'GOOGLE_SHOPPING_CLICKS', 
        'GOOGLE_PMAX_CLICKS', 'GOOGLE_VIDEO_IMPRESSIONS',
        'META_FACEBOOK_IMPRESSIONS', 'META_INSTAGRAM_IMPRESSIONS', 
        'TIKTOK_IMPRESSIONS']
    organic_media = [
        'DIRECT', 'ORGANIC_SEARCH', 'EMAIL', 'REFERRAL', 'ALL_OTHER']
    organic_media_variables = [f'{media}_CLICKS' for media in organic_media]
    control_variables = ['DAY_OF_WEEK', 'IS_WEEKEND', 'MONTH']
    X = mmm_df[media_spend_variables + media_exposure_variables 
               + organic_media_variables + control_variables].copy()
    for column in control_variables:
        X[column] = X[column].astype('category')

    # Perform train-test split considering the holdout column
    holdout = mmm_df['HOLDOUT'].astype(bool)
    X_train = X.loc[~holdout]
    X_test = X.loc[holdout]
    y_train = y.loc[~holdout]
    y_test = y.loc[holdout]

    return X, X_train, X_test, y_train, y_test


def train_ml_model(
        X_train: pd.DataFrame, y_train: pd.Series) -> HistGradientBoostingRegressor:
    '''Train a surrogate ML model to mimic the behaviour of the MMM'''
    print('== MODEL TRAINING ==')

    model = HistGradientBoostingRegressor(
        max_depth=5, learning_rate=0.05, l2_regularization=1.0,
        categorical_features='from_dtype', random_state=3)
    model.fit(X_train, y_train)

    model_name = type(model).__name__
    print(f' -> Successfully trained a surrogate {model_name} model')
    return model
    

def evaluate_ml_model(
        trained_model: HistGradientBoostingRegressor, 
        X_test: pd.DataFrame, y_test: pd.Series) -> None:
    '''Evaluate the surrogate ML model'''
    print('\n== MODEL EVALUATION ==')

    y_pred = trained_model.predict(X_test)
    print(f' -> R2 score: {r2_score(y_test, y_pred):.4f}')


def compute_shap_values(
        trained_model: HistGradientBoostingRegressor, 
        X: pd.DataFrame) -> Explanation:
    '''Extract SHAP values from the trained model through a Tree Explainer'''
    print('== SHAP ANALYSIS ==')

    explainer = shap.TreeExplainer(trained_model)
    shap_values = explainer(X)

    print(' -> Successfully computed SHAP values from a Tree Explainer')
    return shap_values


def plot_summary(shap_values: Explanation, X: pd.DataFrame) -> None:
    '''Display the SHAP summary plot'''
    print('== SHAP SUMMARY PLOT ==')
    fig = plt.figure(figsize=(10, 5))
    
    shap.plots.beeswarm(shap_values, max_display=len(X.columns), show=False)
    plt.title('SHAP Summary Plot')
    
    plt.tight_layout()
    save_plot(fig, 'summary_plot.png')
    plt.show()


def plot_dependence(
        shap_values: Explanation, X: pd.DataFrame, top_k: int = 5) -> None:
    '''Display the SHAP dependence plots for the top-k features'''
    print('== SHAP DEPENDENCE PLOT ==')

    # Compute the mean importance value for each feature
    mean_absolute_values = np.abs(shap_values.values).mean(axis=0)

    # Extract the top-k features
    top_k_features = X.columns[
        np.argsort(mean_absolute_values)[::-1][:top_k]].tolist() 
    
    # Display a dependence plot for each top feature
    for feature in top_k_features:
        shap.plots.scatter(shap_values[:, feature], 
                           color=shap_values, show=False)
        fig = plt.gcf()
        plt.title(f'SHAP Dependence Plot - {feature}')
        
        plt.tight_layout()
        save_plot(fig, f'dependence_plot_{feature.lower()}.png')
        plt.show()
        plt.close(fig)


def plot_feature_importance(shap_values: Explanation, X: pd.DataFrame) -> None:
    '''Display the overall feature importance ranking'''
    print('== SHAP FEATURE IMPORTANCE RANKING ==')
    fig = plt.figure(figsize=(10, 5))
    
    shap.plots.bar(shap_values, max_display=len(X.columns), show=False)
    plt.title('SHAP Feature Importance')
    
    plt.tight_layout()
    save_plot(fig, 'feature_importance.png')
    plt.show()


def compare_shap_importance_with_mmm_contribution(
        shap_values: Explanation, mmm: Meridian, X: pd.DataFrame) -> pd.DataFrame:
    '''Compare SHAP feature importance with MMM media contribution'''
    print('== COMPARISON BETWEEN SHAP IMPORTANCE AND MMM CONTRIBUTION ==')

    # Extract the media contributions from the MMM and compute their averages
    analyzer_instance = analyzer.Analyzer(mmm)
    incremental_contributions = analyzer_instance.incremental_outcome(
        use_posterior=True, aggregate_geos=True, 
        aggregate_times=True, include_non_paid_channels=False)
    mean_media_contributions = (
        incremental_contributions.numpy().mean(axis=(0, 1)))

    # Compute the aggregated mean importance value from SHAP for each channel
    channels = mmm.input_data.media_channel.values
    mean_aggregated_shap_values = []
    for channel in channels:
        channel_columns = [column for column in X.columns 
                                  if column.startswith(channel)]
        shap_importance = np.abs(shap_values.values)[
            :, [X.columns.get_loc(column) 
            for column in channel_columns]].sum(axis=1).mean()
        mean_aggregated_shap_values.append(shap_importance)

    # Compare SHAP importance and MMM contribution considering spending channels
    comparison_df = pd.DataFrame({
        'CHANNEL': channels,
        'SHAP_IMPORTANCE': mean_aggregated_shap_values,
        'IMPORTANCE_PERCENTAGE': (
            np.array(mean_aggregated_shap_values) / 
            sum(mean_aggregated_shap_values)) * 100,
        'MMM_CONTRIBUTION': mean_media_contributions,
        'CONTRIBUTION_PERCENTAGE': (
            mean_media_contributions / mean_media_contributions.sum()) * 100
    })
    return comparison_df