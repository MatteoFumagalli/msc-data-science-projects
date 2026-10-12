# Import libraries
import altair as alt
from IPython.display import display
import numpy as np
import pandas as pd

from meridian.data import data_frame_input_data_builder
from meridian.data.input_data import InputData
from meridian.model import model, spec, prior_distribution
from meridian.model.model import Meridian
from meridian.schema.serde import meridian_serde
import tensorflow_probability as tfp
from meridian.analysis import visualizer

import warnings
warnings.filterwarnings('ignore')

# Define paths
OUTPUTS_CHARTS_DIR = '../outputs/charts/modeling'
OUTPUTS_TABLES_DIR = '../outputs/tables'
OUTPUTS_MODELS_DIR = '../outputs/models'


def save_chart(chart: alt.Chart, file_name: str) -> None:
    '''Save the plot in the visualizations folder or in the current folder'''
    target_path = f'{OUTPUTS_CHARTS_DIR}/{file_name}'
    display(chart)
    try:
        chart.save(target_path)
    except:
        chart.save(file_name)


def save_table(df: pd.DataFrame, file_name: str) -> None:
    '''Save the table in the visualizations folder or in the current folder'''
    target_path = f'{OUTPUTS_TABLES_DIR}/{file_name}'
    display(df)
    try:
        df.to_csv(target_path, index=False)
    except:
        df.to_csv(file_name, index=False)


def save_model(mmm: Meridian, file_name: str) -> None:
    '''Save the trained MMM'''
    file_path = OUTPUTS_MODELS_DIR + '/' + file_name
    try: 
        meridian_serde.save_meridian(mmm, file_path)
    except:
        meridian_serde.save_meridian(mmm, file_name)


def load_model(file_path: str) -> Meridian:
    '''Load the trained MMM'''
    mmm = meridian_serde.load_meridian(file_path)
    return mmm


def load_mmm_dataset(file_path: str) -> pd.DataFrame:
    '''Load the dataset for MMM'''
    mmm_df = pd.read_csv(file_path)
    mmm_df['DATE_DAY'] = pd.to_datetime(mmm_df['DATE_DAY'])
    
    return mmm_df


def define_input_data(mmm_df: pd.DataFrame) -> tuple[InputData, np.ndarray]:
    '''Configure media channels, controls, and target for MMM'''

    # Set default options for date and target
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type='revenue',
        default_time_column='DATE_DAY',
        default_media_time_column='DATE_DAY',
        default_kpi_column='ALL_PURCHASES_ORIGINAL_PRICE'
    )

    # Define media channels and controls
    channels = [
        'GOOGLE_PAID_SEARCH', 'GOOGLE_SHOPPING', 'GOOGLE_PMAX', 
        'GOOGLE_VIDEO', 'META_FACEBOOK', 'META_INSTAGRAM', 'TIKTOK']
    media_exposure_variables = [
        'GOOGLE_PAID_SEARCH_CLICKS', 'GOOGLE_SHOPPING_CLICKS', 
        'GOOGLE_PMAX_CLICKS', 'GOOGLE_VIDEO_IMPRESSIONS',
        'META_FACEBOOK_IMPRESSIONS', 'META_INSTAGRAM_IMPRESSIONS', 
        'TIKTOK_IMPRESSIONS']
    organic_media = [
        'DIRECT', 'ORGANIC_SEARCH', 'EMAIL', 'REFERRAL', 'ALL_OTHER']
    control_variables = ['DAY_OF_WEEK', 'IS_WEEKEND', 'MONTH']

    builder = (
        builder
        .with_kpi(mmm_df)
        .with_media(
            mmm_df, media_cols=media_exposure_variables, 
            media_spend_cols=[f'{channel}_SPEND' for channel in channels], 
            media_channels=channels)
        .with_organic_media(
            mmm_df, 
            organic_media_cols=[f'{media}_CLICKS' for media in organic_media],
            organic_media_channels=organic_media)
        .with_non_media_treatments(
            mmm_df, non_media_treatment_cols=control_variables
        )
    )

    # Define the matrix for holdout observations
    holdout_id = mmm_df['HOLDOUT'].to_numpy()

    # Define the overall input data for the model
    input_data = builder.build()

    return input_data, holdout_id


def train_mmm_model(input_data: InputData, holdout_id: np.ndarray) -> Meridian:
    '''Define and train the MMM'''
    print('== MERIDIAN MARKETING MIX MODELLING ==')

    # Configure the model specification
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(
            loc=[0.5, 0.5, 0.4, -0.2, 0.2, 0.2, 0.1], 
            scale=[0.3, 0.3, 0.3, 0.5, 0.4, 0.4, 0.5]))
    model_spec = spec.ModelSpec(
        prior=prior, holdout_id=holdout_id, 
        media_effects_dist='normal', enable_aks=True)
    
    # Define the Meridian model instance
    mmm = model.Meridian(input_data=input_data, model_spec=model_spec)

    # Perform prior sampling and estimate the posterior distribution
    print('\n -> Performing prior predictive sampling')
    mmm.sample_prior(500)

    print('\n -> Performing posterior sampling')
    mmm.sample_posterior(
        n_chains=10, n_adapt=2000, n_burnin=500, n_keep=1000, seed=3)

    # Save and return the trained model
    save_model(mmm, 'trained_mmm.binpb')
    print('\n -> Correctly saved the trained model')
    return mmm


def inspect_model_convergence(mmm: Meridian) -> None:
    '''Inspect the convergence of the model'''
    print('== MODEL CONVERGENCE ASSESSMENT ==')
    display(mmm.review())


def plot_model_quality(mmm: Meridian) -> None:
    '''Inspect the quality of the model'''
    print('== MODEL QUALITY ASSESSMENT ==\n')

    model_diagnostics = visualizer.ModelDiagnostics(mmm)
    save_chart(model_diagnostics.plot_rhat_boxplot(), 'model_rhat.html')
    print('\n')
    metrics_df = model_diagnostics.predictive_accuracy_table()
    metrics_df['value'] = metrics_df['value'].round(4)
    save_table(metrics_df, 'mmm_performance_metrics_df.csv')
    print('\n')
    
    model_fit = visualizer.ModelFit(mmm)
    save_chart(model_fit.plot_model_fit(), 'model_fit.html')


def plot_adstock_effects(mmm: Meridian) -> None:
    '''Display adstock effects'''
    print('== ADSTOCK EFFECTS ANALYSIS ==\n')

    media_effects = visualizer.MediaEffects(mmm)
    save_chart(media_effects.plot_adstock_decay(), 'adstock_effects.html')


def plot_hill_effects(mmm: Meridian) -> None:
    '''Display hill effects'''
    print('== HILL EFFECTS ANALYSIS ==\n')

    media_effects = visualizer.MediaEffects(mmm)
    hill_curves = media_effects.plot_hill_curves()
    save_chart(hill_curves['media'], 'hill_effects_media.html')
    print('\n')
    save_chart(hill_curves['organic_media'], 'hill_effects_organic_media.html')


def plot_channel_contribution(mmm: Meridian) -> None:
    '''Display the contribution of each channel'''
    print('== CHANNEL CONTRIBUTION ANALYSIS ==\n')

    media_summary = visualizer.MediaSummary(mmm)   
    save_chart(media_summary.plot_channel_contribution_area_chart(),
               'channel_contribution_trend.html')
    print('\n')
    save_chart(media_summary.plot_contribution_waterfall_chart(),
               'channel_contribution_waterfall.html')
    print('\n')
    save_chart(media_summary.plot_spend_vs_contribution(),
               'channel_spend_vs_contribution.html')