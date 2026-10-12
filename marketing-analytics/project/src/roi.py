# Import libraries
import altair as alt
from IPython.display import display
import pandas as pd

from meridian.model.model import Meridian
from meridian.analysis import analyzer, visualizer

# Define paths
OUTPUTS_CHARTS_DIR = '../outputs/charts/roi'
OUTPUTS_TABLES_DIR = '../outputs/tables'


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


def create_analyzer(mmm: Meridian) -> analyzer.Analyzer:
    '''Create an instance of the class Analyzer'''
    analyzer_instance = analyzer.Analyzer(
        model_context=mmm.model_context, inference_data=mmm.inference_data)
    return analyzer_instance


def compute_incremental_contribution(
        analyzer_instance: analyzer.Analyzer) -> pd.DataFrame:
    '''Estimate incremental contribution of each channel'''
    print('== CHANNEL CONTRIBUTION ANALYSIS ==')

    # Extract incremental outcomes
    incremental_contribution_df = (
        analyzer_instance.summary_metrics().incremental_outcome.sel(
            distribution='posterior', metric='mean').to_dataframe().reset_index()) 
    incremental_contribution_df.drop(
        columns=['metric', 'distribution'], inplace=True)
    
    # Compute the percentage of contribution for each channel
    single_channel_mask = incremental_contribution_df['channel'] != 'All Channels'
    total_outcome = incremental_contribution_df.loc[
        single_channel_mask, 'incremental_outcome'].sum()
    incremental_contribution_df.loc[
        single_channel_mask, 'contribution_percentage'] = (
            incremental_contribution_df.loc[
                single_channel_mask, 'incremental_outcome'] / total_outcome * 100)
    incremental_contribution_df.loc[
        ~single_channel_mask, 'contribution_percentage'] = 100.0
    
    # Compute the cumulative percentage of each channel
    incremental_contribution_df.loc[
        single_channel_mask, 'cumulative_percentage'] = (
            incremental_contribution_df.loc[
                single_channel_mask, 'contribution_percentage'].cumsum())
    incremental_contribution_df.loc[
        ~single_channel_mask, 'cumulative_percentage'] = 100.0

    return incremental_contribution_df


def compute_roi(analyzer_instance: analyzer.Analyzer) -> pd.DataFrame:
    '''Compute ROI per each channel together with credible intervals'''
    print('== ROI TABLE ==')

    # Extract ROI mean, median, and bounds of the credible interval
    roi_df = analyzer_instance.summary_metrics().roi.sel(
        distribution='posterior'
        ).drop_vars('distribution').to_dataframe().unstack('metric')
    
    roi_df.columns = [f'{column[0]}_{column[1]}' for column in roi_df.columns]
    roi_df = roi_df.reset_index().round(4)
    save_table(roi_df, 'roi_df.csv')
    return roi_df


def compute_mroi(analyzer_instance: analyzer.Analyzer) -> pd.DataFrame:
    '''Compute mROI per each channel together with credible intervals'''
    print('== MARGINAL ROI TABLE ==')

    # Extract mROI mean, median, and bounds of the credible interval
    mroi_df = analyzer_instance.summary_metrics().mroi.sel(
        distribution='posterior'
        ).drop_vars('distribution').to_dataframe().unstack()
    
    mroi_df.columns = [f'{column[0]}_{column[1]}' for column in mroi_df.columns]
    mroi_df = mroi_df.reset_index().round(4)
    save_table(mroi_df, 'mroi_df.csv')
    return mroi_df


def compute_roi_probability(mmm: Meridian) -> pd.DataFrame:
    '''Compute the posterior probability that a channel ROI is > 1'''
    print('== POSTERIOR PROBABILITY (ROI > 1) TABLE ==')

    roi_probability_df = (mmm.inference_data.posterior['roi_m'] >= 1).mean(
        dim=('chain', 'draw')).to_dataframe(name='P(ROI > 1)').reset_index()
    return roi_probability_df


def plot_response_curves(mmm: Meridian) -> None:
    '''Plot response curves with credible intervals to visualize how each 
    channel saturates as a function of spend'''
    print('== RESPONSE CURVES ANALYSIS ==\n')

    media_effects = visualizer.MediaEffects(mmm)
    save_chart(media_effects.plot_response_curves(
        plot_separately=True, include_ci=True), 'response_curves.html')