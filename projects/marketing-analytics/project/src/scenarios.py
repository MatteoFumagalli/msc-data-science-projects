# Import libraries
from IPython.display import display
import numpy as np
import re
import pandas as pd

from meridian.analysis import optimizer
from meridian.model.model import Meridian

# Define paths
OUTPUTS_TABLES_DIR = '../outputs/tables'


def save_table(df: pd.DataFrame, file_name: str) -> None:
    '''Save the table in the visualizations folder or in the current folder'''
    target_path = f'{OUTPUTS_TABLES_DIR}/{file_name}'
    try:
        df.to_csv(target_path, index=False)
    except:
        df.to_csv(file_name, index=False)


def compute_current_spend_levels(
        mmm: Meridian) -> tuple[list, np.ndarray, float]:
    '''Obtain the current level of spend for each channel and the overall amount'''
    print('== SCENARIO SIMULATION GENERAL SETTING ==')

    daily_media_spend = np.asarray(mmm.input_data.media_spend)
    media_channels = mmm.input_data.media_channel.values.tolist()
    print(f" -> Media channels: {', '.join(media_channels)}")

    media_spend = daily_media_spend.sum(axis=(0, 1))
    total_spend = float(media_spend.sum())
    print(f' -> Total spend across channels: {total_spend:,.2f}')

    return media_channels, media_spend, total_spend


def extract_channel_metrics(
        roi_df: pd.DataFrame, mroi_df: pd.DataFrame) -> pd.DataFrame:
    '''Extract ROI and mROI for each channel'''
    metrics_df = roi_df.merge(mroi_df, on='channel')
    metrics_df = metrics_df[['channel', 'roi_mean', 'mroi_mean']]
    
    return metrics_df


def get_most_performing_channel(metrics_df: pd.DataFrame) -> str:
    '''Define the most performing channel according to mean ROI'''
    most_performing_channel = metrics_df.sort_values(
        'roi_mean', ascending=False).iloc[0]['channel']
    
    return most_performing_channel


def get_saturated_channels(metrics_df: pd.DataFrame) -> list:
    '''Define the list of saturated channels whose mROI is below the median'''
    mroi_median = metrics_df['mroi_mean'].median()
    saturated_channels = metrics_df[
        (metrics_df['mroi_mean'] < mroi_median)]['channel'].tolist()
    
    return saturated_channels


def perform_budget_optimization(
        scenario_approach: str, mmm: Meridian, 
        new_budget: float, media_spend_percentage: list = None) -> dict:
    '''Perform the optimization procedure to better allocate the available budget'''
    budget_optimizer = optimizer.BudgetOptimizer(mmm)

    # Run the optimization
    optimization_results = budget_optimizer.optimize(
        use_posterior=True, fixed_budget=True, 
        budget=new_budget, pct_of_spend=media_spend_percentage)
    
    # Extract useful metrics for scenario comparison
    optimized_metrics = optimization_results.optimized_data
    non_optimized_metrics = optimization_results.nonoptimized_data
    optimized_incremental_revenue = float(
        optimized_metrics.attrs['total_incremental_outcome'])
    non_optimized_incremental_revenue = float(
        non_optimized_metrics.attrs['total_incremental_outcome'])
    roi = float(optimized_metrics.attrs['total_roi'])

    summary_results = {
        'scenario_approach': scenario_approach,
        'budget': new_budget,
        'incremental_revenue': optimized_incremental_revenue,
        'uplift': (
            optimized_incremental_revenue - non_optimized_incremental_revenue),
        'roi': roi
    }
    print(f' -> {scenario_approach}')
    return summary_results


def define_equal_allocation_approach(
        scenario_type: str, mmm: Meridian, 
        media_channels: list, new_budget: float) -> dict:
    '''Define the approach for equal allocation across channels'''
    num_channels = len(media_channels)
    media_spend_percentage = [1 / num_channels] * num_channels
    
    return perform_budget_optimization(
        f'{scenario_type} - Equal Allocation', 
        mmm, new_budget, media_spend_percentage)


def define_highest_mroi_allocation_approach(
        metrics_df: pd.DataFrame, scenario_type: str, 
        mmm: Meridian, new_budget: float) -> dict:
    '''Define the approach for allocation based on highest mROI'''
    mroi = metrics_df['mroi_mean'][:-1].to_numpy()
    media_spend_percentage = (mroi / mroi.sum()).tolist()

    return perform_budget_optimization(
        f'{scenario_type} - mROI Allocation', mmm, 
        new_budget, media_spend_percentage)


def define_saturated_allocation_approach(
        metrics_df: pd.DataFrame, scenario_type: str, mmm: Meridian, 
        media_channels: list, media_spend: np.ndarray, new_budget: float) -> dict:
    '''Define the approach for reduced investment in saturated channels'''
    saturated_channels = get_saturated_channels(metrics_df)
    saturated_indexes = [
        media_channels.index(channel) for channel in saturated_channels]
    
    # Define constraints for each channel
    current_media_spend_percentage = media_spend / media_spend.sum()
    for i in saturated_indexes:
        current_media_spend_percentage[i] *= 0.7
    current_media_spend_percentage /= current_media_spend_percentage.sum()

    return perform_budget_optimization(
        f'{scenario_type} - Saturated Channels Allocation',
        mmm, new_budget, current_media_spend_percentage.tolist())


def optimize_scenario_1(
        mmm: Meridian, media_channels: list, 
        media_spend: np.ndarray, total_spend: float, 
        metrics_df: pd.DataFrame) -> list:
    '''Optimize marketing investment with +20% on total spend'''
    new_budget = total_spend * 1.2
    scenario_type = 'Scenario 1 (+20% total)'
    
    return [
        define_equal_allocation_approach(
            scenario_type, mmm, media_channels, new_budget),
        define_highest_mroi_allocation_approach(
            metrics_df, scenario_type, mmm, new_budget),
        define_saturated_allocation_approach(
            metrics_df, scenario_type, mmm, 
            media_channels, media_spend, new_budget)
    ]


def optimize_scenario_2(
        mmm: Meridian, media_channels: list, 
        media_spend: np.ndarray, total_spend: float, 
        metrics_df: pd.DataFrame) -> list:
    '''Optimize marketing investment with +50% of spend on the
    most performing channel'''

    # Extract the most performing channel and compute the new budget
    most_performing_channel = get_most_performing_channel(metrics_df)
    most_performing_channel_index = media_channels.index(most_performing_channel)
    new_budget = total_spend + media_spend[most_performing_channel_index] * 0.5
    scenario_type = 'Scenario 2 (+50% most performing channel)'
    
    return [
        define_equal_allocation_approach(
            scenario_type, mmm, media_channels, new_budget),
        define_highest_mroi_allocation_approach(
            metrics_df, scenario_type, mmm, new_budget),
        define_saturated_allocation_approach(
            metrics_df, scenario_type, mmm, 
            media_channels, media_spend, new_budget)
    ]


def optimize_scenario_3(
        mmm: Meridian, media_channels: list, 
        media_spend: np.ndarray, total_spend: float, 
        metrics_df: pd.DataFrame) -> list:
    '''Optimize marketing investment with -30% of spend on saturated channels'''

    # Extract the saturated channels and compute the new budget
    saturated_channels = get_saturated_channels(metrics_df)
    saturated_channels_indexes = [
        media_channels.index(channel) for channel in saturated_channels]
    new_budget = total_spend - media_spend[saturated_channels_indexes].sum() * 0.3
    scenario_type = 'Scenario 3 (-30% saturated channels)'
    
    return [
        define_equal_allocation_approach(
            scenario_type, mmm, media_channels, new_budget),
        define_highest_mroi_allocation_approach(
            metrics_df, scenario_type, mmm, new_budget),
        define_saturated_allocation_approach(
            metrics_df, scenario_type, mmm, 
            media_channels, media_spend, new_budget)
    ]


def optimize_scenario_4(
        mmm: Meridian, media_channels: list, 
        media_spend: np.ndarray, total_spend: float, 
        metrics_df: pd.DataFrame) -> list:
    '''Optimize marketing investment with +10% randomly on half of channels'''

    # Extract random channels to increase the spend
    num_channels = len(media_channels)
    random_generator = np.random.default_rng(3)
    random_channels = random_generator.choice(
        num_channels, num_channels // 2, replace=False)
    new_budget = total_spend + media_spend[random_channels].sum() * 0.1
    scenario_type = 'Scenario 4 (+10% randomly on half channels)'
    
    return [
        define_equal_allocation_approach(
            scenario_type, mmm, media_channels, new_budget),
        define_highest_mroi_allocation_approach(
            metrics_df, scenario_type, mmm, new_budget),
        define_saturated_allocation_approach(
            metrics_df, scenario_type, mmm, 
            media_channels, media_spend, new_budget)
    ]


def compare_scenarios(scenarios_results: list) -> pd.DataFrame:
    '''Build a final comparison table among different investment scenarios'''
    print('\n== SCENARIOS COMPARISON ==')

    scenarios_comparison_df = pd.DataFrame([{
        'Scenario': re.sub(
            r'\s*\([^)]*\)', '', scenario_row['scenario_approach']).strip(),
        'Budget': round(scenario_row['budget'], 2),
        'Incremental Revenue': round(scenario_row['incremental_revenue'], 2),
        'Uplift': round(scenario_row['uplift'], 2),
        'ROI': round(scenario_row['roi'], 4)
    } 
    for scenario_row in scenarios_results])

    save_table(scenarios_comparison_df, 'scenarios_comparison_df.csv')
    return scenarios_comparison_df