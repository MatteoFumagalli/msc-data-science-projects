# Import libraries
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns

# Define paths
VISUALIZATIONS_DIR = 'visualizations/eda'

# Define plot theme and parameters
sns.set_theme(style='white')
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# Define paths
OUTPUTS_CHARTS_DIR = '../outputs/charts/eda'


def save_plot(fig: Figure, file_name: str) -> None:
    '''Save the plot in the visualizations folder or in the current folder'''
    target_path = f'{OUTPUTS_CHARTS_DIR}/{file_name}'
    try:
        fig.savefig(target_path, dpi=300, bbox_inches='tight')
    except:
        fig.savefig(file_name, dpi=300, bbox_inches='tight')


def load_organisation_dataset(path: str, organisation_id: str) -> pd.DataFrame:
    '''Load the dataset related to a single organisation for all territories'''
    print('== DATASET LOADING ==')

    df = pd.read_csv(path)
    df['DATE_DAY'] = pd.to_datetime(df['DATE_DAY'])
    
    # Extract the organisation dataset and sort the timeseries by date
    organisation_df = df[
        (df['ORGANISATION_ID'] == organisation_id) & 
        (df['TERRITORY_NAME'] == 'All Territories')].copy()
    organisation_df = organisation_df.sort_values(
        by='DATE_DAY').reset_index(drop=True)
    print(f'Dataset of organisation {organisation_id} correctly loaded')
    
    return organisation_df


def inspect_missing_values(organisation_df: pd.DataFrame) -> pd.DataFrame:
    '''Create a report for inspecting missing values'''
    print('== MISSING VALUES ASSESSMENT ==')
    
    # Compute the number and the percentage of missing values for each column 
    num_missing = [organisation_df[column].isna().sum() 
                   for column in organisation_df.columns]
    percentage_missing = [round(organisation_df[column].isna().mean() * 100, 2) 
                          for column in organisation_df.columns]

    missing_values_df = pd.DataFrame({
        'COLUMN': organisation_df.columns,
        'NUM_MISSING': num_missing,
        'PERCENTAGE_MISSING': percentage_missing
    })
    return missing_values_df


def impute_missing_values(organisation_df: pd.DataFrame) -> pd.DataFrame:
    '''Impute missing values with 0 for media spend, clicks, and impressions'''
    print('== MISSING VALUES IMPUTATION ==')
    
    suffixes = ('_SPEND', '_CLICKS', '_IMPRESSIONS')
    columns_to_impute = [column for column in organisation_df.columns 
                                if column.endswith(suffixes)]
    
    # Perform the imputation
    imputed_df = organisation_df.copy()
    imputed_df[columns_to_impute] = imputed_df[columns_to_impute].fillna(0)

    print('Imputation completed for NaN values '
          'in media spend, clicks, and impressions')
    return imputed_df


def analyse_spending_channels(organisation_df: pd.DataFrame) -> pd.DataFrame:
    '''Analyse the coverage and spending of each media channel'''
    print('== MEDIA CHANNELS ANALYSIS ==')

    # Define the list of media channels
    spending_channels = [
        'GOOGLE_PAID_SEARCH_SPEND', 'GOOGLE_SHOPPING_SPEND',
        'GOOGLE_PMAX_SPEND', 'GOOGLE_DISPLAY_SPEND', 'GOOGLE_VIDEO_SPEND',
        'META_FACEBOOK_SPEND', 'META_INSTAGRAM_SPEND',
        'META_OTHER_SPEND', 'TIKTOK_SPEND']
    available_channels = [channel for channel in spending_channels
                                  if channel in organisation_df.columns]
    
    # Compute global spending across channels
    global_spending = organisation_df[available_channels].sum().sum()

    rows = []
    for channel in available_channels:
        # Compute coverage and overall spending for each channel
        channel_coverage = (
            (organisation_df[channel].notna()) &
            (organisation_df[channel] > 0)).mean()
        num_active_days = (
            (organisation_df[channel].notna()) &
            (organisation_df[channel] > 0)).sum()
        channel_spending = organisation_df[channel].sum()
            
        rows.append({
            'SPENDING_CHANNEL': channel.replace('_SPEND', ''),
            'ACTIVE_DAYS': num_active_days,
            'PERCENTAGE_COVERAGE': round(channel_coverage * 100, 2),
            'TOTAL_SPENDING': round(channel_spending, 2),
            'PERCENTAGE_TOTAL_SPENDING': round(
                (channel_spending / global_spending) * 100, 2) 
                if global_spending > 0 else 0
        })

    media_channels_report_df = pd.DataFrame(rows)
    return media_channels_report_df


def plot_purchases(
        organisation_df: pd.DataFrame, 
        moving_average: bool = False, window: int = 7) -> None:
    '''Display revenue and number of orders, either daily or as a moving average'''
    plot_df = organisation_df.copy()

    # Compute the average over the time window
    if moving_average:
        plot_df['REVENUE'] = (
            plot_df['ALL_PURCHASES_ORIGINAL_PRICE'].rolling(window=window).mean())
        plot_df['ORDERS'] = (
            plot_df['ALL_PURCHASES'].rolling(window=window).mean())
        title = f'Purchasing Trend ({window}-day Moving Average)'
        file_name = 'purchases_trend.png'

    # Display daily fluctuations
    else:
        plot_df['REVENUE'] = plot_df['ALL_PURCHASES_ORIGINAL_PRICE']
        plot_df['ORDERS'] = plot_df['ALL_PURCHASES']
        title = 'Daily Purchasing Trend'
        file_name = 'daily_purchases.png'

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot revenue trend
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Revenue (USD)', color='#91181a')
    sns.lineplot(data=plot_df,
                 x='DATE_DAY', y='REVENUE',
                 color='#91181a', ax=ax1)

    # Plot volume of orders trend
    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of Orders', color='#1a365d')
    sns.lineplot(data=plot_df,
                 x='DATE_DAY', y='ORDERS',
                 color='#1a365d', ax=ax2)
    
    plt.title(title, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, file_name)
    plt.show()


def plot_media_spend(
        organisation_df: pd.DataFrame, 
        moving_average: bool = False, window: int = 7) -> None:
    '''Display media spend, either daily or as a moving average'''
    spending_columns = [column for column in organisation_df.columns
                               if column.endswith('_SPEND')]
    
    # Order columns considering overall media spend
    total_spending = {column: organisation_df[column].sum() 
                      for column in spending_columns}
    sorted_spending_columns = sorted(
        spending_columns, key=lambda x: total_spending[x], reverse=True)
    
    plot_df = organisation_df.copy()
    fig, ax = plt.subplots(figsize=(10, 5))

    # Compute the average over the time window
    if moving_average: 
        for column in sorted_spending_columns:
            plot_df[f'MOVING_AVERAGE_{column}'] = (
                plot_df[column].rolling(window=window).mean())
            label = column.replace('MOVING_AVERAGE_', '').replace('_SPEND', '')
            sns.lineplot(data=plot_df,
                         x='DATE_DAY', y=f'MOVING_AVERAGE_{column}', 
                         ax=ax, label=label)
        title = f'Media Spend Trend ({window}-day Moving Average)'
        file_name = 'media_spend_trend.png'
    
    # Display daily fluctuations
    else:
        for column in sorted_spending_columns:
            label = column.replace('_SPEND', '')
            sns.lineplot(data=plot_df,
                         x='DATE_DAY', y=column, 
                         label=label, ax=ax)
        title = f'Daily Media Spend'
        file_name = 'daily_media_spend.png'
   
    plt.title(title, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Media Spend (USD)')
    plt.legend(loc='upper left', ncol=3, fontsize=8)
    
    plt.tight_layout()
    save_plot(fig, file_name)
    plt.show()


def plot_correlation(organisation_df: pd.DataFrame) -> None:
    '''Display the correlation among a meaningful subset of variables'''

    # Select the list of columns to include inside the plot
    purchase_columns = ['ALL_PURCHASES', 'ALL_PURCHASES_ORIGINAL_PRICE', 
                        'ALL_PURCHASES_GROSS_DISCOUNT']
    suffixes = ('_SPEND', '_CLICKS', '_IMPRESSIONS')

    columns_to_include = [column for column in organisation_df.columns
                          if column.endswith(suffixes) or
                             column in purchase_columns]
    
    fig, ax = plt.subplots(figsize=(10, 8))

    # Compute and display the correlation matrix
    correlation_matrix = organisation_df[columns_to_include].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix,
                mask=mask, annot=False, cmap='RdBu_r', ax=ax)
    plt.title('Correlation Matrix', fontweight='bold')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.tight_layout()
    save_plot(fig, 'correlation_matrix.png')
    plt.show()


def inspect_outliers(organisation_df: pd.DataFrame) -> pd.DataFrame:
    '''Inspect the presence of outliers among a subset of variables'''
    print('== OUTLIERS ASSESSMENT ==')
    
    # Select the list of columns to analyse
    purchase_columns = ['ALL_PURCHASES', 'ALL_PURCHASES_ORIGINAL_PRICE', 
                        'ALL_PURCHASES_GROSS_DISCOUNT']
    columns_to_include = [column for column in organisation_df.columns
                                 if column.endswith('_SPEND') or
                                    column in purchase_columns]

    rows = []
    for column in columns_to_include:
        # Compute quantiles and interquartile range for each column
        column_values = organisation_df[column]
        q1, q2, q3 = column_values.quantile(q=[0.25, 0.5, 0.75])
        iqr = q3 - q1

        # Compute the acceptable bounds and the number of outliers
        lower_bound = max(column_values.min(), q1 - 1.5 * iqr)
        upper_bound = min(column_values.max(), q3 + 1.5 * iqr)
        num_outliers = (
            (column_values < lower_bound) | (column_values > upper_bound)).sum()
        
        rows.append({
            'COLUMN': column,
            'LOWER_BOUND': round(lower_bound, 2),
            'Q1': round(q1, 2),
            'Q2': round(q2, 2),
            'Q3': round(q3, 2),
            'UPPER_BOUND': round(upper_bound, 2),
            'NUM_OUTLIERS': num_outliers,
            'PERCENTAGE_OUTLIERS': round(
                num_outliers / len(column_values) * 100, 2)
        })

    outliers_df = pd.DataFrame(rows)
    return outliers_df