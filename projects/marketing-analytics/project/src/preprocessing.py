# Import libraries
import numpy as np
import pandas as pd

# Define paths
OUTPUTS_TABLES_DIR = '../outputs/tables'


def save_table(df: pd.DataFrame, file_name: str) -> None:
    '''Save the table in the visualizations folder or in the current folder'''
    target_path = f'{OUTPUTS_TABLES_DIR}/{file_name}'
    try:
        df.to_csv(target_path, index=False)
    except:
        df.to_csv(file_name, index=False)


def verify_daily_aggregation(organisation_df: pd.DataFrame) -> None:
    '''Verify whether the dataset contains all consecutive dates'''
    all_dates = pd.date_range(organisation_df['DATE_DAY'].min(), 
                              organisation_df['DATE_DAY'].max(), freq='D')
    num_missing_dates = all_dates.difference(organisation_df['DATE_DAY'])
    print(f'\n -> There are {len(num_missing_dates)} missing dates between '
          f"{organisation_df['DATE_DAY'].min().strftime('%Y-%m-%d')} and "
          f"{organisation_df['DATE_DAY'].max().strftime('%Y-%m-%d')}")


def extract_time_features(organisation_df: pd.DataFrame) -> pd.DataFrame:
    '''Extract temporal features for each date'''
    features_df = organisation_df.copy()
    features_df['DAY_OF_WEEK'] = features_df['DATE_DAY'].dt.dayofweek
    features_df['IS_WEEKEND'] = features_df['DAY_OF_WEEK'].isin([5, 6]).astype(int)
    features_df['MONTH'] = features_df['DATE_DAY'].dt.month

    time_variables = ['DAY_OF_WEEK', 'IS_WEEKEND', 'MONTH']
    print('\n -> The time variables extracted from DATE_DAY are '
          f"{', '.join(time_variables)}")
    return features_df


def verify_missing_values(features_df: pd.DataFrame) -> None:
    '''Verify whether there are remaining missing values'''
    num_missing_values = features_df.isna().sum().sum()
    if num_missing_values == 0:
        print(f'\n -> There are {num_missing_values} missing values. '
              'Manual imputation has already been performed')
    else:
        print(f'\n -> There are {num_missing_values} missing values')


def perform_holdout_indexing(
        features_df: pd.DataFrame, holdout_size: float = 0.25
        ) -> pd.DataFrame:
    '''Create a distributed holdout sample'''
    holdout_df = features_df.copy()
    
    # Compute the index of each holdout instance
    num_holdout = int(round(len(holdout_df) * holdout_size))
    holdout_index = np.linspace(
        0, len(holdout_df)-1, num_holdout, dtype=int)
    
    # Create a holdout column
    holdout_df['HOLDOUT'] = False
    holdout_df.loc[holdout_index, 'HOLDOUT'] = True

    print('\n -> Successfully performed the holdout indexing')
    print(f'    Train set: {len(features_df) - num_holdout} days (75%) '
          f'- Validation set: {num_holdout} days (25%)')
    return holdout_df


def preprocess_dataset(
        organisation_df: pd.DataFrame) -> pd.DataFrame:
    '''Prepare the dataset for MMM modelling'''
    print('== DATASET PREPROCESSING ==')

    # Verify the aggregation of the dataset at daily level
    verify_daily_aggregation(organisation_df)

    # Select the target variable
    target_variable = 'ALL_PURCHASES_ORIGINAL_PRICE'
    print(f'\n -> The target variable is {target_variable}')

    # Create media variables
    media_spend_variables = [
        'GOOGLE_PAID_SEARCH_SPEND', 'GOOGLE_SHOPPING_SPEND', 
        'GOOGLE_PMAX_SPEND', 'GOOGLE_VIDEO_SPEND',
        'META_FACEBOOK_SPEND', 'META_INSTAGRAM_SPEND', 'TIKTOK_SPEND']
    print('\n -> The media spend variables are '
          f"{', '.join(media_spend_variables)}")
    
    media_exposure_variables = [
        'GOOGLE_PAID_SEARCH_CLICKS', 'GOOGLE_SHOPPING_CLICKS', 
        'GOOGLE_PMAX_CLICKS', 'GOOGLE_VIDEO_IMPRESSIONS',
        'META_FACEBOOK_IMPRESSIONS', 'META_INSTAGRAM_IMPRESSIONS', 
        'TIKTOK_IMPRESSIONS']
    print(' -> The media exposure variables are '
          f"{', '.join(media_exposure_variables)}")
    
    channels_to_exclude = ['GOOGLE_DISPLAY', 'META_OTHER']
    print(f" -> The media variables related to {', '.join(channels_to_exclude)} "
          'are excluded as they contribute a share close to 0% of the total spend')

    # Create control variables
    control_clicks_variables = [
        'DIRECT_CLICKS', 'ORGANIC_SEARCH_CLICKS', 'EMAIL_CLICKS',
        'REFERRAL_CLICKS', 'ALL_OTHER_CLICKS']
    print('\n -> The organic media variables used as control are '
          f"{', '.join(control_clicks_variables)}")
    control_clicks_to_exclude = 'BRANDED_SEARCH_CLICKS'
    print(f' -> The control variable {control_clicks_to_exclude} is excluded '
          'due to the high presence of missing values, which is around 98%')

     # Extract time variables
    features_df = extract_time_features(organisation_df)

    # Handle missing values
    verify_missing_values(features_df)

    # Refine the structure of the both sets
    columns_to_keep = (
        ['DATE_DAY', target_variable] + 
        media_spend_variables + media_exposure_variables +
        control_clicks_variables + 
        ['DAY_OF_WEEK', 'IS_WEEKEND', 'MONTH']
    )
    feature_engineering_df = features_df[columns_to_keep].copy()
    feature_engineering_df['DATE_DAY'] = pd.to_datetime(
        feature_engineering_df['DATE_DAY'])
    
    # Perform the holdout indexing
    mmm_df = perform_holdout_indexing(feature_engineering_df)
    save_table(mmm_df, 'mmm_df.csv')

    return mmm_df