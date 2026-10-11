# Library imports
import numpy as np
import pandas as pd
from IPython.display import display

# Define paths
DATA_DIR = 'raw_data'


def load_csv(file_path: str) -> pd.DataFrame:
    '''
    Load a .csv file and return the corresponding DataFrame
    '''
    return pd.read_csv(file_path)


def load_all_data() -> dict[str, pd.DataFrame]:
    '''
    Load a set of .csv files into a dictionary
    '''
    file_names = [
        'CATEGORY', 'PRODUCT', 'PRICE_CHANGE', 'CHANNEL', 
        'STORE', 'EXTERNAL_FACTOR', 'CUSTOMER', 'SESSION',
        'SUPPORT_TICKET', 'MARKETING_CHANNEL', 'CAMPAIGN_TOUCH',
        'PROMOTION', 'ORDER', 'ORDER_ITEM'         
    ]

    print(f'== DATAFRAMES LOADING ==')
    dataframes = {}
    for file_name in file_names:
        file_path = f'{DATA_DIR}/{file_name}.csv'

        try:
            dataframes[file_name.lower()] = load_csv(file_path)
            print(f" -> DataFrame {file_name.replace('_', ' ').title()} "
                  'loaded successfully')
        except Exception as e:
            print(' -> Error when loading DataFrame '
                  f"{file_name.replace('_', ' ').title()}: {e}")
            raise
    
    print(f'\nSuccessfully loaded all {len(dataframes)} DataFrames')
    return dataframes


def define_table_summary(dataframes: dict[str, pd.DataFrame]) -> None:
    '''
    Provide a brief description of each DataFrame
    '''
    descriptions = {
        'category': 'Macro-classification of products',
        'product': 'Catalogue of available products with brand and SKU',
        'price_change': 'Weekly history of price updates per product',
        'channel': 'Sales channels for purchasing',
        'store': 'Physical retail locations and types',
        'external_factor': 'Environmental and marketing conditions per store',
        'customer': 'Customer base profiles',
        'session': 'Digital activity and online behaviour of customers',
        'support_ticket': 'Customer service logs with resolution and CSAT score',
        'marketing_channel': 'Media platforms for advertising',
        'campaign_touch': 'Marketing touchpoints per customer across channels',
        'promotion': 'Promotional campaigns with discounts and offers',
        'order': 'Transaction records of purchases',
        'order_item': 'Specific item line within each order',
    }

    summary = []
    for file_name, df in dataframes.items():
        # Display the name, the number of rows and columns, and the description
        summary.append({
            'DataFrame Name': file_name.replace('_', ' ').capitalize(),
            'Records': len(df),
            'Columns': len(df.columns),
            'Description': descriptions.get(
                file_name, ' -> No description provided')
        })
    
    print(f'== TABLE SUMMARY ==')
    summary_df = pd.DataFrame(summary)
    display(summary_df)


def define_data_dictionary(dataframes: dict[str, pd.DataFrame]) -> None:
    '''
    Provide a dictionary of the columns within of each DataFrame
    '''
    for file_name, df in dataframes.items():
        dictionary = []
        for column in df.columns:
            # Display the name, the type, and the number of unique values
            dictionary.append({
                'Variable Name': column,
                'Type': df[column].dtype,
                'Unique Values': df[column].nunique()
            })

        print('== DATA DICTIONARY - TABLE: '
              f"{file_name.replace('_', ' ').upper()} ==")
        dictionary_df = pd.DataFrame(dictionary)
        display(dictionary_df)
        print('\n')


def analyse_missing_values(dataframes: dict[str, pd.DataFrame]) -> None:
    '''
    Analyse the distribution of missing values in each DataFrame
    '''
    missing_report = []
    for file_name, df in dataframes.items():
        # Extract only columns with missing values
        missing_counts = df.isna().sum()
        columns_with_nan = missing_counts[missing_counts > 0]

        for column_name, num_missing in columns_with_nan.items():
            percentage = round((num_missing / len(df)) * 100, 2)
            missing_report.append({
                'DataFrame Name': file_name,
                'Column Name': column_name,
                'Missing Values': num_missing,
                'Percentage (%)': percentage
            })

    print('== MISSING VALUES ==')
    if missing_report:
        missing_df = pd.DataFrame(missing_report)
        display(missing_df)
    else:
        print(' -> No missing values found across any table')


def preprocess_missing_values(
        dataframes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    '''
    Transform missing values from NaN to None for PostgreSQL compatibility
    '''
    print('== MISSING VALUES PREPROCESSING ==')
    try:
        dataframes['order_item']['promo_id'] = (
            dataframes['order_item']['promo_id'].astype(object).where(
                dataframes['order_item']['promo_id'].notna(), None)
        )

        print(' -> Successfully preprocessed column promo_id '
              'within DataFrame order_item')
        return dataframes
    except Exception as e:
        print(' -> Error when processing column promo_id '
              f'within DataFrame order_item: {e}')
        raise


def parse_date(values: pd.Series, format: str) -> pd.Series:
    '''
    Convert a date column into the format yyyy-mm-dd 
    '''
    return pd.to_datetime(values, format=format).dt.normalize()


def preprocess_date_time(
        dataframes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    '''
    Perform necessary preprocessing steps for date and timestamp values
    '''
    print('\n== DATE AND TIME PREPROCESSING ==')

    # Handle data formats
    date_configurations = [
        ('customer', 'signup_date', '%Y-%m-%d'),
        ('external_factor', 'factor_date', '%Y-%m-%d'),
        ('support_ticket', 'created_date', '%Y-%m-%d'),
        ('campaign_touch', 'touch_date', '%Y-%m-%d'),
        ('promotion', 'start_date', '%Y-%m-%d'),
        ('promotion', 'end_date', '%Y-%m-%d'),
        ('order', 'order_date', '%d-%m-%Y')
    ]
    
    for file_name, column, date_format in date_configurations:
        try:
            dataframes[file_name][column] = parse_date(
                dataframes[file_name][column], date_format)
            print(f' -> Processed column {column} within '
                  f'DataFrame {file_name}')
        except Exception as e:
            print(f' -> Error when processing column {column} within '
                  f'DataFrame {file_name}: {e}')
            raise

    # Handle timestamp
    try:    
        dataframes['session']['session_start'] = pd.to_datetime(
            dataframes['session']['session_start'])
        print(' -> Processed column session_start within DataFrame session')
    except Exception as e:
        print(' -> Error when processing column session_start '
              f'within DataFrame session: {e}')
        raise

    print('\nSuccessfully preprocessed date and time values')
    return dataframes


def preprocess_item_discount(
        dataframes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    '''
    Perform necessary preprocessing steps when discount exceeds line revenue
    '''
    print('\n== ITEM DISCOUNT PREPROCESSING ==')

    try:
        order_item_df = dataframes['order_item'].copy()
        order_df = dataframes['order'][['order_id', 'year', 'week']]
        price_df = dataframes['price_change'][[
            'product_id', 'year', 'week', 'unit_price']].copy()
        df = order_item_df.merge(order_df, on='order_id', how='left')
        
        # Create a temporal label for extracting the price
        df['time_label'] = df['year'] * 100 + df['week']
        df = df.sort_values('time_label')
        price_df['time_label'] = price_df['year'] * 100 + price_df['week']
        price_df = price_df.sort_values('time_label')

        # Search for the last available price or for the nearest in the future
        df_backward = pd.merge_asof(
            df, price_df[['product_id', 'time_label', 'unit_price']],
            on='time_label', by='product_id', direction='backward'
        )
        df_forward = pd.merge_asof(
            df, price_df[['product_id', 'time_label', 'unit_price']],
            on='time_label', by='product_id', direction='forward'
        )
        df_backward['unit_price'] = df_backward[
            'unit_price'].fillna(df_forward['unit_price'])
        df_corrected = df_backward.set_index(order_item_df.index).sort_index()

        line_revenue = df_corrected['quantity'] * df_corrected['unit_price']
        # If discount is greater than total revenue, it is capped at revenue
        discount_greater_revenue = (
            line_revenue.notna() &
            (order_item_df['item_discount'] > line_revenue)
        )
        num_corrected = discount_greater_revenue.sum()

        discount_corrected = np.where(
            discount_greater_revenue,
            line_revenue,
            order_item_df['item_discount']
        )
        dataframes['order_item']['item_discount'] = discount_corrected
        if num_corrected > 0:
            print(f' -> Successfully corrected {num_corrected} rows where '
                  'item_discount exceeded line revenue')
        else:
            print(' -> No discount correction needed')
        return dataframes
    except Exception as e:
        print(f' -> Error when processing item discount: {e}')
        raise


def check_price_timeline_anomalies(dataframes: dict[str, pd.DataFrame]) -> None:
    '''Detect products ordered before their first price is registered or
    products without a registered price'''
    print('== PRODUCT PRICE TIMELINE CHECK ==')

    try:
        order_item_df = dataframes['order_item']
        order_df = dataframes['order'][['order_id', 'year', 'week']]
        price_df = dataframes['price_change'][[
            'product_id', 'year', 'week']].copy()

        # Extract the date of first order for each product
        df = order_item_df.merge(order_df, on='order_id', how='left')
        df['time_label'] = df['year'] * 100 + df['week']

        # Extract the date of first price for each product
        price_df['time_label'] = price_df['year'] * 100 + price_df['week']
        first_price_df = price_df.groupby(
            'product_id')['time_label'].min().reset_index()
        first_price_df.columns = ['product_id', 'first_price_time']

        # Compute the number of products with a price after order or without it
        timeline_df = df.merge(first_price_df, on='product_id', how='left')
        rows_without_price = timeline_df[
            timeline_df['first_price_time'].isna()].shape[0]
        total_order_rows = len(df)
        perc_rows_without_price = (rows_without_price / total_order_rows) * 100

        price_after_order = (
            timeline_df['first_price_time'].notna() & 
            (timeline_df['time_label'] < timeline_df['first_price_time'])
        )
        rows_after_order = timeline_df[price_after_order].shape[0]
        perc_rows_after_order = (rows_after_order / total_order_rows) * 100
        
        if rows_after_order > 0 or rows_without_price > 0:
            if rows_after_order > 0:
                print(f' -> Found {rows_after_order} order rows with a price '
                      'registered only after the order '
                      f'({perc_rows_after_order:.2f}% of total transactions)')
            if rows_without_price > 0:
                print(f' -> Found {rows_without_price} order rows without a '
                      f'registered price ({perc_rows_without_price:.2f}% of '
                      'total transactions)')
        else:
            print('-> All product prices are aligned with order timelines')
    except Exception as e:
        print(f' -> Error when running price timeline check: {e}')
        raise


def normalize_campaign_channels(
        dataframes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    '''
    Map channels in campaign_touch to marketing_channel ids for integrity
    '''
    print('== DATABASE INTEGRITY PREPROCESSING ==')

    try:
        channel_mapping = dict(zip(
            dataframes['marketing_channel']['channel_name'], 
            dataframes['marketing_channel']['channel_id'])
        )
        dataframes['campaign_touch']['channel'] = (
            dataframes['campaign_touch']['channel'].map(channel_mapping))
        dataframes['campaign_touch'] = dataframes['campaign_touch'].rename(
            columns={'channel': 'channel_id'})

        print(' -> Successfully aligned campaign_touch with marketing_channel')
        return dataframes
    except Exception as e:
        print(' -> Error when aligning campaign_touch with '
              f'marketing_channel: {e}')
        raise


def display_summary_statistics(dataframes: dict[str, pd.DataFrame]) -> None:
    '''
    Display summary statistics for all variables involved
    '''
    print('== SUMMARY STATISTICS ==')

    for table_name, df in dataframes.items():
        print(f"\nTable: {table_name.replace('_', ' ').upper()}")

        # Compute statistics for numerical variables
        numerical_columns = df.select_dtypes(
            include=['number']).columns.tolist()
        if numerical_columns:
            print('\n -> Numerical variables')
            numerical_stats = df[numerical_columns].describe().round(2)
            display(numerical_stats)
        else:
            print('\n -> No numerical variables found')

        # Compute statistics for temporal variables
        temporal_columns = df.select_dtypes(
            include=['datetime', 'datetime64']).columns.tolist()
        if temporal_columns:
            print('\n -> Temporal variables')
            for column in temporal_columns:
                print(f"{column}: {df[column].min().strftime('%Y-%m-%d')} "
                      f"-> {df[column].max().strftime('%Y-%m-%d')}")
        else:
            print('\n -> No temporal variables found')

        # Compute statistics for categorical variables
        categorical_columns = df.select_dtypes(
            include=['object', 'category', 'str']).columns.tolist()
        filtered_categorical_columns = [
            category for category in categorical_columns 
            if not category.endswith('_id') and category != 'sku_name']
        if filtered_categorical_columns:
            print('\n -> Categorical variables')
            for column in filtered_categorical_columns:
                print(f'\nVariable: {column}')
                counts = df[column].value_counts()
                percentages = df[column].value_counts(normalize=True) * 100
                categorical_counts = pd.DataFrame({
                    'Count': counts,
                    'Percentage (%)': percentages.round(2)
                })
                display(categorical_counts)
        else:
            print('\n -> No categorical variables found')
        print('\n')


def check_duplicates(dataframes: dict[str, pd.DataFrame]) -> None:
    '''
    Check whether there are duplicate rows or primary keys
    '''
    print('== DATA QUALITY DUPLICATES CHECK ==')

    primary_keys = {
        'category': ['category_id'],
        'product': ['product_id'],
        'price_change': ['product_id', 'year', 'week'],
        'channel': ['channel_id'],
        'store': ['store_id'],
        'external_factor': ['factor_id'],
        'customer': ['customer_id'],
        'session': ['session_id'],
        'support_ticket': ['ticket_id'],
        'campaign_touch': ['touch_id'],
        'marketing_channel': ['channel_id'],
        'promotion': ['promo_id'],
        'order': ['order_id'],
        'order_item': ['order_item_id']
    }

    for table_name, df in dataframes.items():
        print(f"\nTable: {table_name.replace('_', ' ').upper()}")

        # Check duplicate rows
        num_duplicates = df.duplicated().sum()
        if num_duplicates > 0:
            print(f' -> Duplicate rows: {num_duplicates}')
        else:
            print(f' -> There are no duplicate rows')

        # Check duplicated primary keys
        primary_key_columns = primary_keys.get(table_name, [])
        if primary_key_columns:
            primary_key_duplicates = df.duplicated(
                subset=primary_key_columns).sum()
            if primary_key_duplicates > 0:
                print(' -> Primary key duplicate violations: '
                    f'{primary_key_duplicates}')
            else:
                print(f' -> There are no duplicate primary keys')
        else:
            print(' -> Error when searching the primary key: '
                  f"{', '.join(primary_key_columns)}")