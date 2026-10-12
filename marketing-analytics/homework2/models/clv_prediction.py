# Library imports
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from psycopg2.extensions import connection
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import (
    plot_frequency_recency_matrix,
    plot_probability_alive_matrix)
from lifetimes.utils import (
    calibration_and_holdout_data, summary_data_from_transaction_data)

# Define paths
VISUALIZATIONS_DIR = 'visualizations/clv_prediction'

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
    target_path = f'{VISUALIZATIONS_DIR}/{file_name}'
    try:
        fig.savefig(target_path, dpi=300, bbox_inches='tight')
    except:
        fig.savefig(file_name, dpi=300, bbox_inches='tight')


def extract_transactions_data(conn: connection) -> pd.DataFrame:
    '''Extract a summary record for each order'''
    query = '''
        SELECT o.customer_id, o.order_date, o.order_id,
        SUM((oi.quantity * COALESCE (
                (
                    SELECT pc.unit_price
                    FROM price_change AS pc
                    WHERE pc.product_id = oi.product_id
                        AND (pc.year < o.year 
                            OR (pc.year = o.year AND pc.week <= o.week))
                    ORDER BY pc.year DESC, pc.week DESC
                    LIMIT 1                
                ),
                (
                    SELECT pc.unit_price
                    FROM price_change AS pc
                    WHERE pc.product_id = oi.product_id
                    ORDER BY pc.year ASC, pc.week ASC
                    LIMIT 1 
                )          
            )) - oi.item_discount)::FLOAT AS monetary
        FROM "order" AS o
            JOIN order_item AS oi ON oi.order_id = o.order_id
        GROUP BY o.customer_id, o.order_date, o.order_id
        ORDER BY o.customer_id, o.order_date;
    '''

    try:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
            column_names = [description[0] for description in cur.description]
        df = pd.DataFrame(result, columns=column_names)
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['monetary'] = df['monetary'].astype(float)
        return df
    except Exception as e:
        print(f'Database error when executing the query: {e}')
        return pd.DataFrame(
            columns=['customer_id', 'order_date', 'order_id', 'monetary'])
    

def split_calibration_holdout_df(
        transactions_df: pd.DataFrame, 
        calibration_ratio: float = 0.75) -> pd.DataFrame:
    '''Split transactions into calibration (75%) and holdout (25%) periods'''
    # Extract important dates to perform the split
    date_start = transactions_df['order_date'].min()
    date_end = transactions_df['order_date'].max()
    number_days = (date_end - date_start).days
    number_calibration_days = int(number_days * calibration_ratio)
    calibration_end_date = (
        date_start + pd.Timedelta(days=number_calibration_days))
    
    print('== CALIBRATION AND HOLDOUT SPLIT ==')
    print(f' -> Observation window: {number_days} days')
    print(f" -> Calibration period (75%): from {date_start.strftime('%Y-%m-%d')}"
          f" to {calibration_end_date.strftime('%Y-%m-%d')}")
    print(' -> Holdout period (25%): from '
          f"{(calibration_end_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')}"
          f" to {date_end.strftime('%Y-%m-%d')}")
    
    calibration_holdout_df = calibration_and_holdout_data(
        transactions_df,
        customer_id_col='customer_id',
        datetime_col='order_date',
        calibration_period_end=calibration_end_date,
        observation_period_end=date_end,
        freq='D',
        monetary_value_col='monetary'
    )

    return calibration_holdout_df
    

def train_predictive_models(
        calibration_holdout_df: pd.DataFrame,
        penalizer_coefficient: float = 0.01
        ) -> tuple[BetaGeoFitter, GammaGammaFitter]:
    '''Train the BG/NBD model and the Gamma-Gamma model on the calibration 
    period. For the latter, check whether frequency and monetary value 
    are not correlated by using the Pearson correlation coefficient'''
    print('== PREDICTIVE MODELS TRAINING ==')
    
    print(' -> BG/NBD Model')
    bgf = BetaGeoFitter(penalizer_coef=penalizer_coefficient)
    bgf.fit(calibration_holdout_df['frequency_cal'],
            calibration_holdout_df['recency_cal'],
            calibration_holdout_df['T_cal'])

    print(f"   Parameters: r={bgf.params_['r']:.4f}, "
        f"\u03b1={bgf.params_['alpha']:.4f}, "
        f"a={bgf.params_['a']:.4f}, b={bgf.params_['b']:.4f}")
    
    print('\n -> Gamma-Gamma Model')
    # Extract customers with repeat purchase
    repeated_customers = calibration_holdout_df[
        (calibration_holdout_df['frequency_cal'] > 0) & 
        (calibration_holdout_df['monetary_value_cal'] > 0)]
    corr_coeff = repeated_customers[
        ['frequency_cal', 'monetary_value_cal']].corr().iloc[0, 1]
    print('   Pearson correlation coefficient between frequency and monetary: '
          f'{corr_coeff:.4f}')
    
    ggf = GammaGammaFitter(penalizer_coef=penalizer_coefficient)
    ggf.fit(repeated_customers['frequency_cal'],
            repeated_customers['monetary_value_cal'])

    print(f"   Parameters: p={ggf.params_['p']:.4f}, "
        f"q={ggf.params_['q']:.4f}, v={ggf.params_['v']:.4f}")
    
    return bgf, ggf


def evaluate_predictive_models(
    calibration_holdout_df: pd.DataFrame,
    bgf: BetaGeoFitter, ggf: GammaGammaFitter) -> dict[str, float]:
    '''Evaluate model performance (RMSE, MAE, R-squared) on the holdout period'''
    print('== PREDICTIVE MODELS EVALUATION ==')
    holdout_duration = int(calibration_holdout_df['duration_holdout'].iloc[0])

    # Compute expected frequency
    actual_frequency = calibration_holdout_df['frequency_holdout']
    predicted_frequency = (
        bgf.conditional_expected_number_of_purchases_up_to_time(
            holdout_duration,
            calibration_holdout_df['frequency_cal'], 
            calibration_holdout_df['recency_cal'],
            calibration_holdout_df['T_cal']
        )).fillna(0)
    
    # Evaluate transactional accuracy at individual level
    rmse_frequency = np.sqrt(
        mean_squared_error(actual_frequency, predicted_frequency))
    mae_frequency = mean_absolute_error(actual_frequency, predicted_frequency)
    r2_frequency = r2_score(actual_frequency, predicted_frequency)   
   
    # Exclude inactive customers during holdout period for the Gamma-Gamma model
    active_test_customers = calibration_holdout_df[
        (calibration_holdout_df['frequency_cal'] > 0) &
        (calibration_holdout_df['frequency_holdout'] > 0) &
        (calibration_holdout_df['monetary_value_holdout'] > 0)].copy()
    
    # Compute expected monetary value
    actual_monetary = active_test_customers['monetary_value_holdout']
    predicted_monetary = (
        ggf.conditional_expected_average_profit(
            active_test_customers['frequency_cal'], 
            active_test_customers['monetary_value_cal']
        )).fillna(active_test_customers['monetary_value_cal'].mean())
    
    # Evaluate monetary accuracy at population level
    mean_actual_monetary = actual_monetary.mean()
    mean_predicted_monetary = predicted_monetary.mean()

    # Evalute combined total spend accuracy
    actual_total_spend = (
        calibration_holdout_df['monetary_value_holdout'] *
        (calibration_holdout_df['frequency_holdout'] + 1)).fillna(0)    
    global_predicted_monetary = (
        ggf.conditional_expected_average_profit(
            calibration_holdout_df['frequency_cal'], 
            calibration_holdout_df['monetary_value_cal']
        )).fillna(calibration_holdout_df['monetary_value_cal'].mean())
    predicted_total_spend = predicted_frequency * global_predicted_monetary    
    rmse_combined = np.sqrt(
        mean_squared_error(actual_total_spend, predicted_total_spend))
    mae_combined = mean_absolute_error(actual_total_spend, predicted_total_spend)
    r2_combined = r2_score(actual_total_spend, predicted_total_spend)

    # Save evaluation metrics
    metrics = {
        'rmse_frequency': round(rmse_frequency, 4),
        'mae_frequency': round(mae_frequency, 4),
        'r2_frequency': round(r2_frequency, 4),
        'rmse_combined_spend': round(rmse_combined, 4),
        'mae_combined_spend': round(mae_combined, 4),
        'r2_combined_spend': round(r2_combined, 4)
    }

    print(' -> BG/NBD Model')
    print(f"   RMSE: {metrics['rmse_frequency']}")
    print(f"   MAE: {metrics['mae_frequency']}")
    print(f"   R^2: {metrics['r2_frequency']}")

    print('\n -> Gamma-Gamma Model')
    print(f'     Mean actual spending: \u20B9 {mean_actual_monetary:.2f}')
    print(f'     Mean predicted spending: \u20B9 {mean_predicted_monetary:.2f}')

    print('\n -> Combined Performance')
    print(f"   RMSE: {metrics['rmse_combined_spend']}")
    print(f"   MAE: {metrics['mae_combined_spend']}")
    print(f"   R^2: {metrics['r2_combined_spend']}")

    return metrics


def predict_clv_customer_table(
    transactions_df: pd.DataFrame,
    bgf: BetaGeoFitter, ggf: GammaGammaFitter,
    horizon_months: int = 6, 
    discount_rate_monthly: float = 0.01) -> pd.DataFrame:
    '''Compute predicted CLV, expected number of purchases and monetary value, 
    and probability of being active for each customer over a 6-month window'''
    print(f'== CLV PREDICTION TABLE ({horizon_months} MONTHS HORIZON) ==')

    summary_df = summary_data_from_transaction_data(
        transactions_df,
        customer_id_col='customer_id',
        datetime_col='order_date',
        monetary_value_col='monetary',
        observation_period_end=transactions_df['order_date'].max()
    )
    summary_df = summary_df.reset_index()
    horizon_days = horizon_months * 30

    # Compute expected frequency in the next 180 days
    summary_df['expected_frequency'] = (
        bgf.conditional_expected_number_of_purchases_up_to_time(
            horizon_days,
            summary_df['frequency'], summary_df['recency'], summary_df['T']
        ))
    
    # Compute the probability of being active
    summary_df['p_alive'] = bgf.conditional_probability_alive(
        summary_df['frequency'], summary_df['recency'], summary_df['T']
    )

    # Compute expected average monetary value
    summary_df['expected_average_monetary'] = (
        ggf.conditional_expected_average_profit(
            summary_df['frequency'], summary_df['monetary_value']
        ).fillna(summary_df['monetary_value'].mean()))
    
    # Compute the customer lifetime value at 6 months
    valid_gamma_gamma = (
        (summary_df['frequency']) > 0 & (summary_df['monetary_value'] > 0))
    summary_df['predicted_clv'] = np.nan
    summary_df.loc[valid_gamma_gamma, 'predicted_clv'] = (
        ggf.customer_lifetime_value(
            bgf, 
            summary_df.loc[valid_gamma_gamma, 'frequency'], 
            summary_df.loc[valid_gamma_gamma, 'recency'],
            summary_df.loc[valid_gamma_gamma, 'T'], 
            summary_df.loc[valid_gamma_gamma, 'monetary_value'],
            time=horizon_months,
            discount_rate=discount_rate_monthly,
            freq='D'
        ))
    summary_df['predicted_clv'] = summary_df['predicted_clv'].fillna(0)

    print(' -> Successfully computed the predicted CLV customer table')

    summary_df['frequency'] = summary_df['frequency'].astype(int)
    summary_df['recency'] = summary_df['recency'].astype(int)
    summary_df['T'] = summary_df['T'].astype(int)

    summary_df['expected_frequency'] = summary_df['expected_frequency'].round(2)
    summary_df['p_alive'] = summary_df['p_alive'].round(4)
    summary_df['expected_average_monetary'] = (
        summary_df['expected_average_monetary'].round(2))
    summary_df['predicted_clv'] = summary_df['predicted_clv'].round(2)
    summary_df['monetary_value'] = summary_df['monetary_value'].round(2)
    
    print('\n== TOP 10 CUSTOMERS BY PREDICTED CLV '
          F'({horizon_months} MONTHS HORIZON) ==')
    display(summary_df.sort_values(
        by='predicted_clv', ascending=False).head(10).reset_index(drop=True))
    
    return summary_df


def plot_predicted_clv_distribution(clv_prediction_df: pd.DataFrame) -> None:
    '''Display a histogram for the distribution of predicted CLV'''
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        clv_prediction_df['predicted_clv'], 
        bins=40, color='#91181a', ax=ax)
    ax.set_title(f'Predicted CLV Distribution (6 months horizon)')
    ax.set_xlabel('Predicted CLV ($\u20B9$)')
    ax.set_ylabel('Number of Customers')
    fig.tight_layout()
    save_plot(fig, 'clv_distribution.png')
    plt.show()


def plot_actual_and_predicted_spend(
    calibration_holdout_df: pd.DataFrame,
    bgf: BetaGeoFitter, ggf: GammaGammaFitter) -> None:
    '''Display a scatterplot of actual vs predicted spend in the holdout period'''
    # Compute expected frequency and monetary value
    holdout_duration = int(calibration_holdout_df['duration_holdout'].iloc[0])
    predicted_frequency = (
        bgf.conditional_expected_number_of_purchases_up_to_time(
            holdout_duration,
            calibration_holdout_df['frequency_cal'], 
            calibration_holdout_df['recency_cal'],
            calibration_holdout_df['T_cal']
        )).fillna(0)
    
    actual_total_spend = (
        calibration_holdout_df['monetary_value_holdout'] *
        (calibration_holdout_df['frequency_holdout'] + 1)).fillna(0)
    global_predicted_monetary = (
        ggf.conditional_expected_average_profit(
            calibration_holdout_df['frequency_cal'], 
            calibration_holdout_df['monetary_value_cal']
        )).fillna(calibration_holdout_df['monetary_value_cal'].mean())
    predicted_total_spend = predicted_frequency * global_predicted_monetary    

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(actual_total_spend,
               predicted_total_spend,
               alpha=0.4, color='#91181a', label='Customers')
    max_value = max(actual_total_spend.max(), predicted_total_spend.max())
    ax.plot([0, max_value], [0, max_value],
            '--', color='gray', label='Perfect Prediction')

    ax.set_title('Actual vs Predicted Combined Spend (Holdout Period)')
    ax.set_xlabel('Actual Total Spend ($\u20B9$)')
    ax.set_ylabel('Predicted Total Spend ($\u20B9$)')
    ax.legend()

    fig.tight_layout()
    save_plot(fig, 'actual_and_predicted_spend.png')
    plt.show()


def plot_frequency_recency(bgf: BetaGeoFitter) -> None:
    '''Display the frequency-recency matrix as a heatmap'''
    fig = plt.figure(figsize=(10, 5))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)

        plot_frequency_recency_matrix(bgf, cmap='rocket')
    plt.title('Expected Number of Future Purchases for 1 Unit of Time,\n'
              'by Frequency and Recency of a Customer', y=1.05)
    plt.tight_layout()
    save_plot(fig, 'frequency_recency.png')
    plt.show()


def plot_probability_alive(bgf: BetaGeoFitter) -> None:
    '''Display the probability alive matrix as a heatmap'''
    fig = plt.figure(figsize=(10, 5))       
    plot_probability_alive_matrix(bgf, cmap='rocket')
    plt.title('Probability Customer is Alive,\n'
              'by Frequency and Recency of a Customer', y=1.05)
    fig.tight_layout()
    save_plot(fig, 'probability_alive.png')
    plt.show()