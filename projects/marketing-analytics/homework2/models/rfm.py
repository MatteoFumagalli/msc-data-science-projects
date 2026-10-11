# Library imports
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from psycopg2.extensions import connection
import seaborn as sns

# Define paths
VISUALIZATIONS_DIR = 'visualizations/rfm'

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


def compute_rfm_metrics(conn: connection) -> pd.DataFrame:
    '''Compute RFM (recency, frequency, monetary) metrics'''
    query = '''
        WITH spending_details AS (
            SELECT o.order_id, o.customer_id, o.order_date,
                (oi.quantity * COALESCE (
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
                )) - oi.item_discount AS spending
            FROM "order" AS o
                JOIN order_item AS oi ON oi.order_id = o.order_id
        ),
             last_date AS (
            SELECT MAX(o.order_date) AS max_date
            FROM "order" AS o 
        )
        
        SELECT sd.customer_id,
            (ld.max_date - MAX(sd.order_date))::INT AS recency,
            COUNT(DISTINCT sd.order_id) AS frequency,
            SUM(sd.spending)::FLOAT AS monetary
        FROM spending_details AS sd
            CROSS JOIN last_date AS ld 
        GROUP BY sd.customer_id, ld.max_date
        ORDER BY sd.customer_id;
    '''

    try:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
            column_names = [description[0] for description in cur.description]
        return pd.DataFrame(result, columns=column_names)
    except Exception as e:
        print(f'Database error when executing the query: {e}')
        return pd.DataFrame(
            columns=['customer_id', 'recency', 'frequency', 'monetary'])


def assign_ordinal_scores(
        values: pd.Series, n_bins: int = 5, 
        ascending_order: bool = True) -> pd.Series:
    '''Assign ordinal scores from 1 to 5 using quantiles'''
    labels = list(range(1, n_bins + 1))
    ordinal_scores = pd.cut(
        values.rank(method='first'), 
        bins=n_bins, labels=labels).astype(int)
    
    if not ascending_order:
        ordinal_scores = (n_bins + 1) - ordinal_scores
    return ordinal_scores 


def compute_rfm_scores(rfm_df: pd.DataFrame) -> pd.DataFrame:
    '''Compute the overall RFM scoring as R x 100 + F x 10 + M'''
    rfm_df['recency_score'] = assign_ordinal_scores(
        rfm_df['recency'], ascending_order=False)
    rfm_df['frequency_score'] = assign_ordinal_scores(rfm_df['frequency'])
    rfm_df['monetary_score'] = assign_ordinal_scores(rfm_df['monetary'])

    rfm_df['rfm_score'] = (
        rfm_df['recency_score'] * 100 + 
        rfm_df['frequency_score'] * 10 + 
        rfm_df['monetary_score'])
    
    return rfm_df


def get_rfm_customer_table(conn: connection) -> pd.DataFrame:
    '''Compute the RFM metrics and obtain the customer-level RFM table'''
    print('== RFM ANALYSIS ==')
    rfm_df = compute_rfm_metrics(conn)

    if rfm_df.empty:
        print(' -> Error when computing RFM metrics')
        return rfm_df
    print(' -> Successfully computed RFM metrics')
    rfm_df = compute_rfm_scores(rfm_df)
    print(' -> Successfully computed RFM overall scores')
    return rfm_df


def plot_continuous_rfm_metrics_distribution(
        rfm_df: pd.DataFrame) -> None:
    '''Display a histogram and a boxplot for the distribution of each
    continuous RFM metric'''
    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex='col')
    plot_configurations = [
        {
            'column': 'recency',
            'title': 'Recency Distribution',
            'xlabel': 'Days since Last Purchase',
            'color': '#3b1848',
            'discrete': False
        },
        {
            'column': 'frequency',
            'title': 'Frequency Distribution',
            'xlabel': 'Total Number of Orders',
            'color': '#91181a',
            'discrete': True
        },
        {
            'column': 'monetary',
            'title': 'Monetary Distribution',
            'xlabel': 'Total Spending ($\u20B9$)',
            'color': '#f68f46',
            'discrete': False
        }
    ]

    for i, configuration in enumerate(plot_configurations):
        sns.histplot(data=rfm_df, 
                     x=configuration['column'], 
                     discrete=configuration['discrete'], 
                     ax=axes[0, i], color=configuration['color'])
        axes[0, i].set_title(configuration['title'], fontsize=12)
        axes[0, i].set_xlabel(configuration['xlabel'])
        axes[0, i].set_ylabel('Number of Customers')

        sns.boxplot(data=rfm_df,
                    x=configuration['column'],
                    ax=axes[1, i], color=configuration['color'])
        axes[1, i].set_xlabel(configuration['xlabel'])
        axes[1, i].set_ylabel('')

    plt.suptitle('RFM Metrics Continuous Distributions', fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'continuous_rfm_metrics_distribution.png')
    plt.show()


def plot_ordinal_rfm_metrics_distribution(rfm_df: pd.DataFrame) -> None:
    '''Display a countplot for the distribution of each ordinal RFM metric'''
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    plot_configurations = [
        {
            'column': 'recency_score',
            'title': 'Recency Score',
            'color': '#3b1848'
        },
        {
            'column': 'frequency_score',
            'title': 'Frequency Score',
            'color': '#91181a'
        },
        {
            'column': 'monetary_score',
            'title': 'Monetary Score',
            'color': '#f68f46'
        }
    ]

    for i, configuration in enumerate(plot_configurations):
        sns.countplot(data=rfm_df,
                      x=configuration['column'],
                      ax=axes[i], color=configuration['color'])
        axes[i].set_title(configuration['title'], fontsize=12)
        axes[i].set_xlabel('Score (1=worst, 5=best)')
        axes[i].set_ylabel('Number of Customers' if i == 0 else '')
    
    plt.suptitle('RFM Metrics Ordinal Distributions', fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'ordinal_rfm_metrics_distribution.png')
    plt.show()


def compute_rfm_statistics_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    '''Compute summary statistics for each metric score level'''
    metrics = [
        ('recency_score', 'recency', 'Days since Last Purchase'),
        ('frequency_score', 'frequency', 'Total Number of Orders'),
        ('monetary_score', 'monetary', 'Total Spending (\u20B9)')
    ]

    print('== SUMMARY STATISTICS FOR METRIC SCORES ==')
    for score, metric, description in metrics:
        summary = rfm_df.groupby(score).agg(
            min=(metric, 'min'),
            mean=(metric, 'mean'),
            max=(metric, 'max')
        ).round(2)

        print(f'\n -> {metric.capitalize()}: {description}')
        display(summary)


def get_customer_details(conn: connection) -> pd.DataFrame:
    '''Extract customer demographic details for additional analysis'''
    query = '''
        SELECT c.*
        FROM customer AS c;
    '''

    try:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
            column_names = [description[0] for description in cur.description]
        return pd.DataFrame(result, columns=column_names)
    except Exception as e:
        print(f'Database error when executing the query: {e}')
        return pd.DataFrame(
            columns=['customer_id', 'signup_date', 'city', 
                     'state', 'segment', 'status'])


def get_top10_rfm_customer_table(
        conn: connection, rfm_df: pd.DataFrame) -> pd.DataFrame:
    '''Obtain the top 10 customers ranked by RFM score. In case of ties, they
    are sub-ranked by their monetary value, and if necessary by frequency'''
    print('== TOP 10 CUSTOMERS BY RFM SCORE ==')

    # Rank customers by RFM score. In case of ties, use monetary or frequency
    sorted_rfm_df = rfm_df.sort_values(
        by=['rfm_score', 'monetary', 'frequency'],
        ascending=[False, False, False]).reset_index(drop=True)
    top10_df = sorted_rfm_df.head(10).copy()

    # Extract customer details to add contextual information
    customer_details_df = get_customer_details(conn)
    top10_df = top10_df.merge(customer_details_df, on='customer_id', how='left')
    columns = ['customer_id', 'rfm_score', 'recency', 'frequency', 'monetary',
               'segment', 'status', 'city', 'state']
    
    return top10_df[columns]
 

def compute_rfm_distribution_table(rfm_df: pd.DataFrame) -> pd.DataFrame:
    '''Compute the number of customer for each RFM score'''
    rfm_distribution_df = rfm_df.groupby('rfm_score').agg(
        customers=('customer_id', 'count')
    ).sort_values(by='rfm_score', ascending=False).reset_index()

    num_customers = rfm_distribution_df['customers'].sum()
    rfm_distribution_df['percentage'] = (
        rfm_distribution_df['customers'] / num_customers * 100).round(2)
    return rfm_distribution_df


def plot_rfm_score_distribution(rfm_distribution_df: pd.DataFrame) -> None:
    '''Display a histogram for the distribution of overall RFM score'''
    rfm_distribution_df = rfm_distribution_df.sort_values('rfm_score')
    rfm_distribution_df['label'] = rfm_distribution_df['rfm_score'].astype(
        str).apply(lambda x: '\n'.join(list(x)))

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=rfm_distribution_df,
                x='label', y='customers',
                hue='label', palette='rocket_r', 
                legend=False, ax=ax)
    ax.set_title('RFM Score Distribution')
    ax.set_xlabel('Score (111=worst, 555=best)')
    ax.set_ylabel('Number of Customers')

    plt.tight_layout()
    save_plot(fig, 'rfm_score_distribution.png')
    plt.show()