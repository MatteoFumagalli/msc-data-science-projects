# Library imports
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from psycopg2.extensions import connection
import seaborn as sns

# Define paths
VISUALIZATIONS_DIR = 'visualizations/clv_analysis'

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


def compute_clv_metrics(conn: connection) -> pd.DataFrame:
    '''Compute individual average purchase revenue (apv) and purchase frequency 
    (pf), together with global churn rate to estimate customer lifetime value'''
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
        ),
             customer_details AS (
            SELECT sd.customer_id,
                COUNT(DISTINCT sd.order_id) AS total_transactions,
                SUM(sd.spending)::FLOAT AS total_revenue,
                (ld.max_date - MAX(sd.order_date))::INT AS recency
            FROM spending_details AS sd
                CROSS JOIN last_date AS ld
            GROUP BY sd.customer_id, ld.max_date
        ),
             global_churn_rate AS (
            SELECT COUNT(*)::FLOAT AS total_customers,
                SUM(CASE WHEN cd.recency > 90 
                    THEN 1 ELSE 0 END)::FLOAT AS churned_customers
            FROM customer_details AS cd
        )

        SELECT cd.customer_id,
            cd.total_revenue, cd.total_transactions,
            (cd.total_revenue / cd.total_transactions)::FLOAT AS apv,
            cd.total_transactions::FLOAT AS pf,
            (gcr.churned_customers / gcr.total_customers)::FLOAT AS churn_rate,
            CASE WHEN cd.recency > 90 
                THEN 'Churned' ELSE 'Active' END AS customer_status
        FROM customer_details AS cd
            CROSS JOIN global_churn_rate AS gcr
        ORDER BY cd.customer_id;
    '''

    try:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
            column_names = [description[0] for description in cur.description]
        df = pd.DataFrame(result, columns=column_names)
        df['clv'] = (df['apv'] * df['pf']) / df['churn_rate']
        return df
    except Exception as e:
        print(f'Database error when executing the query: {e}')
        return pd.DataFrame(
            columns=['customer_id', 'total_revenue', 'total_transactions', 
                     'apv', 'pf', 'churn_rate', 'customer_status', 'clv'])
    

def get_clv_customer_table(conn: connection) -> pd.DataFrame:
    '''Compute the CLV metrics and obtain the customer-level CLV table'''
    print('== CLV ANALYSIS ==')
    clv_df = compute_clv_metrics(conn)

    if clv_df.empty:
        print(' -> Error when computing CLV metrics')
        return clv_df
    print(' -> Successfully computed CLV metrics and CLV score')

    # Compute global metrics
    number_customers = len(clv_df)
    number_transactions = clv_df['total_transactions'].sum()
    global_revenue = clv_df['total_revenue'].sum()

    global_apv = (global_revenue / number_transactions 
                  if number_transactions > 0 else 0)
    global_pf = (number_transactions / number_customers 
                 if number_customers > 0 else 0)
    global_churn_rate = clv_df['churn_rate'].iloc[0]
    global_clv = ((global_apv * global_pf) / global_churn_rate 
                  if global_churn_rate > 0 else 0)

    print('\n== CLV METHODOLOGY ==')
    print(f' -> Total Customers: {number_customers}')
    print(f' -> Total Revenue: \u20B9 {global_revenue:.2f}')
    print(f' -> Total Transactions: {number_transactions}')

    print(f'\n -> APV = Revenue / Transactions: \u20B9 {global_apv:.2f}')
    print(f' -> PF = Transactions / Customers: {global_pf:.2f}')
    print(' -> Churn Rate = Churned Customers / Total Customers: '
          f'{global_churn_rate * 100:.2f}%')
    print(f' -> CLV = (APV x PF) / Churn Rate: \u20B9 {global_clv:.2f}')

    return clv_df


def get_customer_details(conn: connection) -> pd.DataFrame:
    '''Extract customer demographic details for additional analysis'''
    query = '''
        SELECT c.*
        FROM customer AS c   
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


def plot_clv_distribution(clv_df: pd.DataFrame) -> None:
    '''Display a histogram for the distribution of customer lifetime value'''
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(data=clv_df, x='clv', 
                 bins=40, ax=ax, color='#91181a')
    ax.set_title('Customer Lifetime Value Distribution')
    ax.set_xlabel('Estimated CLV ($\u20B9$)')
    ax.set_ylabel('Number of Customers')

    plt.tight_layout()
    save_plot(fig, 'clv_distribution.png')
    plt.show()


def plot_clv_distribution_per_status(clv_df: pd.DataFrame) -> None:
    '''Display a boxplot for the customer lifetime according to the status'''
    fig, ax = plt.subplots(figsize=(10, 5))
    color = {'Active': '#f68f46', 'Churned': '#91181a'}

    sns.boxplot(data=clv_df,
                x='customer_status', y='clv', hue='customer_status',
                ax=ax, palette=color, legend=False)
    ax.set_title('Customer Lifetime Value per Customer Status')
    ax.set_xlabel('Customer Status')
    ax.set_ylabel('Estimated CLV ($\u20B9$)')

    plt.tight_layout()
    save_plot(fig, 'clv_distribution_per_status.png')
    plt.show()



def plot_revenue_and_transactions(clv_df: pd.DataFrame) -> None:
    '''Display a scatterplot comparing total revenue and transactions, while
    also using CLV for setting the color'''
    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(
        data=clv_df,
        x='total_transactions', y='total_revenue',
        c='clv', cmap='rocket_r',
        alpha=0.7, edgecolors='white', linewidths=0.5
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Estimated CLV ($\u20B9$)')
    ax.set_title('Revenue vs Transactions per Customer')
    ax.set_xlabel('Total Transactions')
    ax.set_ylabel('Total Revenue ($\u20B9$)')

    plt.tight_layout()
    save_plot(fig, 'clv_revenue_and_transactions.png')
    plt.show()


def get_top20_clv_customer_table(
        conn: connection, clv_df: pd.DataFrame) -> pd.DataFrame:
    '''Obtain the top 20 customers ranked by customer lifetime value'''
    print('== TOP 20 CUSTOMERS BY CLV ==')

    # Rank customers by CLV in decreasing order
    sorted_clv_df = clv_df.sort_values(
        by=['clv'], ascending=False).reset_index(drop=True)
    top20_df = sorted_clv_df.head(20).copy()

    # Extract customer details to add contextual information
    customer_details_df = get_customer_details(conn)
    top20_df = top20_df.merge(customer_details_df, on='customer_id', how='left')
    columns = ['customer_id', 'clv', 'apv', 'pf',
               'segment', 'customer_status', 'city', 'state']
    top20_df['clv'] = top20_df['clv'].round(2)
    top20_df['apv'] = top20_df['apv'].round(2)
    top20_df['pf'] = top20_df['pf'].astype(int)
    
    return top20_df[columns]