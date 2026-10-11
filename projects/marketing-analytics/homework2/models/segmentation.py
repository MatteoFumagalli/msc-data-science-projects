# Library imports
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from psycopg2.extensions import connection
import seaborn as sns

# Define paths
VISUALIZATIONS_DIR = 'visualizations/segmentation'

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


def assign_segment(clv_prediction_df: pd.DataFrame) -> pd.DataFrame:
    '''Assign a segment to each customer based on predicted CLV using quantiles'''
    print('== CUSTOMER SEGMENTATION ==')
   
    quantiles = [0, 0.2, 0.5, 0.8, 1]
    labels = ['Bronze', 'Silver', 'Gold', 'Platinum']
    segmented_df = clv_prediction_df.copy()
    segmented_df['customer_segment'] = pd.qcut(
        segmented_df['predicted_clv'], q=quantiles, labels=labels)
    print(' -> Successfully assigned customers to segments: '
          'Bronze, Silver, Gold, and Platinum')
    
    return segmented_df


def extract_customer_information(
        conn: connection, segmented_df: pd.DataFrame) -> pd.DataFrame:
    '''Extract additional consumer information to enrich the available data'''
    query = '''
        WITH category_counts AS (
            SELECT o.customer_id, cat.category_name, 
                SUM(oi.quantity) AS total_quantity
            FROM "order" AS o
            JOIN order_item AS oi ON oi.order_id = o.order_id
            JOIN product AS p ON p.product_id = oi.product_id
            JOIN category AS cat ON cat.category_id = p.category_id
            GROUP BY o.customer_id, cat.category_name
        ),
             top_categories AS (
            SELECT DISTINCT ON (cc.customer_id)
                cc.customer_id, cc.category_name
            FROM category_counts AS cc
            ORDER BY cc.customer_id, cc.total_quantity DESC
        )

        SELECT c.customer_id, c.city, c.state, c.segment AS customer_type, 
            c.status, tc.category_name AS top_category
        FROM customer AS c
        LEFT JOIN top_categories AS tc ON c.customer_id = tc.customer_id;
    '''

    try:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
            column_names = [description[0] for description in cur.description]
        enriched_df = pd.DataFrame(result, columns=column_names)
        enriched_df = segmented_df.merge(
            enriched_df, on='customer_id', how='left')
        print(' -> Successfully extracted additional customer information')
        return enriched_df
    except Exception as e:
        print(f'Database error when executing the query: {e}')
        return segmented_df
    

def get_segment_summary(enriched_df: pd.DataFrame) -> pd.DataFrame:
    '''Obtain a summary for each customer segment'''
    segment_summary_df = enriched_df.groupby(
        'customer_segment', observed=True).agg(
            num_customers = ('customer_id', 'count'),
            avg_spending = ('monetary_value', 'mean'),
            avg_purchase_frequency = ('frequency', 'mean'),
            avg_prob_alive = ('p_alive', 'mean'),
            top_category = ('top_category', 
                            lambda x: x.mode()[0] 
                                if not x.mode().empty else 'N/A'),
            top_city = ('city', 
                        lambda x: x.mode()[0] if not x.mode().empty else 'N/A'),
            top_state = ('state', 
                        lambda x: x.mode()[0] if not x.mode().empty else 'N/A'),
            top_customer_type = ('customer_type', 
                                lambda x: x.mode()[0] 
                                if not x.mode().empty else 'N/A')
    )

    print('\n == SEGMENT SUMMARY TABLE ==')
    segment_summary_df = segment_summary_df.reset_index().round(2)
    display(segment_summary_df)

    return segment_summary_df


def plot_segment_clv_distribution(segmented_df: pd.DataFrame) -> None:
    '''Display a boxplot for the CLV distribution of each segment'''
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.boxplot(data=segmented_df, 
                x='customer_segment', y='predicted_clv',
                hue='customer_segment', palette='rocket_r', 
                ax=ax, legend=False)
    ax.set_title('CLV Distribution per Segment')
    ax.set_xlabel('Customer Segment')
    ax.set_ylabel('Predicted CLV (\u20B9)')

    plt.tight_layout()
    save_plot(fig, 'clv_distribution_per_segment.png')
    plt.show()


def plot_segment_clv_metrics(segment_summary_df: pd.DataFrame) -> None:
    '''Display a bar chart for both average spending and purchase frequency 
    of each segment'''
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Display average spending
    sns.barplot(data=segment_summary_df,
                x='customer_segment', y='avg_spending',
                hue='customer_segment', palette='rocket_r',
                ax=axes[0], legend=False)
    axes[0].set_title('Average Spending', fontsize=12)
    axes[0].set_xlabel('Customer Segment')
    axes[0].set_ylabel('Average Spending (\u20B9)')

    # Display purchase frequency
    sns.barplot(data=segment_summary_df,
                x='customer_segment', y='avg_purchase_frequency',
                hue='customer_segment', palette='rocket_r',
                ax=axes[1], legend=False)
    axes[1].set_title('Average Purchase Frequency', fontsize=12)
    axes[1].set_xlabel('Customer Segment')
    axes[1].set_ylabel('Average Purchase Frequency')

    plt.suptitle('CLV Metrics per Segment', fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'clv_metrics_per_segment.png')
    plt.show()


def plot_segment_top_categories(enriched_df: pd.DataFrame) -> None:
    '''Display a heatmap of category preferences for each segment'''
    contingency_table = pd.crosstab(
        index=enriched_df['customer_segment'], 
        columns=enriched_df['top_category'],
        normalize='index')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(contingency_table, annot=True, 
                fmt='.0%', cmap='rocket_r', ax=ax)
    ax.set_title('Top Category Preference Distribution per Segment')
    ax.set_xlabel('Top Category')
    ax.set_ylabel('Customer Segment')
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    save_plot(fig, 'top_category_preferences_per_segment.png')
    plt.show()


def plot_segment_customer_type(enriched_df: pd.DataFrame) -> None:
    '''Display a bar chart of customer types for each segment'''
    contingency_table = pd.crosstab(
        index=enriched_df['customer_segment'], 
        columns=enriched_df['customer_type'],
        normalize='index') * 100
    
    fig, ax = plt.subplots(figsize=(10, 5))
    contingency_table.plot(
        kind='bar', stacked=True, colormap='rocket_r', ax=ax)
    
    ax.set_title('Customer Type per Segment')
    ax.set_xlabel('Customer Segment')
    ax.set_ylabel('Percentage of Customers (%)')
    ax.tick_params(axis='x', rotation=0)

    plt.legend(title='Customer Type',
               bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot(fig, 'customer_type_per_segment.png')
    plt.show()