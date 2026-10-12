# Library imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import pandas as pd
from psycopg2.extensions import connection
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


def save_plot(fig: Figure, file_name: str) -> None:
    '''Save the plot in the visualizations folder or in the current folder'''
    target_path = f'{VISUALIZATIONS_DIR}/{file_name}'
    try:
        fig.savefig(target_path, dpi=300, bbox_inches='tight')
    except:
        fig.savefig(file_name, dpi=300, bbox_inches='tight')


def compute_spending_distribution(conn: connection) -> pd.DataFrame:
    '''Compute the total spending of each customer'''
    query = '''
        WITH spending_details AS (
            SELECT o.customer_id, 
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
            FROM order_item AS oi
                JOIN "order" AS o ON o.order_id = oi.order_id
        )

        SELECT sp.customer_id, SUM(sp.spending) AS total_spending
        FROM spending_details AS sp
        GROUP BY sp.customer_id;
    '''

    try:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
            column_names = [description[0] for description in cur.description]
        return pd.DataFrame(result, columns=column_names)
    except Exception as e:
        print(f'Database error when executing the query: {e}')
        return pd.DataFrame(columns=['customer_id', 'total_spending'])


def plot_spending_distribution(conn: connection) -> None:
    '''Display a boxplot of the total spending distribution of customers'''
    spending_df = compute_spending_distribution(conn)

    fig = plt.figure(figsize=(8, 3))
    sns.boxplot(x=spending_df['total_spending'], color='#91181a')
    plt.title('Customer Spending Distribution', fontweight='bold')
    plt.xlabel('Amount ($\u20B9$)')
    plt.tight_layout()
    save_plot(fig, 'spending_distribution.png')
    plt.show()

    return spending_df


def compute_purchase_frequency_distribution(conn: connection) -> pd.DataFrame:
    '''Compute the purchase frequency distribution per each customer'''
    query = '''
        SELECT o.customer_id, COUNT(*) as num_orders
        FROM "order" AS o
        GROUP BY o.customer_id;
    '''

    try:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
            column_names = [description[0] for description in cur.description]
        return pd.DataFrame(result, columns=column_names)
    except Exception as e:
        print(f'Database error when executing the query: {e}')
        return pd.DataFrame(columns=['customer_id', 'num_orders'])


def plot_purchase_frequency_distribution(conn: connection) -> None:
    '''Display a histogram of the purchase frequency'''
    purchase_frequency_df = compute_purchase_frequency_distribution(conn)

    fig = plt.figure(figsize=(8, 4))
    sns.histplot(data=purchase_frequency_df,
                 x='num_orders', discrete=True,
                 color='#91181a')
    
    plt.title('Purchase Frequency Distribution', fontweight='bold')
    plt.xlabel('Number of Orders')
    plt.ylabel('Number of Customers')
    plt.tight_layout()
    save_plot(fig, 'purchase_frequency_distribution.png')
    plt.show()

    return purchase_frequency_df


def compute_revenue_per_category(conn: connection) -> pd.DataFrame:
    '''Compute the total revenue associated with each product category'''
    query = '''
        WITH category_details AS (
            SELECT c.category_name, 
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
                )) - oi.item_discount AS revenue
            FROM order_item AS oi
                JOIN "order" AS o ON o.order_id = oi.order_id
                JOIN product AS p ON p.product_id = oi.product_id
                JOIN category AS c ON c.category_id = p.category_id
        )

        SELECT cd.category_name, SUM(cd.revenue) AS total_revenue
        FROM category_details AS cd
        GROUP BY cd.category_name
        ORDER BY total_revenue DESC;
    '''

    try:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
            column_names = [description[0] for description in cur.description]
        return pd.DataFrame(result, columns=column_names)
    except Exception as e:
        print(f'Database error when executing the query: {e}')
        return pd.DataFrame(columns=['category_name', 'total_revenue'])


def plot_revenue_per_category(conn: connection) -> None:
    '''Display a bar chart of the total revenue per category'''
    revenue_per_category_df = compute_revenue_per_category(conn)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=revenue_per_category_df,
                x='total_revenue', y='category_name',
                color='#91181a', ax=ax)
    sns.despine(ax=ax, top=True, right=True)

    plt.title('Total Revenue per Product Category', fontweight='bold')
    plt.xlabel('Revenue ($\u20B9$)')
    plt.ylabel('Product Category')
    plt.tight_layout()
    save_plot(fig, 'revenue_per_category.png')
    plt.show()


def compute_customer_segments(conn: connection) -> pd.DataFrame:
    '''Compute the distribution of segments for active and churned customers'''
    query = '''
        SELECT c.segment, c.status, COUNT(c.customer_id) AS num_customers
        FROM customer AS c
        GROUP BY c.segment, c.status
        ORDER BY c.segment, c.status;
    '''

    try:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
            column_names = [description[0] for description in cur.description]
        return pd.DataFrame(result, columns=column_names)
    except Exception as e:
        print(f'Database error when executing the query: {e}')
        return pd.DataFrame(columns=['segment', 'status', 'num_customers'])


def plot_customer_segments_per_status(conn: connection) -> None:
    '''Display a pie chart of the segment distribution for each status'''
    segments_df = compute_customer_segments(conn)
    unique_status = segments_df['status'].unique()
    num_status = len(unique_status)

    fig, axes = plt.subplots(1, num_status, figsize=(8, 4))
    if num_status == 1:
        axes = [axes]
    
    # Map colors to segments
    unique_segments = sorted(segments_df['segment'].unique())
    palette = sns.color_palette('rocket', len(unique_segments))
    segment_colors = dict(zip(unique_segments, palette))

    # Define a pie chart for each status
    for i, status_name in enumerate(unique_status):
        segment_df = segments_df[segments_df['status'] == status_name]
        segment_df = segment_df.sort_values('segment')
        colors = [segment_colors[segment] for segment in segment_df['segment']]

        _, _, autotexts = axes[i].pie(
            segment_df['num_customers'],
            labels=segment_df['segment'],
            autopct='%1.0f%%',
            startangle=90,
            colors=colors)
        axes[i].set_title(f'Customer Status: {status_name.capitalize()}', 
                          fontsize=12)

        for autotext in autotexts:
            autotext.set_color('white')
    
    plt.suptitle('Segment Composition Comparison', fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'customer_segments.png')
    plt.show()


def compute_revenue_concentration(conn: connection) -> pd.DataFrame:
    '''Compute the revenue concentration among customers'''
    query = '''
        WITH spending_details AS (
            SELECT o.customer_id, 
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
            FROM order_item AS oi
                JOIN "order" AS o ON o.order_id = oi.order_id
        ),
             customer_totals AS (
            SELECT sp.customer_id, SUM(sp.spending) AS total_spending
            FROM spending_details AS sp
            GROUP BY sp.customer_id
        ), 
             ordered_concentration AS (
            SELECT ct.customer_id, ct.total_spending,
                CUME_DIST() OVER (ORDER BY ct.total_spending ASC) * 100.0 
                    AS customer_percentage,
                SUM(ct.total_spending) OVER (ORDER BY ct.total_spending ASC)
                    AS cumulative_spending,
                SUM(ct.total_spending) OVER () AS total_revenue
            FROM customer_totals as ct
        )

        SELECT oc.customer_percentage,
            (oc.cumulative_spending / oc.total_revenue) * 100.0 
                AS revenue_percentage
        FROM ordered_concentration as oc
        ORDER BY customer_percentage ASC;
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
            columns=['customer_percentage', 'revenue_percentage'])


def plot_revenue_concentration(conn: connection) -> None:
    '''Display the cumulative revenue curve across customers'''
    revenue_concentration_df = compute_revenue_concentration(conn)

    fig = plt.figure(figsize=(5, 5))
    # Plot the cumulative revenue curve
    plt.plot(revenue_concentration_df['customer_percentage'],
             revenue_concentration_df['revenue_percentage'],
             label='Cumulative Revenue',
             color='#91181a', linewidth=2)
    
    # Plot the reference line representing perfect equality
    plt.plot([0, 100], [0, 100], linestyle='--',
             color='gray', label='Perfect Equality')
    
    plt.title('Revenue Concentration', fontweight='bold')
    plt.xlabel('Cumulative Percentage of Customers')
    plt.ylabel('Cumulative Percentage of Revenue')
    plt.legend(loc='upper left')
    plt.tight_layout()
    save_plot(fig, 'revenue_concentration.png')
    plt.show()


def compute_time_trends(conn: connection) -> pd.DataFrame:
    '''Compute weekly revenue and order trends over time'''
    query = '''
        WITH weekly_details AS (
            SELECT o.year, o.week, o.order_id,
                (oi.quantity * (
                    SELECT pc.unit_price
                    FROM price_change AS pc
                    WHERE pc.product_id = oi.product_id
                        AND (pc.year < o.year 
                            OR (pc.year = o.year AND pc.week <= o.week))
                    ORDER BY pc.year DESC, pc.week DESC
                    LIMIT 1                
                )) - oi.item_discount AS spending
            FROM order_item AS oi
                JOIN "order" AS o ON o.order_id = oi.order_id
        )

        SELECT wd.year, wd.week, 
            SUM(wd.spending) AS weekly_revenue,
            COUNT(DISTINCT wd.order_id) AS weekly_orders
        FROM weekly_details AS wd
        GROUP BY year, week
        ORDER BY year ASC, week ASC;  
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
            columns=['year', 'week', 'weekly_revenue', 'weekly_orders'])


def plot_time_trends(conn: connection) -> None:
    '''Display a line chart of weekly trends of revenues and number of orders'''
    time_trends_df = compute_time_trends(conn)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot revenue trend
    ax1.set_xlabel('Year (week)')
    ax1.set_ylabel('Revenue ($\u20B9$)', color='#91181a')
    sns.lineplot(data=time_trends_df, 
                 x=time_trends_df.index, y='weekly_revenue',
                 color='#91181a', ax=ax1)

    # Plot volume of orders trend
    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of Orders', color='#1a365d')
    sns.lineplot(data=time_trends_df,
                 x=time_trends_df.index, y='weekly_orders',
                 color='#1a365d', ax=ax2)
    
    # Display a maximum of 8 labels on the x-axis
    time_trends_df['year_week'] = (
        time_trends_df['year'].astype(str) + 
        ' (w' + time_trends_df['week'].astype(str) + ')')
    
    locator = ticker.MaxNLocator(nbins=5, integer=True)
    ticks = locator.tick_values(0, len(time_trends_df) - 1)
    ticks = set([int(tick) for tick in ticks 
                 if 0 <= tick <= len(time_trends_df)])
    ticks.add(0)
    ticks.add(len(time_trends_df) - 1)
    sorted_ticks = sorted(list(ticks))
    labels = [time_trends_df['year_week'].iloc[tick] for tick in sorted_ticks]
    ax1.set_xticks(sorted_ticks)
    ax1.set_xticklabels(labels)

    plt.title('Weekly Purchasing Trends', fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'time_trends.png')
    plt.show()