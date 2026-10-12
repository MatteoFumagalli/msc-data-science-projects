-- Table: CATEGORY
CREATE TABLE IF NOT EXISTS category (
    category_id CHAR(6) PRIMARY KEY,
    category_name VARCHAR(40) NOT NULL UNIQUE
);


-- Table: PRODUCT
CREATE TABLE IF NOT EXISTS product (
    product_id CHAR(9) PRIMARY KEY,
    category_id CHAR(6) NOT NULL 
        REFERENCES category(category_id) ON DELETE RESTRICT,
    brand VARCHAR(20) NOT NULL,
    sku_name VARCHAR(100) NOT NULL
);


-- Table: PRICE_CHANGE
CREATE TABLE IF NOT EXISTS price_change (
    product_id CHAR(9) NOT NULL 
        REFERENCES product(product_id) ON DELETE CASCADE,
    year SMALLINT NOT NULL 
        CHECK (year <= EXTRACT(YEAR FROM CURRENT_DATE)),
    week SMALLINT NOT NULL 
        CHECK (week between 1 AND 53),
    unit_price NUMERIC(7, 2) NOT NULL
        CHECK (unit_price > 0),
    PRIMARY KEY (product_id, year, week)
);


-- Table: CHANNEL
CREATE TABLE IF NOT EXISTS channel (
    channel_id CHAR(5) PRIMARY KEY,
    channel_name VARCHAR(20) NOT NULL
);


-- Table: STORE
CREATE TABLE IF NOT EXISTS store (
    store_id CHAR(6) PRIMARY KEY,
    store_city VARCHAR(30) NOT NULL,
    store_state VARCHAR(30) NOT NULL,
    store_type VARCHAR(20) NOT NULL
        CHECK (store_type IN (
            'General Trade', 'Hypermarket', 'Modern Trade',
            'Supermarket', 'Warehouse'))
);


-- Table: EXTERNAL_FACTOR
CREATE TABLE IF NOT EXISTS external_factor (
    factor_id CHAR(10) PRIMARY KEY,
    store_id CHAR(6) NOT NULL
        REFERENCES store(store_id) ON DELETE RESTRICT,
    factor_date DATE NOT NULL,
    is_holiday BOOLEAN NOT NULL DEFAULT FALSE,
    temp_c NUMERIC(4, 2),
    rainfall_mm NUMERIC(5, 2) CHECK (rainfall_mm >= 0),
    trend_index NUMERIC(5, 2),
    cpi_index NUMERIC(5, 2),
    year SMALLINT NOT NULL 
        CHECK (year <= EXTRACT(YEAR FROM CURRENT_DATE)),
    week SMALLINT NOT NULL 
        CHECK (week between 1 AND 53),
    UNIQUE (store_id, factor_date)
);


-- Table: CUSTOMER
CREATE TABLE IF NOT EXISTS customer (
    customer_id CHAR(10) PRIMARY KEY,
    signup_date DATE NOT NULL,
    city VARCHAR(30) NOT NULL,
    state VARCHAR(30) NOT NULL,
    segment VARCHAR(20) NOT NULL
        CHECK (segment IN (
            'Health-focused', 'Mainstream', 'Premium', 'Value')),
    status VARCHAR(10) NOT NULL
        CHECK (status IN ('active', 'churned'))
);


-- Table: SESSION
CREATE TABLE IF NOT EXISTS session (
    session_id CHAR(11) PRIMARY KEY,
    customer_id CHAR(10) NOT NULL 
        REFERENCES customer(customer_id) ON DELETE CASCADE,
    session_start TIMESTAMP NOT NULL,
    session_duration_sec INTEGER NOT NULL
        CHECK (session_duration_sec >= 0),
    pages_viewed SMALLINT NOT NULL
        CHECK (pages_viewed >= 0), 
    device VARCHAR(10) NOT NULL
        CHECK (device IN ('Desktop', 'Mobile', 'Tablet')),
    referrer VARCHAR(20)
        CHECK (referrer IN (
            'Affiliate', 'Direct', 'Email', 'Google Search',
            'Instagram', 'Marketplace', 'Push', 'YouTube'))
);


-- Table: SUPPORT_TICKET
CREATE TABLE IF NOT EXISTS support_ticket (
    ticket_id CHAR(9) PRIMARY KEY,
    customer_id CHAR(10) NOT NULL
        REFERENCES customer(customer_id) ON DELETE CASCADE,
    created_date DATE NOT NULL,
    issue_type VARCHAR(25) NOT NULL
        CHECK (issue_type IN (
            'Coupon/Promo Issue', 'Damaged Item', 'Delivery Delay',
            'Payment Issue', 'Quality Concern', 
            'Refund/Return', 'Wrong Item')),
    status VARCHAR(15) NOT NULL
        CHECK (status IN ('Escalated', 'Open', 'Resolved')),
    resolution_time_hr INTEGER
        CHECK (resolution_time_hr >= 0),
    csat_score SMALLINT
        CHECK (csat_score BETWEEN 1 AND 5)
);


-- Table: MARKETING CHANNEL
CREATE TABLE IF NOT EXISTS marketing_channel(
    channel_id CHAR(6) PRIMARY KEY NOT NULL,
    channel_name VARCHAR(20) NOT NULL,
    channel_type VARCHAR(10) NOT NULL
        CHECK (channel_type IN ('Offline', 'Owned', 'Paid')),
    cost_tier VARCHAR(10) NOT NULL
        CHECK (cost_tier IN ('High', 'Low', 'Medium', 'Very High'))
);


-- Table: CAMPAIGN_TOUCH
CREATE TABLE IF NOT EXISTS campaign_touch (
    touch_id CHAR(11) PRIMARY KEY,
    customer_id CHAR(10) NOT NULL
        REFERENCES customer(customer_id) ON DELETE CASCADE,
    touch_date DATE NOT NULL,
    channel_id CHAR(6) 
        REFERENCES marketing_channel(channel_id) ON DELETE SET NULL,
    campaign_name VARCHAR(30) NOT NULL,
    outcome VARCHAR(15) NOT NULL
        CHECK (outcome IN ('Clicked', 'Converted', 'Ignored'))
);


-- Table: PROMOTION
CREATE TABLE IF NOT EXISTS promotion (
    promo_id CHAR(7) PRIMARY KEY,
    promo_type VARCHAR(10) NOT NULL
        CHECK (promo_type IN ('BOGO', 'FLAT_OFF', 'PCT_OFF')),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    discount_value NUMERIC(5, 2),

    CONSTRAINT promo_dates CHECK (end_date >= start_date)
);


-- Table: ORDER
CREATE TABLE IF NOT EXISTS "order" (
    order_id CHAR(11) PRIMARY KEY,
    customer_id CHAR(10) NOT NULL
        REFERENCES customer(customer_id) ON DELETE RESTRICT,
    order_date DATE NOT NULL,
    store_id CHAR(6) NOT NULL
        REFERENCES store(store_id) ON DELETE RESTRICT,
    channel_id CHAR(5) NOT NULL
        REFERENCES channel(channel_id) ON DELETE RESTRICT,
    payment_type VARCHAR(20) NOT NULL
        CHECK (payment_type IN (
            'Cash on Delivery', 'Credit Card', 
            'Debit Card', 'UPI', 'Wallet')),
    year SMALLINT NOT NULL 
        CHECK (year <= EXTRACT(YEAR FROM CURRENT_DATE)),
    week SMALLINT NOT NULL 
        CHECK (week between 1 AND 53)
);


-- Table: ORDER_ITEM
CREATE TABLE IF NOT EXISTS order_item (
    order_item_id CHAR(10) PRIMARY KEY,
    order_id CHAR(11) NOT NULL
        REFERENCES "order"(order_id) ON DELETE CASCADE,
    product_id CHAR(9) NOT NULL
        REFERENCES product(product_id) ON DELETE RESTRICT,
    promo_id CHAR(7) 
        REFERENCES promotion(promo_id) ON DELETE SET NULL,
    quantity SMALLINT NOT NULL
        CHECK (quantity > 0),
    item_discount NUMERIC(6, 2) NOT NULL DEFAULT 0
        CHECK (item_discount >= 0)
);