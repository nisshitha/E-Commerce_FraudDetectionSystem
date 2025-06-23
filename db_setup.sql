REATE DATABASE f_db;
use f_db;
CREATE TABLE transac (
    id INT AUTO_INCREMENT PRIMARY KEY,
    transaction_amount FLOAT,
    customer_age INT,
    account_age_days INT,
    transaction_hour INT,
    source VARCHAR(50),
    browser VARCHAR(50),
    sex VARCHAR(10),
    payment_method VARCHAR(50),
    product_category VARCHAR(100),
    quantity INT,
    device_used VARCHAR(50),
    address_match TINYINT(1),
    prediction_label VARCHAR(50),
    confidence FLOAT,
    risk_score FLOAT
);
SELECT * FROM transac;
