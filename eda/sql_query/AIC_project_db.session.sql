-- Table Initializations
-- PostgreSQL
-- By: Taylor Ueda

-- 1. Amazon
CREATE TABLE amazon_data (
    id INT,
    adj_close NUMERIC,
    close NUMERIC,
    high NUMERIC,
    low NUMERIC,
    open NUMERIC,
    volume BIGINT,
    company VARCHAR(50),
    date DATE,
    target INT,
    score INT,
    comments INT,
    cleaned_text TEXT,
    sentiment VARCHAR(20),
    sentiment_score NUMERIC
);

-- 2. Apple
CREATE TABLE apple_data (
    id INT,
    adj_close NUMERIC,
    close NUMERIC,
    high NUMERIC,
    low NUMERIC,
    open NUMERIC,
    volume BIGINT,
    company VARCHAR(50),
    date DATE,
    target INT,
    score INT,
    comments INT,
    cleaned_text TEXT,
    sentiment VARCHAR(20),
    sentiment_score NUMERIC
);

-- Tesla
CREATE TABLE tesla_data (
    id INT,
    adj_close NUMERIC,
    close NUMERIC,
    high NUMERIC,
    low NUMERIC,
    open NUMERIC,
    volume BIGINT,
    company VARCHAR(50),
    date DATE,
    target INT,
    score INT,
    comments INT,
    cleaned_text TEXT,
    sentiment VARCHAR(20),
    sentiment_score NUMERIC
);

-- Must copy local path FROM YOUR COMPUTER
-- The path below does NOT apply to your local path

COPY amazon_data FROM '/Users/taylorueda/Desktop/Projects/Stock Predictor/AI-Collective-AIC-Team-7/datasets/Amazon_data.csv' WITH (FORMAT CSV, HEADER);
COPY apple_data FROM '/Users/taylorueda/Desktop/Projects/Stock Predictor/AI-Collective-AIC-Team-7/datasets/Apple_data.csv' WITH (FORMAT CSV, HEADER);
COPY tesla_data FROM '/Users/taylorueda/Desktop/Projects/Stock Predictor/AI-Collective-AIC-Team-7/datasets/Tesla_data.csv' WITH (FORMAT CSV, HEADER);

------------ Querying goes below ------------

-- Example
SELECT
    amazon_data.company AS amazon_company,
    amazon_data.date AS amazon_date,
    amazon_data.close AS amazon_close,
    apple_data.company AS apple_company,
    apple_data.date AS apple_date,
    apple_data.close AS apple_close
FROM amazon_data
LEFT JOIN apple_data
    ON amazon_data.date = apple_data.date
WHERE amazon_data.close < 20 
  AND apple_data.close < 20
ORDER BY amazon_data.close DESC
LIMIT 500;