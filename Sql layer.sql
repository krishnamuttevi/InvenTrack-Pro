CREATE DATABASE inventory_db;
USE inventory_db;

-- Products
CREATE TABLE products (
  product_id VARCHAR(10) PRIMARY KEY,
  product_name VARCHAR(100),
  category VARCHAR(50),
  reorder_level INT,
  unit_price FLOAT
);

-- Warehouses
CREATE TABLE warehouses (
  warehouse_id VARCHAR(10) PRIMARY KEY,
  location_name VARCHAR(100)
);

-- Suppliers
CREATE TABLE suppliers (
  supplier_id VARCHAR(10) PRIMARY KEY,
  supplier_name VARCHAR(100),
  quality_score FLOAT,
  avg_lead_time_days INT
);

-- Inventory
CREATE TABLE inventory (
  inventory_id VARCHAR(20) PRIMARY KEY,
  product_id VARCHAR(10),
  warehouse_id VARCHAR(10),
  quantity_available INT,
  last_updated DATE,
  FOREIGN KEY (product_id) REFERENCES products(product_id),
  FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id)
);

-- Orders
CREATE TABLE orders (
  order_id VARCHAR(20) PRIMARY KEY,
  product_id VARCHAR(10),
  supplier_id VARCHAR(10),
  quantity_ordered INT,
  order_date DATE,
  delivery_date DATE,
  FOREIGN KEY (product_id) REFERENCES products(product_id),
  FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
);

SHOW VARIABLES LIKE 'secure_file_priv';

LOAD DATA INFILE "C:/ProgramData/MySQL/MySQL Server 9.1/Uploads/products.csv"
INTO TABLE products
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


SELECT i.*, t.total_count
FROM (
    SELECT * FROM products LIMIT 5
) AS i
JOIN (
    SELECT COUNT(*) AS total_count FROM products
) AS t;


LOAD DATA INFILE "C:/ProgramData/MySQL/MySQL Server 9.1/Uploads/warehouses.csv"
INTO TABLE warehouses
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


LOAD DATA INFILE "C:/ProgramData/MySQL/MySQL Server 9.1/Uploads/suppliers.csv"
INTO TABLE suppliers
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

select i.*,t.total_count 
from (
select * from suppliers limit 5
) as i
join 
(
select count(*) as total_count from suppliers
) as t;

LOAD DATA INFILE "C:/ProgramData/MySQL/MySQL Server 9.1/Uploads/inventory.csv"
INTO TABLE inventory
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


SELECT i.*, t.total_count
FROM (
    SELECT * FROM inventory LIMIT 5
) AS i
JOIN (
    SELECT COUNT(*) AS total_count FROM inventory
) AS t;




LOAD DATA INFILE "C:/ProgramData/MySQL/MySQL Server 9.1/Uploads/orders.csv"
INTO TABLE orders
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


select i.*,t.tc from
(
select * from orders limit 5
) as i
join
(
select count(*) as tc from orders
) as t;


CREATE or replace VIEW inventory_health AS
SELECT 
    i.inventory_id, i.product_id, p.category, i.warehouse_id, 
    
    CASE 
        WHEN i.quantity_available <= 0 THEN 'OUT_OF_STOCK'
        WHEN i.quantity_available <= p.reorder_level THEN 'REORDER_NEEDED'
        WHEN i.quantity_available <= (p.reorder_level * 1.5) THEN 'LOW_STOCK'
        ELSE 'ADEQUATE'
    END AS stock_status,
    
    DATEDIFF(CURDATE(), i.last_updated) AS days_since_update,
    
    (i.quantity_available * p.unit_price) AS stock_value,
    
    CASE WHEN i.quantity_available <= p.reorder_level THEN 1 ELSE 0 END AS reorder_flag
FROM inventory i
JOIN products p ON i.product_id = p.product_id
JOIN warehouses w ON i.warehouse_id = w.warehouse_id;

select * from inventory_health;


CREATE or replace VIEW supplier_performance AS
SELECT 
    s.supplier_id, s.supplier_name,
      
    COUNT(CASE WHEN o.delivery_date IS NOT NULL THEN 1 END) AS delivered_orders,
    COUNT(CASE WHEN o.delivery_date <= o.order_date + INTERVAL s.avg_lead_time_days DAY THEN 1 END) AS on_time_deliveries,
    
    ROUND(
        COUNT(CASE WHEN o.delivery_date <= o.order_date + INTERVAL s.avg_lead_time_days DAY THEN 1 END) * 100.0 /
        NULLIF(COUNT(CASE WHEN o.delivery_date IS NOT NULL THEN 1 END), 0), 2
    ) AS on_time_delivery_rate,
    
    ROUND(AVG(CASE WHEN o.delivery_date IS NOT NULL 
              THEN DATEDIFF(o.delivery_date, o.order_date) END), 1) AS actual_avg_lead_time
    
FROM suppliers s
LEFT JOIN orders o ON s.supplier_id = o.supplier_id
LEFT JOIN products p ON o.product_id = p.product_id
GROUP BY s.supplier_id, s.supplier_name;

select * from supplier_performance;


CREATE or replace VIEW product_movement AS
SELECT 
    p.product_id, p.category,
    
    COALESCE(SUM(i.quantity_available), 0) AS total_current_stock,
    
    COALESCE(AVG(o.quantity_ordered), 0) AS avg_order_size,
    
    CASE 
        WHEN COUNT(o.order_id) = 0 THEN 'NO_MOVEMENT'
        WHEN COUNT(o.order_id) >= 10 THEN 'FAST_MOVING'
        WHEN COUNT(o.order_id) >= 3 THEN 'MEDIUM_MOVING'
        ELSE 'SLOW_MOVING'
    END AS movement_category,
       
    -- order id (90 days) assume 30 days per month
    ROUND(COUNT(o.order_id) / 3.0, 2) AS orders_per_month,
    
    MAX(o.order_date) AS last_order_date

FROM products p
LEFT JOIN inventory i ON p.product_id = i.product_id
LEFT JOIN orders o ON p.product_id = o.product_id 
    AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
GROUP BY p.product_id, p.category;

select * from product_movement;

CREATE OR REPLACE VIEW procurement_summary AS
SELECT 
    --  Inventory Metrics
    COUNT(DISTINCT ih.product_id) AS total_products,
    COUNT(DISTINCT ih.warehouse_id) AS total_warehouses,
    SUM(ih.stock_value) AS total_inventory_value,
    COUNT(CASE WHEN ih.reorder_flag = 1 THEN 1 END) AS products_needing_reorder,
    COUNT(CASE WHEN ih.stock_status = 'OUT_OF_STOCK' THEN 1 END) AS stockout_products,
    
    --  Movement Analysis
    COUNT(CASE WHEN pm.movement_category = 'FAST_MOVING' THEN 1 END) AS fast_moving_products,
    COUNT(CASE WHEN pm.movement_category = 'SLOW_MOVING' THEN 1 END) AS slow_moving_products,
    COUNT(CASE WHEN pm.movement_category = 'NO_MOVEMENT' THEN 1 END) AS no_movement_products,
    
    --  Supplier Metrics
    COUNT(DISTINCT sp.supplier_id) AS active_suppliers,
    ROUND(AVG(sp.on_time_delivery_rate), 2) AS avg_on_time_delivery_rate,
    ROUND(AVG(sp.actual_avg_lead_time), 1) AS avg_actual_lead_time,
    
    -- Recency
    MAX(ih.days_since_update) AS max_days_since_update

FROM inventory_health ih
LEFT JOIN product_movement pm ON ih.product_id = pm.product_id
LEFT JOIN supplier_performance sp ON 1=1;  -- join without conditions to summarize supplier KPIs globally

select * from procurement_summary;


-- Create comprehensive master view for ML analysis
CREATE OR REPLACE VIEW master_inventory_analysis AS
WITH product_counts AS (
    SELECT 
        o.supplier_id,
        o.product_id,
        COUNT(*) AS order_count
    FROM orders o
    GROUP BY o.supplier_id, o.product_id
),
top_supplier_products AS (
    SELECT 
        supplier_id,
        product_id
    FROM (
        SELECT 
            pc.*,
            ROW_NUMBER() OVER (PARTITION BY pc.supplier_id ORDER BY pc.order_count DESC) AS rn
        FROM product_counts pc
    ) ranked
    WHERE rn = 1
)

SELECT 
    -- Product Information
    p.product_id,
    p.product_name,
    p.category,
    p.reorder_level,
    p.unit_price,
    
    -- Inventory Health Metrics
    ih.warehouse_id,
    
    ih.stock_status,
    ih.days_since_update,
    ih.stock_value,
    ih.reorder_flag,
    
    -- Product Movement Metrics
    pm.total_current_stock,
    pm.avg_order_size,
    pm.movement_category,
    pm.orders_per_month,
    pm.last_order_date,
    DATEDIFF(CURDATE(), pm.last_order_date) AS days_since_last_order,
    
    -- Supplier Performance
    sp.supplier_id,
    sp.supplier_name,
    sp.on_time_delivery_rate,
    sp.actual_avg_lead_time,
    
    -- Warehouse Information
    w.location_name,
	(ih.stock_value / NULLIF(p.unit_price, 0)) AS quantity_available,

    -- Calculated Features for ML
    CASE 
        WHEN ih.quantity_available <= 0 THEN 1
        WHEN ih.quantity_available <= p.reorder_level THEN 2
        WHEN ih.quantity_available <= (p.reorder_level * 1.5) THEN 3
        ELSE 4
    END AS stock_level_numeric,
    
    CASE 
        WHEN pm.movement_category = 'FAST_MOVING' THEN 4
        WHEN pm.movement_category = 'MEDIUM_MOVING' THEN 3
        WHEN pm.movement_category = 'SLOW_MOVING' THEN 2
        WHEN pm.movement_category = 'NO_MOVEMENT' THEN 1
        ELSE 1
    END AS movement_score,
    
    -- Seasonality indicators
    MONTH(CURDATE()) AS current_month,
    QUARTER(CURDATE()) AS current_quarter,
    
    -- Target variable for ML (1 = needs reorder, 0 = doesn't need reorder)
    CASE 
        WHEN ih.reorder_flag = 1 OR ih.quantity_available <= 0 THEN 1
        ELSE 0
    END AS needs_reorder

FROM products p
LEFT JOIN inventory_health ih ON p.product_id = ih.product_id
LEFT JOIN product_movement pm ON p.product_id = pm.product_id
LEFT JOIN top_supplier_products tsp ON p.product_id = tsp.product_id
LEFT JOIN supplier_performance sp ON tsp.supplier_id = sp.supplier_id
LEFT JOIN warehouses w ON ih.warehouse_id = w.warehouse_id;


-- Export query for Python analysis
SELECT * FROM master_inventory_analysis;