from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from joblib import load
import io
import os
from typing import List, Optional, Dict
import json
import tempfile
import shutil
from pathlib import Path
import logging
import traceback
import re
import pandas as pd
from io import StringIO
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
from dotenv import load_dotenv
import os
from io import StringIO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="InvenTrack Pro", version="2.0.0")

# Try to mount static files if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Load environment variables from .env
load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_BLOB_CONTAINER = os.getenv('AZURE_BLOB_CONTAINER')

def get_csv(nam):
    """Get CSV from Azure Blob Storage with error handling"""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=AZURE_BLOB_CONTAINER, blob=nam + ".csv")
        
        blob_data = blob_client.download_blob().readall().decode('utf-8')
        df = pd.read_csv(StringIO(blob_data))
        return df    
    except ResourceNotFoundError:
        logger.warning(f"Blob '{nam}.csv' not found in Azure storage. Returning empty DataFrame.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error retrieving blob '{nam}': {e}")
        return pd.DataFrame()

def save_csvs(df, nam):
    """Save CSV to Azure Blob Storage"""
    try:
        output = StringIO()
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=AZURE_BLOB_CONTAINER, blob=nam + ".csv")
        df.to_csv(output, index=False)
        csv_data = output.getvalue()
        # Upload the CSV string
        blob_client.upload_blob(csv_data, overwrite=True)
        logger.info(f"Successfully saved {nam}.csv to Azure storage")
    except Exception as e:
        logger.error(f"Error saving {nam}.csv to Azure storage: {e}")
        raise

#OUTPUT_CSV = r"D:\swastik\.0. Geakminds\3. Machine Learning\use case projects\inventpro powerbi\python_layer.csv"

# Global variables to store uploaded data
uploaded_data = {
    'products_df': None,
    'inventory_df': None,
    'warehouses_df': None,
    'orders_df': None,
    'suppliers_df': None,
    'final_results': None,
    'temp_entries': []
}

# Categories for dropdown
CATEGORIES = ['Electronics', 'Furniture', 'Clothing', 'Food', 'Tools', 'Books', 'Sports', 'Home & Garden']

# Initialize default data - will be loaded safely
DEFAULT_WAREHOUSES = {}
DEFAULT_SUPPLIERS = {}

def initialize_defaults():
    """Initialize default warehouses and suppliers with error handling"""
    global DEFAULT_WAREHOUSES, DEFAULT_SUPPLIERS
    
    try:
        # Try to load warehouses
        warehouses_df = get_csv('warehouses')
        if not warehouses_df.empty and 'warehouse_id' in warehouses_df.columns:
            DEFAULT_WAREHOUSES = (
                warehouses_df.set_index('warehouse_id')
                .rename(columns={'location_name': 'name'})
                .to_dict(orient='index')
            )
            logger.info(f"Loaded {len(DEFAULT_WAREHOUSES)} default warehouses")
        else:
            # Create sample default warehouses if none exist
            DEFAULT_WAREHOUSES = {
                'WH001': {'name': 'Main Warehouse'},
                'WH002': {'name': 'Secondary Warehouse'},
                'WH003': {'name': 'Backup Warehouse'}
            }
            logger.info("Using sample default warehouses")
    except Exception as e:
        logger.error(f"Error loading warehouses: {e}")
        DEFAULT_WAREHOUSES = {
            'WH001': {'name': 'Main Warehouse'},
            'WH002': {'name': 'Secondary Warehouse'}
        }
    
    try:
        # Try to load suppliers
        suppliers_df = get_csv('suppliers')
        if not suppliers_df.empty and 'supplier_id' in suppliers_df.columns:
            DEFAULT_SUPPLIERS = (
                suppliers_df.set_index('supplier_id')
                .rename(columns={'supplier_name': 'name'})
                .to_dict(orient='index')
            )
            logger.info(f"Loaded {len(DEFAULT_SUPPLIERS)} default suppliers")
        else:
            # Create sample default suppliers if none exist
            DEFAULT_SUPPLIERS = {
                'S001': {'name': 'Default Supplier 1', 'quality_score': 85.0, 'avg_lead_time_days': 7},
                'S002': {'name': 'Default Supplier 2', 'quality_score': 80.0, 'avg_lead_time_days': 10},
                'S000': {'name': 'Unknown Supplier', 'quality_score': 75.0, 'avg_lead_time_days': 14}
            }
            logger.info("Using sample default suppliers")
    except Exception as e:
        logger.error(f"Error loading suppliers: {e}")
        DEFAULT_SUPPLIERS = {
            'S001': {'name': 'Default Supplier 1', 'quality_score': 85.0, 'avg_lead_time_days': 7},
            'S000': {'name': 'Unknown Supplier', 'quality_score': 75.0, 'avg_lead_time_days': 14}
        }

def load_existing_data():
    """Load existing CSV data if available with error handling"""
    global uploaded_data
    
    try:
        uploaded_data['products_df'] = get_csv("products")
        if not uploaded_data['products_df'].empty:
            logger.info(f"Loaded existing products data: {len(uploaded_data['products_df'])} records")
        else:
            logger.info("No existing products data found")
    except Exception as e:
        logger.error(f"Error loading products: {e}")
        uploaded_data['products_df'] = None
    
    try:
        uploaded_data['inventory_df'] = get_csv("inventory")
        if not uploaded_data['inventory_df'].empty:
            logger.info(f"Loaded existing inventory data: {len(uploaded_data['inventory_df'])} records")
        else:
            logger.info("No existing inventory data found")
    except Exception as e:
        logger.error(f"Error loading inventory: {e}")
        uploaded_data['inventory_df'] = None
    
    try:
        uploaded_data['warehouses_df'] = get_csv('warehouses')
        if not uploaded_data['warehouses_df'].empty:
            logger.info(f"Loaded existing warehouses data: {len(uploaded_data['warehouses_df'])} records")
        else:
            logger.info("No existing warehouses data found")
    except Exception as e:
        logger.error(f"Error loading warehouses: {e}")
        uploaded_data['warehouses_df'] = None
    
    try:
        uploaded_data['suppliers_df'] = get_csv("suppliers")
        if not uploaded_data['suppliers_df'].empty:
            logger.info(f"Loaded existing suppliers data: {len(uploaded_data['suppliers_df'])} records")
        else:
            logger.info("No existing suppliers data found")
    except Exception as e:
        logger.error(f"Error loading suppliers: {e}")
        uploaded_data['suppliers_df'] = None
    
    try:
        uploaded_data['orders_df'] = get_csv("orders")
        if not uploaded_data['orders_df'].empty:
            logger.info(f"Loaded existing orders data: {len(uploaded_data['orders_df'])} records")
        else:
            logger.info("No existing orders data found")
    except Exception as e:
        logger.error(f"Error loading orders: {e}")
        uploaded_data['orders_df'] = None

# Initialize defaults and load existing data on startup
initialize_defaults()
load_existing_data()

def get_supplier_info(supplier_id: str) -> dict:
    """Get supplier information from defaults or existing data"""
    # First check default mappings
    if supplier_id in DEFAULT_SUPPLIERS:
        return {
            'supplier_name': DEFAULT_SUPPLIERS[supplier_id]['name'],
            'quality_score': DEFAULT_SUPPLIERS[supplier_id]['quality_score'],
            'avg_lead_time_days': DEFAULT_SUPPLIERS[supplier_id]['avg_lead_time_days']
        }
    
    # Then check uploaded data
    if uploaded_data.get('suppliers_df') is not None and not uploaded_data['suppliers_df'].empty:
        supplier_row = uploaded_data['suppliers_df'][uploaded_data['suppliers_df']['supplier_id'] == supplier_id]
        if not supplier_row.empty:
            return {
                'supplier_name': supplier_row.iloc[0]['supplier_name'],
                'quality_score': supplier_row.iloc[0]['quality_score'],
                'avg_lead_time_days': supplier_row.iloc[0]['avg_lead_time_days']
            }
    
    # Return default values for new supplier
    return {
        'supplier_name': f'New Supplier {supplier_id}',
        'quality_score': 80.0,
        'avg_lead_time_days': 7
    }

def get_warehouse_info(warehouse_id: str) -> dict:
    """Get warehouse information from defaults or existing data"""
    if warehouse_id in DEFAULT_WAREHOUSES:
        return {
            'location_name': DEFAULT_WAREHOUSES[warehouse_id]['name']
        }

    if uploaded_data.get('warehouses_df') is not None and not uploaded_data['warehouses_df'].empty:
        warehouse_row = uploaded_data['warehouses_df'][uploaded_data['warehouses_df']['warehouse_id'] == warehouse_id]
        if not warehouse_row.empty:
            return {
                'location_name': warehouse_row.iloc[0]['location_name']
            }

    return None

def get_product_info(product_id: str) -> dict:
    """Get product information from existing data"""
    if uploaded_data['products_df'] is not None and not uploaded_data['products_df'].empty:
        product_row = uploaded_data['products_df'][uploaded_data['products_df']['product_id'] == product_id]
        if not product_row.empty:
            return product_row.iloc[0].to_dict()
    return None

def append_to_csv(file_path: str, new_df: pd.DataFrame, key_columns: list):
    """Append new data to CSV file, avoiding duplicates based on key columns"""
    try:
        existing_df = get_csv(file_path)
        if existing_df.empty:
            # If no existing data, just save the new data
            save_csvs(new_df, file_path)
            logger.info(f"Created new {file_path}.csv with {len(new_df)} records")
        else:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Remove duplicates based on key columns
            combined_df = combined_df.drop_duplicates(subset=key_columns, keep='last')
            # Save back to file
            save_csvs(combined_df, file_path)
            logger.info(f"Appended {len(new_df)} records to {file_path}, total records: {len(combined_df)}")
    except Exception as e:
        logger.error(f"Error appending to {file_path}: {e}")
        raise

def add_to_dataset(df_name: str, new_data: dict):
    """Add new data to the appropriate dataset if it doesn't exist"""
    global uploaded_data
    
    key_columns = {
        'products_df': ['product_id'],
        'suppliers_df': ['supplier_id'],
        'warehouses_df': ['warehouse_id'],
        'inventory_df': ['product_id', 'warehouse_id'],
        'orders_df': ['order_id']
    }
    
    if uploaded_data[df_name] is None or uploaded_data[df_name].empty:
        # Create new dataframe if none exists
        uploaded_data[df_name] = pd.DataFrame([new_data])
        logger.info(f"Created new {df_name} with entry")
    else:
        # Check if entry already exists
        keys = key_columns[df_name]
        existing_mask = True
        for key in keys:
            if key in new_data:
                existing_mask = existing_mask & (uploaded_data[df_name][key] == new_data[key])
        
        if not existing_mask.any():
            # Append new entry
            new_row_df = pd.DataFrame([new_data])
            uploaded_data[df_name] = pd.concat([
                uploaded_data[df_name],
                new_row_df
            ], ignore_index=True)
            logger.info(f"Added new entry to {df_name}")
        else:
            # Update existing entry
            for col, value in new_data.items():
                uploaded_data[df_name].loc[existing_mask, col] = value
            logger.info(f"Updated existing entry in {df_name}")

def clean_for_json(obj):
    """Clean data structure for JSON serialization by removing NaN and inf values"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return 0 if isinstance(obj, np.integer) else 0.0
        return int(obj) if isinstance(obj, np.integer) else float(obj)
    elif pd.isna(obj):
        return None
    elif obj is None or str(obj).lower() in ['nan', 'none', 'nat']:
        return ""
    else:
        return obj

def create_final_view(products_df, inventory_df, warehouses_df, orders_df, suppliers_df):
    """Create the final view by merging all dataframes and calculating derived fields"""
    try:
        logger.info("Creating final view...")
        
        # Handle empty dataframes
        if inventory_df.empty:
            logger.warning("Inventory dataframe is empty")
            return pd.DataFrame()
        
        df = inventory_df.copy()
        
        # Ensure proper data types
        df['quantity_available'] = pd.to_numeric(df['quantity_available'], errors='coerce').fillna(0)
        
        # Merge with products
        if not products_df.empty:
            df = df.merge(products_df, on='product_id', how='left')
        df['unit_price'] = pd.to_numeric(df.get('unit_price', 0), errors='coerce').fillna(0)
        df['reorder_level'] = pd.to_numeric(df.get('reorder_level', 10), errors='coerce').fillna(10)
        
        # Merge with warehouses
        if not warehouses_df.empty:
            df = df.merge(warehouses_df, on='warehouse_id', how='left')
        
        # Date calculations - handle various date formats
        try:
            df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
            # If conversion failed, use today's date
            df['last_updated'] = df['last_updated'].fillna(pd.Timestamp.now())
        except Exception as e:
            logger.warning(f"Date conversion issue: {e}")
            df['last_updated'] = pd.Timestamp.now()
        
        today = pd.Timestamp.now().normalize()
        df['days_since_update'] = (today - df['last_updated']).dt.days
        df['stock_value'] = df['quantity_available'] * df['unit_price']
        
        # Stock status function
        def get_stock_status(row):
            try:
                qty = float(row['quantity_available'])
                reorder = float(row['reorder_level'])
                if qty <= 0:
                    return 'OUT_OF_STOCK'
                elif qty <= reorder:
                    return 'REORDER_NEEDED'
                elif qty <= (reorder * 1.5):
                    return 'LOW_STOCK'
                else:
                    return 'ADEQUATE'
            except (ValueError, TypeError):
                return 'UNKNOWN'
        
        df['stock_status'] = df.apply(get_stock_status, axis=1)
        
        # Total stock calculation - sum across all warehouses for each product
        try:
            total_stock = inventory_df.groupby('product_id')['quantity_available'].sum().reset_index()
            total_stock.columns = ['product_id', 'total_current_stock']
            df = df.merge(total_stock, on='product_id', how='left')
            df['total_current_stock'] = df['total_current_stock'].fillna(0)
        except Exception as e:
            logger.warning(f"Total stock calculation error: {e}")
            df['total_current_stock'] = df['quantity_available']
        
        # Recent orders analysis (90 days)
        cutoff_date = datetime.now() - timedelta(days=90)
        
        if not orders_df.empty:
            try:
                # Ensure proper date conversion
                orders_df_copy = orders_df.copy()
                orders_df_copy['order_date'] = pd.to_datetime(orders_df_copy['order_date'], errors='coerce')
                orders_df_copy = orders_df_copy.dropna(subset=['order_date'])
                
                if len(orders_df_copy) > 0:
                    recent_orders = orders_df_copy[orders_df_copy['order_date'] >= cutoff_date]
                    
                    # Average order size calculation
                    if len(recent_orders) > 0:
                        avg_orders = recent_orders.groupby('product_id')['quantity_ordered'].mean().reset_index()
                        avg_orders.columns = ['product_id', 'avg_order_size']
                        avg_orders['avg_order_size'] = avg_orders['avg_order_size'].round(4)
                    else:
                        avg_orders = pd.DataFrame(columns=['product_id', 'avg_order_size'])
                    
                    # Order counts and movement analysis
                    order_counts = recent_orders.groupby('product_id').size().reset_index(name='order_count_90days')
                    
                    def get_movement_category(count):
                        if count >= 10: return 'FAST_MOVING'
                        elif count >= 3: return 'MEDIUM_MOVING'
                        elif count >= 1: return 'SLOW_MOVING'
                        else: return 'NO_MOVEMENT'
                    
                    def get_movement_score(count):
                        if count >= 10: return 4
                        elif count >= 3: return 3
                        elif count >= 1: return 2
                        else: return 1
                    
                    order_counts['movement_category'] = order_counts['order_count_90days'].apply(get_movement_category)
                    order_counts['movement_score'] = order_counts['order_count_90days'].apply(get_movement_score)
                    order_counts['orders_per_month'] = round(order_counts['order_count_90days'] / 3.0, 2)
                    
                    # Last order date
                    last_orders = orders_df_copy.groupby('product_id')['order_date'].max().reset_index()
                    last_orders.columns = ['product_id', 'last_order_date']
                    
                    # Recent supplier info
                    recent_supplier = orders_df_copy.sort_values('order_date').drop_duplicates('product_id', keep='last')[
                        ['product_id', 'supplier_id']
                    ]
                    if len(recent_supplier) > 0 and not suppliers_df.empty:
                        recent_supplier = recent_supplier.merge(suppliers_df, on='supplier_id', how='left')
                    
                else:
                    # No recent orders
                    avg_orders = pd.DataFrame(columns=['product_id', 'avg_order_size'])
                    order_counts = pd.DataFrame(columns=['product_id', 'movement_category', 'movement_score', 'orders_per_month'])
                    last_orders = pd.DataFrame(columns=['product_id', 'last_order_date'])
                    recent_supplier = pd.DataFrame(columns=['product_id', 'supplier_id', 'supplier_name'])
                    
            except Exception as e:
                logger.error(f"Error processing orders: {e}")
                # Fallback to empty dataframes
                avg_orders = pd.DataFrame(columns=['product_id', 'avg_order_size'])
                order_counts = pd.DataFrame(columns=['product_id', 'movement_category', 'movement_score', 'orders_per_month'])
                last_orders = pd.DataFrame(columns=['product_id', 'last_order_date'])
                recent_supplier = pd.DataFrame(columns=['product_id', 'supplier_id', 'supplier_name'])
        else:
            # Empty orders dataframe
            avg_orders = pd.DataFrame(columns=['product_id', 'avg_order_size'])
            order_counts = pd.DataFrame(columns=['product_id', 'movement_category', 'movement_score', 'orders_per_month'])
            last_orders = pd.DataFrame(columns=['product_id', 'last_order_date'])
            recent_supplier = pd.DataFrame(columns=['product_id', 'supplier_id', 'supplier_name'])
        
        # Merge all calculated fields
        df = df.merge(avg_orders, on='product_id', how='left')
        df['avg_order_size'] = df['avg_order_size'].fillna(0)
        
        df = df.merge(order_counts[['product_id', 'movement_category', 'movement_score', 'orders_per_month']], 
                      on='product_id', how='left')
        df['movement_category'] = df['movement_category'].fillna('NO_MOVEMENT')
        df['movement_score'] = df['movement_score'].fillna(1)
        df['orders_per_month'] = df['orders_per_month'].fillna(0)
        
        df = df.merge(last_orders, on='product_id', how='left')
        if 'last_order_date' in df.columns and not df['last_order_date'].empty:
            df['last_order_date'] = pd.to_datetime(df['last_order_date'], errors='coerce')
            df['days_since_last_order'] = (today - df['last_order_date']).dt.days
            df['days_since_last_order'] = df['days_since_last_order'].fillna(9999).astype(int)
        else:
            df['last_order_date'] = pd.NaT
            df['days_since_last_order'] = 9999
        
        # Format dates for display
        df['last_order_date'] = df['last_order_date'].dt.strftime('%m/%d/%Y').str.lstrip('0').str.replace('/0', '/')
        df['last_updated'] = df['last_updated'].dt.strftime('%m/%d/%Y').str.lstrip('0').str.replace('/0', '/')
        
        # Supplier information
        df = df.merge(recent_supplier[['product_id', 'supplier_id', 'supplier_name']], on='product_id', how='left')
        df['supplier_id'] = df['supplier_id'].fillna('')
        df['supplier_name'] = df['supplier_name'].fillna('')
        
        # Stock level numeric
        def get_stock_level_numeric(row):
            try:
                qty = float(row['quantity_available'])
                reorder = float(row['reorder_level'])
                if qty <= 0: return 1
                elif qty <= reorder: return 2
                elif qty <= (reorder * 1.5): return 3
                else: return 4
            except (ValueError, TypeError):
                return 1
        
        df['stock_level_numeric'] = df.apply(get_stock_level_numeric, axis=1)
        
        # Date fields
        df['current_month'] = datetime.now().month
        df['current_quarter'] = (datetime.now().month - 1) // 3 + 1
        
        # Handle NaN values that can cause issues in JSON serialization and calculations
        df = df.replace([np.inf, -np.inf], 0)
        
        # For numeric columns, fill NaN with 0 and ensure proper data types
        numeric_cols = ['quantity_available', 'unit_price', 'reorder_level', 'days_since_update', 
                       'stock_value', 'total_current_stock', 'avg_order_size', 'orders_per_month',
                       'days_since_last_order', 'stock_level_numeric', 'movement_score',
                       'current_month', 'current_quarter']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                # Ensure no infinite values
                df[col] = df[col].replace([np.inf, -np.inf], 0)
        
        # String columns - handle NaN and None values
        string_cols = ['product_name', 'category', 'warehouse_id', 'stock_status',
                      'movement_category', 'supplier_id', 'supplier_name', 'location_name']
        
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('').replace('nan', '').replace('None', '')
        
        logger.info(f"Final view created successfully with {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error in create_final_view: {e}")
        logger.error(traceback.format_exc())
        raise e

def encode_custom_input(df, label_mappings):
    """Encode categorical variables using provided mappings"""
    try:
        encoded_df = df.copy()
        for column in df.columns:
            if column in label_mappings:
                encoded_df[column] = df[column].map(label_mappings[column]).fillna(-1)
        return encoded_df
    except Exception as e:
        logger.error(f"Error in encode_custom_input: {e}")
        return df

def nxt_pred(df):
    """Make predictions using the trained model"""
    try:
        model_path = r'random_forest_model.joblib'
        mappings_path = r'label_mappings.joblib'
        features_path = r"feature_col.joblib"
        
        if not all(os.path.exists(p) for p in [model_path, mappings_path, features_path]):
            logger.warning("Model files not found, using fallback prediction")
            df['needs_reorder'] = (df['quantity_available'] <= df['reorder_level']).astype(int)
            return df
        
        loaded_model = load(model_path)
        label_mappings = load(mappings_path)
        feature_cols = load(features_path)
        
        # Remove inventory_id from feature_cols if it exists
        if 'inventory_id' in feature_cols:
            feature_cols = [col for col in feature_cols if col != 'inventory_id']
        
        # Check if required features exist
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing features for prediction: {missing_features}")
            df['needs_reorder'] = (df['quantity_available'] <= df['reorder_level']).astype(int)
            return df
        
        custom_encoded_df = encode_custom_input(df, label_mappings)
        custom_encoded_df = custom_encoded_df[feature_cols]
        
        # Fill any remaining NaN values
        custom_encoded_df = custom_encoded_df.fillna(0)
        
        preds = loaded_model.predict(custom_encoded_df)
        df['needs_reorder'] = preds
        
        logger.info("Predictions completed successfully")
        return df
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        # Fallback to simple rule-based prediction
        df['needs_reorder'] = (df['quantity_available'] <= df['reorder_level']).astype(int)
        return df

def create_single_entry_dataframe_enhanced(entry_data: dict):
    """Enhanced version with better data mapping and default values"""
    try:
        # Check if product exists
        existing_product = get_product_info(entry_data['product_id'])
        
        if existing_product:
            # Use existing product data, override only if new values provided
            products_df = pd.DataFrame([{
                'product_id': entry_data['product_id'],
                'product_name': entry_data.get('product_name', existing_product['product_name']),
                'category': entry_data.get('category', existing_product['category']),
                'reorder_level': entry_data.get('reorder_level', existing_product['reorder_level']),
                'unit_price': entry_data.get('unit_price', existing_product['unit_price'])
            }])
        else:
            # New product - add to dataset
            new_product = {
                'product_id': entry_data['product_id'],
                'product_name': entry_data['product_name'],
                'category': entry_data['category'],
                'reorder_level': entry_data.get('reorder_level', 10),
                'unit_price': entry_data.get('unit_price', 0.0)
            }
            products_df = pd.DataFrame([new_product])
            add_to_dataset('products_df', new_product)
        
        # Check if warehouse exists using defaults first
        warehouse_info = get_warehouse_info(entry_data['warehouse_id'])
        
        if warehouse_info:
            warehouses_df = pd.DataFrame([{
                'warehouse_id': entry_data['warehouse_id'],
                'location_name': entry_data.get('location_name', warehouse_info['location_name'])
            }])
        else:
            # New warehouse - add to dataset
            new_warehouse = {
                'warehouse_id': entry_data['warehouse_id'],
                'location_name': entry_data['location_name']
            }
            warehouses_df = pd.DataFrame([new_warehouse])
            add_to_dataset('warehouses_df', new_warehouse)
        
        # Get supplier info (from defaults or existing)
        supplier_info = get_supplier_info(entry_data.get('supplier_id', 'S000'))
        
        suppliers_df = pd.DataFrame([{
            'supplier_id': entry_data.get('supplier_id', 'S000'),
            'supplier_name': entry_data.get('supplier_name', supplier_info['supplier_name']),
            'quality_score': entry_data.get('quality_score', supplier_info['quality_score']),
            'avg_lead_time_days': entry_data.get('avg_lead_time_days', supplier_info['avg_lead_time_days'])
        }])
        
        # If supplier doesn't exist in defaults or data, add to dataset
        if entry_data.get('supplier_id', 'S000') not in DEFAULT_SUPPLIERS:
            add_to_dataset('suppliers_df', suppliers_df.iloc[0].to_dict())
        
        # Create inventory entry
        inventory_df = pd.DataFrame([{
            'product_id': entry_data['product_id'],
            'warehouse_id': entry_data['warehouse_id'],
            'quantity_available': entry_data.get('quantity_available', 0),
            'last_updated': entry_data.get('last_updated', datetime.now().strftime('%Y-%m-%d'))
        }])
        
        # Create orders dataframe if order info is provided
        if entry_data.get('order_id') and entry_data.get('order_date'):
            orders_df = pd.DataFrame([{
                'order_id': entry_data['order_id'],
                'product_id': entry_data['product_id'],
                'supplier_id': entry_data.get('supplier_id', 'S000'),
                'quantity_ordered': entry_data.get('quantity_ordered', 0),
                'order_date': entry_data.get('order_date', datetime.now().strftime('%Y-%m-%d')),
                'delivery_date': entry_data.get(
                    'delivery_date',
                    (
                        datetime.strptime(
                            entry_data.get('order_date', datetime.now().strftime('%Y-%m-%d')), '%Y-%m-%d'
                        ) + timedelta(days=int(supplier_info['avg_lead_time_days']))
                    ).strftime('%Y-%m-%d')
                )
            }])
        else:
            orders_df = pd.DataFrame(columns=['order_id', 'product_id', 'supplier_id', 'quantity_ordered', 'order_date', 'delivery_date'])
        
        # Process through the same pipeline
        final_df = create_final_view(products_df, inventory_df, warehouses_df, orders_df, suppliers_df)
        final_df = nxt_pred(final_df)
        
        return final_df
        
    except Exception as e:
        logger.error(f"Error in create_single_entry_dataframe_enhanced: {e}")
        raise e

def get_existing_data_for_entry(entry_data: dict) -> dict:
    """Get existing data that can be auto-filled for a new entry"""
    auto_fill_data = {}
    
    # Auto-fill product data if exists
    if entry_data.get('product_id'):
        product_info = get_product_info(entry_data['product_id'])
        if product_info:
            auto_fill_data.update({
                'product_name': product_info['product_name'],
                'category': product_info['category'],
                'reorder_level': product_info['reorder_level'],
                'unit_price': product_info['unit_price']
            })
    
    # Auto-fill supplier data from defaults or existing
    if entry_data.get('supplier_id'):
        supplier_info = get_supplier_info(entry_data['supplier_id'])
        auto_fill_data.update({
            'supplier_name': supplier_info['supplier_name'],
            'quality_score': supplier_info['quality_score'],
            'avg_lead_time_days': supplier_info['avg_lead_time_days']
        })
    
    # Auto-fill warehouse data from defaults or existing
    if entry_data.get('warehouse_id'):
        warehouse_info = get_warehouse_info(entry_data['warehouse_id'])
        if warehouse_info:
            auto_fill_data['location_name'] = warehouse_info['location_name']
    
    return auto_fill_data

# Output columns definition
OUTPUT_COLUMNS = [
    'product_id', 'product_name', 'category', 'reorder_level', 'unit_price',
    'last_updated', 'warehouse_id', 'stock_status',
    'days_since_update', 'stock_value', 'total_current_stock', 'avg_order_size',
    'movement_category', 'orders_per_month', 'last_order_date',
    'days_since_last_order', 'supplier_id', 'supplier_name', 'location_name',
    'quantity_available', 'stock_level_numeric', 'movement_score',
    'current_month', 'current_quarter', 'needs_reorder'
]

def format_output_df(df):
    """Format the output DataFrame to match the required structure"""
    try:
        # Create a copy to avoid modifying the original
        output_df = df.copy()
        
        # First, handle all NaN and infinite values
        output_df = output_df.replace([np.inf, -np.inf], 0)
        output_df = output_df.fillna(0)
        
        # Ensure all required columns exist
        for col in OUTPUT_COLUMNS:
            if col not in output_df.columns:
                if col in ['supplier_id', 'supplier_name']:
                    output_df[col] = ''
                elif col in ['avg_order_size', 'orders_per_month', 'days_since_last_order', 'movement_score']:
                    output_df[col] = 0
                elif col == 'movement_category':
                    output_df[col] = 'NO_MOVEMENT'
                elif col == 'stock_status':
                    output_df[col] = 'UNKNOWN'
                else:
                    output_df[col] = 0
        
        # Select only the required columns in the correct order
        output_df = output_df[OUTPUT_COLUMNS].copy()
        
        # Clean numeric data types and handle edge cases
        numeric_cols = ['unit_price', 'stock_value', 'avg_order_size', 'orders_per_month']
        for col in numeric_cols:
            if col in output_df.columns:
                output_df[col] = pd.to_numeric(output_df[col], errors='coerce').fillna(0)
                # Replace any remaining inf values
                output_df[col] = output_df[col].replace([np.inf, -np.inf], 0)
        
        # Ensure integer columns are properly formatted
        int_cols = ['reorder_level', 'days_since_update', 'total_current_stock', 'days_since_last_order',
                   'quantity_available', 'stock_level_numeric', 'movement_score', 'current_month', 
                   'current_quarter', 'needs_reorder']
        for col in int_cols:
            if col in output_df.columns:
                output_df[col] = pd.to_numeric(output_df[col], errors='coerce').fillna(0)
                output_df[col] = output_df[col].replace([np.inf, -np.inf], 0).astype(int)
        
        # Handle string columns
        string_cols = ['product_id', 'product_name', 'category', 'warehouse_id', 
                      'stock_status', 'movement_category', 'supplier_id', 'supplier_name', 'location_name']
        for col in string_cols:
            if col in output_df.columns:
                output_df[col] = output_df[col].astype(str).replace('nan', '').replace('None', '')
        
        # Format date columns to match expected format (M/D/YYYY)
        date_columns = ['last_updated', 'last_order_date']
        for col in date_columns:
            if col in output_df.columns:
                if pd.api.types.is_datetime64_any_dtype(output_df[col]):
                    # Handle NaT values first
                    mask_nat = output_df[col].isna()
                    
                    # Format valid dates to M/D/YYYY
                    try:
                        output_df.loc[~mask_nat, col] = output_df.loc[~mask_nat, col].dt.strftime('%-m/%-d/%Y')
                    except:
                        # Fallback for systems that don't support %-
                        output_df.loc[~mask_nat, col] = output_df.loc[~mask_nat, col].dt.strftime('%m/%d/%Y').str.lstrip('0').str.replace('/0', '/')
                    
                    # Handle NaT values
                    if col == 'last_order_date':
                        # For last_order_date, if days_since_last_order is 9999, set to empty
                        mask_no_order = output_df['days_since_last_order'] == 9999
                        output_df.loc[mask_no_order | mask_nat, col] = ''
                    else:
                        output_df.loc[mask_nat, col] = ''
                else:
                    # If not datetime, convert to string and clean
                    output_df[col] = output_df[col].astype(str).replace('nan', '').replace('None', '').replace('NaT', '')
        
        return output_df
        
    except Exception as e:
        logger.error(f"Error in format_output_df: {e}")
        logger.error(traceback.format_exc())
        # Return the original dataframe if formatting fails
        return df

def save_updated_datasets():
    """Save updated datasets back to CSV files, appending new data"""
    try:
        if uploaded_data['products_df'] is not None and not uploaded_data['products_df'].empty:
            append_to_csv('products', uploaded_data['products_df'], ['product_id'])
        
        if uploaded_data['suppliers_df'] is not None and not uploaded_data['suppliers_df'].empty:
            append_to_csv('suppliers', uploaded_data['suppliers_df'], ['supplier_id'])
        
        if uploaded_data['warehouses_df'] is not None and not uploaded_data['warehouses_df'].empty:
            append_to_csv('warehouses', uploaded_data['warehouses_df'], ['warehouse_id'])
        
        if uploaded_data['inventory_df'] is not None and not uploaded_data['inventory_df'].empty:
            append_to_csv('inventory', uploaded_data['inventory_df'], ['product_id', 'warehouse_id'])
        
        if uploaded_data['orders_df'] is not None and not uploaded_data['orders_df'].empty:
            append_to_csv('orders', uploaded_data['orders_df'], ['order_id'])
        
        logger.info("Updated datasets saved successfully")
    
    except Exception as e:
        logger.error(f"Error saving updated datasets: {e}")

@app.get("/")
async def main():
    file_path = "static/index.html"
    if not os.path.exists(file_path):
        # Return a simple HTML page if index.html is not found
        return HTMLResponse(content="""
        <html>
            <head><title>InvenTrack Pro API</title></head>
            <body>
                <h1>InvenTrack Pro API v2.0.0</h1>
                <p>Please change the execution to inventpro powerbi directory <a href="/docs">/docs</a></p>
            </body>
        </html>
        """)
    return FileResponse(file_path, media_type="text/html")

@app.post("/upload-csvs/")
async def upload_csvs(
    products: UploadFile = File(...),
    inventory: UploadFile = File(...),
    warehouses: UploadFile = File(...),
    orders: UploadFile = File(...),
    suppliers: UploadFile = File(...)
):
    """Upload and parse CSV files with safe handling"""
    try:
        files = {
            'products_df': products,
            'inventory_df': inventory,
            'warehouses_df': warehouses,
            'orders_df': orders,
            'suppliers_df': suppliers
        }
        
        for key, file in files.items():
            try:
                content = await file.read()
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                
                # If data already exists, append instead of replacing
                if uploaded_data[key] is not None and not uploaded_data[key].empty:
                    # Get key columns for deduplication
                    key_cols = {
                        'products_df': ['product_id'],
                        'inventory_df': ['product_id', 'warehouse_id'],
                        'warehouses_df': ['warehouse_id'],
                        'orders_df': ['order_id'],
                        'suppliers_df': ['supplier_id']
                    }[key]
                    
                    # Combine with existing data
                    uploaded_data[key] = pd.concat([uploaded_data[key], df], ignore_index=True)
                    # Remove duplicates
                    uploaded_data[key] = uploaded_data[key].drop_duplicates(subset=key_cols, keep='last')
                else:
                    uploaded_data[key] = df
                
                logger.info(f"Successfully loaded {key} with {len(df)} new records, total: {len(uploaded_data[key])}")
            except Exception as e:
                logger.error(f"Error loading {key}: {e}")
                raise HTTPException(status_code=400, detail=f"Error processing {key}: {str(e)}")
        
        return {"message": "Files uploaded successfully", "status": "success"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"General upload error: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing files: {str(e)}")

@app.post("/process-data/")
async def process_data():
    """Process the uploaded data and make predictions"""
    try:
        # Check if all required data is available
        required_dfs = ['products_df', 'inventory_df', 'warehouses_df', 'orders_df', 'suppliers_df']
        for df_name in required_dfs:
            if uploaded_data[df_name] is None or uploaded_data[df_name].empty:
                raise HTTPException(status_code=400, detail=f"Missing data: {df_name}")
        
        logger.info("Starting data processing...")
        
        # Create final view
        pre_final_df = create_final_view(
            uploaded_data['products_df'],
            uploaded_data['inventory_df'],
            uploaded_data['warehouses_df'],
            uploaded_data['orders_df'],
            uploaded_data['suppliers_df']
        )
        logger.info(f"Created final view with {len(pre_final_df)} records")
        
        # Make predictions
        final_df = nxt_pred(pre_final_df)
        logger.info("Predictions completed")
        
        # Format for CSV
        output_df = format_output_df(final_df)
        save_csvs(output_df, 'python_layer')
        logger.info(f"Data successfully written to database")
            
        # Also save updated datasets
        save_updated_datasets()
        
        # Save preview for frontend
        uploaded_data['final_results'] = output_df
        
        # Create preview
        preview_df = final_df.head(20).copy()
        preview_df = preview_df.replace([np.inf, -np.inf], 0)
        preview_df = preview_df.fillna(0)
        
        # Convert for JSON
        for col in preview_df.columns:
            if pd.api.types.is_datetime64_any_dtype(preview_df[col]):
                preview_df[col] = preview_df[col].dt.strftime('%Y-%m-%d').fillna('')
            elif pd.api.types.is_numeric_dtype(preview_df[col]):
                preview_df[col] = pd.to_numeric(preview_df[col], errors='coerce').fillna(0)
                if preview_df[col].dtype == 'int64':
                    preview_df[col] = preview_df[col].astype(int)
                else:
                    preview_df[col] = preview_df[col].astype(float)
            else:
                preview_df[col] = preview_df[col].astype(str).replace('nan', '').replace('None', '')
        
        return {
            "message": "Data processed successfully",
            "status": "success",
            "records_processed": len(final_df),
            "preview": clean_for_json(preview_df.to_dict('records'))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.post("/get-autofill-data/")
async def get_autofill_data(entry_data: dict):
    """Get auto-fill suggestions based on defaults and existing data"""
    try:
        auto_fill_data = get_existing_data_for_entry(entry_data)
        
        # Check existence in defaults first, then in data
        existing_product = (
            uploaded_data['products_df'] is not None and 
            not uploaded_data['products_df'].empty and
            entry_data.get('product_id', '') in uploaded_data['products_df']['product_id'].values
        )
        
        existing_supplier = (
            entry_data.get('supplier_id', '') in DEFAULT_SUPPLIERS or
            (uploaded_data['suppliers_df'] is not None and 
             not uploaded_data['suppliers_df'].empty and
             entry_data.get('supplier_id', '') in uploaded_data['suppliers_df']['supplier_id'].values)
        )
        
        existing_warehouse = (
            entry_data.get('warehouse_id', '') in DEFAULT_WAREHOUSES or
            (uploaded_data['warehouses_df'] is not None and 
             not uploaded_data['warehouses_df'].empty and
             entry_data.get('warehouse_id', '') in uploaded_data['warehouses_df']['warehouse_id'].values)
        )
        
        return {
            "status": "success",
            "auto_fill_data": auto_fill_data,
            "existing_product": existing_product,
            "existing_supplier": existing_supplier,
            "existing_warehouse": existing_warehouse
        }
    except Exception as e:
        logger.error(f"Error getting autofill data: {e}")
        return {"status": "error", "auto_fill_data": {}}

@app.get("/get-dropdown-data/")
async def get_dropdown_data():
    """Get data for populating dropdowns including defaults"""
    try:
        dropdown_data = {
            "products": [],
            "suppliers": [],
            "warehouses": [],
            "categories": CATEGORIES
        }
        
        # Products from uploaded data
        if uploaded_data['products_df'] is not None and not uploaded_data['products_df'].empty:
            dropdown_data["products"] = uploaded_data['products_df'][['product_id', 'product_name']].to_dict('records')
        
        # Suppliers - combine defaults with uploaded
        supplier_list = []
        
        # Add default suppliers
        for supplier_id, info in DEFAULT_SUPPLIERS.items():
            supplier_list.append({
                'supplier_id': supplier_id,
                'supplier_name': info['name']
            })
        
        # Add uploaded suppliers not in defaults
        if uploaded_data['suppliers_df'] is not None and not uploaded_data['suppliers_df'].empty:
            for _, row in uploaded_data['suppliers_df'].iterrows():
                if row['supplier_id'] not in DEFAULT_SUPPLIERS:
                    supplier_list.append({
                        'supplier_id': row['supplier_id'],
                        'supplier_name': row['supplier_name']
                    })
        
        dropdown_data["suppliers"] = supplier_list
        
        # Warehouses - combine defaults with uploaded
        warehouse_list = []
        
        # Add default warehouses
        for warehouse_id, info in DEFAULT_WAREHOUSES.items():
            warehouse_list.append({
                'warehouse_id': warehouse_id,
                'location_name': info['name']
            })
        
        # Add uploaded warehouses not in defaults
        if uploaded_data['warehouses_df'] is not None and not uploaded_data['warehouses_df'].empty:
            for _, row in uploaded_data['warehouses_df'].iterrows():
                if row['warehouse_id'] not in DEFAULT_WAREHOUSES:
                    warehouse_list.append({
                        'warehouse_id': row['warehouse_id'],
                        'location_name': row['location_name']
                    })
        
        dropdown_data["warehouses"] = warehouse_list
        
        return dropdown_data
    
    except Exception as e:
        logger.error(f"Error getting dropdown data: {e}")
        return {"products": [], "suppliers": [], "warehouses": [], "categories": CATEGORIES}

@app.post("/add-single-entry/")
async def add_single_entry(entry_data: dict):
    """Add a single entry to temporary storage with enhanced mapping"""
    try:
        # Validate entry data
        required_fields = ['product_id', 'warehouse_id']
        missing_fields = [field for field in required_fields if field not in entry_data or not entry_data[field]]
        
        if missing_fields:
            raise HTTPException(status_code=400, detail=f"Missing required fields: {missing_fields}")
        
        # Get auto-fill data
        auto_fill_data = get_existing_data_for_entry(entry_data)
        
        # Merge with provided data (provided data takes precedence)
        for key, value in auto_fill_data.items():
            if key not in entry_data or not entry_data[key]:
                entry_data[key] = value
        
        # Add to temporary entries
        uploaded_data['temp_entries'].append(entry_data)
        
        return {
            "message": "Entry added to temporary storage",
            "status": "success",
            "temp_entries": uploaded_data['temp_entries'],
            "auto_filled_fields": list(auto_fill_data.keys())
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding entry: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding entry: {str(e)}")

@app.post("/predict-single-entry/")
async def predict_single_entry(entry_data: dict):
    """Make prediction for a single entry without saving"""
    try:
        # Create dataframe and make prediction
        final_df = create_single_entry_dataframe_enhanced(entry_data)
        
        # Get the prediction result
        preview_row = final_df.iloc[0].to_dict()
        
        # Convert datetime objects to strings for JSON serialization
        for key, value in preview_row.items():
            if pd.api.types.is_datetime64_any_dtype(type(value)) or pd.isna(value):
                if pd.isna(value):
                    preview_row[key] = ''
                else:
                    preview_row[key] = pd.to_datetime(value).strftime('%Y-%m-%d') if pd.notna(value) else ''
            elif isinstance(value, (np.integer, np.floating)):
                # Convert numpy types to Python native types
                if np.isnan(value) or np.isinf(value):
                    preview_row[key] = 0
                else:
                    preview_row[key] = float(value) if isinstance(value, np.floating) else int(value)
            elif value is None or str(value).lower() == 'nan':
                preview_row[key] = ''
        
        return {
            "message": "Prediction complete",
            "status": "success",
            "needs_reorder": int(preview_row.get('needs_reorder', 0)),
            "preview": clean_for_json(preview_row)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.get("/get-temp-entries/")
async def get_temp_entries():
    """Get all temporary entries"""
    return {"temp_entries": uploaded_data['temp_entries']}

@app.delete("/remove-temp-entry/{index}")
async def remove_temp_entry(index: int):
    """Remove a specific temporary entry"""
    try:
        if 0 <= index < len(uploaded_data['temp_entries']):
            uploaded_data['temp_entries'].pop(index)
            return {"message": "Entry removed", "temp_entries": uploaded_data['temp_entries']}
        else:
            raise HTTPException(status_code=404, detail="Entry not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing entry: {e}")
        raise HTTPException(status_code=500, detail=f"Error removing entry: {str(e)}")

@app.delete("/clear-temp-entries/")
async def clear_temp_entries():
    """Clear all temporary entries"""
    uploaded_data['temp_entries'] = []
    return {"message": "All temporary entries cleared", "status": "success"}

@app.post("/save-temp-to-database/")
async def save_temp_to_database():
    """Save temporary entries to the database/CSV file"""
    try:
        if not uploaded_data['temp_entries']:
            raise HTTPException(status_code=400, detail="No temporary entries to save")
        
        all_results = []
        
        for i, entry in enumerate(uploaded_data['temp_entries']):
            try:
                if not isinstance(entry, dict):
                    raise ValueError(f"Invalid entry at index {i}: {entry}")
                
                final_df = create_single_entry_dataframe_enhanced(entry)
                if final_df.empty:
                    raise ValueError(f"Generated empty DataFrame from entry: {entry}")
                
                all_results.append(final_df)
                
            except Exception as e:
                logger.error(f"Error processing entry {i}: {e}")
                continue
        
        if not all_results:
            raise HTTPException(status_code=400, detail="No valid entries could be processed")
        
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        logger.info(f"Combined {len(all_results)} entries into DataFrame with {len(combined_df)} rows")
        
        # Format the output
        formatted_df = format_output_df(combined_df)
        
        try:
            existing_df = get_csv('python_layer')
            if not existing_df.empty:
                # Append new data
                formatted_df = pd.concat([existing_df, formatted_df], ignore_index=True)
                # Remove duplicates based on product_id and warehouse_id
                formatted_df = formatted_df.drop_duplicates(subset=['product_id', 'warehouse_id'], keep='last')
            
            # Write to CSV
            save_csvs(formatted_df, 'python_layer')
            logger.info(f"Successfully saved {len(all_results)} records to database")
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            # Still continue with other operations
        
        # Save updated datasets
        save_updated_datasets()
        
        # Clear temporary entries after successful processing
        entries_saved = len(uploaded_data['temp_entries'])
        uploaded_data['temp_entries'] = []
        
        # Create preview for response
        preview_df = formatted_df.tail(min(3, len(formatted_df))).copy()
        
        # Clean all data for JSON serialization
        preview_df = preview_df.replace([np.inf, -np.inf], 0)
        preview_df = preview_df.fillna(0)
        
        # Handle data types for JSON compliance
        for col in preview_df.columns:
            if pd.api.types.is_datetime64_any_dtype(preview_df[col]):
                preview_df[col] = preview_df[col].dt.strftime('%Y-%m-%d').fillna('')
            elif pd.api.types.is_numeric_dtype(preview_df[col]):
                preview_df[col] = pd.to_numeric(preview_df[col], errors='coerce').fillna(0)
                if preview_df[col].dtype == 'int64':
                    preview_df[col] = preview_df[col].astype(int)
                else:
                    preview_df[col] = preview_df[col].astype(float)
            else:
                preview_df[col] = preview_df[col].astype(str).replace('nan', '').replace('None', '')
        
        return {
            "message": f"Successfully saved {entries_saved} entries to database",
            "status": "success",
            "entries_saved": entries_saved,
            "saved_file": 'to db',
            "records_processed": len(formatted_df),
            "preview": clean_for_json(preview_df.to_dict(orient='records'))
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving to database: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error saving to database: {str(e)}")

@app.get("/download-results/")
async def download_results():
    """Download the processed results as CSV with proper formatting"""
    try:
        if uploaded_data.get('final_results') is None:
            raise HTTPException(status_code=404, detail="No processed results available. Please process data first.")
        
        # Create a clean copy of the results
        final_df = uploaded_data['final_results'].copy()
        
        # Ensure we have all required columns
        for col in OUTPUT_COLUMNS:
            if col not in final_df.columns:
                if col in ['supplier_id', 'supplier_name']:
                    final_df[col] = ''
                elif col in ['avg_order_size', 'orders_per_month', 'days_since_last_order', 'movement_score']:
                    final_df[col] = 0
                elif col == 'movement_category':
                    final_df[col] = 'NO_MOVEMENT'
                elif col == 'stock_status':
                    final_df[col] = 'UNKNOWN'
                else:
                    final_df[col] = 0
        
        # Select and order columns exactly as specified
        final_df = final_df[OUTPUT_COLUMNS]
        
        # Format date columns properly
        date_columns = ['last_updated', 'last_order_date']
        for col in date_columns:
            if col in final_df.columns:
                if pd.api.types.is_datetime64_any_dtype(final_df[col]):
                    # Format to M/D/YYYY to match expected output
                    try:
                        final_df[col] = final_df[col].dt.strftime('%-m/%-d/%Y')
                    except:
                        final_df[col] = final_df[col].dt.strftime('%m/%d/%Y')
                elif col == 'last_order_date':
                    # Handle cases where there's no last order date
                    mask = final_df['days_since_last_order'] == 9999
                    final_df.loc[mask, col] = ''
        
        # Convert to CSV string
        csv_string = final_df.to_csv(index=False)
        
        # Create the response with proper headers
        return Response(
            content=csv_string,
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    f"attachment; "
                    f"filename=inventory_results_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading results: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading results: {str(e)}")

@app.get("/get-stats/")
async def get_stats():
    """Get current statistics of the data"""
    try:
        stats = {}
        
        if uploaded_data.get('products_df') is not None and not uploaded_data['products_df'].empty:
            stats['total_products'] = len(uploaded_data['products_df'])
        
        if uploaded_data.get('inventory_df') is not None and not uploaded_data['inventory_df'].empty:
            stats['total_inventory_records'] = len(uploaded_data['inventory_df'])
        
        if uploaded_data.get('final_results') is not None and not uploaded_data['final_results'].empty:
            df = uploaded_data['final_results']
            stats['out_of_stock'] = len(df[df['stock_status'] == 'OUT_OF_STOCK'])
            stats['reorder_needed'] = len(df[df['stock_status'] == 'REORDER_NEEDED'])
            stats['low_stock'] = len(df[df['stock_status'] == 'LOW_STOCK'])
            stats['adequate_stock'] = len(df[df['stock_status'] == 'ADEQUATE'])
            
            if 'stock_value' in df.columns:
                total_value = df['stock_value'].sum()
                stats['total_stock_value'] = float(total_value) if pd.notna(total_value) else 0.0
            
            if 'needs_reorder' in df.columns:
                predicted_reorders = df['needs_reorder'].sum()
                stats['predicted_reorders'] = int(predicted_reorders) if pd.notna(predicted_reorders) else 0
        
        return stats
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"error": f"Error getting stats: {str(e)}"}

@app.get("/status/")
async def get_status():
    """Get current status of uploaded data"""
    try:
        status = {}
        for key, df in uploaded_data.items():
            if key == 'temp_entries':
                status[key] = {
                    "loaded": len(uploaded_data['temp_entries']) > 0,
                    "count": len(uploaded_data['temp_entries'])
                }
            elif df is not None and not df.empty:
                status[key] = {
                    "loaded": True,
                    "rows": len(df),
                    "columns": list(df.columns)
                }
            else:
                status[key] = {"loaded": False}
        
        return status
    
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {"error": f"Error getting status: {str(e)}"}

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

# Initialize CSV files with sample data if they don't exist
@app.on_event("startup")
async def startup_event():
    """Initialize sample data if no data exists in Azure storage"""
    try:
        # Check if basic CSV files exist, if not create sample data
        sample_products = pd.DataFrame([
            {'product_id': 'P001', 'product_name': 'Sample Product 1', 'category': 'Electronics', 'reorder_level': 10, 'unit_price': 25.99},
            {'product_id': 'P002', 'product_name': 'Sample Product 2', 'category': 'Furniture', 'reorder_level': 5, 'unit_price': 199.99}
        ])
        
        sample_warehouses = pd.DataFrame([
            {'warehouse_id': 'WH001', 'location_name': 'Main Warehouse'},
            {'warehouse_id': 'WH002', 'location_name': 'Secondary Warehouse'}
        ])
        
        sample_suppliers = pd.DataFrame([
            {'supplier_id': 'S001', 'supplier_name': 'Default Supplier 1', 'quality_score': 85.0, 'avg_lead_time_days': 7},
            {'supplier_id': 'S002', 'supplier_name': 'Default Supplier 2', 'quality_score': 80.0, 'avg_lead_time_days': 10}
        ])
        
        sample_inventory = pd.DataFrame([
            {'product_id': 'P001', 'warehouse_id': 'WH001', 'quantity_available': 25, 'last_updated': '2025-01-15'},
            {'product_id': 'P002', 'warehouse_id': 'WH001', 'quantity_available': 3, 'last_updated': '2025-01-15'}
        ])
        
        sample_orders = pd.DataFrame([
            {'order_id': 'O001', 'product_id': 'P001', 'supplier_id': 'S001', 'quantity_ordered': 50, 'order_date': '2025-01-10', 'delivery_date': '2025-01-17'},
            {'order_id': 'O002', 'product_id': 'P002', 'supplier_id': 'S002', 'quantity_ordered': 10, 'order_date': '2025-01-12', 'delivery_date': '2025-01-22'}
        ])
        
        # Try to save sample data if files don't exist
        for name, sample_df in [
            ('products', sample_products),
            ('warehouses', sample_warehouses), 
            ('suppliers', sample_suppliers),
            ('inventory', sample_inventory),
            ('orders', sample_orders)
        ]:
            try:
                existing_df = get_csv(name)
                if existing_df.empty:
                    save_csvs(sample_df, name)
                    logger.info(f"Created sample {name} data")
            except Exception as e:
                logger.warning(f"Could not create sample {name} data: {e}")
        
        # Reload data after potential sample creation
        initialize_defaults()
        load_existing_data()
        
        logger.info("Startup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")

if __name__ == "__main__":
    print("InvenTrack Pro running at http://localhost:8001")
    print("API Documentation available at http://localhost:8001/docs")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)