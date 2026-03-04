# ============================================================
# STEP 1 — Import Libraries
# ============================================================
import pandas as pd
import numpy as np
import ast
import os

print("Libraries loaded successfully!")

# ============================================================
# STEP 2 — Load the 3 Datasets
# ============================================================
print("\nLoading datasets...")

interactions = pd.read_csv("raw_data/interactions.csv")
customers    = pd.read_csv("raw_data/customers_with_income.csv")
products     = pd.read_csv("raw_data/products.csv")

print(f"Interactions : {interactions.shape[0]} rows, {interactions.shape[1]} columns")
print(f"Customers    : {customers.shape[0]} rows, {customers.shape[1]} columns")
print(f"Products     : {products.shape[0]} rows, {products.shape[1]} columns")

# ============================================================
# STEP 3 — Clean Each Dataset
# ============================================================
print("\nCleaning datasets...")

# --- Clean interactions ---
interactions.drop_duplicates(inplace=True)
interactions.rename(columns={"productEvent_Type": "event_type"}, inplace=True)
interactions["Timestamp"] = pd.to_datetime(interactions["Timestamp"])
interactions["hour"]        = interactions["Timestamp"].dt.hour
interactions["day_of_week"] = interactions["Timestamp"].dt.dayofweek
interactions["month"]       = interactions["Timestamp"].dt.month
print("  Interactions cleaned!")

# --- Clean customers ---
customers.drop(columns=["Unnamed: 10"], inplace=True, errors="ignore")
customers.drop_duplicates(inplace=True)

def parse_list(val):
    try:
        return ast.literal_eval(val)
    except:
        return []

customers["Browsing_History"]  = customers["Browsing_History"].apply(parse_list)
customers["Purchase_History"]  = customers["Purchase_History"].apply(parse_list)
customers["browse_count"]   = customers["Browsing_History"].apply(len)
customers["purchase_count"] = customers["Purchase_History"].apply(len)
print("  Customers cleaned!")

# --- Clean products ---
products.drop(columns=["Unnamed: 13", "Unnamed: 14"], inplace=True, errors="ignore")
products.drop_duplicates(inplace=True)
products["Similar_Product_List"] = products["Similar_Product_List"].apply(parse_list)
products["similar_product_count"] = products["Similar_Product_List"].apply(len)
print("  Products cleaned!")

# ============================================================
# STEP 4 — Merge the 3 Datasets
# ============================================================
print("\nMerging datasets...")

df = pd.merge(interactions, products, on="Product_ID", how="left")
print(f"  After merging interactions + products : {df.shape[0]} rows")

df = pd.merge(df, customers, on="Customer_ID", how="left")
print(f"  After merging with customers          : {df.shape[0]} rows")

# ============================================================
# STEP 5 — Add New Derived Features
# ============================================================
print("\nAdding derived features...")

# 1. already_purchased flag
def already_purchased(row):
    try:
        return row["Category_x"] in row["Purchase_History"]
    except:
        return False

df["already_purchased"] = df.apply(already_purchased, axis=1)

# 2. interaction score
event_weights = {
    "view":        1,
    "add_to_cart": 3,
    "purchase":    5,
    "review":      4
}
df["interaction_score"] = df["event_type"].map(event_weights)

# 3. price segment
def price_segment(price):
    if price <= 500:
        return "budget"
    elif price <= 1500:
        return "mid"
    else:
        return "premium"

df["price_segment"] = df["Price"].apply(price_segment)

# 4. engagement level
def engagement_level(seconds):
    if seconds < 30:
        return "low"
    elif seconds < 120:
        return "medium"
    else:
        return "high"

df["engagement_level"] = df["Time_Spent_Seconds"].apply(engagement_level)
print("  Derived features added!")

# ============================================================
# STEP 6 — Final Cleanup
# ============================================================
print("\nFinal cleanup...")

df.drop(columns=["Browsing_History", "Purchase_History",
                  "Similar_Product_List"], inplace=True, errors="ignore")
df.drop(columns=["Category_y"], inplace=True, errors="ignore")
df.rename(columns={"Category_x": "Category"}, inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"  Final dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"  Columns: {df.columns.tolist()}")

# ============================================================
# STEP 7 — Save the Final Dataset
# ============================================================
print("\nSaving final dataset...")

os.makedirs("output", exist_ok=True)
df.to_csv("output/ecom_recommendation_dataset.csv", index=False)

print("\nDone! File saved to output/ecom_recommendation_dataset.csv")
print("\nFinal Dataset Preview:")
print(df.head(3))