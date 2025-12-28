import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('data/placement.db')
    
    # Check count
    count = pd.read_sql("SELECT COUNT(*) as count FROM company_visits", conn)
    print(f"Total rows in company_visits: {count['count'].iloc[0]}")
    
    # Check years
    years = pd.read_sql("SELECT DISTINCT placement_year FROM company_visits", conn)
    print(f"Available years: {years['placement_year'].tolist()}")
    
    # Check specific query used in app
    query = "SELECT DISTINCT company_name FROM company_visits WHERE placement_year IN ('2021', '2022', '2021-22', '2022-23')"
    companies = pd.read_sql(query, conn)
    print(f"Companies found with app query: {len(companies)}")
    if not companies.empty:
        print(f"Sample companies: {companies['company_name'].head().tolist()}")
        
    conn.close()

except Exception as e:
    print(f"Error: {e}")
