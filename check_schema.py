import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('data/placement.db')
    cursor = conn.cursor()
    
    # Get all tables and views
    cursor.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')")
    tables = cursor.fetchall()
    print("Tables and Views:")
    for t in tables:
        print(f"- {t[0]} ({t[1]})")
        
    print("\n--- Columns in student_placement_summary ---")
    try:
        # Check columns of the summary view if it exists
        df = pd.read_sql("SELECT * FROM student_placement_summary LIMIT 1", conn)
        print(df.columns.tolist())
    except Exception as e:
        print(f"Could not read student_placement_summary: {e}")

    conn.close()
except Exception as e:
    print(f"Database error: {e}")
