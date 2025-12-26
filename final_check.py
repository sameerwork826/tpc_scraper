import sqlite3
import os

DB_PATH = "data/placement.db"

def check():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = """
    SELECT e.id, e.company_name, e.event_type, COUNT(es.student_id) 
    FROM events e 
    LEFT JOIN event_students es ON e.id = es.event_id 
    WHERE e.company_name LIKE '%Harness%' 
    GROUP BY e.id
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    
    print(f"Checking Harness data in {DB_PATH}:")
    for row in rows:
        print(f"Event {row[0]}: {row[1]} | {row[2]} | Count: {row[3]}")
    
    conn.close()

if __name__ == "__main__":
    check()
