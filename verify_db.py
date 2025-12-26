
import sqlite3
import pandas as pd

with open("verification_result.txt", "w") as f:
    conn = sqlite3.connect('data/placement.db')

    # 1. Check Commonwealth Bank Normalization
    f.write("--- Checking Company Normalization ---\n")
    companies = pd.read_sql("SELECT DISTINCT company_name FROM events WHERE company_name LIKE '%Common%' OR company_name LIKE '%Wealth%'", conn)
    f.write(companies.to_string())
    f.write("\n\n")

    # 2. Check Sub-Topic splitting (Waitlists)
    f.write("--- Checking Sub-Topics (Waitlist) ---\n")
    waitlists = pd.read_sql("SELECT event_type FROM events WHERE event_type LIKE '%Waitlist%' LIMIT 10", conn)
    f.write(waitlists.to_string())
    f.write("\n\n")

    # 3. Check APT Portfolio specifically for Title usage
    f.write("--- Checking APT Portfolio Title ---\n")
    apt = pd.read_sql("SELECT DISTINCT event_type FROM events WHERE company_name LIKE '%APT Portfolio%'", conn)
    f.write(apt.to_string())
    f.write("\n")

    conn.close()
