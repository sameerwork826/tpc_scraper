
import pandas as pd
import json
import re

def test_load_data():
    try:
        df = pd.read_csv('alot_LM (2).csv', skipinitialspace=True)
        print(f"Columns: {df.columns.tolist()}")
        
        with open('branch_mapping.json', 'r') as f:
            mapping = json.load(f)
        email_codes = mapping['email_codes']
        
        def extract_branch(email):
            user_part = str(email).split('@')[0]
            match = re.search(r'\.([a-z]{2,4})(\d{2})$', user_part)
            if match:
                b_code = match.group(1).lower()
                return email_codes.get(b_code, b_code.upper())
            return 'Unknown'

        df['branch'] = df['email'].apply(extract_branch)
        
        print(f"Sample branch counts:\n{df['branch'].value_counts().head()}")
        print("Test PASSED")
    except Exception as e:
        print(f"Test FAILED: {e}")

if __name__ == '__main__':
    test_load_data()
