import os
import json
import sqlite3
import re

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw_topics")
DB_PATH = os.path.join(DATA_DIR, "placement.db")
BRANCH_MAP_FILE = "branch_mapping.json"

def load_branch_map():
    if os.path.exists(BRANCH_MAP_FILE):
        with open(BRANCH_MAP_FILE, "r") as f:
            return json.load(f)
    return {"roll_codes": {}, "email_codes": {}}

BRANCH_MAP = load_branch_map()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS students') # Old table
    c.execute('DROP TABLE IF EXISTS event_students') # New link table
    c.execute('DROP TABLE IF EXISTS events')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT,
            event_type TEXT,
            raw_filename TEXT,
            topic_url TEXT
        )
    ''')
    
    # Master Student Table (Unique Identity)
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_no TEXT UNIQUE,
            email TEXT,
            name TEXT,
            branch TEXT,
            year TEXT
        )
    ''')
    
    # Link Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS event_students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            event_id INTEGER,
            raw_line TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id),
            FOREIGN KEY(event_id) REFERENCES events(id)
        )
    ''')
    conn.commit()
    return conn

def extract_name_from_email(email):
    # sameern.wanjari.cd.phy21@iitbhu.ac.in -> Sameern Wanjari
    try:
        user_part = email.split("@")[0]
        # Remove branch/year part if present at end (e.g. .phy21)
        user_part = re.sub(r'\.[a-z]{2,4}\d{2}$', '', user_part)
        
        # Split by dots or underscores
        parts = re.split(r'[._]', user_part)
        
        # Filter out common junk suffixes matching 'cd', 'rs', 'id' if they appear
        # The user specifically mentioned "cd"
        blacklist = {'cd', 'id', 'rs', 'phe', 'che', 'mec'} 
        
        clean_parts = []
        for p in parts:
            if p.lower() in blacklist: 
                continue
            if len(p) > 1 and not p.isdigit():
                clean_parts.append(p.capitalize())
                
        return " ".join(clean_parts)
    except:
        return None

def parse_filename(filename):
    name = filename.replace(".json", "")
    parts = name.split("_", 1)
    if len(parts) > 1 and parts[0].isdigit():
        clean_name = parts[1]
    else:
        clean_name = name
        
    # Heuristics for Event Type
    clean_name_lower = clean_name.lower()
    event_type = "Unknown"
    company = clean_name
    
    is_intern = "intern" in clean_name_lower
    
    if "offer" in clean_name_lower:
        event_type = "Internship Offers" if is_intern else "FT Offers"
    elif "interview" in clean_name_lower:
        event_type = "Internship Interview Shortlist" if is_intern else "FT Interview Shortlist"
    elif "test" in clean_name_lower:
        event_type = "Internship Test Shortlist" if is_intern else "FT Test Shortlist"
    elif "gd" in clean_name_lower or "group discussion" in clean_name_lower:
         event_type = "Internship GD Shortlist" if is_intern else "FT GD Shortlist"
    
    # Specific fix for pre-placement
    if "pre-placement" in clean_name_lower or "ppo" in clean_name_lower:
         event_type = "Pre-Placement Offers"

    # Extract Company Name cleaner
    # Remove known suffixes
    suffixes = [
        "_FT_Offers", "_FT_Interview_Shortlist", "_FT_Test_Shortlist", "_Offers", " FT Offers", " FT Interview Shortlist",
        "_Internship_Offers", "_Internship_Interview_Shortlist", "_Internship_Test_Shortlist", " Internship Offers",
        "_Internship_Offer", " Internship Offer", "_FT_Offer", " FT Offer"
    ]
    # Also generic removal of event type words if suffix didn't catch it
    # But be careful not to remove part of company name
    
    for s in suffixes:
        if s in company:
             company = company.replace(s, "")
        elif s.lower() in company.lower(): # Case insensitive try
             pattern = re.compile(re.escape(s), re.IGNORECASE)
             company = pattern.sub("", company)

    company = company.replace("_", " ").strip()
    return company, event_type

def get_branch_year(roll_no, email):
    year = None
    branch = None
    
    # Logic 1: From Roll No (21174028 -> Year 21, Branch Code 17)
    if roll_no and len(roll_no) == 8:
        year_code = roll_no[:2]
        branch_code = roll_no[2:4]
        year = "20" + year_code
        branch = BRANCH_MAP["roll_codes"].get(branch_code, "Unknown")
        
    # Logic 2: From Email if Roll failed or as backup
    if not branch or branch == "Unknown":
        if email:
            # Try to find branch code before '@' (e.g. .phy21@)
            match = re.search(r'\.([a-z]{2,4})(\d{2})@', email)
            if match:
                b_code = match.group(1)
                y_code = match.group(2)
                if not year: year = "20" + y_code
                branch = BRANCH_MAP["email_codes"].get(b_code, b_code.upper())
                
    return branch, year

def extract_students(text, default_event_type):
    students = []
    lines = text.split('\n')
    
    roll_pattern = re.compile(r'\b(\d{8})\b')
    email_pattern = re.compile(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        roll_match = roll_pattern.search(line)
        email_match = email_pattern.search(line)
        
        roll = roll_match.group(1) if roll_match else None
        email = email_match.group(1) if email_match else None
        
        if not roll and not email:
            continue
            
        # Name extraction attempt
        temp_line = line
        if roll: temp_line = temp_line.replace(roll, "")
        if email: temp_line = temp_line.replace(email, "")
        
        # Clean up delimiters
        temp_line = re.sub(r'[-–,:\t]', ' ', temp_line).strip()
        
        # Check if remaining text looks like a name
        name = temp_line
        if len(name) < 3: # If only 1-2 chars remain, assume no name in text
             name = None
        
        # If no name found in text, extract from email
        if not name and email:
            name = extract_name_from_email(email)
            
        if not name:
            name = "Unknown Student"
            
        branch, year = get_branch_year(roll, email)
        
        students.append({
            "name": name,
            "roll_no": roll,
            "email": email,
            "branch": branch,
            "year": year,
            "raw_line": line
        })
    
    return students

def process_files():
    conn = init_db()
    c = conn.cursor()
    
    files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".json")])
    print(f"Processing {len(files)} files...")
    
    student_cache = {} # Map roll_no -> student_id
    
    for fname in files:
        fpath = os.path.join(RAW_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        
        company, event_type = parse_filename(fname)
        topic_url = data.get("url", "")
        raw_text = data.get("raw_text", "")
        
        # 1. Insert Event
        c.execute("INSERT INTO events (company_name, event_type, raw_filename, topic_url) VALUES (?, ?, ?, ?)",
                  (company, event_type, fname, topic_url))
        event_id = c.lastrowid
        
        # 2. Extract Students
        extracted = extract_students(raw_text, event_type)
        
        for s in extracted:
            roll = s['roll_no']
            name = s['name']
            email = s['email']
            branch = s['branch']
            year = s['year']
            
            if not roll:
                # If no roll, skip for now. User said "maintain properly" which usually implies identifying by Roll.
                # Or create a fake roll? No, let's skip unidentifiable records to keep DB clean as requested.
                continue
                
            # Upsert Student
            # Check if exists in cache or DB
            if roll not in student_cache:
                c.execute("SELECT id, name, email FROM students WHERE roll_no = ?", (roll,))
                row = c.fetchone()
                if row:
                    student_cache[roll] = row[0]
                    # Optional: Update Name/Email if better one found?
                    # Keep existing for stability, or update if current name is "Unknown"
                    if row[1] == "Unknown Student" and name != "Unknown Student":
                         c.execute("UPDATE students SET name = ? WHERE id = ?", (name, row[0]))
                else:
                    # Insert new
                    c.execute("INSERT INTO students (roll_no, email, name, branch, year) VALUES (?, ?, ?, ?, ?)",
                              (roll, email, name, branch, year))
                    student_cache[roll] = c.lastrowid
            
            student_id = student_cache[roll]
            
            # Link to Event
            c.execute("INSERT INTO event_students (student_id, event_id, raw_line) VALUES (?, ?, ?)",
                      (student_id, event_id, s['raw_line']))

    conn.commit()
    conn.close()
    print("Database rebuilt with Normalized Schema.")

if __name__ == "__main__":
    process_files()
