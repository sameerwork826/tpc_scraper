import os
import json
import sqlite3
import re

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw_topics")
DB_PATH = os.path.join(DATA_DIR, "placement.db")
BRANCH_MAP_FILE = "branch_mapping.json"

COMPANY_MAPPINGS = {
    "Commonwealth Bank": "Commonwealth Bank of Australia",
    "Commmon Wealth Bank": "Commonwealth Bank of Australia",
    "Common Wealth Bank": "Commonwealth Bank of Australia",
    "Samsung": "Samsung",
    "SRID": "Samsung",
    "SRIN": "Samsung",
    "Sca Technologies": "SCA Technologies",
    "L&T": "L&T",
    "Larsen": "L&T",
    "Nation with Namo": "Nation with Namo",
    "DE Shaw": "D. E. Shaw",
    "Mckinsey": "McKinsey",
    "ICICI": "ICICI",
    "MyKaarma": "myKaarma",
    "Willings": "Willings",
    "Zomato": "Zomato",
    "Flipkart": "Flipkart",
    "Applied Materials": "Applied Materials",
    "Hilabs": "HiLabs",
}

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
    
    # Create tables if not exist (including company_ctc/visits which might be persistent)
    c.execute('''
        CREATE TABLE IF NOT EXISTS company_ctc (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT UNIQUE,
            ctc_lpa REAL,
            ctc_inr REAL,
            inhand_lpa REAL,
            inhand_inr REAL
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS company_visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT,
            role TEXT,
            placement_year TEXT,
            ctc_lpa REAL,
            ctc_inr REAL,
            location TEXT,
            jd_link TEXT,
            eligibility_cgpa REAL
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
            year TEXT,
            cpi REAL
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
    # Create Denormalized VIEW for Analysis
    try:
        c.execute("DROP VIEW IF EXISTS student_placement_summary")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("DROP TABLE IF EXISTS student_placement_summary")
    except sqlite3.OperationalError:
        pass
        
    c.execute("""
        CREATE VIEW student_placement_summary AS
        SELECT s.name, s.roll_no, s.branch, s.year, s.cpi, s.email,
               e.company_name, e.event_type, e.topic_url, e.raw_filename,
               cc.ctc_lpa, cc.ctc_inr, cc.inhand_lpa, cc.inhand_inr
        FROM students s
        JOIN event_students es ON s.id = es.student_id
        JOIN events e ON es.event_id = e.id
        LEFT JOIN company_ctc cc ON e.company_name = cc.company_name
    """)

    conn.commit()
    return conn

def extract_name_from_email(email):
    # sameern.wanjari.cd.phy21@iitbhu.ac.in -> Sameer Wanjari
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
        
        # Fix: ensure we don't return an empty string if all parts were blacklisted (unlikely but safe)
        if not clean_parts:
             return user_part
             
        return " ".join(clean_parts)
    except:
        return None

def normalize_company(name):
    clean = name.strip()
    # Check exact map
    for k, v in COMPANY_MAPPINGS.items():
        if k.lower() in clean.lower():
            return v
    return clean

def get_event_metadata(filename, topic_title):
    # Prefer topic_title if available and not empty, else filename
    source_text = topic_title if topic_title and len(topic_title) > 5 else filename
    source_text = source_text.replace(".json", "").replace("_", " ")
    
    # 1. Strip common prefixes like [Updated], (UPDATED)
    source_text = re.sub(r'^(\[| \()?UPDATED(\]| \))?\s*', '', source_text, flags=re.IGNORECASE).strip()
    source_text = re.sub(r'^\d{3}[_\s]+', '', source_text).strip() # Remove 001_ or 220_ prefixes
    
    clean_name_lower = source_text.lower()
    event_type = "Unknown"
    company = source_text
    
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
    # Remove known suffixes - expanded list
    suffixes = [
        r"\s+FT\s+Offers.*", r"\s+FT\s+Interview\s+Shortlist.*", r"\s+FT\s+Test\s+Shortlist.*",
        r"\s+Internship\s+Offers.*", r"\s+Internship\s+Interview\s+Shortlist.*", r"\s+Internship\s+Test\s+Shortlist.*",
        r"\s+Internship\s+Offer.*", r"\s+FT\s+Offer.*",
        r"\s+Interview\s+Shortlist.*", r"\s+Test\s+Shortlist.*", r"\s+GD\s+Shortlist.*",
        r"\s+Pre-Placement\s+Offers.*", r"\s+and\s+Acceptance.*", r"\s+&\s+Waitlist.*",
        r"\s+Placement\s+Test.*", r"\s+Placement\s+Interview.*", r"\s+Shortlist.*",
        r"\s+Offers.*", r"\s+Waitlist.*"
    ]
    
    # Use regex for cleaner suffix removal
    for s in suffixes:
        company = re.sub(s, "", company, flags=re.IGNORECASE).strip()

    company = company.replace("_", " ").strip()
    normalized_company = normalize_company(company)
    
    return normalized_company, event_type

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
        
        if not roll:
            continue
            
        # Name extraction attempt
        temp_line = line
        if roll: temp_line = temp_line.replace(roll, "")
        if email: temp_line = temp_line.replace(email, "")
        
        # Clean up delimiters
        temp_line = re.sub(r'[-–,:\t]', ' ', temp_line).strip()
        
        # Check if remaining text looks like a name
        name = temp_line
        # Relaxed check: Allow 2 chars (e.g. "Al", "Bo")
        if len(name) < 2: 
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
    
    # Load CPI mapping
    roll_to_cpi = {}
    if os.path.exists('alot_LM (2).csv'):
        import pandas as pd
        cpi_df = pd.read_csv('alot_LM (2).csv', skipinitialspace=True)
        # Convert rollno to string and strip space
        cpi_df['rollno'] = cpi_df['rollno'].astype(str).str.strip()
        roll_to_cpi = dict(zip(cpi_df['rollno'], cpi_df['cpi']))
    
    student_cache = {} # Map roll_no -> student_id
    
    for fname in files:
        fpath = os.path.join(RAW_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        
        topic_title = data.get("topic_title", "")
        raw_text = data.get("raw_text", "")
        topic_url = data.get("url", "")
        
        base_company, base_event_type = get_event_metadata(fname, topic_title)
        
        # SUB-TOPIC SPLITTING LOGIC
        # Detect sections like "Waitlisted:", "Extended Shortlist:", etc.
        # We will split text into chunks.
        
        # Markers to look for. Case insensitive regex.
        markers = {
            "waitlist": "Waitlist",
            "extended": "Extended",
            "interview shortlist": "Interview Shortlist", # Explicit reset
            "gd shortlist": "GD Shortlist"
        }
        
        # Simple finite state approach?
        # Or just regex split?
        # Let's try to identify if the text has multiple sections.
        
        chunks = [] # List of (subtitle, text_content)
        
        # Split by double newline to get paragraphs, check for headers
        # This is heuristics based.
        
        # Better: Search for headers and slice text.
        # But headers might be "Please find waitlist below:" or just "Waitlist:"
        
        # Let's do a simple line-by-line scan to assign "current state"
        lines = raw_text.split('\n')
        current_suffix = "" # Default (Main Event)
        current_chunk_lines = []
        
        for line in lines:
            lower_line = line.lower().strip()
            
            # Check for header-like lines (short, contains keywords)
            # e.g. "Waitlisted:" or "Waitlist Students"
            if len(lower_line) < 50: 
                new_suffix = None
                if "waitlist" in lower_line:
                    new_suffix = " - Waitlist"
                elif "extended" in lower_line:
                    new_suffix = " - Extended"
                elif "shortlist" in lower_line and "interview" in lower_line:
                    new_suffix = "" # Reset to main
                
                if new_suffix is not None:
                    # Flush current chunk
                    if current_chunk_lines:
                        chunks.append((current_suffix, "\n".join(current_chunk_lines)))
                    current_suffix = new_suffix
                    current_chunk_lines = [] # Don't include the header line itself in the text? 
                    # If we exclude, we might lose data if it's on same line. 
                    # But usually headers are separate.
                    continue

            current_chunk_lines.append(line)
            
        # Flush last chunk
        if current_chunk_lines:
            chunks.append((current_suffix, "\n".join(current_chunk_lines)))
            
        
        # Process Chunks
        for suffix, text_content in chunks:
            # If chunk is empty or purely whitespace, skip
            if not text_content.strip():
                continue
                
            # Determine specific event type for this chunk
            specific_event_type = base_event_type + suffix
            
            # Extract Students
            extracted = extract_students(text_content, specific_event_type)
            
            if not extracted:
                continue

            # 1. Insert Event
            # If multiple chunks, we create multiple events for the same file.
            # This is fine, distinct by ID.
            c.execute("INSERT INTO events (company_name, event_type, raw_filename, topic_url) VALUES (?, ?, ?, ?)",
                      (base_company, specific_event_type, fname, topic_url))
            event_id = c.lastrowid
            
            for s in extracted:
                roll = s['roll_no']
                name = s['name']
                email = s['email']
                branch = s['branch']
                year = s['year']
                
                if not roll:
                    continue
                    
                # Upsert Student
                student_id = None
                
                # Try to find by roll
                c.execute("SELECT id, name FROM students WHERE roll_no = ?", (roll,))
                row = c.fetchone()
                if row:
                    student_id = row[0]
                    # Update name if previously "Unknown"
                    if row[1] == "Unknown Student" and name != "Unknown Student":
                         c.execute("UPDATE students SET name = ? WHERE id = ?", (name, student_id))
                    # Update email if missing
                    if email:
                        c.execute("UPDATE students SET email = ? WHERE id = ? AND email IS NULL", (email, student_id))
                
                # Finally Insert if still not found
                if not student_id:
                    # Get CPI from mapping if available
                    student_cpi = roll_to_cpi.get(str(roll))
                    c.execute("INSERT INTO students (roll_no, email, name, branch, year, cpi) VALUES (?, ?, ?, ?, ?, ?)",
                              (roll, email, name, branch, year, student_cpi))
                    student_id = c.lastrowid
                    if roll: student_cache[roll] = student_id
                
                # Link to Event
                c.execute("INSERT INTO event_students (student_id, event_id, raw_line) VALUES (?, ?, ?)",
                          (student_id, event_id, s['raw_line']))

    conn.commit()
    conn.close()
    print("Database rebuilt with Normalized Schema and Sub-Topics.")

if __name__ == "__main__":
    process_files()
