import streamlit as st
import sqlite3
from google import genai
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Configure API
# Database Configuration
DB_PATH = "data/placement.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn

def run_query(query, params=None):
    conn = get_db_connection()
    try:
        if params:
            df = pd.read_sql_query(query, conn, params=params)
        else:
            df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        conn.close()
        return f"Error: {e}"

# ... (run_query ends)

def generate_sql(question):
    # Schema Definition for the LLM
    schema = """
    Table: events
    Columns: id (INTEGER), company_name (TEXT), event_type (TEXT), raw_filename (TEXT), topic_url (TEXT)
    
    Table: students
    Columns: id (INTEGER), roll_no (TEXT), email (TEXT), name (TEXT), branch (TEXT), year (TEXT)
    
    Table: event_students
    Columns: id (INTEGER), student_id (INTEGER), event_id (INTEGER), raw_line (TEXT)
    Foreign Keys: student_id -> students.id, event_id -> events.id
    """
    
    prompt = f"""
    You are a SQL Expert. Convert the following natural language question into a SQL query for a SQLite database.
    
    Database Schema:
    {schema}
    
    CRITICAL RULES:
    1. Return ONLY the SQL query. No markdown, no explanation.
    2. **Joins are Usage**: 
       To find a student's events: `JOIN event_students es ON s.id = es.student_id JOIN events e ON es.event_id = e.id`
    3. **ROBUST NAME MATCHING (IMPORTANT)**: 
       - Users might provide only part of a name (e.g., "Sameer Wanjari" for "Sameer Nandesh Wanjari").
       - NEVER use `name LIKE '%First Last%'`.
       - ALWAYS split the name into parts and match each part separately using AND.
       - Example: For "Sameer Wanjari", use: `s.name LIKE '%Sameer%' AND s.name LIKE '%Wanjari%'`.
    4. Case Insensitive: `LIKE` in SQLite is case-insensitive for ASCII, but ensure logic holds.
    5. "Placed" = e.event_type contains 'Offer' or 'PPO' or 'Pre-Placement'.
    6. "Interview Shortlist" = e.event_type contains 'Interview'.
    7. "Test Shortlist" = e.event_type contains 'Test'.
    8. **Branches**: 'branch' column in `students` table contains values like 'CSE', 'Physics'.
    9. **Counts vs Lists**: 
       - If asked "How many" ONLY, use `COUNT(DISTINCT s.roll_no)`.
       - If asked "How many" AND "Names/Who/List", use `SELECT DISTINCT s.name, e.company_name...`.
    10. Select columns: `students.name`, `students.roll_no`, `students.branch`, `events.company_name`, `events.event_type`.
    
    Question: {question}
    SQL:
    """
    
    # Needs client instance - ensure client is initialized before this if using global, 
    # OR pass client. But client is initialized later in script. 
    # Better to move client init UP or lazily use os.getenv
    
    # Wait, client is initialized at line 91. 
    # If I define this function here (line 32), it captures 'client' from global scope?
    # No, it looks up 'client' when CALLED.
    # It is called inside the loop at line 120.
    # At line 120, 'client' (line 91) IS defined. So this is safe.
    
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=prompt
    )
    sql = response.text.replace("```sql", "").replace("```", "").strip()
    return sql

def generate_natural_answer(question, sql, df):
    # safe-guard for large results
    if len(df) > 50:
        data_context = df.head(50).to_markdown(index=False) + f"\n...(and {len(df)-50} more rows)"
    else:
        data_context = df.to_markdown(index=False)

    prompt = f"""
    You are a helpful assistant for the IIT BHU Placement Cell.
    
    User Question: {question}
    Executed SQL: {sql}
    Result Data:
    {data_context}
    
    Task: Answer the user's question naturally based ONLY on the result data.
    
    SPECIAL FORMAT FOR "ANALYSIS" REQUESTS:
    If the user asks for an "analysis" or "overview" of a student, you MUST follow this format:
    1. **Interview Shortlist Summary**: "Lastname was shortlisted for X companies for Interviews." (Count unique companies from events where event_type contains 'Interview').
       - IGNORE 'Test Shortlists' for this count unless explicitly asked.
    2. **Interview Shortlist Details**: List the companies (e.g., "• Company A \n • Company B").
    3. **Offers**: Clearly state any 'FT Offers' or 'Internship Offers' received.
    
    General Rules:
    - If the result is a specific status, state it clearly.
    - If the dataframe is empty, say "I couldn't find any records matching that."
    - Use formatting like bold or bullet points for readability.
    - Do NOT mention "SQL" or "dataframe" in the final answer.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=prompt
    )
    return response.text

# Streamlit UI
st.set_page_config(page_title="Placement Query Bot", page_icon="🎓", layout="wide")

# Sidebar Configuration
with st.sidebar:
    st.title("🎓 TPC Bot")
    st.markdown("**Created by: Sameer Wanjari**")
    st.markdown("---")
    
    # API Key Handling
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.warning("⚠️ Gemini API Key Missing")
        st.caption("AI features (Chat) require a key. Explorer works without it.")
        user_api_key = st.text_input("Enter Gemini API Key", type="password", help="Get it from: https://aistudio.google.com/app/apikey")
        if user_api_key:
            os.environ["GOOGLE_API_KEY"] = user_api_key
            st.success("Key set!")
            st.rerun()
    else:
        st.success("✅ API Key Active")
        if st.toggle("Change Gemini API Key"):
            new_key = st.text_input("New Gemini API Key", type="password")
            if new_key:
                os.environ["GOOGLE_API_KEY"] = new_key
                st.rerun()
    
    st.markdown("---")

    # Database Stats
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT roll_no) FROM students")
    total_students = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT company_name) FROM events")
    total_companies = c.fetchone()[0]
    conn.close()

    st.header("📊 Stats")
    st.metric("Total Students", total_students)
    st.metric("Total Companies", total_companies)
    st.markdown("---")
    
    # Data Refresh
    st.header("⚙️ Data")
    if st.button("🔄 Refresh DB"):
        with st.spinner("Processing..."):
            try:
                import process_data
                process_data.process_files()
                st.success("Done! Reloading...")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

# Initialize Client
api_key = os.getenv("GOOGLE_API_KEY")
client = None
if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Gemini Client: {e}")

# Main Interface Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "🔍 Student Explorer", "🏢 Company Explorer"])

# --- TAB 1: CHAT ---
with tab1:
    st.header("Ask anything about placements")
    st.markdown("Examples: *'Analysis of Sameer Wanjari'*, *'How many Physics students got offers?'*")

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not client:
        st.warning("⚠️ **Gemini API Key is missing!**")
        st.info("You can still use the **Student Explorer** tab to browse data manually.")
        st.markdown("To enable AI Chat:")
        st.markdown("1. Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).")
        st.markdown("2. Enter it in the sidebar.")
    
    # Only show chat input if client is available
    elif prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # 1. Generate SQL
                sql_query = generate_sql(prompt)
                
                # 2. Execute SQL
                result = run_query(sql_query)
                
                if isinstance(result, pd.DataFrame):
                    # 3. Generate Natural Language Answer
                    nl_response = generate_natural_answer(prompt, sql_query, result)
                    message_placeholder.markdown(nl_response)
                    
                    # Save to history
                    st.session_state.messages.append({"role": "assistant", "content": nl_response})
                    
                    with st.expander("View Technical Details (SQL & Data)"):
                        st.code(sql_query, language="sql")
                        st.dataframe(result)
                else:
                    message_placeholder.error(result)
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {result}"})
                    
            except Exception as e:
                message_placeholder.error(f"An error occurred: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {e}"})

# --- TAB 2: EXPLORER ---
with tab2:
    st.header("Student Profile Explorer")
    
    conn = get_db_connection()
    
    # 1. Filters
    col1, col2 = st.columns(2)
    with col1:
        branches = pd.read_sql("SELECT DISTINCT branch FROM students WHERE branch IS NOT NULL ORDER BY branch", conn)['branch'].tolist()
        selected_branch = st.selectbox("Filter by Branch", ["All"] + branches)
    
    with col2:
        years = pd.read_sql("SELECT DISTINCT year FROM students WHERE year IS NOT NULL ORDER BY year", conn)['year'].tolist()
        selected_year = st.selectbox("Filter by Year", ["All"] + years)
    
    # 2. Student Selector
    query = "SELECT DISTINCT name, roll_no FROM students WHERE 1=1"
    params = []
    if selected_branch != "All":
        query += " AND branch = ?"
        params.append(selected_branch)
    if selected_year != "All":
        query += " AND year = ?"
        params.append(selected_year)
    
    query += " ORDER BY name"
    
    students_df = pd.read_sql(query, conn, params=params)
    
    if students_df.empty:
        st.warning("No students found with filters.")
    else:
        # Create display label "Name (Roll)"
        student_options = [f"{row['name']} ({row['roll_no']})" for _, row in students_df.iterrows()]
        selected_student_str = st.selectbox("Select Student", student_options, index=None, placeholder="Type to search...")
        
        if selected_student_str:
            # Extract Roll
            roll_no = selected_student_str.split("(")[-1].strip(")")
            
            st.markdown("---")
            st.subheader(f"Profile: {selected_student_str}")
            
            # Fetch History
            history_query = """
                SELECT e.company_name, e.event_type, e.topic_url
                FROM event_students es
                JOIN students s ON es.student_id = s.id
                JOIN events e ON es.event_id = e.id
                WHERE s.roll_no = ?
                ORDER BY e.event_type, e.company_name
            """
            history = pd.read_sql(history_query, conn, params=[roll_no])
            
            if not history.empty:
                # Summary Metrics
                offers = history[history['event_type'].str.contains('Offer', case=False)]
                interviews = history[history['event_type'].str.contains('Interview', case=False)]
                tests = history[history['event_type'].str.contains('Test', case=False)]
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Offers", len(offers))
                m2.metric("Interviews", len(interviews))
                m3.metric("Tests", len(tests))
                
                # Detailed Timeline
                st.write("#### 📅 Event Timeline")
                
                # Group by type for cleaner view
                for etype in history['event_type'].unique():
                    with st.expander(f"{etype} ({len(history[history['event_type']==etype])})", expanded=True):
                        subset = history[history['event_type'] == etype]
                        for _, row in subset.iterrows():
                            # Markdown list with link
                            if row['topic_url']:
                                st.markdown(f"- [{row['company_name']}]({row['topic_url']})")
                            else:
                                st.markdown(f"- {row['company_name']}")
            else:
                st.info("No recorded events for this student.")
    
    conn.close()

# --- TAB 3: COMPANY EXPLORER ---
with tab3:
    st.header("🏢 Company Explorer")
    conn = get_db_connection()

    # 1. Company Selector
    companies = pd.read_sql("SELECT DISTINCT company_name FROM events ORDER BY company_name", conn)['company_name'].tolist()
    if not companies:
        st.warning("No companies found.")
    else:
        selected_company = st.selectbox("Select Company", companies, index=None, placeholder="Choose a company...")

        if selected_company:
            st.markdown("---")
            st.subheader(f"Results for: {selected_company}")

            # Fetch relevant events and students
            # We need to distinguish between FT and Intern
            
            # Get IDs of events for this company
            events_df = pd.read_sql("SELECT id, event_type, topic_url FROM events WHERE company_name = ?", conn, params=[selected_company])
            
            if events_df.empty:
                st.info("No events found for this company.")
            else:
                # Separate Full-Time and Internship Events
                ft_events_df = events_df[~events_df['event_type'].str.contains("Internship|Intern", case=False, regex=True)]
                intern_events_df = events_df[events_df['event_type'].str.contains("Internship|Intern", case=False, regex=True)]
                
                def display_events_table(events_subset, section_title):
                    if events_subset.empty:
                        return
                        
                    st.subheader(section_title)
                    # Get unique event types in this subset
                    unique_types = events_subset['event_type'].unique()
                    
                    for etype in sorted(unique_types):
                        # Filter events for this specific type
                        matched_ids = events_subset[events_subset['event_type'] == etype]['id'].tolist()
                        
                        # Query students
                        placeholders = ','.join(['?'] * len(matched_ids))
                        q = f"""
                            SELECT DISTINCT s.name, s.roll_no, s.branch, s.year, e.event_type, e.topic_url
                            FROM event_students es
                            JOIN students s ON es.student_id = s.id
                            JOIN events e ON es.event_id = e.id
                            WHERE es.event_id IN ({placeholders})
                            ORDER BY s.name
                        """
                        # Get URL for this type (take first one, usually identical for same event type/company combo if merged)
                        # Actually event_type might differ if we have multiple files with same type? 
                        # But grouping by event_type is safer.
                        
                        results = pd.read_sql(q, conn, params=matched_ids)
                        
                        if not results.empty:
                            with st.expander(f"{etype} ({len(results)})", expanded=False):
                                # Show Source Link if available
                                links = events_subset[events_subset['event_type'] == etype]['topic_url'].unique()
                                if len(links) > 0 and links[0]:
                                    st.markdown(f"🔗 **[View Original Forum Post]({links[0]})**")
                                
                                display_df = results[['name', 'roll_no', 'branch', 'year']].copy()
                                display_df.columns = ["Name", "Roll No", "Branch", "Year"]
                                st.dataframe(display_df, hide_index=True, use_container_width=True)

                display_events_table(ft_events_df, "🎓 Full-Time")
                display_events_table(intern_events_df, "💼 Internship")

    conn.close()
