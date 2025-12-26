import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load data



@st.cache_data
def load_data():
    # Use skipinitialspace=True to handle ' email', ' rollno' etc.
    df = pd.read_csv('alot_LM (2).csv', skipinitialspace=True)
    # Ensure columns exist and select them
    df = df[['email', 'rollno', 'cpi']]
    
    with open('branch_mapping.json', 'r') as f:
        mapping = json.load(f)
    email_codes = mapping['email_codes']
    
    # Robust branch extraction using regex (matches process_data.py logic)
    def extract_branch(email):
        # sameern.wanjari.cd.phy21@itbhu.ac.in -> PHY
        # Remove itbhu.ac.in or any domain
        user_part = str(email).split('@')[0]
        # Match the pattern before '@' (e.g., .phy21)
        import re
        match = re.search(r'\.([a-z]{2,4})(\d{2})$', user_part)
        if match:
            b_code = match.group(1).lower()
            return email_codes.get(b_code, b_code.upper())
        return 'Unknown'

    df['branch'] = df['email'].apply(extract_branch)
    
    return df
df = load_data()

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Select Feature', [
    'Overall Stats', 
    'Branch-wise Stats', 
    'CPI Plots', 
    'Student Search', 
    'CPI Range Filter', 
    'Top Students per Branch', 
    'Branch Comparisons'
])

# Overall Stats
if page == 'Overall Stats':
    st.title('Overall Stats')
    st.write(f"Total Students: {len(df)}")
    st.write(f"Average CPI: {df['cpi'].mean():.2f}")
    st.write(f"Max CPI: {df['cpi'].max()}")
    st.write(f"Min CPI: {df['cpi'].min()}")
    st.write("Branch Distribution:")
    st.dataframe(df['branch'].value_counts().reset_index(name='Count'))

# Branch-wise Stats
elif page == 'Branch-wise Stats':
    st.title('Branch-wise Stats')
    branches = df['branch'].unique()
    selected_branch = st.selectbox('Select Branch', branches)
    branch_df = df[df['branch'] == selected_branch]
    st.write(f"Students in {selected_branch}: {len(branch_df)}")
    st.write(f"Avg CPI: {branch_df['cpi'].mean():.2f}")
    st.write(f"Max CPI: {branch_df['cpi'].max()}")
    st.write(f"Min CPI: {branch_df['cpi'].min()}")
    st.dataframe(branch_df[['rollno', 'email', 'cpi']])

# CPI Plots
elif page == 'CPI Plots':
    st.title('CPI Plots')
    plot_type = st.selectbox('Plot Type', ['Histogram', 'Boxplot'])
    branches = st.multiselect('Select Branches', df['branch'].unique(), default=df['branch'].unique()[:3])
    plot_df = df[df['branch'].isin(branches)]
    
    if plot_type == 'Histogram':
        fig, ax = plt.subplots()
        sns.histplot(data=plot_df, x='cpi', hue='branch', multiple='stack', ax=ax)
        st.pyplot(fig)
    elif plot_type == 'Boxplot':
        fig, ax = plt.subplots()
        sns.boxplot(data=plot_df, x='branch', y='cpi', ax=ax)
        st.pyplot(fig)

# Student Search
elif page == 'Student Search':
    st.title('Student Search')
    search_type = st.radio('Search by', ['Roll No', 'Email'])
    query = st.text_input('Enter Query')
    if query:
        if search_type == 'Roll No':
            try:
                search_val = int(query)
                result = df[df['rollno'] == search_val]
            except ValueError:
                result = pd.DataFrame()
        else:
            result = df[df['email'].str.contains(query, case=False, na=False)]
        
        if not result.empty:
            st.dataframe(result[['rollno', 'email', 'branch', 'cpi']])
        else:
            st.write('No results found.')

# CPI Range Filter
elif page == 'CPI Range Filter':
    st.title('CPI Range Filter')
    min_cpi, max_cpi = st.slider('Select CPI Range', float(df['cpi'].min()), float(df['cpi'].max()), (7.0, 9.0))
    branches = st.multiselect('Filter by Branches (optional)', df['branch'].unique())
    filtered = df[(df['cpi'] >= min_cpi) & (df['cpi'] <= max_cpi)]
    if branches:
        filtered = filtered[filtered['branch'].isin(branches)]
    st.write(f"Students in range: {len(filtered)}")
    st.dataframe(filtered[['rollno', 'email', 'branch', 'cpi']])

# Top Students per Branch
elif page == 'Top Students per Branch':
    st.title('Top Students per Branch')
    n = st.number_input('Top N', min_value=1, value=5)
    branches = st.multiselect('Select Branches', df['branch'].unique(), default=df['branch'].unique())
    for branch in branches:
        st.subheader(branch)
        top = df[df['branch'] == branch].nlargest(n, 'cpi')[['rollno', 'email', 'cpi']]
        st.dataframe(top)

# Branch Comparisons
elif page == 'Branch Comparisons':
    st.title('Branch Comparisons')
    branches = st.multiselect('Select Branches to Compare', df['branch'].unique(), default=df['branch'].unique()[:2])
    if len(branches) >= 2:
        comp_df = df[df['branch'].isin(branches)].groupby('branch')['cpi'].agg(['mean', 'max', 'min', 'count']).reset_index()
        st.dataframe(comp_df)
        fig, ax = plt.subplots()
        sns.barplot(data=comp_df, x='branch', y='mean', ax=ax)
        ax.set_ylabel('Avg CPI')
        st.pyplot(fig)
    else:
        st.write('Select at least 2 branches.')