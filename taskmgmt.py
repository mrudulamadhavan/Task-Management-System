import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("task_dataset.csv")
    return df

df = load_data()

# ----------------------------
# Streamlit UI
# ----------------------------

# Set page title and logo
st.set_page_config(page_title="Task Management System", page_icon="🗂️", layout="centered")


st.title(" 🗂️   Task Management System   🗂️")

# Description
st.markdown(
    "<h4 style='text-align: left; font-weight: bold;'>Effortlessly manage your tasks: Prioritize, Track, and Download </h4>",
    unsafe_allow_html=True)
st.markdown(
    "<h5 style='text-align: center; font-weight: bold;'>All in one click! 👍</h5>",
    unsafe_allow_html=True
)

st.write("")
st.write("""To view assigned tasks, select your name below :""")

# ----------------------------
# Clean up & Prepare Data
# ----------------------------
priority_order = ['High', 'Medium', 'Low']
df['Priority'] = df['Priority'].apply(lambda x: x if x in priority_order else 'Low')
df['Priority_Rank'] = df['Priority'].map({'High': 1, 'Medium': 2, 'Low': 3})


# Randomly reassign tasks for demo (optional, or use actual data)
team_members = df['Assigned_To'].unique().tolist()


# Shuffle and assign tasks
df_assignment = df.copy()
workload = {member: 0 for member in team_members}
df_assignment = df_assignment.sample(frac=1, random_state=42)

for priority in priority_order:
    tasks = df_assignment[df_assignment['Priority'] == priority]
    for idx, row in tasks.iterrows():
        assignee = min(workload, key=workload.get)
        workload[assignee] += 1
        df_assignment.at[idx, 'Assigned_To'] = assignee

# ----------------------------
# User Selection and Display
# ----------------------------
selected_employee = st.selectbox("Employee Name", sorted(df_assignment['Assigned_To'].unique()))

# Filter and display tasks
filtered_tasks = df_assignment[df_assignment['Assigned_To'] == selected_employee]
filtered_tasks = filtered_tasks.sort_values(by='Priority_Rank').head(5).reset_index(drop=True)

st.subheader(f"Top 5 Tasks Assigned to {selected_employee}")
st.dataframe(filtered_tasks[['Task_ID', 'Description', 'Deadline', 'Priority']])



# ----------------------------
# Download Assigned Tasks
# ----------------------------
csv_data = filtered_tasks.to_csv(index=False)
st.download_button(
    label="📥 Download My Tasks",
    data=csv_data,
    file_name=f"{selected_employee}_tasks.csv",
    mime='text/csv'
)
