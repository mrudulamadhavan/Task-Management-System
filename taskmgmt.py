import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords for NLP
nltk.download('stopwords')

# Function to clean text (NLP preprocessing)
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    stop_words = set(stopwords.words('english'))  # Get stopwords
    return ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords

# Load dataset
@st.cache
def load_data():
    # Assuming the dataset is in the same directory as the script
    df = pd.read_csv("task_dataset.csv")
    return df

df = load_data()

# Streamlit Sidebar for User Input
st.sidebar.header("AI-Powered Task Management")
st.sidebar.markdown("""
    - **Priority Distribution**: Visualize task priorities.
    - **Task Completion Status**: View the status of tasks.
    - **Task Assignment**: Assign tasks to team members.
    - **Model Evaluation**: View evaluation results for predictive models.
""")

# Display Data Overview
st.header("Task Management Data Overview")
st.write(df.head())
st.markdown("**Data Information**:")
st.write(df.info())

# Task Distribution Visualizations
st.subheader("Task Distribution")

# Priority Distribution Plot
st.subheader("Priority Distribution")
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(x='Priority', data=df, palette='viridis', hue='Priority', ax=ax)
st.pyplot(fig)

# Task Completion Status Plot
st.subheader("Task Completion Status")
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.countplot(x='Completed', data=df, palette='coolwarm', hue='Completed', ax=ax2)
st.pyplot(fig2)

# Text Preprocessing (NLP)
st.subheader("Text Preprocessing: Clean Task Descriptions")

# Apply the cleaning function to the Description column
df['Clean_Description'] = df['Description'].apply(clean_text)
st.write("Sample of Cleaned Descriptions:")
st.write(df[['Description', 'Clean_Description']].sample(8).reset_index(drop=True))

# Model Development and Evaluation (Random Forest)
st.subheader("Predict Task Priority Using Random Forest")

# TF-IDF Vectorizer for Description Text
tfidf = TfidfVectorizer()
X_text = tfidf.fit_transform(df['Clean_Description'])
y = df['Priority']

# Split data for training/testing
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Display Model Evaluation
st.write("Classification Report (Random Forest):")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Reds', ax=ax3)
st.pyplot(fig3)

# Task Assignment Simulation
st.subheader("Task Assignment to Team Members")

# Get unique team members
team_members = df['Assigned_To'].unique()

# Randomly select 7 team members from the unique list
random_team_members = random.sample(list(team_members), 7)

# Initialize workload dictionary
workload = {member: 0 for member in random_team_members}

# Shuffle rows to randomize task assignment
df_assignment = df.sample(frac=1, random_state=42)

# Ensure Priority column has valid values
priority_order = ['High', 'Medium', 'Low']
df_assignment['Priority'] = df_assignment['Priority'].apply(lambda x: x if x in priority_order else 'Low')  # Default to 'Low'

# Assign tasks based on priority (High > Medium > Low)
assigned_tasks = []

for priority in priority_order:
    tasks = df_assignment[df_assignment['Priority'] == priority]
    for idx, row in tasks.iterrows():
        assignee = min(workload, key=workload.get)
        workload[assignee] += 1
        df_assignment.at[idx, 'Assigned_To'] = assignee
        assigned_tasks.append((row['Task_ID'], assignee, priority))

# Display Task Assignment Results
st.write("Task Assignment Complete!")
st.write("Workload per Member:")
for member, count in workload.items():
    st.write(f"{member}: {count} tasks")

# Preview of Assigned Tasks
st.write("Sample of Assigned Tasks:")
st.write(df_assignment[['Task_ID', 'Description', 'Priority', 'Assigned_To']].sample(10).reset_index(drop=True))

# -------- New Section: User Task Selection --------
# User selects their name from the list
st.subheader("Select Your Name and View Assigned Tasks")

selected_member = st.selectbox("Choose your name from the list", random_team_members)

# Filter tasks assigned to the selected member
assigned_to_member = df_assignment[df_assignment['Assigned_To'] == selected_member]

# Display the first 3 assigned tasks to the selected team member
st.write(f"### {selected_member}'s Assigned Tasks:")

# Limit to 3 tasks
assigned_tasks_sample = assigned_to_member[['Task_ID', 'Description', 'Deadline', 'Priority']].head(3)

# Display tasks with columns (Task Name, Deadline, Priority)
st.write(assigned_tasks_sample)

# -------- End of New Section --------

# Conclusion
st.markdown("""
    ### Summary:
    - Task priority distribution and completion status have been visualized.
    - The Random Forest model was trained to predict task priority based on task descriptions.
    - Task assignment to 7 randomly selected team members has been simulated.
    - You can now select your name and view the first 3 tasks assigned to you.
""")

# Option to download the updated DataFrame
st.sidebar.download_button(
    label="Download Updated Task Data",
    data=df_assignment.to_csv(index=False),
    file_name="assigned_task_data.csv",
    mime="text/csv"
)
