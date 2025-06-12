import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import string

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Dataset parameters
n_rows = 60000

# Define realistic categories and data
task_categories = [
    'Development', 'Testing', 'Documentation', 'Design', 'Research', 
    'Marketing', 'Sales', 'Customer Support', 'Operations', 'HR',
    'Finance', 'Legal', 'Project Management', 'Data Analysis', 'Infrastructure'
]

priorities = ['Low', 'Medium', 'High', 'Critical']
statuses = ['Not Started', 'In Progress', 'On Hold', 'Completed', 'Cancelled']
departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations', 'Legal']

# User pools for assignment
users = [f'user_{i:03d}' for i in range(1, 201)]  # 200 users
managers = [f'manager_{i:02d}' for i in range(1, 31)]  # 30 managers

# Task description templates for more realistic text
task_templates = [
    "Implement {} functionality for the {} module",
    "Review and update {} documentation for {} process",
    "Design {} interface for {} application",
    "Test {} feature in {} environment",
    "Analyze {} data for {} optimization",
    "Create {} report for {} stakeholders",
    "Fix {} bug in {} system",
    "Develop {} strategy for {} campaign",
    "Conduct {} research on {} market trends",
    "Optimize {} performance in {} database",
    "Configure {} settings for {} deployment",
    "Validate {} requirements for {} project",
    "Prepare {} presentation for {} meeting",
    "Monitor {} metrics for {} dashboard",
    "Integrate {} API with {} service"
]

action_words = ['new', 'advanced', 'automated', 'enhanced', 'streamlined', 'comprehensive', 'critical', 'urgent', 'routine', 'complex']
subject_words = ['user authentication', 'payment processing', 'data migration', 'security audit', 'performance monitoring', 'client onboarding', 'inventory management', 'workflow automation', 'system integration', 'compliance check']

def generate_task_description():
    """Generate realistic task descriptions"""
    template = random.choice(task_templates)
    action = random.choice(action_words)
    subject = random.choice(subject_words)
    return template.format(action, subject)

def generate_realistic_dates():
    """Generate realistic date ranges"""
    # Create date range from 6 months ago to 3 months in future
    start_date = datetime.now() - timedelta(days=180)
    end_date = datetime.now() + timedelta(days=90)
    
    created_date = start_date + timedelta(days=random.randint(0, 270))
    
    # Due date is typically 1-30 days after creation
    due_offset = np.random.exponential(7) + 1  # Exponential distribution with mean ~8 days
    due_date = created_date + timedelta(days=min(int(due_offset), 60))
    
    # Completion date (if completed)
    completion_date = None
    if random.random() < 0.6:  # 60% tasks are completed
        completion_offset = random.uniform(0.5, 1.5) * (due_date - created_date).days
        completion_date = created_date + timedelta(days=completion_offset)
    
    return created_date, due_date, completion_date

# Generate the dataset
print("Generating task management dataset...")

data = []
for i in range(n_rows):
    if i % 10000 == 0:
        print(f"Generated {i} rows...")
    
    # Basic task information
    task_id = f"TASK_{i+1:06d}"
    title = f"Task {i+1}: {generate_task_description()}"
    description = generate_task_description() + ". " + generate_task_description().lower()
    category = random.choice(task_categories)
    
    # Priority with realistic distribution (more medium/low than high/critical)
    priority_weights = [0.3, 0.4, 0.25, 0.05]  # Low, Medium, High, Critical
    priority = np.random.choice(priorities, p=priority_weights)
    
    # Status with realistic distribution
    status_weights = [0.15, 0.35, 0.1, 0.35, 0.05]  # Not Started, In Progress, On Hold, Completed, Cancelled
    status = np.random.choice(statuses, p=status_weights)
    
    # Dates
    created_date, due_date, completion_date = generate_realistic_dates()
    
    # User assignment
    assigned_to = random.choice(users)
    created_by = random.choice(users)
    manager = random.choice(managers)
    department = random.choice(departments)
    
    # Effort estimation (in hours) with some outliers
    if random.random() < 0.05:  # 5% outliers
        estimated_hours = random.uniform(100, 500)
    else:
        estimated_hours = np.random.lognormal(mean=2.5, sigma=0.8)  # Log-normal distribution
    
    # Actual hours (if task is completed or in progress)
    actual_hours = None
    if status in ['Completed', 'In Progress']:
        # Actual hours typically differ from estimates
        variance = random.uniform(0.7, 1.5)
        actual_hours = estimated_hours * variance
        if random.random() < 0.03:  # 3% extreme outliers
            actual_hours *= random.uniform(2, 5)
    
    # Complexity score (1-10)
    complexity = min(10, max(1, np.random.normal(5, 2)))
    
    # Dependencies count
    dependencies = max(0, int(np.random.poisson(1.2)))
    
    # User workload (current tasks assigned)
    current_workload = max(0, int(np.random.poisson(8)))
    
    # User experience level (1-10)
    user_experience = min(10, max(1, np.random.normal(6, 1.5)))
    
    # Task age in days
    task_age = (datetime.now() - created_date).days
    
    # Overdue flag
    is_overdue = due_date < datetime.now() and status not in ['Completed', 'Cancelled']
    
    data.append({
        'task_id': task_id,
        'title': title,
        'description': description,
        'category': category,
        'priority': priority,
        'status': status,
        'created_date': created_date.strftime('%Y-%m-%d'),
        'due_date': due_date.strftime('%Y-%m-%d'),
        'completion_date': completion_date.strftime('%Y-%m-%d') if completion_date else None,
        'assigned_to': assigned_to,
        'created_by': created_by,
        'manager': manager,
        'department': department,
        'estimated_hours': round(estimated_hours, 2),
        'actual_hours': round(actual_hours, 2) if actual_hours else None,
        'complexity_score': round(complexity, 1),
        'dependencies_count': dependencies,
        'user_current_workload': current_workload,
        'user_experience_level': round(user_experience, 1),
        'task_age_days': task_age,
        'is_overdue': is_overdue
    })

# Create DataFrame
df = pd.DataFrame(data)

# Introduce realistic missing values
print("Introducing missing values...")

# Missing completion dates for non-completed tasks (realistic)
df.loc[df['status'].isin(['Not Started', 'In Progress', 'On Hold']), 'completion_date'] = None

# Missing actual hours for not started tasks
df.loc[df['status'] == 'Not Started', 'actual_hours'] = None

# Random missing values (realistic percentages)
missing_patterns = {
    'description': 0.02,  # 2% missing descriptions
    'estimated_hours': 0.05,  # 5% missing estimates
    'complexity_score': 0.08,  # 8% missing complexity scores
    'user_experience_level': 0.03,  # 3% missing user experience
    'manager': 0.01,  # 1% missing manager assignment
}

for column, missing_rate in missing_patterns.items():
    missing_indices = np.random.choice(df.index, size=int(len(df) * missing_rate), replace=False)
    df.loc[missing_indices, column] = None

# Add some data quality issues
print("Adding data quality issues...")

# Duplicate task titles (realistic scenario)
duplicate_indices = np.random.choice(df.index, size=100, replace=False)
for idx in duplicate_indices:
    duplicate_target = random.choice(df.index)
    df.at[idx, 'title'] = df.at[duplicate_target, 'title'] + " (Copy)"

# Inconsistent category naming
inconsistent_indices = np.random.choice(df.index, size=50, replace=False)
for idx in inconsistent_indices:
    if df.at[idx, 'category'] == 'Development':
        df.at[idx, 'category'] = random.choice(['Dev', 'Software Development', 'development'])

# Save dataset
print("Saving dataset...")
df.to_csv('task_management_dataset.csv', index=False)

# Display dataset information
print(f"\nDataset Generation Complete!")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nDataset Info:")
print(f"- Shape: {df.shape}")
print(f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\nColumn Details:")
for col in df.columns:
    null_count = df[col].isnull().sum()
    null_pct = (null_count / len(df)) * 100
    print(f"- {col}: {null_count} missing ({null_pct:.1f}%)")

print(f"\nSample Statistics:")
print(f"- Unique categories: {df['category'].nunique()}")
print(f"- Unique users: {df['assigned_to'].nunique()}")
print(f"- Date range: {df['created_date'].min()} to {df['created_date'].max()}")
print(f"- Priority distribution:\n{df['priority'].value_counts()}")
print(f"- Status distribution:\n{df['status'].value_counts()}")

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nDataset saved as 'task_management_dataset.csv'")
print("Ready for EDA and preprocessing!")