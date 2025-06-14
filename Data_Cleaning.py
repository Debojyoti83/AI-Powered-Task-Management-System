import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("TASK MANAGEMENT DATASET - DATA CLEANING")
print("="*50)

# Load the dataset
df = pd.read_csv('task_management_dataset.csv')
print(f"Original dataset shape: {df.shape}")

# Create a copy for cleaning
df_cleaned = df.copy()

# Convert date columns to datetime
date_cols = ['created_date', 'due_date', 'completion_date']
for col in date_cols:
    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')

print(f"Date columns converted to datetime format")

# 1. HANDLE MISSING VALUES
print("\n1. HANDLING MISSING VALUES")
print("-" * 30)

print("Missing values before cleaning:")
missing_before = df_cleaned.isnull().sum()
print(missing_before[missing_before > 0])

# Strategy for handling missing values:

# 1.1 Description - Forward fill from title or use default
print("\n1.1 Cleaning missing descriptions...")
missing_desc_mask = df_cleaned['description'].isnull()
print(f"Missing descriptions: {missing_desc_mask.sum()}")

# Create default descriptions based on category and title
def create_default_description(row):
    if pd.isna(row['description']):
        return f"Task related to {row['category'].lower()} - {row['title'][:50]}..."
    return row['description']

df_cleaned['description'] = df_cleaned.apply(create_default_description, axis=1)
print(f"Missing descriptions after cleaning: {df_cleaned['description'].isnull().sum()}")

# 1.2 Manager - Forward fill or assign default manager per department
print("\n1.2 Cleaning missing managers...")
missing_manager_mask = df_cleaned['manager'].isnull()
print(f"Missing managers: {missing_manager_mask.sum()}")

# Assign default managers per department
dept_manager_map = {
    'Engineering': 'manager_01',
    'Marketing': 'manager_02', 
    'Sales': 'manager_03',
    'HR': 'manager_04',
    'Finance': 'manager_05',
    'Operations': 'manager_06',
    'Legal': 'manager_07'
}

df_cleaned['manager'] = df_cleaned.apply(
    lambda row: dept_manager_map.get(row['department'], 'manager_01') if pd.isna(row['manager']) else row['manager'], 
    axis=1
)
print(f"Missing managers after cleaning: {df_cleaned['manager'].isnull().sum()}")

# 1.3 Estimated hours - Use median by category and priority
print("\n1.3 Cleaning missing estimated hours...")
missing_est_hours = df_cleaned['estimated_hours'].isnull().sum()
print(f"Missing estimated hours: {missing_est_hours}")

# Fill with median by category and priority
for category in df_cleaned['category'].unique():
    for priority in df_cleaned['priority'].unique():
        mask = (df_cleaned['category'] == category) & (df_cleaned['priority'] == priority) & df_cleaned['estimated_hours'].isnull()
        if mask.any():
            median_hours = df_cleaned[(df_cleaned['category'] == category) & (df_cleaned['priority'] == priority)]['estimated_hours'].median()
            if pd.isna(median_hours):
                median_hours = df_cleaned['estimated_hours'].median()  # Fallback to overall median
            df_cleaned.loc[mask, 'estimated_hours'] = median_hours

print(f"Missing estimated hours after cleaning: {df_cleaned['estimated_hours'].isnull().sum()}")

# 1.4 Complexity score - Use mean by category
print("\n1.4 Cleaning missing complexity scores...")
missing_complexity = df_cleaned['complexity_score'].isnull().sum()
print(f"Missing complexity scores: {missing_complexity}")

for category in df_cleaned['category'].unique():
    mask = (df_cleaned['category'] == category) & df_cleaned['complexity_score'].isnull()
    if mask.any():
        mean_complexity = df_cleaned[df_cleaned['category'] == category]['complexity_score'].mean()
        if pd.isna(mean_complexity):
            mean_complexity = df_cleaned['complexity_score'].mean()  # Fallback
        df_cleaned.loc[mask, 'complexity_score'] = round(mean_complexity, 1)

print(f"Missing complexity scores after cleaning: {df_cleaned['complexity_score'].isnull().sum()}")

# 1.5 User experience level - Use mean by department
print("\n1.5 Cleaning missing user experience levels...")
missing_exp = df_cleaned['user_experience_level'].isnull().sum()
print(f"Missing user experience levels: {missing_exp}")

for dept in df_cleaned['department'].unique():
    mask = (df_cleaned['department'] == dept) & df_cleaned['user_experience_level'].isnull()
    if mask.any():
        mean_exp = df_cleaned[df_cleaned['department'] == dept]['user_experience_level'].mean()
        if pd.isna(mean_exp):
            mean_exp = df_cleaned['user_experience_level'].mean()  # Fallback
        df_cleaned.loc[mask, 'user_experience_level'] = round(mean_exp, 1)

print(f"Missing user experience levels after cleaning: {df_cleaned['user_experience_level'].isnull().sum()}")

# 1.6 Actual hours - Keep missing for tasks not started/completed (this is expected)
print("\n1.6 Actual hours - keeping realistic missing values")
# Only fill actual hours for 'In Progress' and 'Completed' tasks that are missing
in_progress_completed = df_cleaned['status'].isin(['In Progress', 'Completed'])
missing_actual_hours = df_cleaned['actual_hours'].isnull()

mask = in_progress_completed & missing_actual_hours
if mask.any():
    # Use estimated hours with some variance for missing actual hours
    df_cleaned.loc[mask, 'actual_hours'] = df_cleaned.loc[mask, 'estimated_hours'] * np.random.normal(1.1, 0.3, mask.sum())

print(f"Actual hours filled for in-progress/completed tasks: {mask.sum()}")

# 1.7 Completion date - Keep missing for non-completed tasks (this is expected)
print("\n1.7 Completion dates - keeping realistic missing values")
print(f"Completion dates missing (expected for non-completed): {df_cleaned['completion_date'].isnull().sum()}")

# 2. STANDARDIZE CATEGORICAL DATA
print("\n2. STANDARDIZING CATEGORICAL DATA")
print("-" * 35)

# 2.1 Fix inconsistent category names
print("\n2.1 Fixing category inconsistencies...")
category_mapping = {
    'development': 'Development',
    'Dev': 'Development', 
    'Software Development': 'Development'
}

df_cleaned['category'] = df_cleaned['category'].replace(category_mapping)
print(f"Unique categories after cleaning: {df_cleaned['category'].nunique()}")
print(f"Categories: {sorted(df_cleaned['category'].unique())}")

# 2.2 Standardize priority order
print("\n2.2 Standardizing priority levels...")
priority_order = ['Low', 'Medium', 'High', 'Critical']
df_cleaned['priority'] = pd.Categorical(df_cleaned['priority'], categories=priority_order, ordered=True)
print(f"Priority levels: {df_cleaned['priority'].cat.categories.tolist()}")

# 2.3 Standardize status order
print("\n2.3 Standardizing status levels...")
status_order = ['Not Started', 'In Progress', 'On Hold', 'Completed', 'Cancelled']
df_cleaned['status'] = pd.Categorical(df_cleaned['status'], categories=status_order, ordered=True)
print(f"Status levels: {df_cleaned['status'].cat.categories.tolist()}")

# 3. HANDLE OUTLIERS
print("\n3. HANDLING OUTLIERS")
print("-" * 20)

def cap_outliers(series, method='iqr', factor=1.5):
    """Cap outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    outliers_before = ((series < lower_bound) | (series > upper_bound)).sum()
    series_capped = series.clip(lower_bound, upper_bound)
    outliers_after = ((series_capped < lower_bound) | (series_capped > upper_bound)).sum()
    
    return series_capped, outliers_before, outliers_after

# 3.1 Handle estimated hours outliers
print("\n3.1 Handling estimated hours outliers...")
df_cleaned['estimated_hours'], est_before, est_after = cap_outliers(df_cleaned['estimated_hours'])
print(f"Estimated hours outliers: {est_before} -> {est_after}")

# 3.2 Handle actual hours outliers  
print("\n3.2 Handling actual hours outliers...")
actual_hours_clean = df_cleaned['actual_hours'].dropna()
actual_hours_capped, act_before, act_after = cap_outliers(actual_hours_clean)
df_cleaned.loc[df_cleaned['actual_hours'].notna(), 'actual_hours'] = actual_hours_capped
print(f"Actual hours outliers: {act_before} -> {act_after}")

# 3.3 Handle user workload outliers
print("\n3.3 Handling user workload outliers...")
df_cleaned['user_current_workload'], work_before, work_after = cap_outliers(df_cleaned['user_current_workload'])
print(f"User workload outliers: {work_before} -> {work_after}")

# 4. CREATE DERIVED FEATURES
print("\n4. CREATING DERIVED FEATURES")
print("-" * 28)

# 4.1 Task duration (for completed tasks)
print("\n4.1 Creating task duration feature...")
completed_mask = df_cleaned['status'] == 'Completed'
df_cleaned['task_duration_days'] = np.nan
df_cleaned.loc[completed_mask, 'task_duration_days'] = (
    df_cleaned.loc[completed_mask, 'completion_date'] - 
    df_cleaned.loc[completed_mask, 'created_date']
).dt.days

print(f"Task duration calculated for {completed_mask.sum()} completed tasks")

# 4.2 Effort variance (actual vs estimated)
print("\n4.2 Creating effort variance feature...")
has_both_hours = df_cleaned['actual_hours'].notna() & df_cleaned['estimated_hours'].notna()
df_cleaned['effort_variance_ratio'] = np.nan
df_cleaned.loc[has_both_hours, 'effort_variance_ratio'] = (
    df_cleaned.loc[has_both_hours, 'actual_hours'] / 
    df_cleaned.loc[has_both_hours, 'estimated_hours']
)

print(f"Effort variance calculated for {has_both_hours.sum()} tasks")

# 4.3 Priority score (numeric)
print("\n4.3 Creating numeric priority score...")
priority_scores = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
df_cleaned['priority_score'] = df_cleaned['priority'].map(priority_scores)

# 4.4 Time to due date
print("\n4.4 Creating time to due date feature...")
current_date = datetime.now()
df_cleaned['days_to_due'] = (df_cleaned['due_date'] - current_date).dt.days

# 4.5 Workload intensity (complexity / estimated hours)
print("\n4.5 Creating workload intensity feature...")
has_complexity_hours = df_cleaned['complexity_score'].notna() & df_cleaned['estimated_hours'].notna()
df_cleaned['workload_intensity'] = np.nan
df_cleaned.loc[has_complexity_hours, 'workload_intensity'] = (
    df_cleaned.loc[has_complexity_hours, 'complexity_score'] / 
    df_cleaned.loc[has_complexity_hours, 'estimated_hours']
)

print(f"Workload intensity calculated for {has_complexity_hours.sum()} tasks")

# 5. DATA VALIDATION
print("\n5. DATA VALIDATION")
print("-" * 18)

print("\n5.1 Checking data consistency...")
# Check date logic
invalid_dates = df_cleaned['due_date'] < df_cleaned['created_date']
print(f"Tasks with due date before creation: {invalid_dates.sum()}")

# Check negative values
negative_hours = df_cleaned['estimated_hours'] < 0
print(f"Tasks with negative estimated hours: {negative_hours.sum()}")

# Check complexity range
invalid_complexity = (df_cleaned['complexity_score'] < 1) | (df_cleaned['complexity_score'] > 10)
print(f"Tasks with invalid complexity scores: {invalid_complexity.sum()}")

# Check experience range
invalid_experience = (df_cleaned['user_experience_level'] < 1) | (df_cleaned['user_experience_level'] > 10)
print(f"Users with invalid experience levels: {invalid_experience.sum()}")

# 6. FINAL CLEANING SUMMARY
print("\n6. FINAL CLEANING SUMMARY")
print("-" * 26)

missing_after = df_cleaned.isnull().sum()
print("Missing values after cleaning:")
print(missing_after[missing_after > 0])

print(f"\nDataset shape after cleaning: {df_cleaned.shape}")
print(f"Original shape: {df.shape}")

# Data types summary
print(f"\nData types after cleaning:")
print(df_cleaned.dtypes)

# Save cleaned dataset
df_cleaned.to_csv('task_management_dataset_cleaned.csv', index=False)
print(f"\n✅ Cleaned dataset saved as 'task_management_dataset_cleaned.csv'")

# 7. CLEANING STATISTICS
print("\n7. CLEANING STATISTICS")
print("-" * 21)

print("Missing Value Reductions:")
missing_comparison = pd.DataFrame({
    'Before': missing_before,
    'After': missing_after,
    'Reduction': missing_before - missing_after
})
print(missing_comparison[missing_comparison['Before'] > 0])

print(f"\nOutlier Reductions:")
print(f"- Estimated hours: {est_before} -> {est_after}")
print(f"- Actual hours: {act_before} -> {act_after}")
print(f"- User workload: {work_before} -> {work_after}")

print(f"\nNew Features Created:")
new_features = ['task_duration_days', 'effort_variance_ratio', 'priority_score', 
                'days_to_due', 'workload_intensity']
for feature in new_features:
    non_null_count = df_cleaned[feature].count()
    print(f"- {feature}: {non_null_count} non-null values")

# Quick preview of cleaned data
print(f"\nFirst 5 rows of cleaned data:")
print(df_cleaned[['task_id', 'category', 'priority', 'status', 'estimated_hours', 'complexity_score']].head())