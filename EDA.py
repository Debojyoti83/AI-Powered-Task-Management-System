import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

# Load the dataset
print("Loading task management dataset...")
df = pd.read_csv('task_management_dataset.csv')

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print("="*60)

# 1. BASIC DATASET OVERVIEW
print("\n1. DATASET OVERVIEW")
print("="*40)

# Basic info
print(f"Dataset dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Data types
print(f"\nData Types:")
print(df.dtypes)

# Convert date columns
date_cols = ['created_date', 'due_date', 'completion_date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

print(f"\nBasic Statistics:")
print(df.describe())

# 2. MISSING VALUES ANALYSIS
print("\n\n2. MISSING VALUES ANALYSIS")
print("="*40)

missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_data.index,
    'Missing_Count': missing_data.values,
    'Missing_Percentage': missing_percent.values
}).sort_values('Missing_Percentage', ascending=False)

print("Missing Values Summary:")
print(missing_df[missing_df['Missing_Count'] > 0])

# Visualize missing values
plt.figure(figsize=(15, 10))

# Missing values heatmap
plt.subplot(2, 2, 1)
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.xticks(rotation=45)

# Missing values bar plot
plt.subplot(2, 2, 2)
missing_df_filtered = missing_df[missing_df['Missing_Count'] > 0]
plt.bar(range(len(missing_df_filtered)), missing_df_filtered['Missing_Percentage'])
plt.xticks(range(len(missing_df_filtered)), missing_df_filtered['Column'], rotation=45)
plt.ylabel('Missing Percentage (%)')
plt.title('Missing Values by Column')

plt.tight_layout()
plt.show()

# 3. CATEGORICAL VARIABLES ANALYSIS
print("\n\n3. CATEGORICAL VARIABLES ANALYSIS")
print("="*40)

categorical_cols = ['category', 'priority', 'status', 'department']

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, col in enumerate(categorical_cols):
    # Value counts
    counts = df[col].value_counts()
    print(f"\n{col.upper()} Distribution:")
    print(counts)
    print(f"Unique values: {df[col].nunique()}")

    # Plot
    counts.plot(kind='bar', ax=axes[i])
    axes[i].set_title(f'{col.title()} Distribution')
    axes[i].set_xlabel(col.title())
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 4. NUMERICAL VARIABLES ANALYSIS
print("\n\n4. NUMERICAL VARIABLES ANALYSIS")
print("="*40)

numerical_cols = ['estimated_hours', 'actual_hours', 'complexity_score',
                 'dependencies_count', 'user_current_workload', 'user_experience_level', 'task_age_days']

# Statistical summary
print("Numerical Variables Summary:")
print(df[numerical_cols].describe())

# Detect outliers using IQR method
print("\nOutlier Detection (IQR Method):")
for col in numerical_cols:
    if df[col].notna().sum() > 0:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        print(f"- {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")

# Distribution plots
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

for i, col in enumerate(numerical_cols):
    if i < len(axes):
        # Histogram
        df[col].hist(bins=50, ax=axes[i], alpha=0.7)
        axes[i].set_title(f'{col.replace("_", " ").title()} Distribution')
        axes[i].set_xlabel(col.replace("_", " ").title())
        axes[i].set_ylabel('Frequency')

# Box plots for outlier visualization
plt.figure(figsize=(15, 10))
df[numerical_cols].boxplot(ax=plt.gca())
plt.title('Box Plots - Numerical Variables (Outlier Detection)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. TIME SERIES ANALYSIS
print("\n\n5. TIME SERIES ANALYSIS")
print("="*40)

# Task creation over time
df['created_month'] = df['created_date'].dt.to_period('M')
monthly_tasks = df.groupby('created_month').size()

print("Task Creation Trends:")
print(f"Date range: {df['created_date'].min()} to {df['created_date'].max()}")
print(f"Peak month: {monthly_tasks.idxmax()} ({monthly_tasks.max()} tasks)")
print(f"Lowest month: {monthly_tasks.idxmin()} ({monthly_tasks.min()} tasks)")

plt.figure(figsize=(15, 8))

# Tasks created over time
plt.subplot(2, 2, 1)
monthly_tasks.plot(kind='line', marker='o')
plt.title('Tasks Created Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Tasks')
plt.xticks(rotation=45)

# Task completion analysis
completed_tasks = df[df['status'] == 'Completed'].copy()
if len(completed_tasks) > 0:
    completed_tasks['completion_month'] = completed_tasks['completion_date'].dt.to_period('M')
    monthly_completions = completed_tasks.groupby('completion_month').size()

    plt.subplot(2, 2, 2)
    monthly_completions.plot(kind='line', marker='s', color='green')
    plt.title('Task Completions Over Time')
    plt.xlabel('Month')
    plt.ylabel('Completed Tasks')
    plt.xticks(rotation=45)

# Task duration analysis
completed_tasks['duration_days'] = (completed_tasks['completion_date'] - completed_tasks['created_date']).dt.days
print(f"\nTask Duration Statistics (Completed Tasks):")
print(completed_tasks['duration_days'].describe())

plt.subplot(2, 2, 3)
completed_tasks['duration_days'].hist(bins=30, alpha=0.7, color='orange')
plt.title('Task Duration Distribution (Days)')
plt.xlabel('Duration (Days)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 6. PRIORITY VS OTHER VARIABLES
print("\n\n6. PRIORITY ANALYSIS")
print("="*40)

# Priority vs complexity
priority_complexity = df.groupby('priority')['complexity_score'].agg(['mean', 'std', 'count'])
print("Priority vs Complexity Score:")
print(priority_complexity)

# Priority vs estimated hours
priority_hours = df.groupby('priority')['estimated_hours'].agg(['mean', 'std', 'count'])
print("\nPriority vs Estimated Hours:")
print(priority_hours)

plt.figure(figsize=(15, 10))

# Priority vs Complexity
plt.subplot(2, 3, 1)
sns.boxplot(data=df, x='priority', y='complexity_score', order=['Low', 'Medium', 'High', 'Critical'])
plt.title('Priority vs Complexity Score')
plt.xticks(rotation=45)

# Priority vs Estimated Hours
plt.subplot(2, 3, 2)
sns.boxplot(data=df, x='priority', y='estimated_hours', order=['Low', 'Medium', 'High', 'Critical'])
plt.title('Priority vs Estimated Hours')
plt.xticks(rotation=45)

# Priority vs Status
plt.subplot(2, 3, 3)
priority_status = pd.crosstab(df['priority'], df['status'])
priority_status_pct = priority_status.div(priority_status.sum(axis=1), axis=0) * 100
sns.heatmap(priority_status_pct, annot=True, fmt='.1f', cmap='YlOrRd')
plt.title('Priority vs Status (%)')

plt.tight_layout()
plt.show()

# 7. USER AND WORKLOAD ANALYSIS
print("\n\n7. USER AND WORKLOAD ANALYSIS")
print("="*40)

# User workload distribution
print("User Workload Statistics:")
print(df['user_current_workload'].describe())

# Top users by task count
user_task_counts = df['assigned_to'].value_counts().head(10)
print(f"\nTop 10 Users by Task Assignment:")
print(user_task_counts)

# Experience vs performance
experience_performance = df.groupby(pd.cut(df['user_experience_level'], bins=5)).agg({
    'complexity_score': 'mean',
    'estimated_hours': 'mean',
    'task_id': 'count'
}).round(2)
print(f"\nUser Experience vs Task Characteristics:")
print(experience_performance)

plt.figure(figsize=(15, 8))

# Workload distribution
plt.subplot(2, 3, 1)
df['user_current_workload'].hist(bins=20, alpha=0.7)
plt.title('User Workload Distribution')
plt.xlabel('Current Workload')
plt.ylabel('Frequency')

# Experience level distribution
plt.subplot(2, 3, 2)
df['user_experience_level'].hist(bins=20, alpha=0.7, color='green')
plt.title('User Experience Level Distribution')
plt.xlabel('Experience Level')
plt.ylabel('Frequency')

# Department workload
dept_workload = df.groupby('department').size().sort_values(ascending=True)
plt.subplot(2, 3, 3)
dept_workload.plot(kind='barh')
plt.title('Tasks by Department')
plt.xlabel('Number of Tasks')

plt.tight_layout()
plt.show()

# 8. CORRELATION ANALYSIS
print("\n\n8. CORRELATION ANALYSIS")
print("="*40)

# Select numerical columns for correlation
corr_cols = ['estimated_hours', 'actual_hours', 'complexity_score',
             'dependencies_count', 'user_current_workload', 'user_experience_level', 'task_age_days']

correlation_matrix = df[corr_cols].corr()
print("Correlation Matrix:")
print(correlation_matrix.round(3))

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.3f')
plt.title('Correlation Matrix - Numerical Variables')
plt.tight_layout()
plt.show()

# 9. DATA QUALITY ISSUES
print("\n\n9. DATA QUALITY ASSESSMENT")
print("="*40)

# Check for duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# Check for impossible values
print("\nData Quality Checks:")
print(f"- Tasks with completion date before creation: {(df['completion_date'] < df['created_date']).sum()}")
print(f"- Tasks with due date before creation: {(df['due_date'] < df['created_date']).sum()}")
print(f"- Negative estimated hours: {(df['estimated_hours'] < 0).sum()}")
print(f"- Negative actual hours: {(df['actual_hours'] < 0).sum()}")

# Check overdue tasks
overdue_tasks = df[df['is_overdue'] == True]
print(f"- Overdue tasks: {len(overdue_tasks)} ({len(overdue_tasks)/len(df)*100:.2f}%)")

# 10. KEY INSIGHTS SUMMARY
print("\n\n10. KEY INSIGHTS SUMMARY")
print("="*40)

print("📊 DATASET OVERVIEW:")
print(f"   • Total tasks: {len(df):,}")
print(f"   • Date range: {df['created_date'].min().strftime('%Y-%m-%d')} to {df['created_date'].max().strftime('%Y-%m-%d')}")
print(f"   • Categories: {df['category'].nunique()}")
print(f"   • Users: {df['assigned_to'].nunique()}")

print("\n🔍 DATA QUALITY:")
print(f"   • Missing values: {df.isnull().sum().sum():,} ({df.isnull().sum().sum()/(len(df)*len(df.columns))*100:.2f}% of total)")
print(f"   • Most missing: {missing_df.iloc[0]['Column']} ({missing_df.iloc[0]['Missing_Percentage']:.1f}%)")
print(f"   • Overdue tasks: {overdue_tasks.shape[0]:,}")

print("\n📈 TASK DISTRIBUTION:")
print(f"   • Most common priority: {df['priority'].mode()[0]} ({df['priority'].value_counts().iloc[0]:,} tasks)")
print(f"   • Most common status: {df['status'].mode()[0]} ({df['status'].value_counts().iloc[0]:,} tasks)")
print(f"   • Completion rate: {(df['status'] == 'Completed').sum()/len(df)*100:.1f}%")

print("\n⏱️ EFFORT ANALYSIS:")
print(f"   • Avg estimated hours: {df['estimated_hours'].mean():.1f}")
print(f"   • Avg complexity score: {df['complexity_score'].mean():.1f}")
print(f"   • Avg task duration: {completed_tasks['duration_days'].mean():.1f} days")

print("\n👥 USER INSIGHTS:")
print(f"   • Avg user workload: {df['user_current_workload'].mean():.1f} tasks")
print(f"   • Avg user experience: {df['user_experience_level'].mean():.1f}/10")
print(f"   • Most active department: {df['department'].mode()[0]}")

plt.show()