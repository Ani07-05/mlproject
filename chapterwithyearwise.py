"""
JEE Chapter Weightage Prediction Model
This script analyzes historical JEE data and predicts weightage for each chapter for upcoming year.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('corrected_dataset.csv')

# Data exploration
print(f"Dataset shape: {df.shape}")
print(df.info())
print("Data sample:")
print(df.head())

# Check for missing values and basic statistics
print("\nMissing values:")
print(df.isnull().sum())
print("\nMarks by subject:")
print(df.groupby('subject')['marks'].describe())

# Instead of using ML to predict marks, we'll use a direct historical analysis approach
# This will ensure more variation in marks based on actual patterns

# Create historical importance analysis
print("\nAnalyzing historical importance of chapters...")

# Analyze data by focusing more on recent years (2015-2024)
recent_years = list(range(2015, 2025))
older_years = list(range(2002, 2015))

# Weight recent years more heavily
recent_df = df[df['year'].isin(recent_years)].copy()
older_df = df[df['year'].isin(older_years)].copy()

# Calculate importance metrics
def calculate_chapter_importance(subject_df):
    # Group by chapter
    chapter_stats = subject_df.groupby('chapter').agg({
        'marks': ['mean', 'sum', 'count', 'max']
    })
    
    chapter_stats.columns = ['avg_marks', 'total_marks', 'frequency', 'max_marks']
    
    # Calculate various importance metrics
    chapter_stats['importance_score'] = (
        chapter_stats['avg_marks'] * 0.3 +  # Average marks
        chapter_stats['frequency'] * 0.3 +  # How often it appears
        chapter_stats['max_marks'] * 0.4    # Maximum marks ever given
    )
    
    # Calculate historical variation (higher variation = more important)
    chapter_variation = {}
    for chapter in chapter_stats.index:
        chapter_data = subject_df[subject_df['chapter'] == chapter]
        if len(chapter_data) > 1:
            chapter_variation[chapter] = chapter_data['marks'].std()
        else:
            chapter_variation[chapter] = 0
    
    chapter_stats['variation'] = pd.Series(chapter_variation)
    chapter_stats['importance_score'] += chapter_stats['variation'] * 0.5
    
    return chapter_stats.sort_values('importance_score', ascending=False)

# Calculate importance by subject
subject_importance = {}
for subject in df['subject'].unique():
    # Calculate importance from recent data (weighted higher)
    recent_subject_df = recent_df[recent_df['subject'] == subject]
    recent_importance = calculate_chapter_importance(recent_subject_df)
    
    # Calculate importance from older data
    older_subject_df = older_df[older_df['subject'] == subject]
    older_importance = calculate_chapter_importance(older_subject_df)
    
    # Combine them with more weight to recent years
    if not recent_importance.empty and not older_importance.empty:
        # Get all chapters from both periods - convert set to list
        all_chapters = list(set(recent_importance.index) | set(older_importance.index))
        
        combined_importance = pd.DataFrame(index=all_chapters)
        
        # Fill with data from both periods, applying weights
        for chapter in all_chapters:
            recent_score = recent_importance.loc[chapter, 'importance_score'] if chapter in recent_importance.index else 0
            older_score = older_importance.loc[chapter, 'importance_score'] if chapter in older_importance.index else 0
            
            # Weight recent years 3x more than older years
            combined_importance.loc[chapter, 'importance_score'] = recent_score * 0.75 + older_score * 0.25
            
            # Get other metrics from recent if available, otherwise from older
            for col in ['avg_marks', 'total_marks', 'frequency', 'max_marks', 'variation']:
                if chapter in recent_importance.index:
                    combined_importance.loc[chapter, col] = recent_importance.loc[chapter, col]
                elif chapter in older_importance.index:
                    combined_importance.loc[chapter, col] = older_importance.loc[chapter, col]
                else:
                    combined_importance.loc[chapter, col] = 0
        
        subject_importance[subject] = combined_importance.sort_values('importance_score', ascending=False)
    else:
        # If either is empty, use the non-empty one
        subject_importance[subject] = recent_importance if not recent_importance.empty else older_importance

# Create training data for ML models to compare
# We'll use this for generating confusion matrices and model comparison graphs
print("\nCreating training data for ML model comparison...")

# Enhance the dataset with features that help differentiate between chapters
enhanced_df = df.copy()

# Add feature: historical average marks for this chapter
chapter_avg = df.groupby('chapter')['marks'].mean().to_dict()
enhanced_df['chapter_avg_mark'] = enhanced_df['chapter'].map(chapter_avg)

# Add feature: historical frequency of the chapter
chapter_freq = df.groupby('chapter')['marks'].count().to_dict()
enhanced_df['chapter_frequency'] = enhanced_df['chapter'].map(chapter_freq)

# Add feature: historical maximum marks for this chapter
chapter_max = df.groupby('chapter')['marks'].max().to_dict()
enhanced_df['chapter_max_mark'] = enhanced_df['chapter'].map(chapter_max)

# Calculate trend metrics - how much each chapter's marks changed over time
def calculate_trend(chapter_data):
    if len(chapter_data) >= 5:  # Need at least a few data points
        # Use a simple linear regression to capture trend
        x = chapter_data['year'].values.reshape(-1, 1)
        y = chapter_data['avg_marks'].values
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(x, y)
        return model.coef_[0]  # Slope of the regression line
    return 0

year_trend = df.groupby(['year', 'chapter']).agg({
    'marks': ['mean', 'sum', 'count']
}).reset_index()
year_trend.columns = ['year', 'chapter', 'avg_marks', 'total_marks', 'frequency']

chapter_trends = {}
for chapter in df['chapter'].unique():
    chapter_data = year_trend[year_trend['chapter'] == chapter]
    trend = calculate_trend(chapter_data)
    chapter_trends[chapter] = trend

# Add feature: historical trend (increasing/decreasing importance)
enhanced_df['chapter_trend'] = enhanced_df['chapter'].map(chapter_trends)

# Add feature: recent years importance (last 5 years)
recent_avg = recent_df.groupby('chapter')['marks'].mean().to_dict()
enhanced_df['recent_importance'] = enhanced_df['chapter'].map(recent_avg)

# Add feature: has this chapter had high marks in any year
enhanced_df['had_high_marks'] = (enhanced_df['chapter_max_mark'] >= 8).astype(int)

# Add feature: consistency of the chapter's appearance
chapter_years = {}
for chapter in df['chapter'].unique():
    chapter_data = df[df['chapter'] == chapter]
    years_count = chapter_data['year'].nunique()
    chapter_years[chapter] = years_count / df['year'].nunique()  # Normalize by total years
enhanced_df['chapter_consistency'] = enhanced_df['chapter'].map(chapter_years)

# Fill any NaN values
enhanced_df['chapter_trend'].fillna(0, inplace=True)
enhanced_df['recent_importance'].fillna(enhanced_df['chapter_avg_mark'], inplace=True)
enhanced_df['chapter_consistency'].fillna(0, inplace=True)

# Create a categorical target variable for classification confusion matrix
# We'll bin the marks into categories: low (4), medium (8), high (12+)
def categorize_marks(mark):
    if mark <= 4:
        return 'low'
    elif mark <= 8:
        return 'medium'
    else:
        return 'high'

enhanced_df['mark_category'] = enhanced_df['marks'].apply(categorize_marks)

# Prepare data for modeling
X = enhanced_df[['year', 'chapter', 'subject', 'chapter_avg_mark', 'chapter_frequency', 
               'chapter_max_mark', 'chapter_trend', 'recent_importance', 'had_high_marks',
               'chapter_consistency']]
y = enhanced_df['marks']
y_cat = enhanced_df['mark_category']

# Create pipeline for preprocessing with imputer for numeric features
categorical_features = ['chapter', 'subject']
numeric_features = ['year', 'chapter_avg_mark', 'chapter_frequency', 'chapter_max_mark', 
                  'chapter_trend', 'recent_importance', 'had_high_marks',
                  'chapter_consistency']

# Add imputer to handle any remaining NaN values
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
_, _, y_cat_train, y_cat_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# Define models to evaluate with better hyperparameters
print("\nTraining multiple models for comparison...")
models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42),
    'SVM': SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale'),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, 
                                               min_samples_split=5, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
}

# Evaluate models using cross-validation
results = {}
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"Training {name}...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Cross-validation scores
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    
    results[name] = {
        'Mean RMSE': rmse_scores.mean(),
        'Std RMSE': rmse_scores.std()
    }
    
    # Fit model on entire training set
    pipeline.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Store results
    results[name].update({
        'Test MSE': mse,
        'Test RMSE': rmse,
        'Test R²': r2,
        'Test MAE': mae
    })
    
    # For confusion matrix, convert predictions to categories
    y_pred_cat = np.array([categorize_marks(mark) for mark in y_pred])
    
    # Calculate and store confusion matrix
    cm = confusion_matrix(y_cat_test, y_pred_cat, labels=['low', 'medium', 'high'])
    results[name]['confusion_matrix'] = cm
    results[name]['y_pred'] = y_pred
    results[name]['y_pred_cat'] = y_pred_cat

# Display model performance comparison
results_df = pd.DataFrame({name: {k: v for k, v in values.items() if k != 'confusion_matrix' and k != 'y_pred' and k != 'y_pred_cat'} 
                          for name, values in results.items()}).T
print("\nModel Comparison:")
print(results_df)

# Select best model based on lowest test RMSE
best_model_name = results_df['Test RMSE'].idxmin()
print(f"\nBest model: {best_model_name} with Test RMSE = {results_df.loc[best_model_name, 'Test RMSE']:.4f}")

# Create visualization of model comparison
plt.figure(figsize=(12, 10))

# Plot 1: Test RMSE comparison
plt.subplot(2, 2, 1)
plt.bar(results_df.index, results_df['Test RMSE'], color='skyblue')
plt.title('Test RMSE by Model (Lower is Better)')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
for i, v in enumerate(results_df['Test RMSE']):
    plt.text(i, v + 0.05, f'{v:.2f}', ha='center')

# Plot 2: R² comparison
plt.subplot(2, 2, 2)
plt.bar(results_df.index, results_df['Test R²'], color='lightgreen')
plt.title('R² Score by Model (Higher is Better)')
plt.ylabel('R²')
plt.xticks(rotation=45)
for i, v in enumerate(results_df['Test R²']):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center')

# Plot 3: Model Stability (CV RMSE and Std)
plt.subplot(2, 2, 3)
x = np.arange(len(results_df.index))
width = 0.35
plt.bar(x - width/2, results_df['Mean RMSE'], width, label='Mean CV RMSE', color='coral')
plt.bar(x + width/2, results_df['Std RMSE'], width, label='Std CV RMSE', color='lightblue')
plt.title('Model Stability in Cross-Validation')
plt.xticks(x, results_df.index, rotation=45)
plt.legend()

# Plot 4: MAE comparison
plt.subplot(2, 2, 4)
plt.bar(results_df.index, results_df['Test MAE'], color='plum')
plt.title('Test MAE by Model (Lower is Better)')
plt.ylabel('MAE')
plt.xticks(rotation=45)
for i, v in enumerate(results_df['Test MAE']):
    plt.text(i, v + 0.05, f'{v:.2f}', ha='center')

plt.tight_layout()
plt.savefig('model_comparison.png')
print("Model comparison graph saved as 'model_comparison.png'")

# Plot confusion matrix for the best model
plt.figure(figsize=(10, 8))
best_cm = results[best_model_name]['confusion_matrix']
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['low', 'medium', 'high'], 
           yticklabels=['low', 'medium', 'high'])
plt.title(f'Confusion Matrix for {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")

# Print confusion matrix in text form
print(f"\nConfusion Matrix for {best_model_name}:")
print("Rows: Actual, Columns: Predicted (low, medium, high)")
print(best_cm)

# Calculate classification report
best_pred_cat = results[best_model_name]['y_pred_cat']
print("\nClassification Report:")
print(classification_report(y_cat_test, best_pred_cat))

# Function to manually allocate marks based on historical importance
def allocate_marks_with_variation(importance_df, total_marks=100):
    """
    Allocate marks to chapters with deliberate variation based on importance
    """
    result_df = importance_df.copy()
    
    # Get the number of chapters
    num_chapters = len(result_df)
    
    # Sort by importance
    sorted_df = result_df.sort_values('importance_score', ascending=False)
    
    # Initialize allocated marks
    sorted_df['allocated_marks'] = 0
    
    # Define mark tiers - create varied mark distribution
    # Top 10% get 12 marks
    # Next 20% get 8 marks
    # Next 30% get 4 marks
    # Bottom 40% get 0 marks
    
    # Calculate cutoffs
    top_tier_cutoff = int(num_chapters * 0.1)
    second_tier_cutoff = int(num_chapters * 0.3)  # top 10% + next 20%
    third_tier_cutoff = int(num_chapters * 0.6)   # top 10% + next 20% + next 30%
    
    # Ensure at least one chapter in each tier
    top_tier_cutoff = max(top_tier_cutoff, 1)
    second_tier_cutoff = max(second_tier_cutoff, top_tier_cutoff + 1)
    third_tier_cutoff = max(third_tier_cutoff, second_tier_cutoff + 1)
    
    # Allocate marks by tier
    indices = sorted_df.index.tolist()
    
    # Top tier: 12 marks
    for i in range(min(top_tier_cutoff, len(indices))):
        sorted_df.loc[indices[i], 'allocated_marks'] = 12
    
    # Second tier: 8 marks
    for i in range(top_tier_cutoff, min(second_tier_cutoff, len(indices))):
        sorted_df.loc[indices[i], 'allocated_marks'] = 8
    
    # Third tier: 4 marks
    for i in range(second_tier_cutoff, min(third_tier_cutoff, len(indices))):
        sorted_df.loc[indices[i], 'allocated_marks'] = 4
    
    # Calculate current total
    current_total = sorted_df['allocated_marks'].sum()
    
    # Adjust to get exactly 100 marks
    if current_total != total_marks:
        diff = total_marks - current_total
        
        if diff > 0:  # Need to add marks
            # Sort chapters by importance that have less than 12 marks
            potential_increase = sorted_df[sorted_df['allocated_marks'] < 12].sort_values('importance_score', ascending=False)
            
            remaining = diff
            for idx in potential_increase.index:
                if remaining <= 0:
                    break
                    
                current_marks = sorted_df.loc[idx, 'allocated_marks']
                # Add marks in increments of 4
                if current_marks == 0:
                    add_marks = 4
                elif current_marks == 4:
                    add_marks = 4  # 4 -> 8
                elif current_marks == 8:
                    add_marks = 4  # 8 -> 12
                else:
                    continue  # Skip if already at 12 or higher
                
                if add_marks <= remaining:
                    sorted_df.loc[idx, 'allocated_marks'] += add_marks
                    remaining -= add_marks
                    
        elif diff < 0:  # Need to remove marks
            # Sort chapters by importance (ascending) that have marks
            potential_decrease = sorted_df[sorted_df['allocated_marks'] > 0].sort_values('importance_score')
            
            remaining = abs(diff)
            for idx in potential_decrease.index:
                if remaining <= 0:
                    break
                    
                current_marks = sorted_df.loc[idx, 'allocated_marks']
                # Remove marks in increments of 4
                if current_marks == 4:
                    sub_marks = 4
                elif current_marks == 8:
                    sub_marks = 4  # 8 -> 4
                elif current_marks == 12:
                    sub_marks = 4  # 12 -> 8
                else:
                    continue
                
                if sub_marks <= remaining:
                    sorted_df.loc[idx, 'allocated_marks'] -= sub_marks
                    remaining -= sub_marks
    
    # Calculate number of questions
    sorted_df['no_of_questions'] = sorted_df['allocated_marks'] // 4
    
    # Calculate weightage percentage
    sorted_df['weightage_percent'] = (sorted_df['allocated_marks'] / total_marks) * 100
    
    return sorted_df

# Apply mark allocation for each subject
final_results = {}
for subject, importance_df in subject_importance.items():
    final_results[subject] = allocate_marks_with_variation(importance_df)

# Combine results into a single dataframe
result_rows = []
for subject, result_df in final_results.items():
    for chapter, row in result_df.iterrows():
        if row['allocated_marks'] > 0:  # Only include chapters with marks
            result_rows.append({
                'subject': subject,
                'topic': chapter,
                'no_of_questions': int(row['no_of_questions']),
                'marks': row['allocated_marks'],
                'weightage_percent': row['weightage_percent']
            })
        else:
            # Add rows with 0 marks for completeness
            result_rows.append({
                'subject': subject,
                'topic': chapter,
                'no_of_questions': 0,
                'marks': 0,
                'weightage_percent': 0
            })

# Create final dataframe
final_df = pd.DataFrame(result_rows)

# Sort by subject and marks (descending)
final_df = final_df.sort_values(['subject', 'marks', 'weightage_percent'], ascending=[True, False, False])

# Verify total marks per subject
print("\nTotal marks by subject:")
print(final_df.groupby('subject')['marks'].sum())

# Verify number of questions per subject
print("\nNumber of questions by subject:")
print(final_df.groupby('subject')['no_of_questions'].sum())

# Save results to CSV
final_df.to_csv('topic_weightage_prediction_2025.csv', index=False)
print("\nPredictions saved to 'topic_weightage_prediction_2025.csv'")

# Visualize the predictions
plt.figure(figsize=(15, 10))

# Create a horizontal bar chart for each subject
subjects = final_df['subject'].unique()
for i, subject in enumerate(subjects):
    plt.subplot(len(subjects), 1, i+1)
    
    # Get data for this subject
    subject_df = final_df[final_df['subject'] == subject]
    subject_df = subject_df[subject_df['marks'] > 0]  # Only show topics with marks
    subject_df = subject_df.sort_values('marks', ascending=True)  # Ascending for horizontal bars
    
    # Create horizontal bar chart
    bars = plt.barh(subject_df['topic'], subject_df['marks'], color='skyblue')
    
    # Add labels and title
    plt.title(f'{subject.upper()} - Topic Weightage')
    plt.xlabel('Marks')
    
    # Add marks as text labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.0f}', ha='left', va='center')
    
    # Adjust y-axis to fit all topics
    plt.tight_layout()

plt.savefig('topic_weightage_visualization.png')
print("Topic weightage visualization saved as 'topic_weightage_visualization.png'")

# Print the results
for subject in final_df['subject'].unique():
    subject_df = final_df[final_df['subject'] == subject]
    print(f"\n{subject.upper()} TOPIC WEIGHTAGE:")
    print("="*80)
    print(f"{'Topic':<40} | {'No. of Questions':<15} | {'Marks':<10} | {'Weightage (%)':<15}")
    print("-"*80)
    
    for _, row in subject_df.iterrows():
        print(f"{row['topic']:<40} | {row['no_of_questions']:<15} | {row['marks']:<10} | {row['weightage_percent']:.2f}")

print("\nPrediction and analysis completed!")