import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix)
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


def load_data(file_path="e:/IML/mlproject/cleaned_chapter_weightage.csv"):
    """Load and prepare the dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def prepare_data(df):
    """Prepare data for modeling"""
    # Create features from subject and chapter
    X = df[['subject', 'chapter']]
    
    # Create target variable - for simplicity we'll create a synthetic one if not available
    if 'marks' in df.columns:
        y = df['marks']
    else:
        # Create synthetic target based on topic frequency as a baseline
        topic_importance = df.groupby(['subject', 'chapter']).size().reset_index(name='frequency')
        df = pd.merge(df, topic_importance, on=['subject', 'chapter'], how='left')
        # Normalize to typical JEE marking scheme (4-mark questions)
        df['synthetic_marks'] = np.round(df['frequency'] / df['frequency'].max() * 16)
        y = df['synthetic_marks']
        print("Created synthetic target variable based on topic frequency")

    # Encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X)
    feature_names = encoder.get_feature_names_out(['subject', 'chapter'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    return X_encoded, y, X_train, X_test, y_train, y_test, encoder, feature_names


def build_models():
    """Define and return a dictionary of models for comparison"""
    return {
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42),
        'Linear Regression': LinearRegression(),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
        'Support Vector Machine': SVR(kernel='rbf', C=10, epsilon=0.2)
    }


def perform_cross_validation(models, X_encoded, y):
    """Perform k-fold cross-validation on all models"""
    print("Performing k-fold cross-validation...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_encoded, y, cv=kfold, scoring='neg_mean_absolute_error')
        cv_results[name] = -cv_scores.mean()  # Convert negative MAE to positive
        print(f"{name} - Cross-validation MAE: {-cv_scores.mean():.4f} (std: {cv_scores.std():.4f})")
    
    return cv_results, kfold


def evaluate_models(models, X_train, X_test, y_train, y_test, cv_results):
    """Train and evaluate all models"""
    print("\nTraining and evaluating models...")
    model_metrics = []
    best_model = None
    best_model_name = None
    lowest_mae = float('inf')

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate regression metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Store metrics
        model_metrics.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'CV_MAE': cv_results[name]
        })
        
        # Track best model based on cross-validation score
        if cv_results[name] < lowest_mae:
            lowest_mae = cv_results[name]
            best_model = model
            best_model_name = name

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(model_metrics)
    print("\nModel Comparison:")
    print(metrics_df.to_string(index=False))
    print(f"\nBest model: {best_model_name} with Cross-Validation MAE: {lowest_mae:.4f}")
    
    return best_model, best_model_name, metrics_df


def create_confusion_matrix(best_model, best_model_name, X_test, y_test):
    """Generate and plot confusion matrix"""
    print("\nGenerating classification metrics...")
    y_pred_best = best_model.predict(X_test)

    # Convert continuous values to discrete bins (standard JEE marking)
    y_test_binned = np.round(y_test / 4) * 4
    y_pred_binned = np.round(y_pred_best / 4) * 4

    # Create confusion matrix
    cm = confusion_matrix(y_test_binned, y_pred_binned)
    
    # Get unique class labels
    class_labels = sorted(np.unique(np.concatenate([y_test_binned, y_pred_binned])))
    print(f"Class labels (mark categories): {class_labels}")

    # Plot confusion matrix with proper labels
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=class_labels, 
                yticklabels=class_labels)
    plt.xlabel('Predicted Marks')
    plt.ylabel('Actual Marks')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.savefig('e:/IML/mlproject/confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # Calculate classification metrics
    unique_bins = np.unique(np.concatenate([y_test_binned, y_pred_binned]))
    bin_mapping = {val: idx for idx, val in enumerate(unique_bins)}
    y_test_class = np.array([bin_mapping[val] for val in y_test_binned])
    y_pred_class = np.array([bin_mapping[val] for val in y_pred_binned])

    accuracy = accuracy_score(y_test_class, y_pred_class)
    precision = precision_score(y_test_class, y_pred_class, average='weighted', zero_division=0)
    recall = recall_score(y_test_class, y_pred_class, average='weighted')
    f1 = f1_score(y_test_class, y_pred_class, average='weighted')

    print(f"\nClassification Metrics for {best_model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


def analyze_feature_importance(best_model, best_model_name, feature_names):
    """Analyze and visualize feature importance for tree-based models"""
    if best_model_name in ["Random Forest", "Gradient Boosting"]:
        print("\nAnalyzing feature importance...")
        plt.figure(figsize=(14, 10))
        
        importance = best_model.feature_importances_
        # Sort feature importance in descending order
        idx = importance.argsort()[-20:]  # Get top 20 features
        
        plt.barh(range(len(idx)), importance[idx], align='center')
        plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Feature Importance - {best_model_name}')
        plt.tight_layout()
        plt.savefig('e:/IML/mlproject/feature_importance.png')
        print("Feature importance visualization saved as 'feature_importance.png'")


def make_predictions(best_model, encoder, df):
    """Generate predictions for JEE 2025"""
    print("\nGenerating predictions for JEE 2025...")

    # Get unique subject-chapter combinations
    unique_combinations = df[['subject', 'chapter']].drop_duplicates().copy()
    unique_combinations['year'] = 2025

    # Transform data
    future_encoded = encoder.transform(unique_combinations[['subject', 'chapter']])

    # Make predictions
    predictions = best_model.predict(future_encoded)

    # Ensure predictions are multiples of 4 (standard JEE marking)
    predictions = np.round(predictions / 4) * 4

    # Create final prediction dataframe
    future_data = unique_combinations.copy()
    future_data['predicted_marks'] = predictions

    # Sort by predicted marks (descending) to show most important topics first
    future_data = future_data.sort_values('predicted_marks', ascending=False)

    # Save predictions
    output_path = 'e:/IML/mlproject/optimized_predictions.csv'
    future_data.to_csv(output_path, index=False)
    print(f"Predictions for 2025 JEE saved successfully at {output_path}")
    
    return future_data


def analyze_subject_distribution(future_data):
    """Perform subject-wise analysis of predictions"""
    print("\nPerforming detailed subject analysis...")
    subject_analysis = future_data.groupby('subject')['predicted_marks'].agg(['mean', 'sum', 'count']).reset_index()
    subject_analysis = subject_analysis.sort_values('sum', ascending=False)

    print("\nSubject-wise Prediction Analysis:")
    print(subject_analysis)
    
    # Display top chapters by subject
    print("\nTop chapters by subject:")
    for subject in future_data['subject'].unique():
        top_chapters = future_data[future_data['subject'] == subject].head(5)
        print(f"\n{subject.upper()}:")
        for i, (_, row) in enumerate(top_chapters.iterrows(), 1):
            print(f"  {i}. {row['chapter']} - {row['predicted_marks']} marks")
    
    return subject_analysis


def check_overfitting(X, y, best_model_name):
    """Check for overfitting using learning curves and regularized models"""
    print("\nChecking for overfitting...")
    
    # Create regularized alternatives to Linear Regression
    regularized_models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    # Prepare the figure for learning curves
    plt.figure(figsize=(14, 10))
    
    # Generate learning curves for the standard Linear Regression
    if best_model_name == 'Linear Regression':
        train_sizes, train_scores, test_scores = learning_curve(
            LinearRegression(), X, y, cv=5, scoring='neg_mean_squared_error',
            train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)
        
        # Calculate mean and standard deviation
        train_scores_mean = -np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        # Plot learning curves
        plt.subplot(2, 2, 1)
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.title("Learning Curves (Linear Regression)")
        plt.xlabel("Training examples")
        plt.ylabel("Mean Squared Error")
        plt.legend(loc="best")
        
        # Compare with regularized models
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        position = 2
        
        for name, model in regularized_models.items():
            # Generate learning curves for regularized model
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=kfold, scoring='neg_mean_squared_error',
                train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)
            
            # Calculate mean and standard deviation
            train_scores_mean = -np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = -np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            
            # Plot learning curves for regularized model
            plt.subplot(2, 2, position)
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1, color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
            plt.title(f"Learning Curves ({name})")
            plt.xlabel("Training examples")
            plt.ylabel("Mean Squared Error")
            plt.legend(loc="best")
            
            position += 1
            
            # Evaluate model performance
            cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
            print(f"{name} - Cross-validation MAE: {-cv_scores.mean():.4f} (std: {cv_scores.std():.4f})")
    
    plt.tight_layout()
    plt.savefig('e:/IML/mlproject/overfitting_check.png')
    print("Overfitting visualization saved as 'overfitting_check.png'")
    
    return regularized_models


def main():
    """Main function to run the entire analysis pipeline"""
    print("JEE Topic Weightage Analysis Tool")
    print("=================================")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Prepare data
    X_encoded, y, X_train, X_test, y_train, y_test, encoder, feature_names = prepare_data(df)
    
    # Build models
    models = build_models()
    
    # Cross-validation
    cv_results, kfold = perform_cross_validation(models, X_encoded, y)
    
    # Evaluate models
    best_model, best_model_name, metrics_df = evaluate_models(
        models, X_train, X_test, y_train, y_test, cv_results)
    
    # Check for overfitting if the model is Linear Regression with unusually good performance
    if best_model_name == 'Linear Regression' and cv_results[best_model_name] < 0.01:
        print("\nWARNING: Linear regression model has suspiciously low MAE, checking for overfitting...")
        regularized_models = check_overfitting(X_encoded, y, best_model_name)
        
        # Get the best regularized model
        reg_cv_results = {}
        for name, model in regularized_models.items():
            cv_scores = cross_val_score(model, X_encoded, y, cv=kfold, scoring='neg_mean_absolute_error')
            reg_cv_results[name] = -cv_scores.mean()
        
        # Find best regularized model
        best_reg_name = min(reg_cv_results, key=reg_cv_results.get)
        best_reg_model = regularized_models[best_reg_name]
        best_reg_mae = reg_cv_results[best_reg_name]
        
        # If the best regularized model has reasonable error (not suspiciously low), use it instead
        if best_reg_mae > 0.01:
            print(f"\nSwitching to {best_reg_name} to avoid overfitting. Cross-validation MAE: {best_reg_mae:.4f}")
            best_model = best_reg_model
            best_model_name = best_reg_name
            
            # Update cv_results with the regularized model
            cv_results[best_reg_name] = best_reg_mae
            
            # Train the regularized model on the full training set
            best_model.fit(X_train, y_train)
    
    # Create confusion matrix
    create_confusion_matrix(best_model, best_model_name, X_test, y_test)
    
    # Analyze feature importance for tree-based models
    analyze_feature_importance(best_model, best_model_name, feature_names)
    
    # Make predictions for JEE 2025
    future_data = make_predictions(best_model, encoder, df)
    
    # Analyze subject distribution
    analyze_subject_distribution(future_data)
    
    # Create a visualization to illustrate the difference between training and test performance
    print("\nPlotting learning curves to visualize model performance...")
    plt.figure(figsize=(10, 6))
    model_to_plot = best_model
    
    train_sizes, train_scores, test_scores = learning_curve(
        model_to_plot, X_encoded, y, cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error',
        random_state=42)
    
    # Calculate mean and standard deviation
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot learning curves
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title(f"Learning Curves ({best_model_name})")
    plt.xlabel("Training examples")
    plt.ylabel("Mean Squared Error")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('e:/IML/mlproject/learning_curve.png')
    
    print("\nAnalysis complete! Check the output files for detailed predictions.")
    print(f"Final model used: {best_model_name}")
    if best_model_name in ['Ridge Regression', 'Lasso Regression', 'ElasticNet']:
        print("Regularized model was chosen to prevent overfitting.")
    elif best_model_name == 'Linear Regression' and cv_results[best_model_name] < 0.01:
        print("WARNING: Linear regression model may be overfitting. Review the learning curves.")


if __name__ == "__main__":
    main()