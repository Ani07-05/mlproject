# JEE Chapter Weightage Prediction Model

## Overview
This project analyzes historical JEE (Joint Entrance Examination) data to predict chapter weightage for upcoming exams. Using advanced machine learning techniques and historical pattern analysis, it identifies which chapters are likely to have higher marks allocation in future JEE papers.

## Features
- **Historical Data Analysis**: Analyzes JEE exam patterns from 2002-2024 with higher weighting for recent years (2015-2024)
- **Multi-Model Comparison**: Evaluates multiple ML models (Random Forest, SVM, Gradient Boosting, XGBoost)
- **Chapter Importance Metrics**: Calculates comprehensive importance scores based on:
  - Average marks
  - Frequency of appearance
  - Maximum marks
  - Historical variation
  - Recent trends
- **Visualization**: Generates insightful graphs for model performance and chapter weightage predictions
- **Structured Predictions**: Provides practical marks allocation with proper question distribution

## Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- pickle

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ani07-05/mlproject.git
cd jee-weightage-prediction
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset
The model requires a dataset with historical JEE question information. The dataset should include:
- Year of exam
- Subject (Physics, Chemistry, Mathematics)
- Chapter/topic
- Marks allocated

Place your dataset file as `corrected_dataset.csv` in the project directory.

## Usage

1. Prepare your dataset as described above.

2. Run the prediction model:
```bash
python chapterwithyearwise.py
```

3. Review the outputs:
   - `topic_weightage_prediction_2025.csv`: Contains detailed predictions
   - `model_comparison.png`: Visual comparison of model performance
   - `confusion_matrix.png`: Confusion matrix for the best model
   - `topic_weightage_visualization.png`: Visual representation of predicted weightage

## Output Interpretation

The model produces a detailed breakdown of:
- Subject-wise marks allocation
- Chapter-wise question distribution
- Expected number of questions per chapter
- Weightage percentage of each chapter

## Model Methodology

1. **Data Preprocessing**: Handles missing values and converts features appropriately
2. **Feature Engineering**: Creates features like historical average, frequency, max marks, and trend analysis
3. **Model Training**: Trains and evaluates multiple regression models
4. **Mark Allocation**: Uses a tiered approach to allocate marks based on importance scores:
   - Top 10% chapters get 12 marks
   - Next 20% get 8 marks
   - Next 30% get 4 marks
   - Bottom 40% get 0 marks
5. **Adjustment**: Fine-tunes the allocation to match the exam's total marks (100)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to improve the model.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Thanks to all contributors who have helped improve this prediction model

