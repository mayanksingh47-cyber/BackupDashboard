import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from datetime import datetime
import logging
import os

# Set the target SLA here
TARGET_SLA = 99.0  # You can easily edit this value

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_process_data(directory):
    logging.info(f"Loading data from {directory}")
    all_data = []
    current_month_data = None
    current_month = datetime.now().month
    total_rows_processed = 0

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)
            
            rows_in_file = len(data)
            total_rows_processed += rows_in_file
            logging.info(f"Processed {rows_in_file} rows from file: {filename}")
            
            if 'Backup Day Month' not in data.columns or 'Backup Day Day' not in data.columns:
                logging.error(f"'Backup Day Month' or 'Backup Day Day' column is missing in {filename}")
                continue

            # Create a datetime column from Backup Day Month and Backup Day Day
            data['Backup Date'] = pd.to_datetime(data['Backup Day Month'].astype(str) + ' ' + data['Backup Day Day'].astype(str), format='%m %d', errors='coerce')
            data['Backup Date'] = data['Backup Date'].apply(lambda x: x.replace(year=datetime.now().year))

            data = data.dropna(subset=['Backup Date'])
            if data.empty:
                logging.error(f"All rows in {filename} were dropped due to invalid dates.")
                continue

            # Check if this file contains data for the current month
            if any(data['Backup Date'].dt.month == current_month):
                if current_month_data is None:
                    current_month_data = data[data['Backup Date'].dt.month == current_month]
                else:
                    current_month_data = pd.concat([current_month_data, data[data['Backup Date'].dt.month == current_month]])
            
            # Add non-current month data to all_data
            previous_month_data = data[data['Backup Date'].dt.month != current_month]
            if not previous_month_data.empty:
                all_data.append(previous_month_data)

    if not all_data:
        logging.error("No valid data found in the previous months' files.")
        return None, None

    if current_month_data is None:
        logging.error("No valid data found for the current month.")
        return None, None

    previous_months_data = pd.concat(all_data, ignore_index=True)
    
    logging.info(f"Total rows processed: {total_rows_processed}")
    logging.info(f"Rows in previous months data: {len(previous_months_data)}")
    logging.info(f"Rows in current month data: {len(current_month_data)}")
    logging.info(f"Current month identified as: {current_month}")

    return previous_months_data, current_month_data

def calculate_sla(data):
    data['day'] = data['Backup Date'].dt.day
    data['month'] = data['Backup Date'].dt.month
    data['day_of_week'] = data['Backup Date'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    data['Job_Volume'] = data.groupby(['Backup Date', 'Client', 'Server'])['Status'].transform('count')
    data['Success_Rate'] = data.groupby(['Backup Date', 'Client', 'Server'])['Status'].transform(lambda x: (x == 'Success').mean())
    
    grouped_data = data.groupby(['Backup Date', 'Client', 'Server'])
    sla_daily = grouped_data.apply(lambda x: pd.Series({
        'SLA': ((x['Status'] == 'Success') | (x['Status'] == 'Partial')).mean() * 100,
        'Total_Jobs': len(x),
        'Success_Rate': (x['Status'] == 'Success').mean(),
        'Partial_Rate': (x['Status'] == 'Partial').mean(),
        'Failure_Rate': (x['Status'] == 'Failed').mean()
    }), include_groups=False).reset_index()
    
    sla_daily = sla_daily.merge(
        data[['Backup Date', 'Client', 'Server', 'day', 'month', 'day_of_week', 'is_weekend']].drop_duplicates(),
        on=['Backup Date', 'Client', 'Server'], how='left'
    )
    
    return sla_daily

def analyze_client_performance(data):
    logging.info("Analyzing client performance")
    client_performance = data.groupby('Client').agg({
        'SLA': 'mean',
        'Total_Jobs': 'sum',
        'Success_Rate': 'mean',
        'Failure_Rate': 'mean'
    }).sort_values('SLA')
    
    worst_performers = client_performance[client_performance['SLA'] < TARGET_SLA].head(5)
    logging.info(f"Top 5 Worst Performing Clients (SLA < {TARGET_SLA}%):")
    logging.info(worst_performers)
    
    return worst_performers

def train_xgboost_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    model = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    logging.info(f"Best parameters: {grid_search.best_params_}")
    
    train_rmse = np.sqrt(mean_squared_error(y_train, best_model.predict(X_train)))
    test_rmse = np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))
    logging.info(f"Train RMSE: {train_rmse:.2f}")
    logging.info(f"Test RMSE: {test_rmse:.2f}")
    
    return best_model, grid_search.best_params_

def predict_current_month_sla(previous_months_data, current_month_data, target_sla):
    logging.info("Predicting current month SLA using XGBoost")
    features = ['day', 'month', 'day_of_week', 'is_weekend', 'Total_Jobs', 'Success_Rate', 'Partial_Rate', 'Failure_Rate']
    
    # Change made here to include current month data for model training
    X = pd.concat([previous_months_data[features], current_month_data[features]], ignore_index=True)
    y = pd.concat([previous_months_data['SLA'], current_month_data['SLA']], ignore_index=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model, best_params = train_xgboost_model(X_scaled, y)

    # Feature importance
    feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    logging.info("Feature Importance:")
    logging.info(feature_importance)

    current_month_sla = current_month_data['SLA'].mean()
    days_in_month = current_month_data['Backup Date'].dt.daysinmonth.iloc[0]
    days_processed = current_month_data['day'].max()
    days_remaining = days_in_month - days_processed

    logging.info(f"Current month SLA: {current_month_sla:.2f}%")
    logging.info(f"Days processed: {days_processed}")
    logging.info(f"Days remaining: {days_remaining}")

    if days_remaining > 0:
        required_sla = (target_sla * days_in_month - current_month_sla * days_processed) / days_remaining
        logging.info(f"Required SLA for remaining days to meet {target_sla}% target: {required_sla:.2f}%")
    else:
        required_sla = None

    # Predict SLA for remaining days
    remaining_days = pd.DataFrame({
        'day': range(days_processed + 1, days_in_month + 1),
        'month': current_month_data['month'].iloc[0],
        'day_of_week': [(current_month_data['Backup Date'].iloc[0].replace(day=d).weekday()) for d in range(days_processed + 1, days_in_month + 1)],
        'is_weekend': [int(current_month_data['Backup Date'].iloc[0].replace(day=d).weekday() in [5, 6]) for d in range(days_processed + 1, days_in_month + 1)],
        'Total_Jobs': current_month_data['Total_Jobs'].mean(),
        'Success_Rate': current_month_data['Success_Rate'].mean(),
        'Partial_Rate': current_month_data['Partial_Rate'].mean(),
        'Failure_Rate': current_month_data['Failure_Rate'].mean()
    })

    # Fit the scaler on all available data (previous and current months)
    X_all = pd.concat([previous_months_data[features], current_month_data[features]], ignore_index=True)
    scaler.fit(X_all)

    # Now scale the remaining days' data
    remaining_days_scaled = scaler.transform(remaining_days[features])

    # Predict SLA for remaining days using the model trained on all data
    predicted_sla = model.predict(remaining_days_scaled)
    
    # Add predicted SLA to remaining_days DataFrame
    remaining_days['predicted_sla'] = predicted_sla

    logging.info("Daily SLA predictions for remaining days:")
    for day, sla in zip(remaining_days['day'], predicted_sla):
        logging.info(f"Day {day}: {sla:.2f}%")
    
    final_predicted_sla = (current_month_sla * days_processed + predicted_sla.sum()) / days_in_month

    logging.info(f"Predicted current month SLA (XGBoost): {final_predicted_sla:.2f}%")

    return final_predicted_sla, current_month_sla, days_processed, days_remaining, required_sla, best_params, feature_importance, remaining_days


def compile_results(previous_months_data, current_month_data, target_sla):
    logging.info("Compiling results")
    results = []

    # Predict current month SLA using XGBoost
    xgb_predicted_sla, current_month_sla, days_processed, days_remaining, required_sla, best_params, feature_importance, remaining_days = predict_current_month_sla(previous_months_data, current_month_data, target_sla)

    # XGBoost prediction
    results.append({
        'Metric': 'Predicted Current Month SLA (XGBoost)',
        'Value': f'{xgb_predicted_sla:.2f}%'
    })

    # Current month SLA
    results.append({
        'Metric': 'Current Month SLA',
        'Value': f'{current_month_sla:.2f}%'
    })

    # Days processed and remaining
    results.append({
        'Metric': 'Days Processed in Current Month',
        'Value': days_processed
    })
    results.append({
        'Metric': 'Days Remaining in Current Month',
        'Value': days_remaining
    })

    # Required SLA for remaining days
    if required_sla is not None:
        results.append({
            'Metric': f'Required SLA for Remaining Days (to meet {target_sla}% target)',
            'Value': f'{required_sla:.2f}%'
        })

    # Best parameters
    results.append({
        'Metric': 'Best XGBoost Parameters',
        'Value': str(best_params)
    })

    # Top 3 important features
    for i, row in feature_importance.head(3).iterrows():
        results.append({
            'Metric': f'Important Feature #{i+1}',
            'Value': f"{row['feature']} (importance: {row['importance']:.4f})"
        })

    # Worst performing clients
    worst_performers = analyze_client_performance(previous_months_data)
    for i, (client, data) in enumerate(worst_performers.iterrows(), 1):
        results.append({
            'Metric': f'Worst Performing Client #{i}',
            'Value': f"{client} (SLA: {data['SLA']:.2f}%, Total Jobs: {data['Total_Jobs']}, Success Rate: {data['Success_Rate']:.2f})"
        })

    # Add remaining days predictions
    for index, row in remaining_days.iterrows():
        results.append({
            'Metric': f'Day {row["day"]} Prediction',
            'Value': f'{row["predicted_sla"]:.2f}%'
        })

    return pd.DataFrame(results)

# Main execution
if __name__ == "__main__":
    try:
        logging.info("Starting script execution")
        
        # Set the directory path
        directory_path = r'C:\python\Cleaned_data'
        
        # Load and process data
        previous_months_data, current_month_data = load_and_process_data(directory_path)
        
        if previous_months_data is None or current_month_data is None:
            logging.error("Failed to load and process data. Exiting script.")
            exit(1)
        
        # Calculate SLA for previous months and current month
        sla_previous_months = calculate_sla(previous_months_data)
        sla_current_month = calculate_sla(current_month_data)
        
        logging.info(f"Rows in SLA previous months data: {len(sla_previous_months)}")
        logging.info(f"Rows in SLA current month data: {len(sla_current_month)}")
        
        # Compile results
        results_df = compile_results(sla_previous_months, sla_current_month, TARGET_SLA)

        # Generate filename with current date
        current_date = datetime.now().strftime("%Y%m%d")
        results_filename = f"SLA_Prediction_Results_{current_date}.csv"

        # Save results to CSV
        results_df.to_csv(results_filename, index=False)
        logging.info(f"All results have been saved to {results_filename}")

        logging.info("Script execution completed successfully")
    except Exception as e:
        logging.error(f"An error occurred during script execution: {str(e)}")
        raise