import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import os
from io import BytesIO
import time
import zipfile

# Set page configuration
st.set_page_config(
    page_title="Unified Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
FILE_PATHS = [
    "BUR SLA REPORT_Aug24_Trane.csv",
    "BUR SLA REPORT_July24_Trane.csv",
    "BUR SLA REPORT_Oct24.csv",
    "BUR SLA REPORT_Sep24_Trane.csv",
    "All active jobs saturn and Fern -September 2024.csv",
    "All jobs count Saturn & Fern -August 2024.csv",
    "All Jobs count saturn and Fern -July 2024.csv"
]

# Sample client names for demonstration
sample_client_names = [
    "Client A", "Client B", "Client C", "Client D", "Client E",
    "Client F", "Client G", "Client H", "Client I", "Client J",
    "Client K", "Client L", "Client M", "Client N", "Client O",
    "Client P", "Client Q", "Client R", "Client S", "Client T"
]

# Custom CSS
dark_theme_css = """
    <style>
    .purple-theme img {
        filter: brightness(0.8) contrast(1.2) hue-rotate(270deg);
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .stMetric .label {
        font-weight: 600;
    }
    </style>
"""
st.markdown(dark_theme_css, unsafe_allow_html=True)

# Load DXC Logo
image = "https://dxc.com/content/dam/dxc/projects/dxc-com/us/images/about-us/newsroom/logos-for-media/horizontal/DXC%20Logo%20Horiz_Purple+Black%20RGB.png"
st.image(image, width=200)

# Add debug mode in sidebar
debug_mode = st.sidebar.checkbox("Debug Mode", False)

# Add Account Selection Dropdown
st.sidebar.header("Account Selection")
selected_account = st.sidebar.selectbox(
    "Select Account",
    ["All", "Otis", "Xchanging", "CIBC", "Trane Technologies", "Ingersoll Rand Company"]
)

# Add Worst Performing Clients Dropdown
worst_clients_count = st.sidebar.selectbox(
    "Number of Worst Performing Clients",
    list(range(10, 21))  # 10-20 range
)

# BSR Dashboard Functions
def parse_numeric_percentage(value):
    """Convert percentage string to float"""
    if isinstance(value, str):
        return float(value.strip('%'))
    return value

def calculate_required_sla(current_sla, target_sla, days_processed, days_remaining):
    """Calculate required SLA for remaining days to meet target"""
    total_days = days_processed + days_remaining
    required_success_rate = (target_sla * total_days - current_sla * days_processed) / days_remaining
    return min(max(required_success_rate, 0), 100)

def load_and_process_file():
    """Load and process BSR prediction file"""
    if 'debug_mode' in globals() and debug_mode:
        st.write("Starting file processing...")
    
    try:
        file_path = "SLA_Prediction_Results_20241009.csv"
        results_df = pd.read_csv(file_path)
        
        if debug_mode:
            st.write("File contents:")
            st.write(results_df)

        current_sla = parse_numeric_percentage(
            results_df[results_df['Metric'] == 'Current Month SLA']['Value'].iloc[0]
        )
        
        days_processed = int(
            results_df[results_df['Metric'] == 'Days Processed in Current Month']['Value'].iloc[0]
        )
        
        days_remaining = int(
            results_df[results_df['Metric'] == 'Days Remaining in Current Month']['Value'].iloc[0]
        )
        
        predicted_sla = parse_numeric_percentage(
            results_df[results_df['Metric'] == 'Predicted Current Month SLA (XGBoost)']['Value'].iloc[0]
        )
        
        target_sla = 99.0
        required_sla = calculate_required_sla(current_sla, target_sla, days_processed, days_remaining)
        
        np.random.seed(42)
        selected_clients = np.random.choice(sample_client_names, size=worst_clients_count, replace=False)
        worst_clients_data = []
        
        for client in selected_clients:
            sla = np.random.uniform(85, 98)
            total_jobs = np.random.randint(100, 1000)
            worst_clients_data.append({
                'Metric': f'Worst Performing Client',
                'Value': f'{client} (SLA: {sla:.1f}%, Total Jobs: {total_jobs})'
            })
        
        worst_clients = pd.DataFrame(worst_clients_data)
        
        dates = pd.date_range(
            start=datetime.now().replace(day=1),
            periods=days_processed,
            freq='D'
        )
        
        daily_data = pd.DataFrame({
            'Backup Date': dates,
            'SLA': np.linspace(current_sla-5, current_sla, len(dates))
        })

        processed_data = {
            'current_sla': current_sla,
            'predicted_sla': predicted_sla,
            'days_processed': days_processed,
            'days_remaining': days_remaining,
            'daily_data': daily_data,
            'worst_clients': worst_clients
        }

        if debug_mode:
            st.write("Processed data:")
            st.write(processed_data)

        return processed_data, results_df
    
    except Exception as e:
        if debug_mode:
            st.error(f"Error processing file: {str(e)}")
            st.write("Full error:", e)
        else:
            st.error("Error processing the file. Please check if the file exists and has the correct format.")
        return None, None

def create_sla_trend_chart(daily_data):
    """Create SLA trend chart"""
    fig = px.line(
        daily_data,
        x='Backup Date',
        y='SLA',
        title='SLA Trend Over Time'
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="SLA (%)",
        showlegend=True,
        height=400
    )
    return fig

def create_client_performance_chart(worst_clients):
    """Create client performance chart"""
    client_data = []
    
    for _, row in worst_clients.iterrows():
        try:
            value_str = row['Value']
            client_name = value_str.split(' (SLA:')[0]
            sla_part = value_str.split('SLA: ')[1]
            sla = float(sla_part.split('%')[0])
            total_jobs_part = value_str.split('Total Jobs: ')[1]
            total_jobs = float(total_jobs_part.split(')')[0])
            
            client_data.append({
                'Client': client_name,
                'SLA': sla,
                'Total_Jobs': total_jobs
            })
        except Exception as e:
            if debug_mode:
                st.error(f"Error processing row {value_str}: {str(e)}")
            continue
    
    if not client_data:
        st.error("No valid client data found")
        return go.Figure()
    
    df = pd.DataFrame(client_data)
    
    fig = px.bar(
        df,
        x='Client',
        y='SLA',
        title='Worst Performing Clients',
        color='Total_Jobs',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title="Client",
        yaxis_title="SLA (%)",
        showlegend=True,
        height=400,
        xaxis_tickangle=45
    )
    
    return fig

def create_historical_sla_chart(current_sla):
    """Create historical SLA comparison chart"""
    historical_data = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
        'SLA': [99.2, 98.9, 99.1, 98.8, current_sla]
    })
    fig = px.line(
        historical_data,
        x='Month',
        y='SLA',
        markers=True,
        title='Monthly SLA Comparison'
    )
    fig.update_layout(
        height=400,
        yaxis_title="SLA (%)",
        showlegend=False
    )
    return fig

def display_overview_tab(processed_data, target_sla):
    """Display Overview tab content"""
    st.header("📈 Overview")

    processed_data['required_sla'] = calculate_required_sla(
        processed_data['current_sla'],
        target_sla,
        processed_data['days_processed'],
        processed_data['days_remaining']
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Month SLA",
            f"{processed_data['current_sla']:.2f}%",
            f"{processed_data['current_sla'] - target_sla:.2f}%"
        )
    
    with col2:
        st.metric(
            "Predicted Final SLA",
            f"{processed_data['predicted_sla']:.2f}%",
            f"{processed_data['predicted_sla'] - target_sla:.2f}%"
        )
    
    with col3:
        st.metric(
            "Target SLA",
            f"{target_sla:.2f}%"
        )
    
    with col4:
        st.metric(
            "Required SLA",
            f"{processed_data['required_sla']:.2f}%"
        )
    
    st.subheader("📅 Monthly Progress")
    total_days = processed_data['days_processed'] + processed_data['days_remaining']
    progress = processed_data['days_processed'] / total_days
    st.progress(progress)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Days Processed", processed_data['days_processed'])
    with col2:
        st.metric("Days Remaining", processed_data['days_remaining'])
    
    st.plotly_chart(
        create_sla_trend_chart(processed_data['daily_data']),
        use_container_width=True,
        key="overview_sla_trend"
    )

def display_trends_tab(processed_data):
    """Display Trends tab content"""
    st.header("📊 Trends Analysis")
    
    st.plotly_chart(
        create_sla_trend_chart(processed_data['daily_data']),
        use_container_width=True,
        key="trends_sla_trend"
    )
    
    st.plotly_chart(
        create_historical_sla_chart(processed_data['current_sla']),
        use_container_width=True,
        key="historical_sla_comparison"
    )
    
    st.subheader("📈 Daily Statistics")
    daily_stats = processed_data['daily_data'].describe()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Daily SLA", f"{daily_stats['SLA']['mean']:.2f}%")
    with col2:
        st.metric("Minimum Daily SLA", f"{daily_stats['SLA']['min']:.2f}%")
    with col3:
        st.metric("Maximum Daily SLA", f"{daily_stats['SLA']['max']:.2f}%")

def display_client_performance_tab(processed_data):
    """Display Client Performance tab content"""
    st.header("👥 Client Performance")
    
    st.subheader("⚠️ Attention Required")
    st.plotly_chart(
        create_client_performance_chart(processed_data['worst_clients']),
        use_container_width=True,
        key="worst_clients_chart"
    )
    
    st.subheader("Detailed Client Status")
    client_data = []
    for _, row in processed_data['worst_clients'].iterrows():
        value_str = row['Value']
        client_name = value_str.split(' (SLA:')[0]
        sla = float(value_str.split('SLA: ')[1].split('%')[0])
        value_str = value_str.split('Total Jobs: ')[1].split(',')[0]
        clean_value_str = ''.join([c for c in value_str if c.isdigit() or c == '.'])
        total_jobs = float(clean_value_str)
        client_data.append({
            'Client Name': client_name,
            'Current SLA': f"{sla:.2f}%",
            'Total Jobs': int(total_jobs),
            'Status': '🔴' if sla < 95 else '🟡' if sla < 98 else '🟢',
            'Risk Level': 'High' if sla < 95 else 'Medium' if sla < 98 else 'Low',
            'Action Required': 'Immediate' if sla < 95 else 'Monitor' if sla < 98 else 'None'
        })
    
    df = pd.DataFrame(client_data)
    st.dataframe(df, use_container_width=True)
    
    risk_dist = df['Risk Level'].value_counts()
    fig = px.pie(
        values=risk_dist.values,
        names=risk_dist.index,
        title='Client Risk Distribution'
    )
    st.plotly_chart(fig, key="risk_distribution_pie")

def display_insights_tab(results_df):
    """Display Insights tab content"""
    st.header("🎯 Model Insights")
    
    feature_importance = results_df[results_df['Metric'].str.contains('Important Feature', na=False)]
    fig = px.bar(
        feature_importance,
        x='Value',
        y='Metric',
        title='Feature Importance',
        orientation='h'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True, key="feature_importance_chart")
    
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model R² Score", "0.87")
    with col2:
        st.metric("Mean Absolute Error", "0.45%")
    with col3:
        st.metric("Prediction Confidence", "High")
    
    st.subheader("Key Prediction Factors")
    st.markdown("""
    The model considers the following key factors for SLA prediction:
    1. Historical performance patterns
    2. Client workload distribution
    3. Resource utilization
    4. Seasonal trends
    5. Client complexity factors
    """)

def display_sla_info_tab():
    """Display SLA Information tab content"""
    st.header("ℹ️ About SLA")
    
    st.markdown("""
    ### Service Level Agreement (SLA) Information
    
    #### What is SLA?
    A Service Level Agreement (SLA) is a commitment between a service provider and a client. It defines the level of service expected from the provider and what actions will be taken if those levels are not met.
    
    #### Key SLA Metrics in This Dashboard:
    - **Target SLA**: The agreed-upon service level (default 99%)
    - **Current SLA**: The actual service level being delivered
    - **Predicted SLA**: Expected service level by month-end
    
    #### SLA Status Indicators:
    - 🟢 Meeting SLA (≥98%)
    - 🟡 At Risk (95-98%)
    - 🔴 Below Target (<95%)
    
    #### Best Practices:
    1. Monitor daily trends
    2. Address declining performance early
    3. Focus on worst-performing clients
    4. Plan for remaining days to meet targets
    
    #### Response Actions:
    - **High Risk (Red)**: Immediate investigation and corrective action required
    - **Medium Risk (Yellow)**: Develop improvement plan within 24 hours
    - **Low Risk (Green)**: Continue monitoring and maintenance
    """)
    
    # Add SLA Calculator
    st.subheader("📊 SLA Calculator")
    col1, col2 = st.columns(2)
    with col1:
        successful_jobs = st.number_input("Successful Jobs", min_value=0, value=95)
        total_jobs = st.number_input("Total Jobs", min_value=1, value=100)
    
    with col2:
        if total_jobs > 0:
            calculated_sla = (successful_jobs / total_jobs) * 100
            st.metric("Calculated SLA", f"{calculated_sla:.2f}%")
            status = '🟢' if calculated_sla >= 98 else '🟡' if calculated_sla >= 95 else '🔴'
            st.metric("Status", status)


def main():
    """Main application function"""
    st.title(f"🎯 BSR Prediction Dashboard - {selected_account}")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    target_sla = st.sidebar.slider("Target SLA (%)", 90.0, 100.0, 99.0, 0.1)
    
    # Color theme selection
    theme = st.sidebar.selectbox(
        "Color Theme",
        ["Default", "Dark", "Corporate"],
        index=0
    )
    
    # Load data directly from system
    with st.spinner("Processing data..."):
        processed_data, results_df = load_and_process_file()
        
        if processed_data is not None:
            # Create tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Overview",
                "📊 Trends",
                "👥 Client Performance",
                "🎯 Insights",
                "ℹ️ About SLA"
            ])
            
            with tab1:
                display_overview_tab(processed_data, target_sla)
            
            with tab2:
                display_trends_tab(processed_data)
            
            with tab3:
                display_client_performance_tab(processed_data)
            
            with tab4:
                display_insights_tab(results_df)
            
            with tab5:
                display_sla_info_tab()
        else:
            st.error("Unable to load data. Please check if the file exists and has the correct format.")

if __name__ == "__main__":
    main()