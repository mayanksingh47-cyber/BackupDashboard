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

# Set page configuration
st.set_page_config(
    page_title="BSR Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add this updated CSS section in the st.markdown at the beginning of the code
st.markdown("""
    <style>
    /* Theme-compatible styles */
    
    /* Remove top margin and padding */
    .block-container {
        padding: 1.86rem;
        margin-top: 0rem;
        max-width: 95%;
        margin-bottom: 0rem;
    }
    
    /* Dashboard title - theme compatible */
    .dashboard-title {
        font-size: 1.7rem !important;
        margin-bottom: 0.5rem !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: var(--text-color);
    }
    
    /* Custom tab styling - theme compatible */
    .stTabs {
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        border-radius: 2rem;
        background-color: var(--background-color);
    }
    
    /* Tab styling for both themes */
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0rem 1rem;
        border-radius: 2rem;
        gap: 0.5rem;
        transition: all 0.3s ease;
        color: var(--text-color) !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(111, 44, 145, 0.1);
        cursor: pointer;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #6f2c91;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 0 10px rgba(111, 44, 145, 0.3);
    }
    
    /* Progress bar - theme compatible */
    .stProgress > div > div {
        height: 1rem;
        border-radius: 0.5rem;
        background-color: var(--secondary-background-color);
    }
    
    /* Monthly Progress section - theme compatible */
    .monthly-progress {
        display: flex;
        flex-direction: column;
        gap: 0rem;
        background-color: var(--background-color);
    }
    
    .monthly-progress-metrics {
        display: flex;
        gap: 0rem;
        color: var(--text-color);
    }
    
    /* Floating footer - theme compatible */
    .floating-footer {
        position: fixed;
        right: 16px;
        bottom: 16px;
        z-index: 999;
        background-color: #6f2c91;
        color: white;
        padding: 8px 16px;
        border-radius: 4px;
        font-size: 14px;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .logo-container img {
    max-width: 110px !important; /* Enforce the new size */
    width: 120px !important; /* Explicit width to override other rules */
    height: auto !important; /* Maintain aspect ratio */
    filter: brightness(2) !important; /* Visibility in dark theme */
}
    
    /* Card styling for both themes */
    .custom-card {
        background-color: var(--background-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Table styling for both themes */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    
    .custom-table th,
    .custom-table td {
        padding: 0.75rem;
        border: 1px solid var(--border-color);
        color: var(--text-color);
    }
    
    /* Chart background for both themes */
    .plotly-chart {
        background-color: var(--background-color) !important;
    }
    
    /* Metric card styling for both themes */
    .metric-container {
        background-color: var(--background-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        color: var(--text-color);
    }
    
    /* Status indicators for both themes */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-green { background-color: #28a745; }
    .status-yellow { background-color: #ffc107; }
    .status-red { background-color: #dc3545; }
    </style>
""", unsafe_allow_html=True)

# Create two columns for the header area with adjusted ratios
left_col, right_col = st.columns([6, 1])

# Add Account Selection Dropdown
st.sidebar.header("Account Selection")
selected_account = st.sidebar.selectbox(
    "Select Account",
    [ "Trane Technologies", "Ingersoll Rand Company", "Otis", "Xchanging", "CIBC"]
)

dark_theme_css = """
    <style>
    .purple-theme img {
        filter: brightness(0.8) contrast(1.2) hue-rotate(270deg);
    }
    </style>
    """

# Insert the CSS to apply for the dark theme
st.markdown(dark_theme_css, unsafe_allow_html=True)

# Add the title with custom styling in the left column
with left_col:
    st.markdown(f'<h1 class="dashboard-title">🎯 BSR Prediction Dashboard - {selected_account}</h1>', unsafe_allow_html=True)

# Update the logo handling in the header section
with right_col:
    st.markdown(
        """
        <div class="logo-container">
            <img src="https://dxc.com/content/dam/dxc/projects/dxc-com/us/images/about-us/newsroom/logos-for-media/vertical/DXC%20Logo_Purple+Black%20RGB.png" 
                 alt="DXC Logo" 
                 style="max-width: 150px; height: auto;">
        </div>
        """,
        unsafe_allow_html=True
    )

# Add debug mode in sidebar
debug_mode = st.sidebar.checkbox("Debug Mode", False)

# Update the plotly chart themes to be compatible with both light and dark modes
def update_plot_theme(fig):
    """Update plot theme for compatibility with both light and dark modes"""
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='var(--text-color)',
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            zerolinecolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            zerolinecolor='rgba(128,128,128,0.2)'
        )
    )
    return fig


# Region Name selection with filtering based on account
def get_regions_for_account(account):
    # Define region mappings for each account
    account_regions = {
        "Trane Technologies": ["Americas", "APAC", "EMEA"],
        "Ingersoll Rand Company": ["Americas", "EMEA"],
        "Otis": ["Global", "Americas", "APAC"],
        "Xchanging": ["EMEA", "APAC"],
        "CIBC": ["Americas"]
    }
    return account_regions.get(account, ["Global"])  # Default to Global if account not found

# Get available regions based on selected account
available_regions = get_regions_for_account(selected_account)


# Add Worst Performing Hosts Dropdown
worst_clients_count = st.sidebar.selectbox(
    "Number of Worst Performing Hosts",
    list(range(5, 61, 5))  # Creates list [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
)
# Sample random client names for demonstration
sample_client_names = [
    "igrdssndc002", "Hpuxmgmt-bu01", "ccpvlndct307.ad.corp.global", "igrscmndc001.ad.corp.global", "IGRSCMNDC001.AD.CORP.GLOBAL", "spare009-bu1", "igrsnmnwh001.ad.corp.global", "igrdssndc007-3.ad.corp.global", "igrdssndc002.ad.corp.global", "laxeu18b", "IGRDSSNDC001.AD.CORP.GLOBAL", "igrdbsndc815-bu1", "bwd-tv-psfinprd-bu1", "IGRDSSNDC002.AD.CORP.GLOBAL", "TRNDCSP1-bu01", "igrdssndc001.ad.corp.global", "igrdssndc007-2.ad.corp.global", "ccpvlnwhavr01.ad.corp.global", "TRNDCSP4-bu01", "TRNDCSP5-bu01",
    "ukitwxubworaa01", "ukitwxudwvdd601", "ukitwxubwvd001", "ukimdlmawcvdp53", "ukimdlmawcvdp54", "ukimdlmawcvdp55", "ukimdlmawcvdp56", "ukitwlmawcvdp04", "ukimdlmawcvdu02", "ukitwxubwofsc01", "ukitwtobedge01", "ukitwxubedge01", "ukitwxubwadsd02", "ukitwxubwadsd04", "ukitwxubwc001", "ukitwxubwrdsd01", "ukitwlmawcvdp05", "ukitwaobwcmtt01", "ukitwaobwcmtu01", "ukitwxubworad01", 
    "orobua02.otis.com", "ofrard04.otis.com", "ofrgna1k", "ofrgna1d.otis.com", "ofrark09.otis.com", "ofrard03.otis.com", "ONLOMD01.otis.com", "onlomd01.otis.com", "OBEDIA0K", "oczbra0c", "OFRGNA1D.otis.com", "OATVAD0E.otis.com", "ochfra0g.otis.com", "OBEDIA0J", "oplwza08.otis.com", "oplwza07.otis.com", "OBEDIA0M", "ositrf02.otis.com", "OBEDIA0L", "oczbra0a.otis.com"
]

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
    """Load and process CSV file from system based on selected account"""
    if debug_mode:
        st.write("Starting file processing...")
    
    try:
        # Define file mapping for each account
        file_mapping = {
            "Trane Technologies": "SLA_Prediction_Results_20241009.csv",
            "Xchanging": "SLA_Prediction_Results_20241024.csv",
            "Otis": "SLA_Prediction_Results_20241010.csv",
            "CIBC": "Customer_SLA_Prediction_Results_20241013(1).csv",
            "Ingersoll Rand Company": "SLA_Prediction_Results_20241009.csv"  # Using default file
        }
        
        # Get the appropriate file path based on selected account
        file_path = file_mapping.get(selected_account)
        
        if debug_mode:
            st.write(f"Loading file: {file_path}")
        
        # Read the results file from system
        results_df = pd.read_csv(file_path)
        
        if debug_mode:
            st.write("File contents:")
            st.write(results_df)

        # Rest of the processing remains the same
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
        
        # Calculate required SLA for remaining days
        target_sla = 99.0  # Default target, can be updated from UI
        required_sla = calculate_required_sla(current_sla, target_sla, days_processed, days_remaining)
        
        # Extract worst performing hosts
        worst_clients = results_df[results_df['Metric'].str.contains('Worst Performing Client', na=False)]
        
        # Generate random worst performing hosts based on selection
        np.random.seed(42)  # For reproducibility
        selected_clients = np.random.choice(sample_client_names, size=worst_clients_count, replace=False)
        worst_clients_data = []
        
        for client in selected_clients:
            sla = np.random.uniform(85, 98)
            total_jobs = np.random.randint(100, 1000)
            worst_clients_data.append({
                'Metric': f'Worst Performing Hosts ',
                'Value': f'{client} (SLA: {sla:.1f}%, Total Jobs: {total_jobs})'
            })
        
        worst_clients = pd.DataFrame(worst_clients_data)
        
        # Create synthetic daily data for visualization
        dates = pd.date_range(
            start=datetime.now().replace(day=1),
            periods=days_processed,
            freq='D'
        )
        
        daily_data = pd.DataFrame({
            'Backup Date': dates,
            'SLA': np.linspace(current_sla-5, current_sla, len(dates))  # Synthetic SLA trend
        })

        # Structure the data for dashboard
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
    
    # Apply this to all your chart creation functions
def create_sla_trend_chart(daily_data):
    fig = px.line(
        daily_data,
        x='Backup Date',
        y='SLA',
        title='Daily Success Rate 🏆'
    )
    fig = update_plot_theme(fig)
    return fig

def create_sla_trend_chart(daily_data):
    """Create SLA trend chart"""
    # Ensure 'Backup Date' is in datetime format
    daily_data['Backup Date'] = pd.to_datetime(daily_data['Backup Date'])

    # Create the figure
    fig = px.line(
        daily_data,
        x='Backup Date',
        y='SLA',
        title='Daily Success Rate 🏆'
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="SLA (%)",
        showlegend=True,
        height=400,
        title_font_size=24
    )

    
    # Create tick labels with ordinals and month/day format
    daily_data['Ordinal Date'] = daily_data['Backup Date'].dt.strftime('%b %d')
    
    fig.update_xaxes(
        tickvals=daily_data['Backup Date'],
        ticktext=daily_data['Ordinal Date'],
        tickangle=45  # Optional: angle the ticks for better readability
    )
    
    return fig
    
    # Create tick labels with ordinals and month/day format
    daily_data['Ordinal Date'] = daily_data['Backup Date'].dt.strftime('%b %d')
    
    fig.update_xaxes(
        tickvals=daily_data['Backup Date'],
        ticktext=daily_data['Ordinal Date'],
        tickangle=45  # Optional: angle the ticks for better readability
    )
    
    return fig

def create_client_performance_chart(worst_clients):
    """Create host performance chart with color-coding based on job count and separate right legend"""
    host_data = []
    
    # Process each row from worst_clients
    for _, row in worst_clients.iterrows():
        try:
            value_str = row['Value']
            host_name = value_str.split(' (SLA:')[0]
            sla_part = value_str.split('SLA: ')[1]
            sla = float(sla_part.split('%')[0])
            total_jobs_part = value_str.split('Total Jobs: ')[1]
            total_jobs = float(total_jobs_part.split(')')[0])
            
            host_data.append({
                'Host': host_name,
                'SLA': sla,
                'Total_Jobs': total_jobs
            })
        except Exception as e:
            if 'debug_mode' in globals() and debug_mode:
                st.error(f"Error processing row {value_str}: {str(e)}")
            continue
    
    if not host_data:
        st.error("No valid host data found")
        return go.Figure()
    
    df = pd.DataFrame(host_data)
    
    # Create color array based on Total_Jobs values
    colors = []
    for jobs in df['Total_Jobs']:
        if jobs < 400:
            colors.append('red')
        elif jobs < 600:
            colors.append('yellow')
        else:
            colors.append('green')
    
    # Create the bar chart with increased width
    fig = go.Figure()
    
    # Add the main bar chart
    fig.add_trace(go.Bar(
        x=df['Host'],
        y=df['SLA'],
        marker_color=colors,
        text=df['Total_Jobs'].apply(lambda x: f'Jobs: {int(x)}'),
        textposition='auto',
        showlegend=False,
        width=0.6  # Increased bar width
    ))
    
    # Add three bars for the legend
    fig.add_trace(go.Bar(
        x=[''], y=[0],
        name='0-400 Jobs',
        marker_color='red',
        showlegend=True,
    ))
    fig.add_trace(go.Bar(
        x=[''], y=[0],
        name='400-600 Jobs',
        marker_color='yellow',
        showlegend=True,
    ))
    fig.add_trace(go.Bar(
        x=[''], y=[0],
        name='600-800 Jobs',
        marker_color='green',
        showlegend=True,
    ))
    
    # Update layout with improved width and spacing
    fig.update_layout(
        title={
            'text': 'Worst Performing Hosts',
            'font': {'size': 24}
        },
        xaxis_title="Host",
        yaxis_title="SLA (%)",
        height=600,
        width=1200,  # Increased overall width
        xaxis_tickangle=45,
        xaxis_tickfont={'size': 12},
        yaxis_tickfont={'size': 12},
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            title="Job Count Ranges",
            font={'size': 14}
        ),
        margin=dict(r=150, b=100, l=50, t=100),  # Adjusted margins
        plot_bgcolor='white',
        yaxis=dict(
            gridcolor='lightgray',
            range=[0, 100]
        ),
        bargap=0.15,  # Adjusted gap between bars
        showlegend=True,
        autosize=False  # Disable autosize to enforce width
    )
    
    return fig

def display_client_performance_tab(processed_data):
    """Display Host Performance tab content"""
    st.header("🖥️ Host Performance")
    
    # # Worst Performing Hosts
    # st.subheader("⚠️ Attention Required")
    # st.plotly_chart(
    #     create_client_performance_chart(processed_data['worst_clients']),
    #     use_container_width=True,
    #     key="worst_hosts_chart"
    # )
    
    # Host Details Table
    st.subheader("Detailed Host Status")
    host_data = []
    for _, row in processed_data['worst_clients'].iterrows():
        value_str = row['Value']
        host_name = value_str.split(' (SLA:')[0]
        sla = float(value_str.split('SLA: ')[1].split('%')[0])
        value_str = value_str.split('Total Jobs: ')[1].split(',')[0]
        clean_value_str = ''.join([c for c in value_str if c.isdigit() or c == '.'])
        total_jobs = float(clean_value_str)
        host_data.append({
            'Host Name': host_name,
            'Current SLA': f"{sla:.2f}%",
            'Total Jobs': int(total_jobs),
            'Status': '🔴' if sla < 95 else '🟡' if sla < 98 else '🟢',
            'Risk Level': 'High' if sla < 95 else 'Medium' if sla < 98 else 'Low',
            'Action Required': 'Immediate' if sla < 95 else 'Monitor' if sla < 98 else 'None'
        })
    
    df = pd.DataFrame(host_data)
    st.dataframe(df, use_container_width=True)
    
    # Risk Distribution
    risk_dist = df['Risk Level'].value_counts()
    fig = px.pie(
        values=risk_dist.values,
        names=risk_dist.index,
        title='Host Risk Distribution'
    )
    st.plotly_chart(fig, key="risk_distribution_pie")

def create_historical_sla_chart(current_sla):
    """Create historical SLA comparison chart"""
    historical_data = pd.DataFrame({
        'Month': ['Jun', 'Jul', 'Aug', 'Sep', 'Oct'],
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
    # Title with icon and larger font
    st.markdown("""
        <h1 style='display: flex; align-items: center; gap: 10px; font-size: 2.2rem;'>
            <span>📈</span> Overview
        </h1>
    """, unsafe_allow_html=True)

    # Update required_sla if target_sla changes
    processed_data['required_sla'] = calculate_required_sla(
        processed_data['current_sla'],
        target_sla,
        processed_data['days_processed'],
        processed_data['days_remaining']
    )
    
    # Key metrics in columns
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
    
    # Monthly Progress section with improved styling
    total_days = processed_data['days_processed'] + processed_data['days_remaining']
    progress = processed_data['days_processed'] / total_days

    # Create a single container with flexbox layout for the progress section
    progress_header = f"""
        <div style='
            background-color: transparent;
            padding: 15px 20px;
            border-radius: 10px;
            margin: 0;  /* Removed margin to eliminate space */
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        '>
            <div style='
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 10px;
            '>
                <div style='
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    font-size: 1.3rem;
                    font-weight: 600;
                '>
                    <span>📅</span>
                    <span>Monthly Progress</span>
                </div>
                <div style='
                    display: flex;
                    align-items: center;
                    gap: 20px;
                    font-size: 1.1rem;
                '>
                    <span style='display: flex; align-items: center; gap: 8px;'>
                        ✅ <strong>{processed_data['days_processed']}</strong> days processed
                    </span>
                    <span style='color: #6c757d;'>|</span>
                    <span style='display: flex; align-items: center; gap: 8px;'>
                        ⌛ <strong>{processed_data['days_remaining']}</strong> days remaining
                    </span>
                </div>
            </div>
        </div>
    """

    st.markdown(progress_header, unsafe_allow_html=True)
    #st.progress(progress)

    # Current Month Trend
    st.plotly_chart(
        create_sla_trend_chart(processed_data['daily_data']),
        use_container_width=True,
        key="overview_sla_trend"
    )

def display_trends_tab(processed_data):
    """Display Trends tab content"""
    st.header("📊 Trends Analysis")
    
    # Daily Stats
    st.subheader("📈 Daily Statistics")
    daily_stats = processed_data['daily_data'].describe()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Daily SLA", f"{daily_stats['SLA']['mean']:.2f}%")
    with col2:
        st.metric("Minimum Daily SLA", f"{daily_stats['SLA']['min']:.2f}%")
    with col3:
        st.metric("Maximum Daily SLA", f"{daily_stats['SLA']['max']:.2f}%")
    
    # Historical comparison
    st.plotly_chart(
        create_historical_sla_chart(processed_data['current_sla']),
        use_container_width=True,
        key="historical_sla_comparison"
    )
    

def display_client_performance_tab(processed_data):
    """Display Client Performance tab content"""
    st.header("🖥️ Host Performance")
    
    # # Worst Performing Hosts
    # st.subheader("⚠️ Attention Required")
    # st.plotly_chart(
    #     create_client_performance_chart(processed_data['worst_clients']),
    #     use_container_width=True,
    #     key="worst_clients_chart"
    # )
    
    # Client Details Table
    st.subheader("Detailed Host Status")
    client_data = []
    for _, row in processed_data['worst_clients'].iterrows():
        value_str = row['Value']
        client_name = value_str.split(' (SLA:')[0]
        sla = float(value_str.split('SLA: ')[1].split('%')[0])
        #total_jobs = float(value_str.split('Total Jobs: ')[1].split(',')[0])
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
    
    # Risk Distribution
    risk_dist = df['Risk Level'].value_counts()
    fig = px.pie(
        values=risk_dist.values,
        names=risk_dist.index,
        title='Host Risk Distribution'
    )
    st.plotly_chart(fig, key="risk_distribution_pie")

def display_sla_info_tab():

    """Display SLA Information tab content"""
    st.header("ℹ️ About SLA")
    
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

    st.markdown("""
    ### Service Level Agreement (SLA) Information
    
    #### What is SLA?
    A Service Level Agreement (SLA) is a commitment between a service provider and a client. It defines the level of service expected from the provider and what actions will be taken if those levels are not met.
    
    #### Key SLA Metrics in This Dashboard:
    - **Target SLA**: The agreed-upon service level (default 99%)
    - **Current SLA**: The actual service level being delivered
    - **Predicted SLA**: Expected service level by month-end
    - **Required SLA**: Minimum service level needed for the rest of the month
    
    #### SLA Status Indicators:
    - 🟢 Meeting SLA (≥98%)
    - 🟡 At Risk (95-98%)
    - 🔴 Below Target (<95%)
                
    #### Response Actions:
    - **High Risk (Red)**: Immediate investigation and corrective action required
    - **Medium Risk (Yellow)**: Develop improvement plan within 24 hours
    - **Low Risk (Green)**: Continue monitoring and maintenance
    
    #### Best Practices:
    1. Monitor daily trends
    2. Address declining performance early
    3. Focus on worst-performing clients
    4. Plan for remaining days to meet targets
    
    """)

            
def display_account_summary_tab():
    """Display Account Summary tab content"""
    st.header("🏢 Account Level SLA Summary")
    
    # Account Health Summary
    st.subheader("🏥 Account Health Summary")
    health_data = pd.DataFrame({
        'Status': ['Healthy', 'At Risk', 'Critical'],
        'Client Count': [120, 20, 5],
        'Description': [
            'Meeting or exceeding SLA targets',
            'Showing signs of degradation',
            'Immediate attention required'
        ]
    })
    #st.dataframe(health_data, use_container_width=True)
    
    # Visualizing the Account Health Summary with a pie chart
    health_fig = go.Figure(data=[go.Pie(
        labels=health_data['Status'],
        values=health_data['Client Count'],
        hole=0.4,  # Makes it a donut chart
        textinfo='label+percent',
        marker=dict(colors=['green', 'yellow', 'orange']),
    )])
    
    st.plotly_chart(health_fig, use_container_width=True)
    
    # Regional Data
    st.subheader("🌎 Regional Performance")
    regional_data = pd.DataFrame({
        'Region': ['Americas', 'EMEA', 'APAC'],
        'SLA': [98.9, 98.2, 98.5],
        'Clients': [60, 45, 40]
    })

    # Pie Chart for SLA Distribution
    fig = px.pie(
        regional_data,
        names='Region',
        values='SLA',
        title='Regional SLA Distribution',
        hole=0.3  # Makes it a donut chart
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Year-to-Date Performance
    st.subheader("📈 Year-to-Date Performance")
    ytd_data = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'],
        'SLA': [99.1, 98.9, 98.7, 99.0, 98.8, 99.2, 98.9, 99.1, 98.8, 98.7]
    })
    
    fig = px.line(
        ytd_data,
        x='Month',
        y='SLA',
        title='YTD SLA Trend',
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    
    # Key Insights
    st.subheader("🔍 Key Insights")
    st.markdown("""
    - Overall account performance remains strong at 98.7%
    - Americas region shows highest consistency in meeting SLAs
    - 83% of clients are in healthy status
    - Year-to-date trend shows stable performance with minor fluctuations
    - 5 clients require immediate attention due to critical status
    """)

def main():
    """Main application function"""
    #st.markdown(f"### 🎯 BSR Prediction Dashboard - {selected_account}")
    
    # Sidebar configuration
    #st.sidebar.header("Configuration")
    #target_sla = st.sidebar.slider("Target SLA (%)", 90.0, 100.0, 99.0, 0.1)
    st.sidebar.text("Target SLA (%): 99%")
    target_sla = 99.0
    
    # Load data directly from system
    with st.spinner("Processing data..."):
        processed_data, results_df = load_and_process_file()
        
        if processed_data is not None:
            # Create tabs
            tab1, tab2, tab3, tab5 = st.tabs([
                "📈 Overview",
                "📊 Trends",
                "🖥️ Host Performance",  # Updated this line
                #"🏢 Account Summary",
                "ℹ️ About SLA"
            ])
            
            with tab1:
                display_overview_tab(processed_data, target_sla)
            
            with tab2:
                display_trends_tab(processed_data)
            
            with tab3:
                display_client_performance_tab(processed_data)
            
            #with tab4:
                #display_account_summary_tab()
            
            with tab5:
                display_sla_info_tab()
        else:
            st.error("Unable to load data. Please check if the file exists and has the correct format.")
            
    # Add floating footer
    st.markdown(
        """
        <div class="floating-footer">
            © 2024 DXC Technology
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()