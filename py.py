"""
AI Customer Upsell Prediction Dashboard - Adapted for Current Directory Structure
Author: Mohanarajan Reddy (@mohanarajanreddy)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Check what's actually available in your directory
def check_available_files():
    """Check what files are actually available in the current directory structure"""
    available_files = {}
    
    # Check for data files
    data_paths = [
        './data/raw/telecom_data.csv',
        './notebooks/data/processed/telecom_processed.csv'
    ]
    available_files['data'] = [path for path in data_paths if os.path.exists(path)]
    
    # Check for model files
    model_paths = [
        './notebooks/models/best_model_xgboost.pkl',
        './notebooks/models/ensemble_model.pkl',
        './notebooks/models/scaler.pkl',
        './notebooks/models/feature_columns.pkl'
    ]
    available_files['models'] = [path for path in model_paths if os.path.exists(path)]
    
    # Check for output/report files
    report_paths = [
        './notebooks/outputs/reports/model_results.json',
        './notebooks/outputs/reports/eda_summary.json',
        './notebooks/outputs/reports/evaluation_summary.txt',
        './notebooks/outputs/reports/customer_segments.csv'
    ]
    available_files['reports'] = [path for path in report_paths if os.path.exists(path)]
    
    return available_files

# Page configuration
st.set_page_config(
    page_title="🎯 AI Customer Upsell Prediction System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .status-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    
    .info-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def load_available_data():
    """Load any available data from the current directory structure"""
    available_files = check_available_files()
    data_info = {
        'raw_data': None,
        'processed_data': None,
        'status': 'no_data',
        'message': 'No data files found'
    }
    
    # Try to load raw data
    if './data/raw/telecom_data.csv' in available_files['data']:
        try:
            data_info['raw_data'] = pd.read_csv('./data/raw/telecom_data.csv')
            data_info['status'] = 'raw_data_available'
            data_info['message'] = f"Raw data loaded: {len(data_info['raw_data']):,} records"
        except Exception as e:
            st.error(f"Error loading raw data: {e}")
    
    # Try to load processed data
    if './notebooks/data/processed/telecom_processed.csv' in available_files['data']:
        try:
            data_info['processed_data'] = pd.read_csv('./notebooks/data/processed/telecom_processed.csv')
            data_info['status'] = 'processed_data_available'  
            data_info['message'] = f"Processed data loaded: {len(data_info['processed_data']):,} records"
        except Exception as e:
            st.error(f"Error loading processed data: {e}")
    
    return data_info, available_files

def create_mock_predictions(df):
    """Create mock predictions for demonstration purposes"""
    np.random.seed(42)
    
    # Create realistic mock predictions based on available features
    n_customers = len(df)
    
    # Base probability influenced by available features
    base_prob = np.random.beta(2, 5, n_customers)
    
    # Adjust based on available features if they exist
    if 'Total_Charges' in df.columns:
        # Higher charges might indicate higher upsell potential
        charge_factor = (df['Total_Charges'] - df['Total_Charges'].min()) / (df['Total_Charges'].max() - df['Total_Charges'].min())
        base_prob = base_prob * 0.7 + charge_factor * 0.3
    
    if 'CustServ Calls' in df.columns:
        # More service calls might indicate dissatisfaction (lower upsell potential)
        service_factor = 1 - (df['CustServ Calls'] / df['CustServ Calls'].max())
        base_prob = base_prob * 0.8 + service_factor * 0.2
    
    # Add predictions to dataframe
    df_with_predictions = df.copy()
    df_with_predictions['ML_Upsell_Probability'] = np.clip(base_prob, 0, 1)
    df_with_predictions['ML_Prediction'] = (base_prob > 0.5).astype(int)
    
    # Create confidence levels
    df_with_predictions['Confidence_Level'] = pd.cut(
        base_prob,
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=['Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
    )
    
    # Create priority levels
    df_with_predictions['ML_Priority'] = pd.cut(
        base_prob,
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH', 'VERY HIGH'],
        include_lowest=True
    )
    
    # Add expected revenue (mock)
    df_with_predictions['Expected_Monthly_Revenue'] = np.random.normal(50, 20, n_customers) * (1 + base_prob)
    df_with_predictions['Expected_Monthly_Revenue'] = np.clip(df_with_predictions['Expected_Monthly_Revenue'], 10, 200)
    
    return df_with_predictions

def show_system_status(available_files):
    """Display current system status"""
    st.markdown("## 🔧 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        data_status = "✅ Available" if available_files['data'] else "❌ Missing"
        st.markdown(f"""
        <div class="status-card {'success-card' if available_files['data'] else 'warning-card'}">
            <h4>📊 Data Files</h4>
            <p>{data_status}</p>
            <small>{len(available_files['data'])} files found</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        model_status = "✅ Available" if available_files['models'] else "❌ Missing"
        st.markdown(f"""
        <div class="status-card {'success-card' if available_files['models'] else 'warning-card'}">
            <h4>🤖 Model Files</h4>
            <p>{model_status}</p>
            <small>{len(available_files['models'])} files found</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        report_status = "✅ Available" if available_files['reports'] else "❌ Missing"
        st.markdown(f"""
        <div class="status-card {'success-card' if available_files['reports'] else 'warning-card'}">
            <h4>📈 Reports</h4>
            <p>{report_status}</p>
            <small>{len(available_files['reports'])} files found</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        try:
            import torch
            gpu_status = "✅ GPU Available" if torch.cuda.is_available() else "⚠️ CPU Only"
            gpu_class = "success-card" if torch.cuda.is_available() else "info-card"
        except ImportError:
            gpu_status = "❌ PyTorch Missing"
            gpu_class = "warning-card"
        
        st.markdown(f"""
        <div class="status-card {gpu_class}">
            <h4>🔥 Compute</h4>
            <p>{gpu_status}</p>
        </div>
        """, unsafe_allow_html=True)

def show_available_files_details(available_files):
    """Show details of available files"""
    with st.expander("📁 Available Files Details"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Data Files:**")
            if available_files['data']:
                for file in available_files['data']:
                    st.write(f"✅ {file}")
            else:
                st.write("❌ No data files found")
        
        with col2:
            st.write("**Model Files:**")
            if available_files['models']:
                for file in available_files['models']:
                    st.write(f"✅ {file}")
            else:
                st.write("❌ No model files found")
        
        with col3:
            st.write("**Report Files:**")
            if available_files['reports']:
                for file in available_files['reports']:
                    st.write(f"✅ {file}")
            else:
                st.write("❌ No report files found")

def show_data_overview(df):
    """Show overview of loaded data"""
    st.markdown("## 📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 Total Records</h3>
            <h2>{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📋 Features</h3>
            <h2>{len(df.columns)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if 'Total_Charges' in df.columns:
            avg_charges = df['Total_Charges'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Avg Charges</h3>
                <h2>USD {avg_charges:.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Data Type</h3>
                <h2>Telecom</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'Account Length' in df.columns:
            avg_length = df['Account Length'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>📅 Avg Account</h3>
                <h2>{avg_length:.0f} days</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔍 Status</h3>
                <h2>Ready</h2>
            </div>
            """, unsafe_allow_html=True)

def show_predictions_dashboard(df_with_predictions):
    """Show predictions dashboard"""
    st.markdown("## 🎯 Prediction Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df_with_predictions)
    high_prob_customers = len(df_with_predictions[df_with_predictions['ML_Upsell_Probability'] > 0.7])
    avg_probability = df_with_predictions['ML_Upsell_Probability'].mean()
    expected_revenue = df_with_predictions['Expected_Monthly_Revenue'].sum()
    
    with col1:
        st.metric("Total Customers", f"{total_customers:,}")
    with col2:
        st.metric("High Probability", f"{high_prob_customers:,}", f"{high_prob_customers/total_customers*100:.1f}%")
    with col3:
        st.metric("Avg Probability", f"{avg_probability:.1%}")
    with col4:
        st.metric("Expected Revenue", f"USD {expected_revenue:,.0f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            df_with_predictions, 
            x='ML_Upsell_Probability',
            title="Upsell Probability Distribution",
            nbins=30,
            color_discrete_sequence=['#667eea']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        confidence_counts = df_with_predictions['Confidence_Level'].value_counts()
        fig_pie = px.pie(
            values=confidence_counts.values,
            names=confidence_counts.index,
            title="Confidence Level Distribution",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Top customers
    st.markdown("### 🏆 Top 10 Customers by Upsell Probability")
    top_customers = df_with_predictions.nlargest(10, 'ML_Upsell_Probability')[
        ['ML_Upsell_Probability', 'Expected_Monthly_Revenue', 'ML_Priority', 'Confidence_Level']
    ].copy()
    top_customers['ML_Upsell_Probability'] = (top_customers['ML_Upsell_Probability'] * 100).round(1)
    top_customers['Expected_Monthly_Revenue'] = top_customers['Expected_Monthly_Revenue'].round(2)
    st.dataframe(top_customers, use_container_width=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 AI Customer Upsell Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("**Adapted for Current Directory Structure** | **Author**: Mohanarajan P")
    
    # Load available data and check system
    data_info, available_files = load_available_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Show system status in sidebar
        st.markdown("### 📊 Quick Status")
        st.write(f"📁 Data Files: {len(available_files['data'])}")
        st.write(f"🤖 Models: {len(available_files['models'])}")
        st.write(f"📈 Reports: {len(available_files['reports'])}")
        
        st.markdown("---")
        
        # Options
        st.markdown("### ⚙️ Options")
        show_file_details = st.checkbox("Show File Details", value=False)
        use_mock_predictions = st.checkbox("Generate Mock Predictions", value=True)
        
        st.markdown("---")
        
        # File upload
        st.markdown("### 📤 Upload Data")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    # Main content
    show_system_status(available_files)
    
    if show_file_details:
        show_available_files_details(available_files)
    
    # Data processing
    df_to_use = None
    
    if uploaded_file is not None:
        try:
            df_to_use = pd.read_csv(uploaded_file)
            st.success(f"✅ Uploaded file loaded: {len(df_to_use):,} records")
        except Exception as e:
            st.error(f"❌ Error loading uploaded file: {e}")
    elif data_info['processed_data'] is not None:
        df_to_use = data_info['processed_data']
        st.info(f"📊 Using processed data: {len(df_to_use):,} records")
    elif data_info['raw_data'] is not None:
        df_to_use = data_info['raw_data']
        st.info(f"📊 Using raw data: {len(df_to_use):,} records")
    
    if df_to_use is not None:
        show_data_overview(df_to_use)
        
        if use_mock_predictions:
            with st.spinner("Generating predictions..."):
                df_with_predictions = create_mock_predictions(df_to_use)
            show_predictions_dashboard(df_with_predictions)
        else:
            st.info("Enable 'Generate Mock Predictions' to see prediction results.")
            
            # Show data preview
            st.markdown("### 🔍 Data Preview")
            st.dataframe(df_to_use.head(10), use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; border: 2px dashed #667eea; border-radius: 15px; margin: 2rem 0;">
            <h2>🚀 Welcome to Your AI Customer Upsell System</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0;">
                Your project structure is ready! To get started:
            </p>
            <div style="text-align: left; max-width: 600px; margin: 0 auto;">
                <h3>📋 Next Steps:</h3>
                <ol>
                    <li><strong>Add data:</strong> Place your telecom data in <code>./data/raw/telecom_data.csv</code></li>
                    <li><strong>Run preprocessing:</strong> Execute <code>01_telecom_data_preprocessing.ipynb</code></li>
                    <li><strong>Train models:</strong> Run <code>03_ensemble_model_training.ipynb</code></li>
                    <li><strong>Upload data:</strong> Use the sidebar to upload a CSV file for immediate analysis</li>
                </ol>
            </div>
            <p><strong>💡 Tip:</strong> Enable "Generate Mock Predictions" to see how the dashboard works with sample data!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Instructions for setup
    with st.expander("🔧 Setup Instructions"):
        st.markdown("""
        ### To fully activate your ML pipeline:
        
        1. **Prepare your data:**
           ```bash
           # Place your telecom dataset here:
           ./data/raw/telecom_data.csv
           ```
        
        2. **Run the notebooks in order:**
           - `notebooks/data_preprocessing/01_telecom_data_preprocessing.ipynb`
           - `notebooks/model_development/03_ensemble_model_training.ipynb` 
           - `notebooks/evaluation/04_model_evaluation.ipynb`
           - `notebooks/exploratory_analysis/06_customer_segmentation.ipynb`
        
        3. **For advanced features:**
           - `notebooks/model_development/05_hyperparameter_optimization.ipynb`
           - `notebooks/model_development/07_deep_learning_models.ipynb`
        
        4. **Your trained models will be saved in:**
           - `./notebooks/models/`
           - `./notebooks/outputs/reports/`
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🎯 <strong>AI Customer Upsell Prediction System</strong> | Adapted for Current Directory Structure</p>
        <p><strong>Author:</strong> Mohanarajan P | <strong>Repository:</strong> <a href="https://github.com/mohanarajanreddy/ai-customer-upsell-prediction">github.com/mohanarajanreddy/ai-customer-upsell-prediction</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
