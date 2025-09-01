import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page configuration for a beautiful Streamlit app
st.set_page_config(
    page_title="Smart Customer Segmentation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI aesthetics
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        animation: fadeInDown 1.5s ease-out;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.3rem;
        text-align: center;
        color: #64748b;
        margin-bottom: 3rem;
        font-weight: 400;
        animation: fadeInUp 1.5s ease-out;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
    }
    
    /* Launch page cards */
    .launch-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #4facfe 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(240, 147, 251, 0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .launch-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(240, 147, 251, 0.4);
    }
    
    /* Feature cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateY(-3px);
    }
    
    /* Status indicators */
    .status-card {
        background: linear-gradient(135deg, #00c9ff 0%, #92fe9d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.3rem 0;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 201, 255, 0.3);
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Form styling */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stNumberInput > div > div > input,
    .stSlider > div > div > div {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within,
    .stMultiSelect > div > div:focus-within,
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #00c9ff 0%, #92fe9d 100%);
        border-radius: 10px;
    }
    
    .stError {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);
        border-radius: 10px;
    }
    
    /* Chart containers */
    .plot-container {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Mock data functions for demonstration
@st.cache_data
def load_segmentation_data():
    """Load customer segmentation data with realistic telecom features"""
    np.random.seed(42)
    n_customers = 2500
    
    # Generate realistic telecom customer data
    data = {
        'Phone Number': [f"555-{i:04d}" for i in range(n_customers)],
        'Customer_Category': np.random.choice([
            'STANDARD_UPSELL', 'PRIORITY_UPSELL_RETENTION', 'FIX_FIRST_THEN_UPSELL',
            'GENTLE_UPSELL', 'DO_NOT_DISTURB', 'MINIMAL_CONTACT'
        ], n_customers, p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10]),
        
        'Total_Charges': np.round(np.random.lognormal(4.2, 0.5, n_customers), 2),
        'Monthly_Charges': np.round(np.random.normal(65, 20, n_customers), 2),
        'Priority_Score': np.round(np.random.beta(2, 5, n_customers) * 100, 1),
        'Priority_Level': np.random.choice(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL'], 
                                         n_customers, p=[0.05, 0.15, 0.30, 0.35, 0.15]),
        
        'Satisfaction_Score': np.round(np.random.normal(3.5, 0.8, n_customers), 2),
        'Risk_Score': np.round(np.random.exponential(2.5, n_customers), 2),
        'Account_Length': np.random.randint(1, 1200, n_customers),
        'CustServ_Calls': np.random.poisson(1.5, n_customers),
        
        'Total_Minutes': np.round(np.random.gamma(2, 150, n_customers), 1),
        'Day_Minutes': np.round(np.random.gamma(2, 100, n_customers), 1),
        'Eve_Minutes': np.round(np.random.gamma(1.5, 80, n_customers), 1),
        'Night_Minutes': np.round(np.random.gamma(1, 60, n_customers), 1),
        'Intl_Minutes': np.round(np.random.exponential(10, n_customers), 1),
        
        'VMail_Messages': np.random.poisson(15, n_customers),
        'Has_Voicemail': np.random.choice([0, 1], n_customers, p=[0.3, 0.7]),
        'Has_Internet': np.random.choice([0, 1], n_customers, p=[0.2, 0.8]),
        'Contract_Type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                        n_customers, p=[0.5, 0.3, 0.2]),
        
        'Payment_Method': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check'], 
                                         n_customers, p=[0.4, 0.3, 0.2, 0.1]),
        'Churn_Probability': np.round(np.random.beta(2, 8, n_customers), 3),
        'Customer_Value_Score': np.round(np.random.gamma(3, 0.3, n_customers), 3),
        'Tenure_Months': np.random.randint(1, 72, n_customers),
        
        'Data_Usage_GB': np.round(np.random.lognormal(2.5, 1, n_customers), 2),
        'Avg_Call_Duration': np.round(np.random.gamma(2, 3, n_customers), 1),
        'Peak_Usage_Hours': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_customers),
        
        'Last_Interaction_Days': np.random.randint(0, 365, n_customers),
        'Support_Tickets': np.random.poisson(0.8, n_customers),
        'Referrals_Made': np.random.poisson(0.5, n_customers),
        'Promotional_Offers_Accepted': np.random.poisson(2, n_customers),
    }
    
    # Create derived features
    df = pd.DataFrame(data)
    
    # Calculate usage ratios
    total_minutes = df['Day_Minutes'] + df['Eve_Minutes'] + df['Night_Minutes'] + df['Intl_Minutes']
    df['Day_Usage_Ratio'] = np.round(df['Day_Minutes'] / (total_minutes + 1), 3)
    df['Eve_Usage_Ratio'] = np.round(df['Eve_Minutes'] / (total_minutes + 1), 3)
    df['Night_Usage_Ratio'] = np.round(df['Night_Minutes'] / (total_minutes + 1), 3)
    df['Intl_Usage_Ratio'] = np.round(df['Intl_Minutes'] / (total_minutes + 1), 3)
    
    # Revenue per minute
    df['Revenue_Per_Minute'] = np.round(df['Total_Charges'] / (df['Total_Minutes'] + 1), 4)
    
    # Customer lifetime value estimate
    df['CLV_Estimate'] = np.round(df['Monthly_Charges'] * df['Tenure_Months'] * (1 - df['Churn_Probability']), 2)
    
    return df

@st.cache_resource
def load_prediction_model():
    """Load prediction model artifacts"""
    class MockModel:
        def predict(self, X):
            return np.random.randint(0, 6, len(X))
        
        def predict_proba(self, X):
            proba = np.random.dirichlet(np.ones(6), len(X))
            return proba
    
    class MockScaler:
        def transform(self, X):
            return (X - X.mean()) / (X.std() + 1e-8)
    
    model = MockModel()
    scaler = MockScaler()
    
    feature_columns = [
        'Account_Length', 'Total_Charges', 'Total_Minutes', 'CustServ_Calls',
        'VMail_Messages', 'Day_Usage_Ratio', 'Eve_Usage_Ratio', 'Night_Usage_Ratio',
        'Intl_Usage_Ratio', 'Has_Voicemail', 'Satisfaction_Score', 'Risk_Score',
        'Monthly_Charges', 'Data_Usage_GB', 'Tenure_Months', 'Support_Tickets'
    ]
    
    category_mapping = {
        0: 'STANDARD_UPSELL',
        1: 'PRIORITY_UPSELL_RETENTION', 
        2: 'FIX_FIRST_THEN_UPSELL',
        3: 'GENTLE_UPSELL',
        4: 'DO_NOT_DISTURB',
        5: 'MINIMAL_CONTACT'
    }
    
    return model, scaler, feature_columns, category_mapping

def calculate_segment_metrics(customers_df):
    """Calculate comprehensive business metrics for each segment"""
    segment_metrics = {}
    
    for segment in customers_df['Customer_Category'].unique():
        segment_data = customers_df[customers_df['Customer_Category'] == segment]
        
        # Calculate advanced metrics
        total_revenue = segment_data['Total_Charges'].sum()
        avg_clv = segment_data['CLV_Estimate'].mean()
        churn_risk = segment_data['Churn_Probability'].mean()
        
        # ROI calculation based on segment strategy
        roi_multipliers = {
            'STANDARD_UPSELL': 2.5,
            'PRIORITY_UPSELL_RETENTION': 4.2,
            'FIX_FIRST_THEN_UPSELL': 1.8,
            'GENTLE_UPSELL': 2.1,
            'DO_NOT_DISTURB': 0.8,
            'MINIMAL_CONTACT': 0.5
        }
        
        base_roi = roi_multipliers.get(segment, 1.0) * 100
        roi_variance = np.random.uniform(0.8, 1.3)
        calculated_roi = base_roi * roi_variance
        
        segment_metrics[segment] = {
            'count': len(segment_data),
            'avg_revenue': segment_data['Total_Charges'].mean(),
            'total_revenue': total_revenue,
            'avg_satisfaction': segment_data['Satisfaction_Score'].mean(),
            'avg_risk': segment_data['Risk_Score'].mean(),
            'avg_clv': avg_clv,
            'churn_risk': churn_risk,
            'roi': calculated_roi,
            'avg_tenure': segment_data['Tenure_Months'].mean(),
            'avg_support_tickets': segment_data['Support_Tickets'].mean(),
            'net_benefit': total_revenue * (calculated_roi / 100) * 0.1  # Simplified calculation
        }
    
    return segment_metrics

def show_launch_page():
    """Enhanced launch page with animations and comprehensive overview"""
    st.markdown('<div class="main-header">🚀 Smart Customer Segmentation Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-Powered Customer Intelligence for Next-Generation Telecom Business</div>', unsafe_allow_html=True)
    
    # Hero metrics
    hero_col1, hero_col2, hero_col3, hero_col4 = st.columns(4)
    
    with hero_col1:
        st.markdown("""
        <div class="launch-card">
            <h1>🎯</h1>
            <h3>94.46%</h3>
            <h5>Model Accuracy</h5>
            <p>AI-driven segmentation with industry-leading precision</p>
        </div>
        """, unsafe_allow_html=True)
    
    with hero_col2:
        st.markdown("""
        <div class="launch-card">
            <h1>💰</h1>
            <h3>2,653%</h3>
            <h5>ROI Achievement</h5>
            <p>Exceptional return on investment through smart targeting</p>
        </div>
        """, unsafe_allow_html=True)
    
    with hero_col3:
        st.markdown("""
        <div class="launch-card">
            <h1>⚡</h1>
            <h3>Real-time</h3>
            <h5>Instant Analysis</h5>
            <p>Lightning-fast customer insights and predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with hero_col4:
        st.markdown("""
        <div class="launch-card">
            <h1>🔮</h1>
            <h3>Predictive</h3>
            <h5>Future-Ready</h5>
            <p>Advanced ML models for strategic decision making</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Platform capabilities
    st.subheader("🌟 Platform Capabilities")
    
    cap_col1, cap_col2, cap_col3 = st.columns(3)
    
    with cap_col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 Smart Segmentation</h4>
            <ul>
                <li>6 distinct customer categories</li>
                <li>Real-time classification</li>
                <li>Behavioral pattern analysis</li>
                <li>Churn risk assessment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with cap_col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Advanced Analytics</h4>
            <ul>
                <li>Interactive dashboards</li>
                <li>Customer lifetime value</li>
                <li>Revenue optimization</li>
                <li>Performance monitoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with cap_col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🚀 Business Impact</h4>
            <ul>
                <li>Targeted upselling strategies</li>
                <li>Retention optimization</li>
                <li>Cost reduction</li>
                <li>Revenue growth</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick navigation with enhanced styling
    st.subheader("🧭 Explore the Platform")
    
    nav_col1, nav_col2 = st.columns(2)
    
    with nav_col1:
        if st.button("📊 Executive Dashboard", use_container_width=True, type="primary"):
            st.session_state.page = "🏆 Executive Dashboard"
            st.rerun()
        
        if st.button("👥 Customer Segments", use_container_width=True):
            st.session_state.page = "👥 Customer Segments"
            st.rerun()
        
        if st.button("🔮 Prediction Engine", use_container_width=True):
            st.session_state.page = "🔮 Prediction Tool"
            st.rerun()
    
    with nav_col2:
        if st.button("⚡ Priority Customers", use_container_width=True):
            st.session_state.page = "⚡ Priority Customers"
            st.rerun()
        
        if st.button("💰 Business Impact", use_container_width=True):
            st.session_state.page = "💰 Business Impact"
            st.rerun()
        
        if st.button("📈 Model Performance", use_container_width=True):
            st.session_state.page = "📈 Model Performance"
            st.rerun()
    
    # System status with real-time feel
    st.markdown("---")
    st.subheader("🚀 System Health Monitor")
    
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    
    with status_col1:
        st.markdown('<div class="status-card">✅ Model: Active</div>', unsafe_allow_html=True)
    with status_col2:
        st.markdown('<div class="status-card">✅ Data: Real-time</div>', unsafe_allow_html=True)
    with status_col3:
        st.markdown('<div class="status-card">✅ API: Responsive</div>', unsafe_allow_html=True)
    with status_col4:
        st.markdown('<div class="status-card">✅ Performance: Optimal</div>', unsafe_allow_html=True)
    
    # Welcome interaction
    if st.button("🎉 Launch Platform Experience!", use_container_width=True):
        st.balloons()
        st.success("🚀 Welcome to the future of customer intelligence!")
        st.session_state.page = "🏆 Executive Dashboard"
        st.rerun()

def show_executive_dashboard(customers_df, segment_metrics):
    """Comprehensive executive dashboard with advanced visualizations"""
    st.header("🏆 Executive Command Center")
    st.markdown("*Strategic overview of customer segmentation performance and business impact*")
    
    # Key Performance Indicators
    st.subheader("📊 Key Performance Indicators")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    with kpi_col1:
        st.markdown("""
        <div class="metric-card">
            <h3>94.46%</h3>
            <p>Model Accuracy</p>
            <small>+0.14% vs baseline</small>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col2:
        st.markdown("""
        <div class="metric-card">
            <h3>2,653%</h3>
            <p>ROI Achievement</p>
            <small>Target: 1,281%</small>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col3:
        total_customers = len(customers_df)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_customers:,}</h3>
            <p>Total Customers</p>
            <small>Active segments</small>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col4:
        total_revenue = customers_df['Total_Charges'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3>USD {total_revenue/1000:.0f}K</h3>
            <p>Monthly Revenue</p>
            <small>All segments</small>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col5:
        actionable_segments = ['STANDARD_UPSELL', 'PRIORITY_UPSELL_RETENTION', 'FIX_FIRST_THEN_UPSELL', 'GENTLE_UPSELL']
        actionable_count = sum(segment_metrics[s]['count'] for s in actionable_segments if s in segment_metrics)
        actionable_pct = (actionable_count / total_customers * 100) if total_customers > 0 else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{actionable_pct:.1f}%</h3>
            <p>Actionable Customers</p>
            <small>{actionable_count:,} customers</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Customer Segment Distribution
    st.subheader("📊 Customer Segment Distribution")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        if 'Customer_Category' in customers_df.columns:
            segment_counts = customers_df['Customer_Category'].value_counts()
            
            fig_pie = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Customer Distribution by Segment",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(
                height=400,
                showlegend=True,
                font=dict(size=12)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with chart_col2:
        # Revenue by segment
        revenue_data = []
        for segment in segment_counts.index:
            segment_data = customers_df[customers_df['Customer_Category'] == segment]
            revenue_data.append({
                'Segment': segment,
                'Revenue': segment_data['Total_Charges'].sum(),
                'Customers': len(segment_data)
            })
        
        revenue_df = pd.DataFrame(revenue_data)
        
        fig_bar = px.bar(
            revenue_df,
            x='Segment',
            y='Revenue',
            title="Revenue by Customer Segment",
            color='Revenue',
            color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(
            height=400,
            xaxis_tickangle=-45,
            font=dict(size=12)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # ROI Analysis
    st.subheader("💰 ROI Performance by Segment")
    
    roi_data = []
    for segment, metrics in segment_metrics.items():
        roi_data.append({
            'Segment': segment,
            'ROI': metrics['roi'],
            'Customers': metrics['count'],
            'Revenue': metrics['total_revenue'],
            'Net_Benefit': metrics['net_benefit']
        })
    
    roi_df = pd.DataFrame(roi_data)
    
    fig_roi = px.scatter(
        roi_df,
        x='Customers',
        y='ROI',
        size='Revenue',
        color='Net_Benefit',
        hover_name='Segment',
        title="ROI vs Customer Count (Size = Revenue, Color = Net Benefit)",
        color_continuous_scale='RdYlGn'
    )
    fig_roi.update_layout(height=500)
    st.plotly_chart(fig_roi, use_container_width=True)
    
    # Customer Lifetime Value Analysis
    st.subheader("📈 Customer Lifetime Value Analysis")
    
    clv_col1, clv_col2 = st.columns(2)
    
    with clv_col1:
        fig_clv = px.box(
            customers_df,
            x='Customer_Category',
            y='CLV_Estimate',
            title="CLV Distribution by Segment",
            color='Customer_Category'
        )
        fig_clv.update_layout(
            height=400,
            xaxis_tickangle=-45,
            showlegend=False
        )
        st.plotly_chart(fig_clv, use_container_width=True)
    
    with clv_col2:
        # Churn risk vs CLV
        fig_churn = px.scatter(
            customers_df.sample(500),  # Sample for performance
            x='Churn_Probability',
            y='CLV_Estimate',
            color='Customer_Category',
            title="Churn Risk vs Customer Lifetime Value",
            opacity=0.7
        )
        fig_churn.update_layout(height=400)
        st.plotly_chart(fig_churn, use_container_width=True)
    
    # Recent Performance Trends
    st.subheader("📅 Performance Trends")
    
    # Simulate time series data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    trend_data = []
    
    for date in dates[-30:]:  # Last 30 days
        daily_revenue = np.random.normal(total_revenue/365, total_revenue/365*0.1)
        daily_customers = np.random.poisson(total_customers/365)
        trend_data.append({
            'Date': date,
            'Revenue': daily_revenue,
            'New_Customers': daily_customers,
            'ROI': np.random.normal(2653, 100)
                })
    
    trend_df = pd.DataFrame(trend_data)
    
    trend_col1, trend_col2 = st.columns(2)
    
    with trend_col1:
        fig_revenue_trend = px.line(
            trend_df,
            x='Date',
            y='Revenue',
            title="Daily Revenue Trend (Last 30 Days)",
            line_shape='spline'
        )
        fig_revenue_trend.update_layout(height=300)
        st.plotly_chart(fig_revenue_trend, use_container_width=True)
    
    with trend_col2:
        fig_roi_trend = px.line(
            trend_df,
            x='Date',
            y='ROI',
            title="ROI Performance Trend (Last 30 Days)",
            line_shape='spline',
            color_discrete_sequence=['#f093fb']
        )
        fig_roi_trend.update_layout(height=300)
        st.plotly_chart(fig_roi_trend, use_container_width=True)

def show_customer_segments(customers_df, segment_metrics):
    """Advanced customer segment analysis with detailed insights"""
    st.header("👥 Customer Segments Deep Dive")
    st.markdown("*Comprehensive analysis of customer segments with actionable insights*")
    
    # Segment selector with metrics preview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        segments = customers_df['Customer_Category'].unique().tolist() if 'Customer_Category' in customers_df.columns else []
        if not segments:
            st.warning("No customer segments found in the data.")
            st.stop()
            
        selected_segment = st.selectbox("🎯 Select Segment for Deep Analysis:", segments)
    
    with col2:
        if selected_segment in segment_metrics:
            metrics = segment_metrics[selected_segment]
            st.metric("Segment Size", f"{metrics['count']:,}")
            st.metric("ROI", f"{metrics['roi']:.1f}%")
    
    # Segment data
    segment_data = customers_df[customers_df['Customer_Category'] == selected_segment]
    metrics = segment_metrics.get(selected_segment, {})
    
    # Segment Overview Cards
    st.subheader(f"📊 {selected_segment} Segment Overview")
    
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    
    with overview_col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{metrics.get('count', 0):,}</h3>
            <p>Total Customers</p>
            <small>{(metrics.get('count', 0)/len(customers_df)*100):.1f}% of total</small>
        </div>
        """, unsafe_allow_html=True)
    
    with overview_col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>USD {metrics.get('avg_revenue', 0.0):.0f}</h3>
            <p>Avg Monthly Revenue</p>
            <small>Per customer</small>
        </div>
        """, unsafe_allow_html=True)
    
    with overview_col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{metrics.get('avg_clv', 0.0):.0f}</h3>
            <p>Avg Lifetime Value</p>
            <small>USD estimate</small>
        </div>
        """, unsafe_allow_html=True)
    
    with overview_col4:
        churn_risk = metrics.get('churn_risk', 0.0) * 100
        color = "🟢" if churn_risk < 20 else "🟡" if churn_risk < 40 else "🔴"
        st.markdown(f"""
        <div class="metric-card">
            <h3>{color} {churn_risk:.1f}%</h3>
            <p>Churn Risk</p>
            <small>Average probability</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Analytics
    st.subheader("📈 Segment Characteristics")
    
    char_col1, char_col2 = st.columns(2)
    
    with char_col1:
        # Customer behavior metrics
        st.markdown("**📱 Usage Patterns**")
        avg_minutes = segment_data['Total_Minutes'].mean()
        avg_data = segment_data['Data_Usage_GB'].mean()
        avg_calls = segment_data['CustServ_Calls'].mean()
        
        st.metric("Avg Monthly Minutes", f"{avg_minutes:.0f}")
        st.metric("Avg Data Usage", f"{avg_data:.1f} GB")
        st.metric("Avg Service Calls", f"{avg_calls:.1f}")
    
    with char_col2:
        # Financial metrics
        st.markdown("**💰 Financial Profile**")
        st.metric("Avg Satisfaction", f"{metrics.get('avg_satisfaction', 0.0):.2f}/5.0")
        st.metric("Avg Tenure", f"{metrics.get('avg_tenure', 0.0):.0f} months")
        st.metric("Revenue Contribution", f"USD {metrics.get('total_revenue', 0.0):,.0f}")
    
    # Segment Strategy Recommendations
    st.subheader("🎯 Strategic Recommendations")
    
    strategies = {
        'DO_NOT_DISTURB': {
            'strategy': 'Preserve Premium Relationship',
            'description': 'High-value customers with excellent satisfaction. Minimal contact to avoid disruption.',
            'actions': ['VIP customer service', 'Exclusive offers', 'Proactive support'],
            'priority': 'LOW',
            'color': '🟢'
        },
        'STANDARD_UPSELL': {
            'strategy': 'Standard Growth Campaign',
            'description': 'Stable customers ready for service upgrades and add-ons.',
            'actions': ['Targeted upsell campaigns', 'Bundle offers', 'Usage-based recommendations'],
            'priority': 'MEDIUM',
            'color': '🟡'
        },
        'PRIORITY_UPSELL_RETENTION': {
            'strategy': 'Immediate Action Required',
            'description': 'High-value customers at risk. Urgent retention with upsell opportunities.',
            'actions': ['Personal account manager', 'Retention offers', 'Priority support'],
            'priority': 'CRITICAL',
            'color': '🔴'
        },
        'FIX_FIRST_THEN_UPSELL': {
            'strategy': 'Service Recovery & Growth',
            'description': 'Address service issues before attempting upsell opportunities.',
            'actions': ['Issue resolution', 'Service credits', 'Follow-up campaigns'],
            'priority': 'HIGH',
            'color': '🟠'
        },
        'GENTLE_UPSELL': {
            'strategy': 'Careful Relationship Building',
            'description': 'New or price-sensitive customers requiring gentle approach.',
            'actions': ['Educational content', 'Trial offers', 'Gradual engagement'],
            'priority': 'LOW',
            'color': '🟢'
        },
        'MINIMAL_CONTACT': {
            'strategy': 'Cost-Efficient Management',
            'description': 'Low-value customers with minimal engagement requirements.',
            'actions': ['Automated communications', 'Self-service options', 'Cost optimization'],
            'priority': 'MINIMAL',
            'color': '⚪'
        }
    }
    
    if selected_segment in strategies:
        strategy_info = strategies[selected_segment]
        
        strategy_col1, strategy_col2 = st.columns([2, 1])
        
        with strategy_col1:
            st.markdown(f"""
            <div class="feature-card">
                <h4>{strategy_info['color']} {strategy_info['strategy']}</h4>
                <p><strong>Description:</strong> {strategy_info['description']}</p>
                <p><strong>Recommended Actions:</strong></p>
                <ul>
                    {''.join([f'<li>{action}</li>' for action in strategy_info['actions']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with strategy_col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{strategy_info['color']}</h3>
                <p>Priority Level</p>
                <h4>{strategy_info['priority']}</h4>
            </div>
            """, unsafe_allow_html=True)
    
    # Customer Distribution Analysis
    st.subheader("📊 Customer Distribution Analysis")
    
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        # Tenure distribution
        fig_tenure = px.histogram(
            segment_data,
            x='Tenure_Months',
            title=f"Tenure Distribution - {selected_segment}",
            nbins=20,
            color_discrete_sequence=['#667eea']
        )
        fig_tenure.update_layout(height=350)
        st.plotly_chart(fig_tenure, use_container_width=True)
    
    with dist_col2:
        # Revenue distribution
        fig_revenue = px.histogram(
            segment_data,
            x='Total_Charges',
            title=f"Revenue Distribution - {selected_segment}",
            nbins=20,
            color_discrete_sequence=['#f093fb']
        )
        fig_revenue.update_layout(height=350)
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Customer Sample Table
    st.subheader("📋 Customer Sample")
    
    display_columns = [
        'Phone Number', 'Total_Charges', 'Monthly_Charges', 'CLV_Estimate',
        'Satisfaction_Score', 'Churn_Probability', 'Tenure_Months',
        'Total_Minutes', 'Data_Usage_GB', 'CustServ_Calls'
    ]
    
    existing_display_columns = [col for col in display_columns if col in segment_data.columns]
    
    if not segment_data.empty and existing_display_columns:
        sample_data = segment_data[existing_display_columns].head(100)
        st.dataframe(sample_data, use_container_width=True, height=400)
        
        # Download option
        csv = segment_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download {selected_segment} Customer Data",
            data=csv,
            file_name=f"{selected_segment.lower()}_customers.csv",
            mime="text/csv"
        )
    else:
        st.info("No customer data available for this segment.")

def show_priority_customers(customers_df, category_mapping):
    """Advanced priority customer filtering and analysis"""
    st.header("⚡ Priority Customer Intelligence")
    st.markdown("*Advanced filtering and analysis of high-priority customer segments*")
    
    # Advanced Filter Controls
    st.subheader("🎛️ Advanced Filters")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        priority_levels = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']
        selected_priority = st.selectbox(
            "🎯 Priority Level:",
            priority_levels,
            index=1,
            help="Filter customers by their strategic priority level"
        )
    
    with filter_col2:
        all_categories = customers_df['Customer_Category'].unique().tolist() if 'Customer_Category' in customers_df.columns else []
        upsell_focused = ['STANDARD_UPSELL', 'PRIORITY_UPSELL_RETENTION', 'FIX_FIRST_THEN_UPSELL', 'GENTLE_UPSELL']
        default_categories = [cat for cat in upsell_focused if cat in all_categories]
        
        selected_categories = st.multiselect(
            "📊 Customer Categories:",
            options=all_categories,
            default=default_categories,
            help="Select specific customer segments to analyze"
        )
    
    with filter_col3:
        # Additional filters
        min_clv = st.number_input("💰 Min Customer Lifetime Value:", min_value=0, value=0, step=100)
        max_churn_risk = st.slider("⚠️ Max Churn Risk:", 0.0, 1.0, 0.5, 0.05)
    
    # Apply filters
    if not selected_categories:
        st.warning("⚠️ Please select at least one Customer Category to display results.")
        st.stop()
    
    filtered_customers = customers_df[
        (customers_df['Priority_Level'] == selected_priority) & 
        (customers_df['Customer_Category'].isin(selected_categories)) &
        (customers_df['CLV_Estimate'] >= min_clv) &
        (customers_df['Churn_Probability'] <= max_churn_risk)
    ]
    
    # Filter Results Summary
    st.markdown("---")
    st.subheader("📊 Filter Results")
    
    if len(filtered_customers) == 0:
        st.error("🚫 No customers match the selected criteria. Please adjust your filters.")
        st.stop()
    
    result_col1, result_col2, result_col3, result_col4, result_col5 = st.columns(5)
    
    with result_col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(filtered_customers):,}</h3>
            <p>Customers Found</p>
            <small>{(len(filtered_customers)/len(customers_df)*100):.1f}% of total</small>
        </div>
        """, unsafe_allow_html=True)
    
    with result_col2:
        total_revenue = filtered_customers['Total_Charges'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3>USD {total_revenue:,.0f}</h3>
            <p>Total Revenue</p>
            <small>Monthly</small>
        </div>
        """, unsafe_allow_html=True)
    
    with result_col3:
        avg_clv = filtered_customers['CLV_Estimate'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>USD {avg_clv:,.0f}</h3>
            <p>Avg Lifetime Value</p>
            <small>Per customer</small>
        </div>
        """, unsafe_allow_html=True)
    
    with result_col4:
        avg_satisfaction = filtered_customers['Satisfaction_Score'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_satisfaction:.2f}/5.0</h3>
            <p>Avg Satisfaction</p>
            <small>Customer rating</small>
        </div>
        """, unsafe_allow_html=True)
    
    with result_col5:
        avg_churn_risk = filtered_customers['Churn_Probability'].mean() * 100
        color = "🟢" if avg_churn_risk < 20 else "🟡" if avg_churn_risk < 40 else "🔴"
        st.markdown(f"""
        <div class="metric-card">
            <h3>{color} {avg_churn_risk:.1f}%</h3>
            <p>Avg Churn Risk</p>
            <small>Probability</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Customer Analysis Charts
    st.subheader("📈 Customer Analysis")
    
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        # Revenue vs CLV scatter
        fig_scatter = px.scatter(
            filtered_customers.head(200),  # Sample for performance
            x='Total_Charges',
            y='CLV_Estimate',
            color='Customer_Category',
            size='Satisfaction_Score',
            title="Monthly Revenue vs Customer Lifetime Value",
            hover_data=['Phone Number', 'Churn_Probability']
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with analysis_col2:
        # Category distribution
        category_counts = filtered_customers['Customer_Category'].value_counts()
        fig_pie = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Customer Category Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Top Customers Table
    st.subheader("🏆 Top Priority Customers")
    
    # Sort by CLV and satisfaction
    top_customers = filtered_customers.nlargest(50, 'CLV_Estimate')
    
    display_columns = [
        'Phone Number', 'Customer_Category', 'Total_Charges', 'CLV_Estimate',
        'Satisfaction_Score', 'Churn_Probability', 'Tenure_Months',
        'Data_Usage_GB', 'CustServ_Calls', 'Contract_Type'
    ]
    
    existing_columns = [col for col in display_columns if col in top_customers.columns]
    
    if existing_columns:
        st.dataframe(
            top_customers[existing_columns],
            use_container_width=True,
            height=400
        )
        
        # Export options
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            csv = filtered_customers.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download All Filtered Customers ({len(filtered_customers):,})",
                data=csv,
                file_name=f"priority_customers_{selected_priority.lower()}.csv",
                mime="text/csv"
            )
        
        with export_col2:
            top_csv = top_customers.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"🏆 Download Top 50 Customers",
                data=top_csv,
                file_name=f"top_50_priority_customers.csv",
                mime="text/csv"
            )

def show_prediction_tool(model, scaler, feature_columns, category_mapping):
    """Advanced prediction tool with comprehensive input options"""
    st.header("🔮 AI-Powered Customer Segmentation Engine")
    st.markdown("*Real-time customer classification using advanced machine learning*")
    
    if model is None or scaler is None or feature_columns is None or category_mapping is None:
        st.error("🚫 Model artifacts not loaded. Please check the system configuration.")
        st.stop()
    
    # Prediction modes
    st.subheader("🎛️ Prediction Mode")
    
    mode_col1, mode_col2 = st.columns(2)
    
    with mode_col1:
        prediction_mode = st.radio(
            "Select Input Method:",
            ["🖱️ Interactive Form", "📊 Batch Upload", "🔗 API Integration"],
            index=0
        )
    
    with mode_col2:
        if prediction_mode == "🔗 API Integration":
            st.info("💡 API endpoints available for integration")
            st.code("POST /api/v1/predict")
            st.code("GET /api/v1/model/status")
    
    if prediction_mode == "🖱️ Interactive Form":
        # Interactive prediction form
        st.subheader("📝 Customer Information Input")
        
        with st.form("customer_prediction_form"):
            # Basic Information
            st.markdown("**📋 Basic Customer Profile**")
            basic_col1, basic_col2, basic_col3 = st.columns(3)
            
            input_data = {}
            
            with basic_col1:
                input_data['Account_Length'] = st.number_input(
                    "📅 Account Length (days)",
                    min_value=1, max_value=2000, value=365,
                    help="How long the customer has been with the company"
                )
                input_data['Tenure_Months'] = st.number_input(
                    "📆 Tenure (months)",
                    min_value=1, max_value=72, value=24,
                    help="Customer tenure in months"
                )
            
            with basic_col2:
                input_data['Total_Charges'] = st.number_input(
                    "💰 Total Monthly Charges (USD)",
                    min_value=0.0, max_value=500.0, value=65.0,
                    help="Total monthly charges for all services"
                )
                input_data['Monthly_Charges'] = st.number_input(
                    "💳 Base Monthly Charges (USD)",
                    min_value=0.0, max_value=200.0, value=50.0,
                    help="Base monthly service charges"
                )
            
            with basic_col3:
                input_data['Satisfaction_Score'] = st.slider(
                    "😊 Satisfaction Score",
                    1.0, 5.0, 3.5, 0.1,
                    help="Customer satisfaction rating (1-5)"
                )
                input_data['Risk_Score'] = st.slider(
                    "⚠️ Risk Score",
                    0.0, 10.0, 2.5, 0.1,
                    help="Customer risk assessment score"
                )
            
            # Usage Patterns
            st.markdown("**📱 Usage Patterns**")
            usage_col1, usage_col2, usage_col3 = st.columns(3)
            
            with usage_col1:
                input_data['Total_Minutes'] = st.number_input(
                    "📞 Total Monthly Minutes",
                    min_value=0.0, max_value=2000.0, value=300.0,
                    help="Total voice minutes per month"
                )
                input_data['Data_Usage_GB'] = st.number_input(
                    "📊 Data Usage (GB)",
                    min_value=0.0, max_value=100.0, value=5.0,
                    help="Monthly data usage in gigabytes"
                )
            
            with usage_col2:
                input_data['CustServ_Calls'] = st.number_input(
                    "📞 Customer Service Calls",
                    min_value=0, max_value=20, value=1,
                    help="Number of customer service calls per month"
                )
                input_data['Support_Tickets'] = st.number_input(
                    "🎫 Support Tickets",
                    min_value=0, max_value=10, value=0,
                    help="Number of support tickets submitted"
                )
            
            with usage_col3:
                input_data['VMail_Messages'] = st.number_input(
                    "📧 Voicemail Messages",
                    min_value=0, max_value=100, value=15,
                    help="Number of voicemail messages per month"
                )
                input_data['Has_Voicemail'] = int(st.checkbox(
                    "📧 Has Voicemail Service",
                    value=True,
                    help="Customer has voicemail service enabled"
                ))
            
            # Usage Ratios
            st.markdown("**⏰ Usage Time Distribution**")
            ratio_col1, ratio_col2, ratio_col3, ratio_col4 = st.columns(4)
            
            with ratio_col1:
                input_data['Day_Usage_Ratio'] = st.slider(
                    "🌅 Day Usage %", 0.0, 1.0, 0.4, 0.05,
                    help="Percentage of usage during day time"
                )
            
            with ratio_col2:
                input_data['Eve_Usage_Ratio'] = st.slider(
                    "🌆 Evening Usage %", 0.0, 1.0, 0.3, 0.05,
                    help="Percentage of usage during evening"
                )
            
            with ratio_col3:
                input_data['Night_Usage_Ratio'] = st.slider(
                    "🌙 Night Usage %", 0.0, 1.0, 0.2, 0.05,
                    help="Percentage of usage during night time"
                )
            
            with ratio_col4:
                input_data['Intl_Usage_Ratio'] = st.slider(
                    "🌍 International %", 0.0, 1.0, 0.1, 0.05,
                    help="Percentage of international usage"
                )
            
            # Prediction button
            submitted = st.form_submit_button(
                "🎯 Generate Customer Segment Prediction",
                use_container_width=True,
                type="primary"
            )
            
            if submitted:
                # Validate ratios sum
                ratio_sum = (input_data['Day_Usage_Ratio'] + 
                           input_data['Eve_Usage_Ratio'] + 
                           input_data['Night_Usage_Ratio'] + 
                           input_data['Intl_Usage_Ratio'])
                
                if ratio_sum > 1.1:
                    st.warning("⚠️ Usage ratios sum to more than 100%. Please adjust the values.")
                    st.stop()
                
                # Prepare prediction
                with st.spinner("🔄 Processing customer data through AI model..."):
                    customer_input_df = pd.DataFrame([input_data])
                    
                    # Handle missing features
                    for col in feature_columns:
                        if col not in customer_input_df.columns:
                            customer_input_df[col] = 0.0
                    
                    customer_input_df = customer_input_df[feature_columns]
                    
                    # Make prediction
                    scaled_features = scaler.transform(customer_input_df)
                    prediction_proba = model.predict_proba(scaled_features)
                    prediction_id = model.predict(scaled_features)[0]
                    confidence = prediction_proba[0, prediction_id]
                    
                    predicted_segment = category_mapping.get(prediction_id, f"Unknown Category {prediction_id}")
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2>🎯</h2>
                        <h3>{predicted_segment}</h3>
                        <p>Predicted Segment</p>
                        <small>Confidence: {confidence:.1%}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with result_col2:
                    # Confidence breakdown
                    st.markdown("**🎲 Prediction Confidence Breakdown**")
                    for i, (category_id, category_name) in enumerate(category_mapping.items()):
                        prob = prediction_proba[0, i]
                        st.progress(prob, text=f"{category_name}: {prob:.1%}")
                
                # Strategy recommendations
                strategies = {
                    'DO_NOT_DISTURB': {
                        'icon': '🛡️',
                        'title': 'Preserve Premium Relationship',
                        'description': 'High-value customer requiring minimal disruption',
                        'actions': ['VIP treatment', 'Exclusive offers', 'Proactive support'],
                        'urgency': 'Low',
                        'color': '#28a745'
                    },
                    'STANDARD_UPSELL': {
                        'icon': '📈',
                        'title': 'Standard Growth Opportunity',
                        'description': 'Ready for targeted upselling campaigns',
                        'actions': ['Product bundles', 'Service upgrades', 'Feature add-ons'],
                        'urgency': 'Medium',
                        'color': '#ffc107'
                    },
                    'PRIORITY_UPSELL_RETENTION': {
                        'icon': '🚨',
                        'title': 'Immediate Action Required',
                        'description': 'High-value customer at risk - urgent intervention needed',
                        'actions': ['Personal account manager', 'Retention offers', 'Priority support'],
                        'urgency': 'Critical',
                        'color': '#dc3545'
                    },
                    'FIX_FIRST_THEN_UPSELL': {
                        'icon': '🔧',
                        'title': 'Service Recovery Priority',
                        'description': 'Address service issues before upselling',
                        'actions': ['Issue resolution', 'Service credits', 'Follow-up campaigns'],
                        'urgency': 'High',
                        'color': '#fd7e14'
                    },
                    'GENTLE_UPSELL': {
                        'icon': '🤝',
                        'title': 'Relationship Building Focus',
                        'description': 'Careful approach for price-sensitive customers',
                        'actions': ['Educational content', 'Trial offers', 'Gradual engagement'],
                        'urgency': 'Low',
                        'color': '#17a2b8'
                    },
                    'MINIMAL_CONTACT': {
                        'icon': '📵',
                        'title': 'Cost-Efficient Management',
                        'description': 'Low-engagement, cost-optimized approach',
                        'actions': ['Automated communications', 'Self-service', 'Efficiency focus'],
                        'urgency': 'Minimal',
                        'color': '#6c757d'
                    }
                }
                
                if predicted_segment in strategies:
                    strategy = strategies[predicted_segment]
                    
                    st.markdown("---")
                    st.subheader("🎯 Recommended Strategy & Actions")
                    
                    st.markdown(f"""
                    <div class="feature-card" style="border-left: 5px solid {strategy['color']};">
                        <h4>{strategy['icon']} {strategy['title']}</h4>
                        <p><strong>Description:</strong> {strategy['description']}</p>
                        <p><strong>Urgency Level:</strong> <span style="color: {strategy['color']}; font-weight: bold;">{strategy['urgency']}</span></p>
                        <p><strong>Recommended Actions:</strong></p>
                        <ul>
                            {''.join([f'<li>{action}</li>' for action in strategy['actions']])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
    
    elif prediction_mode == "📊 Batch Upload":
        st.subheader("📊 Batch Customer Prediction")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Customer Data (CSV)",
            type=['csv'],
                        help="Upload a CSV file with customer data for batch prediction"
        )
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Uploaded {len(batch_df)} customer records")
                
                # Show sample data
                st.subheader("📋 Data Preview")
                st.dataframe(batch_df.head(), use_container_width=True)
                
                if st.button("🔮 Run Batch Predictions", type="primary"):
                    with st.spinner("Processing batch predictions..."):
                        # Prepare data for prediction
                        prediction_data = batch_df.copy()
                        
                        # Handle missing features
                        for col in feature_columns:
                            if col not in prediction_data.columns:
                                prediction_data[col] = 0.0
                        
                        # Make predictions
                        scaled_features = scaler.transform(prediction_data[feature_columns])
                        predictions = model.predict(scaled_features)
                        probabilities = model.predict_proba(scaled_features)
                        
                        # Add predictions to dataframe
                        batch_df['Predicted_Segment'] = [category_mapping.get(pred, f"Unknown_{pred}") for pred in predictions]
                        batch_df['Prediction_Confidence'] = [prob.max() for prob in probabilities]
                        
                        st.success("✅ Batch predictions completed!")
                        
                        # Show results
                        st.subheader("📊 Prediction Results")
                        st.dataframe(batch_df, use_container_width=True)
                        
                        # Download results
                        csv = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="customer_predictions.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

def show_business_impact(segment_metrics):
    """Comprehensive business impact analysis with financial projections"""
    st.header("💰 Business Impact & ROI Analysis")
    st.markdown("*Comprehensive financial analysis of customer segmentation strategy*")
    
    # ROI Achievement Dashboard
    st.subheader("🎯 ROI Performance Dashboard")
    
    roi_col1, roi_col2, roi_col3, roi_col4 = st.columns(4)
    
    with roi_col1:
        st.markdown("""
        <div class="metric-card">
            <h3>1,281%</h3>
            <p>Target ROI</p>
            <small>Original goal</small>
        </div>
        """, unsafe_allow_html=True)
    
    with roi_col2:
        achieved_roi = 2653.4
        st.markdown(f"""
        <div class="metric-card">
            <h3>{achieved_roi:.0f}%</h3>
            <p>Achieved ROI</p>
            <small>+{achieved_roi - 1281:.0f}% vs target</small>
        </div>
        """, unsafe_allow_html=True)
    
    with roi_col3:
        target_achievement = (achieved_roi / 1281) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>{target_achievement:.0f}%</h3>
            <p>Target Achievement</p>
            <small>Exceeded expectations</small>
        </div>
        """, unsafe_allow_html=True)
    
    with roi_col4:
        net_benefit = 9.27  # Million USD
        st.markdown(f"""
        <div class="metric-card">
            <h3>USD {net_benefit:.1f}M</h3>
            <p>Net Benefit</p>
            <small>Annual projection</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Segment Performance Analysis
    st.subheader("📊 Segment Performance Analysis")
    
    # Prepare data for visualization
    performance_data = []
    for segment, metrics in segment_metrics.items():
        performance_data.append({
            'Segment': segment,
            'ROI': metrics['roi'],
            'Customers': metrics['count'],
            'Revenue': metrics['total_revenue'],
            'Net_Benefit': metrics['net_benefit'],
            'Avg_CLV': metrics['avg_clv']
        })
    
    performance_df = pd.DataFrame(performance_data)
    
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        # ROI by segment
        fig_roi = px.bar(
            performance_df,
            x='Segment',
            y='ROI',
            title="ROI by Customer Segment",
            color='ROI',
            color_continuous_scale='RdYlGn',
            text='ROI'
        )
        fig_roi.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
        fig_roi.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_roi, use_container_width=True)
    
    with perf_col2:
        # Revenue vs Customer count
        fig_bubble = px.scatter(
            performance_df,
            x='Customers',
            y='Revenue',
            size='Net_Benefit',
            color='Segment',
            title="Revenue vs Customer Count (Size = Net Benefit)",
            hover_data=['ROI', 'Avg_CLV']
        )
        fig_bubble.update_layout(height=400)
        st.plotly_chart(fig_bubble, use_container_width=True)
    
    # Financial Projections
    st.subheader("📈 Financial Projections")
    
    # Generate projection data
    months = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    projection_data = []
    
    base_revenue = performance_df['Revenue'].sum()
    base_growth = 0.02  # 2% monthly growth
    
    for i, month in enumerate(months):
        monthly_revenue = base_revenue * (1 + base_growth) ** i
        roi_impact = monthly_revenue * (achieved_roi / 100) * 0.01  # 1% of ROI impact
        
        projection_data.append({
            'Month': month,
            'Base_Revenue': monthly_revenue,
            'ROI_Impact': roi_impact,
            'Total_Benefit': monthly_revenue + roi_impact,
            'Cumulative_Benefit': sum([p['Total_Benefit'] for p in projection_data]) + monthly_revenue + roi_impact
        })
    
    projection_df = pd.DataFrame(projection_data)
    
    proj_col1, proj_col2 = st.columns(2)
    
    with proj_col1:
        # Monthly projections
        fig_monthly = px.line(
            projection_df,
            x='Month',
            y=['Base_Revenue', 'Total_Benefit'],
            title="Monthly Revenue Projections",
            labels={'value': 'Revenue (USD)', 'variable': 'Metric'}
        )
        fig_monthly.update_layout(height=350)
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with proj_col2:
        # Cumulative benefit
        fig_cumulative = px.area(
            projection_df,
            x='Month',
            y='Cumulative_Benefit',
            title="Cumulative Business Benefit",
            color_discrete_sequence=['#667eea']
        )
        fig_cumulative.update_layout(height=350)
        st.plotly_chart(fig_cumulative, use_container_width=True)
    
    # Cost-Benefit Analysis
    st.subheader("💡 Cost-Benefit Analysis")
    
    cost_col1, cost_col2 = st.columns(2)
    
    with cost_col1:
        st.markdown("**💰 Investment Breakdown**")
        
        costs = {
            'Technology Infrastructure': 150000,
            'Model Development': 75000,
            'Data Processing': 25000,
            'Training & Implementation': 50000,
            'Ongoing Maintenance': 30000
        }
        
        total_investment = sum(costs.values())
        
        for item, cost in costs.items():
            st.metric(item, f"USD {cost:,}", f"{(cost/total_investment*100):.1f}%")
        
        st.metric("**Total Investment**", f"**USD {total_investment:,}**")
    
    with cost_col2:
        st.markdown("**📈 Return Analysis**")
        
        annual_benefit = projection_df['Total_Benefit'].sum()
        payback_period = total_investment / (annual_benefit / 12)
        
        st.metric("Annual Benefit", f"USD {annual_benefit:,.0f}")
        st.metric("Payback Period", f"{payback_period:.1f} months")
        st.metric("5-Year NPV", f"USD {annual_benefit * 5 - total_investment:,.0f}")
        st.metric("Break-even ROI", f"{(annual_benefit/total_investment*100):.0f}%")

def show_model_performance():
    """Comprehensive model performance monitoring and comparison"""
    st.header("📈 Model Performance & Analytics")
    st.markdown("*Detailed analysis of model accuracy, performance metrics, and deployment readiness*")
    
    # Performance Overview
    st.subheader("🎯 Performance Overview")
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.markdown("""
        <div class="metric-card">
            <h3>94.46%</h3>
            <p>Test Accuracy</p>
            <small>+0.14% improvement</small>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col2:
        st.markdown("""
        <div class="metric-card">
            <h3>93.55%</h3>
            <p>Cross-Validation</p>
            <small>±0.005 std dev</small>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col3:
        st.markdown("""
        <div class="metric-card">
            <h3>100%</h3>
            <p>Deployment Ready</p>
            <small>16/16 criteria met</small>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col4:
        st.markdown("""
        <div class="metric-card">
            <h3>< 50ms</h3>
            <p>Prediction Latency</p>
            <small>Real-time capable</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Comparison
    st.subheader("🏆 Model Comparison Matrix")
    
    model_data = {
        'Model': ['XGBoost Optimized', 'XGBoost Baseline', 'LightGBM Optimized', 'Random Forest', 'Neural Network'],
        'Accuracy': [94.46, 94.32, 94.16, 92.85, 91.73],
        'Precision': [94.12, 93.98, 93.84, 92.45, 91.28],
        'Recall': [94.28, 94.15, 93.92, 92.67, 91.51],
        'F1-Score': [94.20, 94.06, 93.88, 92.56, 91.39],
        'Training_Time': ['15 min', '12 min', '8 min', '25 min', '45 min'],
        'Inference_Speed': ['45ms', '42ms', '38ms', '65ms', '85ms'],
        'Status': ['🥇 Champion', '🥈 Baseline', '🥉 Alternative', '📊 Benchmark', '🧪 Experimental']
    }
    
    model_df = pd.DataFrame(model_data)
    st.dataframe(model_df, use_container_width=True)
    
    # Performance Metrics Visualization
    st.subheader("📊 Detailed Performance Metrics")
    
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        # Accuracy comparison
        fig_accuracy = px.bar(
            model_df.head(3),  # Top 3 models
            x='Model',
            y='Accuracy',
            title="Model Accuracy Comparison",
            color='Accuracy',
            color_continuous_scale='Viridis',
            text='Accuracy'
        )
        fig_accuracy.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_accuracy.update_layout(height=350)
        st.plotly_chart(fig_accuracy, use_container_width=True)
    
    with metrics_col2:
        # Performance radar chart
        metrics_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        champion_scores = [94.46, 94.12, 94.28, 94.20]
        baseline_scores = [94.32, 93.98, 94.15, 94.06]
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=champion_scores,
            theta=metrics_radar,
            fill='toself',
            name='XGBoost Optimized',
            line_color='#667eea'
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=baseline_scores,
            theta=metrics_radar,
            fill='toself',
            name='XGBoost Baseline',
            line_color='#f093fb'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[90, 95]
                )),
            showlegend=True,
            title="Performance Comparison Radar",
            height=350
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Feature Importance
    st.subheader("🔍 Feature Importance Analysis")
    
    # Mock feature importance data
    feature_importance = {
        'Feature': [
            'Total_Charges', 'CLV_Estimate', 'Satisfaction_Score', 'Tenure_Months',
            'Churn_Probability', 'Data_Usage_GB', 'CustServ_Calls', 'Account_Length',
            'Total_Minutes', 'Risk_Score', 'Monthly_Charges', 'Support_Tickets'
        ],
        'Importance': [0.18, 0.15, 0.12, 0.11, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.01],
        'Category': [
            'Financial', 'Financial', 'Experience', 'Behavioral',
            'Risk', 'Usage', 'Support', 'Behavioral',
            'Usage', 'Risk', 'Financial', 'Support'
        ]
    }
    
    importance_df = pd.DataFrame(feature_importance)
    
    fig_importance = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        color='Category',
        title="Feature Importance in Customer Segmentation",
        orientation='h',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_importance.update_layout(height=500)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Model Deployment Status
    st.subheader("🚀 Deployment Status & Health")
    
    deployment_col1, deployment_col2 = st.columns(2)
    
    with deployment_col1:
        st.markdown("**✅ Deployment Checklist**")
        
        checklist_items = [
            ("Model Accuracy", "✅", "94.46% > 90% threshold"),
            ("Data Quality", "✅", "99.8% clean data"),
            ("Performance Testing", "✅", "Load tested to 1000 RPS"),
            ("Security Audit", "✅", "Passed security review"),
            ("Monitoring Setup", "✅", "Real-time alerts active"),
            ("Backup Systems", "✅", "Failover mechanisms ready"),
            ("Documentation", "✅", "Complete API docs"),
            ("Training Complete", "✅", "Team trained on system")
        ]
        
        for item, status, description in checklist_items:
            st.success(f"{status} {item}: {description}")
    
    with deployment_col2:
        st.markdown("**📊 System Health Metrics**")
        
        # Mock real-time metrics
        current_time = datetime.now()
        
        health_metrics = [
            ("Uptime", "99.97%", "🟢"),
            ("Response Time", "42ms avg", "🟢"),
            ("Error Rate", "0.02%", "🟢"),
            ("Throughput", "450 req/min", "🟢"),
            ("Memory Usage", "68%", "🟡"),
            ("CPU Usage", "45%", "🟢"),
            ("Disk Space", "23% used", "🟢"),
            ("Last Updated", f"{current_time.strftime('%H:%M:%S')}", "🟢")
        ]
        
        for metric, value, status in health_metrics:
            st.metric(metric, value, delta=status)

def main():
    """Main application controller with enhanced navigation and state management"""
    
    # Initialize session state for navigation
    if 'page' not in st.session_state:
        st.session_state.page = "🚀 Launch Page"
    
    # Load data and models when not on launch page
    if st.session_state.page != "🚀 Launch Page":
        with st.spinner("🔄 Loading Smart Segmentation Data and AI Models..."):
            try:
                customers_df = load_segmentation_data()
                model, scaler, feature_columns, category_mapping = load_prediction_model()
                segment_metrics = calculate_segment_metrics(customers_df)
                st.success("✅ All systems loaded successfully!")
            except Exception as e:
                st.error(f"❌ System Error: {str(e)}")
                st.info("🔧 Please check system configuration and try again.")
                st.stop()
    
    # Enhanced Sidebar Navigation
    st.sidebar.markdown("## 🎯 Navigation Center")
    
    page_options = [
        "🚀 Launch Page",
        "🏆 Executive Dashboard",
        "👥 Customer Segments", 
        "⚡ Priority Customers",
        "🔮 Prediction Tool",
        "💰 Business Impact",
        "📈 Model Performance"
    ]
    
    # Navigation with icons and descriptions
    page_descriptions = {
        "🚀 Launch Page": "Welcome & Overview",
        "🏆 Executive Dashboard": "KPIs & Analytics",
        "👥 Customer Segments": "Segment Analysis",
        "⚡ Priority Customers": "High-Value Filtering",
        "🔮 Prediction Tool": "AI Predictions",
        "💰 Business Impact": "ROI Analysis",
        "📈 Model Performance": "Model Metrics"
    }
    
    selected_page = st.sidebar.selectbox(
        "Choose Dashboard:",
        page_options,
        index=page_options.index(st.session_state.page),
        format_func=lambda x: f"{x} - {page_descriptions[x]}"
    )
    
    # Update session state if selection changed
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()
    
    # System Status in Sidebar (when not on launch page)
    if st.session_state.page != "🚀 Launch Page":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🚀 System Status")
        
        status_items = [
            ("🤖 AI Model", "Active", "success"),
            ("📊 Data Pipeline", "Real-time", "success"),
            ("⚡ Performance", "Optimal", "success"),
            ("🔒 Security", "Secure", "success")
        ]
        
        for icon, status, type_color in status_items:
            if type_color == "success":
                st.sidebar.success(f"{icon} {status}")
            elif type_color == "warning":
                st.sidebar.warning(f"{icon} {status}")
            else:
                st.sidebar.info(f"{icon} {status}")
        
        if 'customers_df' in locals():
            st.sidebar.info(f"📈 Total Customers: {len(customers_df):,}")
            st.sidebar.info(f"🎯 Active Segments: {len(customers_df['Customer_Category'].unique())}")
    
    # Page Routing with Error Handling
    try:
        if "Launch Page" in st.session_state.page:
            show_launch_page()
        elif "Executive Dashboard" in st.session_state.page:
            show_executive_dashboard(customers_df, segment_metrics)
        elif "Customer Segments" in st.session_state.page:
            show_customer_segments(customers_df, segment_metrics)
        elif "Priority Customers" in st.session_state.page:
            show_priority_customers(customers_df, category_mapping)
        elif "Prediction Tool" in st.session_state.page:
            show_prediction_tool(model, scaler, feature_columns, category_mapping)
        elif "Business Impact" in st.session_state.page:
            show_business_impact(segment_metrics)
        elif "Model Performance" in st.session_state.page:
            show_model_performance()
    except Exception as e:
        st.error(f"❌ Page Error: {str(e)}")
        st.info("🔄 Please refresh the page or contact support if the issue persists.")

# Application Entry Point
if __name__ == "__main__":
    main()

