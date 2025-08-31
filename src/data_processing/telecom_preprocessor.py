import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class TelecomDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def load_and_clean_data(self, file_path):
        """Load and clean the telecom dataset"""
        df = pd.read_csv(file_path)
        
        # Remove duplicates (as mentioned in your plan - 40,729 duplicates)
        print(f"Original dataset shape: {df.shape}")
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Convert Churn to binary
        if 'Churn' in df.columns:
            df['Churn_Binary'] = df['Churn'].map({'FALSE': 0, 'True': 1, False: 0, True: 1})
            df['Churn_Binary'] = df['Churn_Binary'].fillna(0).astype(int)
        
        return df
    
    def engineer_advanced_features(self, df):
        """Create the 43 engineered features as per your plan"""
        
        # Usage Aggregation Features
        df['Total_Minutes'] = df['Day Mins'] + df['Eve Mins'] + df['Night Mins'] + df['Intl Mins']
        df['Total_Calls'] = df['Day Calls'] + df['Eve Calls'] + df['Night Calls'] + df['Intl Calls']
        df['Total_Charges'] = df['Day Charge'] + df['Eve Charge'] + df['Night Charge'] + df['Intl Charge']
        
        # Usage Intensity Features
        df['Avg_Call_Duration'] = df['Total_Minutes'] / (df['Total_Calls'] + 1e-6)
        df['Day_Call_Duration'] = df['Day Mins'] / (df['Day Calls'] + 1e-6)
        df['Eve_Call_Duration'] = df['Eve Mins'] / (df['Eve Calls'] + 1e-6)
        df['Night_Call_Duration'] = df['Night Mins'] / (df['Night Calls'] + 1e-6)
        
        # Usage Pattern Ratios
        df['Day_Usage_Ratio'] = df['Day Mins'] / (df['Total_Minutes'] + 1e-6)
        df['Eve_Usage_Ratio'] = df['Eve Mins'] / (df['Total_Minutes'] + 1e-6)
        df['Night_Usage_Ratio'] = df['Night Mins'] / (df['Total_Minutes'] + 1e-6)
        df['Intl_Usage_Ratio'] = df['Intl Mins'] / (df['Total_Minutes'] + 1e-6)
        
        # Revenue-Based Features
        df['Revenue_Per_Minute'] = df['Total_Charges'] / (df['Total_Minutes'] + 1e-6)
        df['Day_Revenue_Rate'] = df['Day Charge'] / (df['Day Mins'] + 1e-6)
        df['Eve_Revenue_Rate'] = df['Eve Charge'] / (df['Eve Mins'] + 1e-6)
        df['Night_Revenue_Rate'] = df['Night Charge'] / (df['Night Mins'] + 1e-6)
        
        # Customer Lifecycle Features
        df['Account_Length_Months'] = df['Account Length'] / 30.0
        df['Usage_Per_Day'] = df['Total_Minutes'] / (df['Account Length'] + 1)
        df['Revenue_Per_Day'] = df['Total_Charges'] / (df['Account Length'] + 1)
        df['Service_Calls_Per_Month'] = df['CustServ Calls'] / (df['Account_Length_Months'] + 1e-6)
        
        # Behavioral Indicators (Binary Flags)
        df['Is_Heavy_Day_User'] = (df['Day Mins'] > df['Day Mins'].quantile(0.75)).astype(int)
        df['Is_Heavy_Eve_User'] = (df['Eve Mins'] > df['Eve Mins'].quantile(0.75)).astype(int)
        df['Is_Heavy_Night_User'] = (df['Night Mins'] > df['Night Mins'].quantile(0.75)).astype(int)
        df['Is_Intl_User'] = (df['Intl Mins'] > 0).astype(int)
        df['Is_High_Service_User'] = (df['CustServ Calls'] > 2).astype(int)
        df['Has_Voicemail'] = (df['VMail Message'] > 0).astype(int)
        df['Is_High_Value_Customer'] = (df['Total_Charges'] > df['Total_Charges'].quantile(0.8)).astype(int)
        
        # Risk Indicators
        df['Churn_Risk_Score'] = (
            df['CustServ Calls'] * 0.4 +
            (df['Total_Minutes'] < df['Total_Minutes'].quantile(0.25)) * 0.3 +
            (df['Account Length'] < 100) * 0.3
        )
        
        # Advanced ML Features
        # Usage intensity scores
        df['Day_Intensity'] = df['Day Mins'] / (df['Day Calls'] + 1e-6)
        df['Eve_Intensity'] = df['Eve Mins'] / (df['Eve Calls'] + 1e-6)
        df['Night_Intensity'] = df['Night Mins'] / (df['Night Calls'] + 1e-6)
        df['Intl_Intensity'] = df['Intl Mins'] / (df['Intl Calls'] + 1e-6)
        
        # Customer value segments
        df['Customer_Value_Score'] = (
            df['Total_Charges'] * 0.4 +
            df['Account Length'] * 0.3 +
            df['Total_Minutes'] * 0.3
        )
        
        # Engagement metrics
        df['Engagement_Score'] = (
            df['Has_Voicemail'] * 0.2 +
            df['Is_Intl_User'] * 0.3 +
            (df['CustServ Calls'] == 0) * 0.5  # No service calls = good engagement
        )
        
        return df
    
    def create_upsell_features(self, df):
        """Convert churn prediction to upsell opportunities"""
        
        # Upsell propensity based on usage patterns and churn risk
        df['Upsell_Propensity'] = (
            df['Is_High_Value_Customer'] * 0.3 +
            df['Is_Heavy_Day_User'] * 0.2 +
            df['Is_Heavy_Eve_User'] * 0.2 +
            df['Is_Intl_User'] * 0.2 +
            (1 - df['Churn_Risk_Score'] / df['Churn_Risk_Score'].max()) * 0.1
        )
        
        # Recommended products based on usage patterns
        conditions = [
            (df['Is_Heavy_Day_User'] == 1) & (df['Day_Usage_Ratio'] > 0.4),
            (df['Is_Heavy_Eve_User'] == 1) & (df['Eve_Usage_Ratio'] > 0.4),
            (df['Is_Heavy_Night_User'] == 1) & (df['Night_Usage_Ratio'] > 0.4),
            (df['Is_Intl_User'] == 1) & (df['Intl_Usage_Ratio'] > 0.1),
            (df['Is_High_Service_User'] == 1),
            (df['Has_Voicemail'] == 1) & (df['VMail Message'] > 20)
        ]
        
        choices = [
            'Unlimited Day Plan',
            'Evening Special Package',
            'Night Owl Plan',
            'International Package',
            'Premium Support Plan',
            'Enhanced Voicemail Package'
        ]
        
        df['Recommended_Product'] = np.select(conditions, choices, default='Standard Upgrade')
        
        # Expected revenue based on current spending and upsell propensity
        df['Expected_Monthly_Revenue'] = (
            df['Total_Charges'] * (1 + df['Upsell_Propensity']) * 
            np.random.uniform(1.2, 2.5, len(df))  # Upsell multiplier
        )
        
        # Priority levels
        df['Priority'] = pd.cut(
            df['Upsell_Propensity'], 
            bins=[0, 0.25, 0.5, 0.75, 1.0], 
            labels=['LOW', 'MEDIUM', 'HIGH', 'VERY HIGH']
        )
        
        return df
    
    def prepare_ml_features(self, df):
        """Prepare features for ML models"""
        
        # Select features for ML (excluding targets and identifiers)
        feature_columns = [
            'Account Length', 'VMail Message', 'Day Mins', 'Day Calls', 'Day Charge',
            'Eve Mins', 'Eve Calls', 'Eve Charge', 'Night Mins', 'Night Calls', 
            'Night Charge', 'Intl Mins', 'Intl Calls', 'Intl Charge', 'CustServ Calls',
            'Total_Minutes', 'Total_Calls', 'Total_Charges', 'Avg_Call_Duration',
            'Day_Call_Duration', 'Eve_Call_Duration', 'Night_Call_Duration',
            'Day_Usage_Ratio', 'Eve_Usage_Ratio', 'Night_Usage_Ratio', 'Intl_Usage_Ratio',
            'Revenue_Per_Minute', 'Day_Revenue_Rate', 'Eve_Revenue_Rate', 'Night_Revenue_Rate',
            'Account_Length_Months', 'Usage_Per_Day', 'Revenue_Per_Day', 'Service_Calls_Per_Month',
            'Is_Heavy_Day_User', 'Is_Heavy_Eve_User', 'Is_Heavy_Night_User', 'Is_Intl_User',
            'Is_High_Service_User', 'Has_Voicemail', 'Is_High_Value_Customer', 'Churn_Risk_Score',
            'Customer_Value_Score', 'Engagement_Score', 'Upsell_Propensity'
        ]
        
        X = df[feature_columns].fillna(0)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        return X, feature_columns
    
    def detect_anomalies(self, X):
        """Detect anomalous customers for special handling"""
        anomaly_scores = self.anomaly_detector.fit_predict(X)
        return anomaly_scores
    
    def apply_pca(self, X):
        """Apply PCA for dimensionality reduction"""
        X_pca = self.pca.fit_transform(X)
        return X_pca
