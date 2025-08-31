#!/usr/bin/env python3
"""
GPU Setup Verification for AI Customer Upsell Prediction
Author: Mohanarajan P (@mohanarajanreddy)
"""

def verify_setup():
    print("🔍 AI Customer Upsell Prediction - Setup Verification")
    print("=" * 60)
    
    success = 0
    total = 6
    
    # Core packages
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        print("✅ Core ML packages (pandas, numpy, sklearn)")
        success += 1
    except ImportError as e:
        print(f"❌ Core packages failed: {e}")
    
    # Streamlit
    try:
        import streamlit as st
        print(f"✅ Streamlit {st.__version__}")
        success += 1
    except ImportError:
        print("❌ Streamlit not available")
    
    # Visualization
    try:
        import plotly, seaborn, matplotlib
        print("✅ Visualization packages")
        success += 1
    except ImportError:
        print("❌ Visualization packages missing")
    
    # XGBoost
    try:
        import xgboost as xgb
        print(f"✅ XGBoost {xgb.__version__}")
        success += 1
    except ImportError:
        print("❌ XGBoost not available")
    
    # GPU packages
    try:
        import cupy as cp
        import cudf, cuml
        print(f"✅ GPU packages (CuPy {cp.__version__})")
        success += 1
    except ImportError:
        print("⚠️  GPU packages not available (CPU mode)")
    
    # GPU test
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"✅ GPU: {gpu.name} ({gpu.memoryTotal}MB)")
            success += 1
        else:
            print("⚠️  No GPU detected")
    except:
        print("⚠️  GPU detection unavailable")
    
    print("=" * 60)
    print(f"📊 Status: {success}/{total} components ready")
    
    if success >= 4:
        print("🎉 Setup successful! Ready for development.")
        print("\n🚀 Next steps:")
        print("1. streamlit run src/dashboard/app.py")
        print("2. Start building your ML models!")
        return True
    else:
        print("⚠️  Setup incomplete. Check installation.")
        return False

if __name__ == "__main__":
    verify_setup()
