"""
CNC Tool Wear Prediction Dashboard
Interactive visualization for real-time monitoring
Author: Hamidreza Daneshsarand
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import time

# Page config
st.set_page_config(
    page_title="CNC Tool Wear Monitor",
    page_icon="🔧",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('tool_wear_predictor_model.pkl')
        scaler = joblib.load('tool_wear_scaler.pkl')
        features = joblib.load('feature_columns.pkl')
        return model, scaler, features
    except:
        return None, None, None

# Generate sample data
def generate_realtime_data():
    return {
        'spindle_speed_rpm': np.random.uniform(1000, 2000),
        'feed_rate_mm_min': np.random.uniform(100, 200),
        'cutting_depth_mm': np.random.uniform(0.5, 2),
        'cutting_time_hours': np.random.uniform(0, 100),
        'cutting_speed_m_min': np.random.uniform(150, 350),
        'temperature_rise_C': np.random.uniform(150, 350),
        'vibration_mm_s': np.random.uniform(0.3, 3),
        'power_consumption_kW': np.random.uniform(2, 6),
        'surface_roughness_um': np.random.uniform(0.8, 4),
        'coolant_flow_L_min': np.random.uniform(8, 15),
    }

# Main app
def main():
    # Header
    st.title("🏭 CNC Tool Wear Prediction Dashboard")
    st.markdown("**Real-time monitoring system with 98% accuracy**")
    
    # Load model
    model, scaler, features = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please run tool_wear_prediction.py first")
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Control Panel")
    machine_id = st.sidebar.selectbox(
        "Select Machine",
        ["DMG_MORI_01", "MAZAK_02", "OKUMA_03", "HAAS_04"]
    )
    
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 5)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔴 Real-time Monitor", "📊 Analytics", "💰 ROI Calculator", "📈 History"])
    
    with tab1:
        st.header("Real-time Tool Condition")
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Placeholder for live updates
        placeholder = st.empty()
        
        # Real-time monitoring loop
        if st.sidebar.button("🟢 Start Monitoring"):
            for i in range(10):  # Monitor for 10 iterations
                # Get current data
                current_data = generate_realtime_data()
                
                # Prepare for prediction (simplified)
                X = pd.DataFrame([current_data])
                # Add missing features with default values
                for feat in features:
                    if feat not in X.columns:
                        X[feat] = 0
                X = X[features]
                
                # Predict
                X_scaled = scaler.transform(X)
                wear_prob = model.predict_proba(X_scaled)[0][1]
                
                # Determine status
                if wear_prob > 0.7:
                    status = "🔴 CRITICAL"
                    color = "red"
                    action = "Replace Immediately"
                elif wear_prob > 0.5:
                    status = "🟡 WARNING"
                    color = "orange"
                    action = "Schedule Replacement"
                else:
                    status = "🟢 GOOD"
                    color = "green"
                    action = "Continue Operation"
                
                with placeholder.container():
                    # Update metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Tool Wear",
                            f"{wear_prob*100:.1f}%",
                            delta=f"{(wear_prob-0.5)*100:.1f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Vibration",
                            f"{current_data['vibration_mm_s']:.2f} mm/s",
                            delta=None
                        )
                    
                    with col3:
                        st.metric(
                            "Temperature",
                            f"{current_data['temperature_rise_C']:.0f}°C",
                            delta=None
                        )
                    
                    with col4:
                        st.metric(
                            "Status",
                            status.split()[1],
                            delta=None
                        )
                    
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = wear_prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"Tool Wear Level - {machine_id}"},
                        delta = {'reference': 70},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgreen"},
                                {'range': [50, 70], 'color': "lightyellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Action recommendation
                    st.info(f"💡 **Recommendation:** {action}")
                    
                    if wear_prob > 0.7:
                        st.error(f"⚠️ **Alert:** Tool #{i+1} needs immediate replacement!")
                        st.balloons()
                    
                time.sleep(refresh_rate)
    
    with tab2:
        st.header("Performance Analytics")
        
        # Create sample performance data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        performance_data = pd.DataFrame({
            'Date': dates,
            'Accuracy': np.random.uniform(94, 99, 30),
            'Precision': np.random.uniform(95, 99, 30),
            'Recall': np.random.uniform(96, 100, 30)
        })
        
        # Line chart
        fig = px.line(performance_data, x='Date', y=['Accuracy', 'Precision', 'Recall'],
                     title='Model Performance Over Time')
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance")
        features_df = pd.DataFrame({
            'Feature': ['Thermal Load', 'Life Percentage', 'Cutting Time', 'Spindle Speed', 'Vibration'],
            'Importance': [29.07, 13.65, 13.04, 8.55, 3.43]
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(features_df, x='Importance', y='Feature', orientation='h',
                    title='Top 5 Most Important Features')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("💰 ROI Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Parameters")
            num_machines = st.number_input("Number of CNC Machines", 1, 100, 10)
            tool_cost = st.number_input("Cost per Tool (€)", 100, 1000, 500)
            downtime_cost = st.number_input("Downtime Cost per Hour (€)", 500, 2000, 1000)
            implementation_cost = st.number_input("Implementation Cost (€)", 1000, 10000, 5000)
        
        with col2:
            st.subheader("Calculated Savings")
            
            # Calculate savings
            failures_prevented = num_machines * 0.5  # Average 0.5 per machine per month
            tool_savings = failures_prevented * tool_cost
            downtime_savings = failures_prevented * 3 * downtime_cost  # 3 hours average
            monthly_savings = tool_savings + downtime_savings
            yearly_savings = monthly_savings * 12
            roi = ((yearly_savings - implementation_cost) / implementation_cost) * 100
            
            st.metric("Monthly Savings", f"€{monthly_savings:,.0f}")
            st.metric("Yearly Savings", f"€{yearly_savings:,.0f}")
            st.metric("ROI", f"{roi:.1f}%")
            
            # Pie chart of savings
            fig = px.pie(
                values=[tool_savings, downtime_savings],
                names=['Tool Cost Savings', 'Downtime Prevention'],
                title='Savings Breakdown'
            )
            st.plotly_chart(fig)
    
    with tab4:
        st.header("Historical Data")
        
        # Load CSV data
        try:
            df = pd.read_csv('cnc_simulation_data.csv')
            
            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Operations", len(df))
            with col2:
                st.metric("Tools Monitored", df['tool_id'].nunique())
            with col3:
                st.metric("Failure Rate", f"{(df['tool_condition']==2).mean()*100:.1f}%")
            
            # Time series plot
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            daily_stats = df.groupby(df['timestamp'].dt.date)['tool_condition'].value_counts().unstack(fill_value=0)
            
            fig = px.area(daily_stats, title='Tool Condition Distribution Over Time')
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.subheader("Recent Operations")
            st.dataframe(df.tail(10))
            
        except:
            st.info("No historical data available. Run CNC_Carbide_Tool_Data_Generator.py to generate data.")

if __name__ == "__main__":
    main()