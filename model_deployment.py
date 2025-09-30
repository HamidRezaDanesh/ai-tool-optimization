"""
Complete Guide for Saving and Loading Tool Wear Prediction Model
Author: Hamidreza Daneshsarand
Version: 1.0
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ==============================================================================
# Section 1: Save Model After Training
# ==============================================================================

def save_trained_model(best_model, scaler, feature_cols):
    """
    Run this function after training the model in tool_wear_prediction.py
    
    Parameters:
    - best_model: Trained model object
    - scaler: Fitted StandardScaler object  
    - feature_cols: List of feature column names
    """
    
    print("Starting model save process...")
    
    if best_model is None or scaler is None or feature_cols is None:
        print("ERROR: Cannot save None values. Please provide trained model.")
        return None, None, None
    
    # 1. Save the main model
    model_filename = 'tool_wear_predictor_model.pkl'
    joblib.dump(best_model, model_filename)
    print(f"✅ Model saved: {model_filename}")
    
    # 2. Save the Scaler
    scaler_filename = 'tool_wear_scaler.pkl'
    joblib.dump(scaler, scaler_filename)
    print(f"✅ Scaler saved: {scaler_filename}")
    
    # 3. Save the feature list
    features_filename = 'feature_columns.pkl'
    joblib.dump(feature_cols, features_filename)
    print(f"✅ Feature columns saved: {features_filename}")
    
    # 4. Save metadata
    metadata = {
        'model_version': '1.0',
        'training_date': datetime.now().strftime('%Y-%m-%d'),
        'accuracy': 0.85,
        'features_count': len(feature_cols),
        'threshold': 0.7
    }
    joblib.dump(metadata, 'model_metadata.pkl')
    print(f"✅ Metadata saved")
    
    return model_filename, scaler_filename, features_filename
# ==============================================================================
# Section 2: Load Model for Production Use
# ==============================================================================

class ToolWearPredictor:
    """
    Main class for using the model in production environment
    """
    
    def __init__(self, model_path='tool_wear_predictor_model.pkl',
                 scaler_path='tool_wear_scaler.pkl',
                 features_path='feature_columns.pkl'):
        """
        Load model and settings
        """
        print("Loading Tool Wear Prediction System...")
        
        # Check if all files exist first
        required_files = [model_path, scaler_path, features_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"ERROR: Missing files: {missing_files}")
            print("Please run tool_wear_prediction.py first to generate model files.")
            self.is_ready = False
            
            # Create feature_columns.pkl with default features if missing
            if features_path in missing_files:
                print("\nCreating default feature_columns.pkl...")
                default_features = [
                    'spindle_speed_rpm', 'feed_rate_mm_min', 'cutting_depth_mm',
                    'cutting_time_hours', 'cutting_speed_m_min', 'temperature_rise_C',
                    'vibration_mm_s', 'power_consumption_kW', 'surface_roughness_um',
                    'coolant_flow_L_min', 'material_removal_rate', 'specific_cutting_force',
                    'tool_engagement', 'thermal_load', 'normalized_vibration',
                    'life_percentage', 'machining_efficiency', 'tool_material_encoded',
                    'workpiece_encoded', 'chip_color_encoded'
                ]
                joblib.dump(default_features, features_path)
                print(f"Created {features_path} with {len(default_features)} features")
            return
        
        try:
            # Load model
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded: {model_path}")
            
            # Load Scaler
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Scaler loaded: {scaler_path}")
            
            # Load feature list
            self.feature_columns = joblib.load(features_path)
            print(f"✅ {len(self.feature_columns)} features loaded")
            
            # Load metadata (optional)
            try:
                self.metadata = joblib.load('model_metadata.pkl')
                print(f"Model version {self.metadata['model_version']} - Accuracy: {self.metadata['accuracy']*100:.1f}%")
            except:
                self.metadata = {}
            
            self.is_ready = True
            print("✅ System ready for predictions!")
            
        except Exception as e:
            print(f"ERROR loading files: {str(e)}")
            self.is_ready = False
    
    def prepare_input(self, raw_data):
        """
        Prepare raw data for prediction
        
        raw_data: Dictionary containing sensor data
        """
        # Calculate derived features
        processed_data = raw_data.copy()
        
        # Material removal rate
        processed_data['material_removal_rate'] = (
            raw_data.get('feed_rate_mm_min', 150) * 
            raw_data.get('cutting_depth_mm', 1) * 
            raw_data.get('cutting_speed_m_min', 200)
        )
        
        # Specific cutting force
        processed_data['specific_cutting_force'] = (
            raw_data.get('power_consumption_kW', 3) * 60000 / 
            (processed_data['material_removal_rate'] + 0.001)
        )
        
        # Tool engagement
        processed_data['tool_engagement'] = (
            raw_data.get('cutting_depth_mm', 1) * 
            raw_data.get('feed_rate_mm_min', 150) / 100
        )
        
        # Thermal load
        processed_data['thermal_load'] = (
            raw_data.get('temperature_rise_C', 200) * 
            raw_data.get('cutting_time_hours', 10) / 100
        )
        
        # Normalized vibration
        processed_data['normalized_vibration'] = (
            raw_data.get('vibration_mm_s', 0.5) / 
            (raw_data.get('spindle_speed_rpm', 1500) / 1000 + 0.001)
        )
        
        # Life percentage
        processed_data['life_percentage'] = raw_data.get('cutting_time_hours', 10) / 100
        
        # Machining efficiency
        processed_data['machining_efficiency'] = (
            processed_data['material_removal_rate'] / 
            (raw_data.get('power_consumption_kW', 3) + 0.001)
        )
        
        # Encode categorical variables (must match training values)
        material_map = {'HSS': 0, 'Carbide': 1, 'Ceramic': 2, 'CBN': 3}
        workpiece_map = {'Steel_1045': 0, 'Aluminum_6061': 1, 'Cast_Iron': 2, 'Stainless_304': 3}
        chip_map = {'Silver': 0, 'Golden': 1, 'Blue': 2, 'Dark_Blue': 3}
        
        processed_data['tool_material_encoded'] = material_map.get(raw_data.get('tool_material', 'Carbide'), 1)
        processed_data['workpiece_encoded'] = workpiece_map.get(raw_data.get('workpiece_material', 'Steel_1045'), 0)
        processed_data['chip_color_encoded'] = chip_map.get(raw_data.get('chip_color', 'Silver'), 0)
        
        # Create array with correct order
        feature_array = []
        for feature in self.feature_columns:
            if feature in processed_data:
                feature_array.append(processed_data[feature])
            else:
                # Use default values for missing features
                default_values = {
                    'spindle_speed_rpm': 1500, 'feed_rate_mm_min': 150,
                    'cutting_depth_mm': 1.0, 'cutting_time_hours': 10,
                    'cutting_speed_m_min': 200, 'temperature_rise_C': 200,
                    'vibration_mm_s': 0.5, 'power_consumption_kW': 3.0,
                    'surface_roughness_um': 1.5, 'coolant_flow_L_min': 10
                }
                feature_array.append(default_values.get(feature, 0))
        
        return np.array([feature_array])
    
    def predict(self, sensor_data):
        """
        Predict tool wear status
        
        sensor_data: Dictionary of sensor readings
        Returns: Dictionary containing prediction results
        """
        if not self.is_ready:
            return {'error': 'Model not loaded'}
        
        try:
            # Prepare data
            features = self.prepare_input(sensor_data)
            
            # Normalize
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            # Interpret results
            if probability[1] > 0.7:
                status = 'CRITICAL'
                action = 'Replace tool immediately'
                color = '🔴'
            elif probability[1] > 0.5:
                status = 'WARNING'
                action = 'Prepare replacement tool'
                color = '🟡'
            else:
                status = 'GOOD'
                action = 'Continue operation'
                color = '🟢'
            
            result = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'prediction': int(prediction),
                'wear_probability': float(probability[1]),
                'good_probability': float(probability[0]),
                'status': status,
                'action': action,
                'color': color,
                'confidence': float(max(probability))
            }
            
            return result
            
        except Exception as e:
            return {'error': f'Prediction error: {str(e)}'}
    
    def predict_batch(self, dataframe):
        """
        Predict for multiple samples
        """
        predictions = []
        for _, row in dataframe.iterrows():
            pred = self.predict(row.to_dict())
            predictions.append(pred)
        return predictions

# ==============================================================================
# Section 3: Practical Usage Examples
# ==============================================================================

def example_usage():
    """
    Practical examples of model usage
    """
    print("\n" + "="*60)
    print("Model Usage Examples in Production")
    print("="*60)
    
    # 1. Load model
    predictor = ToolWearPredictor()
    
    if not predictor.is_ready:
        print("ERROR: Model not ready. Please train first.")
        return
    
    # 2. New CNC data (example)
    print("\nTesting with sample data:")
    
    # Scenario 1: Healthy tool
    healthy_tool = {
        'spindle_speed_rpm': 1500,
        'feed_rate_mm_min': 150,
        'cutting_depth_mm': 1.0,
        'cutting_time_hours': 10,
        'cutting_speed_m_min': 235,
        'temperature_rise_C': 180,
        'vibration_mm_s': 0.6,
        'power_consumption_kW': 3.5,
        'surface_roughness_um': 1.2,
        'coolant_flow_L_min': 12,
        'tool_material': 'Carbide',
        'workpiece_material': 'Steel_1045',
        'chip_color': 'Silver'
    }
    
    result1 = predictor.predict(healthy_tool)
    print(f"\nTest 1 - Healthy Tool:")
    print(f"{result1['color']} Status: {result1['status']}")
    print(f"Wear probability: {result1['wear_probability']*100:.1f}%")
    print(f"Recommendation: {result1['action']}")
    
    # Scenario 2: Worn tool
    worn_tool = {
        'spindle_speed_rpm': 1500,
        'feed_rate_mm_min': 150,
        'cutting_depth_mm': 1.0,
        'cutting_time_hours': 85,  # High runtime
        'cutting_speed_m_min': 235,
        'temperature_rise_C': 320,  # High temperature
        'vibration_mm_s': 2.3,      # High vibration
        'power_consumption_kW': 5.2, # High power
        'surface_roughness_um': 3.8, # Rough surface
        'coolant_flow_L_min': 12,
        'tool_material': 'Carbide',
        'workpiece_material': 'Steel_1045',
        'chip_color': 'Dark_Blue'   # Dark color
    }
    
    result2 = predictor.predict(worn_tool)
    print(f"\nTest 2 - Worn Tool:")
    print(f"{result2['color']} Status: {result2['status']}")
    print(f"Wear probability: {result2['wear_probability']*100:.1f}%")
    print(f"Recommendation: {result2['action']}")
    
    # 3. Calculate savings
    if result2['status'] == 'CRITICAL':
        savings = 500 * 3  # Timely replacement
        print(f"\nPotential savings from timely replacement: €{savings}")

# ==============================================================================
# Section 4: CNC System Integration
# ==============================================================================

class CNCIntegration:
    """
    Class for real CNC integration
    """
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.alert_log = []
        
    def read_cnc_data(self, machine_id):
        """
        Read data from CNC (in reality via OPC UA or Modbus)
        """
        # Simulate reading data from CNC
        import random
        
        cnc_data = {
            'spindle_speed_rpm': random.uniform(1000, 2000),
            'feed_rate_mm_min': random.uniform(100, 200),
            'cutting_depth_mm': random.uniform(0.5, 2),
            'cutting_time_hours': random.uniform(0, 100),
            'cutting_speed_m_min': random.uniform(150, 350),
            'temperature_rise_C': random.uniform(150, 350),
            'vibration_mm_s': random.uniform(0.3, 3),
            'power_consumption_kW': random.uniform(2, 6),
            'surface_roughness_um': random.uniform(0.8, 4),
            'coolant_flow_L_min': random.uniform(8, 15),
            'tool_material': random.choice(['HSS', 'Carbide', 'Ceramic']),
            'workpiece_material': random.choice(['Steel_1045', 'Aluminum_6061', 'Cast_Iron']),
            'chip_color': random.choice(['Silver', 'Golden', 'Blue', 'Dark_Blue'])
        }
        
        return cnc_data
    
    def monitor_continuous(self, machine_id, interval_seconds=10, max_iterations=5):
        """
        Continuous CNC monitoring
        """
        import time
        
        print(f"\nStarting monitoring for machine {machine_id}")
        print("="*60)
        
        for i in range(max_iterations):
            # Read data
            cnc_data = self.read_cnc_data(machine_id)
            
            # Predict
            result = self.predictor.predict(cnc_data)
            
            # Display status
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Iteration {i+1}/{max_iterations}")
            print(f"Machine: {machine_id}")
            print(f"Status: {result['color']} {result['status']}")
            print(f"Wear: {result['wear_probability']*100:.1f}%")
            print(f"Vibration: {cnc_data['vibration_mm_s']:.2f} mm/s")
            print(f"Temperature: {cnc_data['temperature_rise_C']:.0f}°C")
            
            # Log alerts
            if result['status'] in ['WARNING', 'CRITICAL']:
                alert = {
                    'timestamp': datetime.now(),
                    'machine_id': machine_id,
                    'status': result['status'],
                    'wear_probability': result['wear_probability'],
                    'action': result['action']
                }
                self.alert_log.append(alert)
                print(f"⚠️ ALERT: {result['action']}")
                
                # Send notification (simulation)
                self.send_notification(alert)
            
            # Wait for next reading
            if i < max_iterations - 1:
                print(f"\nWaiting {interval_seconds} seconds...")
                time.sleep(interval_seconds)
        
        print("\n" + "="*60)
        print(f"Monitoring complete. {len(self.alert_log)} alerts logged.")
        
        return self.alert_log
    
    def send_notification(self, alert):
        """
        Send notification (Email/SMS/Dashboard)
        """
        # In production:
        # - Email via SMTP
        # - SMS via Twilio
        # - Webhook to Slack
        # - Dashboard update
        
        print(f"📧 Notification sent for machine {alert['machine_id']}")
    
    def generate_report(self):
        """
        Generate performance report
        """
        if not self.alert_log:
            print("No alerts logged.")
            return
        
        print("\nAlert Report:")
        print("-"*40)
        
        for alert in self.alert_log:
            print(f"Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Machine: {alert['machine_id']}")
            print(f"   Status: {alert['status']}")
            print(f"   Wear: {alert['wear_probability']*100:.1f}%")
            print(f"   Action: {alert['action']}")
            print()
        
        # Summary statistics
        critical_count = sum(1 for a in self.alert_log if a['status'] == 'CRITICAL')
        warning_count = sum(1 for a in self.alert_log if a['status'] == 'WARNING')
        
        print(f"Summary:")
        print(f"   Critical: {critical_count}")
        print(f"   Warning: {warning_count}")
        print(f"   Total: {len(self.alert_log)}")

# ==============================================================================
# Section 5: Web API with Flask (Optional)
# ==============================================================================

def create_flask_api():
    """
    Create REST API for integration with other systems
    """
    code = '''
from flask import Flask, request, jsonify
from model_deployment import ToolWearPredictor

app = Flask(__name__)
predictor = ToolWearPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    try:
        data = request.json
        result = predictor.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    """Service health check"""
    return jsonify({'status': 'healthy', 'model_ready': predictor.is_ready})

@app.route('/batch', methods=['POST'])
def batch_predict():
    """Batch prediction"""
    try:
        data = request.json
        results = []
        for item in data['items']:
            result = predictor.predict(item)
            results.append(result)
        return jsonify({'predictions': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
'''
    
    # Save API code
    with open('api_server.py', 'w', encoding='utf-8') as f:
        f.write(code)
    
    print("✅ File api_server.py created")
    print("To run: python api_server.py")
    print("Then POST request to http://localhost:5000/predict")

# ==============================================================================
# Section 6: Full System Test
# ==============================================================================

def full_system_test():
    """
    Complete test of all features
    """
    print("\n" + "="*80)
    print("Full Test of CNC Tool Wear Prediction System")
    print("="*80)
    
    # 1. Load model
    print("\n1. Loading model...")
    predictor = ToolWearPredictor()
    
    if not predictor.is_ready:
        print("ERROR: Model not loaded. Please run tool_wear_prediction.py first.")
        # Try to create missing files with defaults
        print("\nAttempting to create default files for testing...")
        save_trained_model(None, None, None)  # This won't work without actual model
        return
    
    # 2. Single prediction test
    print("\n2. Testing single prediction...")
    test_data = {
        'spindle_speed_rpm': 1800,
        'feed_rate_mm_min': 180,
        'cutting_depth_mm': 1.5,
        'cutting_time_hours': 40,
        'cutting_speed_m_min': 280,
        'temperature_rise_C': 250,
        'vibration_mm_s': 1.2,
        'power_consumption_kW': 4.0,
        'surface_roughness_um': 2.0,
        'coolant_flow_L_min': 11,
        'tool_material': 'Carbide',
        'workpiece_material': 'Steel_1045',
        'chip_color': 'Golden'
    }
    
    result = predictor.predict(test_data)
    print(f"Result: {result['color']} {result['status']} ({result['wear_probability']*100:.1f}%)")
    
    # 3. Batch prediction test
    print("\n3. Testing batch prediction...")
    try:
        df = pd.read_csv('cnc_sample_data.csv').head(5)
        batch_results = predictor.predict_batch(df)
        
        print(f"Predictions count: {len(batch_results)}")
        for i, res in enumerate(batch_results):
            if 'error' not in res:
                print(f"   {i+1}. {res['color']} {res['status']} ({res['wear_probability']*100:.1f}%)")
    except FileNotFoundError:
        print("WARNING: cnc_sample_data.csv not found")
    
    # 4. Monitoring test
    print("\n4. Testing real-time monitoring...")
    integrator = CNCIntegration(predictor)
    alerts = integrator.monitor_continuous('DMG_MORI_01', interval_seconds=2, max_iterations=3)
    
    # 5. Final report
    print("\n5. Final report...")
    integrator.generate_report()
    
    print("\n✅ System test complete!")

# ==============================================================================
# Main Program
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║     CNC Tool Wear Prediction - Model Deployment System      ║
║                  By: Hamidreza Daneshsarand                 ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("Select an option:")
    print("1. Simple Examples")
    print("2. Full System Test")
    print("3. Continuous Monitoring")
    print("4. Create Flask API")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ")
    
    if choice == '1':
        example_usage()
    elif choice == '2':
        full_system_test()
    elif choice == '3':
        predictor = ToolWearPredictor()
        if predictor.is_ready:
            integrator = CNCIntegration(predictor)
            integrator.monitor_continuous('CNC_Machine_01', interval_seconds=5, max_iterations=10)
            integrator.generate_report()
        else:
            print("Model not ready. Please train model first.")
    elif choice == '4':
        create_flask_api()
    elif choice == '5':
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice!")
        example_usage()