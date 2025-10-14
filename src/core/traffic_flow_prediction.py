"""
Traffic Flow Prediction Module.

This module provides traffic flow prediction and congestion forecasting
using machine learning models based on vehicle counts and temporal features.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import pickle
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ..config.settings import settings


@dataclass
class VehicleCounts:
    """Vehicle count data for traffic flow prediction."""
    
    car_count: int
    bike_count: int
    bus_count: int
    truck_count: int
    total_count: int
    timestamp: datetime


@dataclass
class TrafficPrediction:
    """Traffic flow prediction result."""
    
    predicted_situation: str
    confidence: float
    congestion_level: str
    predicted_vehicle_counts: VehicleCounts
    prediction_time: datetime
    features_used: List[str]


@dataclass
class TrafficFlowResult:
    """Complete traffic flow analysis result."""
    
    current_counts: VehicleCounts
    prediction: TrafficPrediction
    historical_trend: str
    peak_hour_indicator: bool
    processing_time: float


class TrafficFlowPredictor:
    """
    Traffic flow prediction system using machine learning models.
    
    This class provides traffic flow prediction based on vehicle counts,
    temporal features, and historical patterns using Random Forest Classifier.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the traffic flow predictor.
        
        Args:
            config: Optional configuration dictionary to override default settings.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Model settings
        self.model_path = self.config.get(
            "traffic_flow_model_path",
            settings.models.traffic_flow_model_path
        )
        self.confidence_threshold = self.config.get(
            "traffic_flow_confidence_threshold",
            settings.models.traffic_flow_confidence_threshold
        )
        
        # Traffic situation categories
        self.traffic_situations = ["low", "normal", "high", "congested"]
        
        # Initialize models and encoders
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        
        # Historical data for trend analysis
        self.historical_data = []
        self.max_history_size = 100
        
        self._initialize_model()
        self.logger.info("Traffic flow predictor initialized successfully")
    
    def _initialize_model(self) -> None:
        """Initialize or load the traffic flow prediction model."""
        try:
            # Try to load pre-trained model
            if Path(self.model_path).exists():
                self._load_model()
            else:
                # Initialize new model
                self._create_new_model()
                self.logger.info("Created new traffic flow prediction model")
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            self._create_new_model()
    
    def _load_model(self) -> None:
        """Load pre-trained model from file."""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.label_encoder = model_data['label_encoder']
                self.feature_columns = model_data['feature_columns']
            self.logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self._create_new_model()
    
    def _create_new_model(self) -> None:
        """Create a new Random Forest model."""
        try:
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
            self.scaler = StandardScaler()
            self.label_encoder = LabelEncoder()
            self.feature_columns = [
                'car_count', 'bike_count', 'bus_count', 'truck_count',
                'hour', 'day_of_week', 'is_weekend', 'is_peak_hour'
            ]
            self.logger.info("New model created successfully")
        except Exception as e:
            self.logger.error(f"Error creating new model: {e}")
            raise
    
    def _save_model(self) -> None:
        """Save the trained model to file."""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns
            }
            
            # Create directory if it doesn't exist
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            self.logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def extract_temporal_features(self, timestamp: datetime) -> Dict[str, Any]:
        """
        Extract temporal features from timestamp.
        
        Args:
            timestamp: Datetime object.
            
        Returns:
            Dictionary of temporal features.
        """
        try:
            hour = timestamp.hour
            day_of_week = timestamp.weekday()  # 0 = Monday, 6 = Sunday
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Define peak hours (7-9 AM and 5-7 PM)
            is_peak_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
            
            return {
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'is_peak_hour': is_peak_hour
            }
        except Exception as e:
            self.logger.error(f"Error extracting temporal features: {e}")
            return {
                'hour': 12,
                'day_of_week': 0,
                'is_weekend': 0,
                'is_peak_hour': 0
            }
    
    def prepare_features(self, vehicle_counts: VehicleCounts) -> np.ndarray:
        """
        Prepare features for model prediction.
        
        Args:
            vehicle_counts: Vehicle count data.
            
        Returns:
            Feature array ready for model input.
        """
        try:
            # Extract temporal features
            temporal_features = self.extract_temporal_features(vehicle_counts.timestamp)
            
            # Combine all features
            features = [
                vehicle_counts.car_count,
                vehicle_counts.bike_count,
                vehicle_counts.bus_count,
                vehicle_counts.truck_count,
                temporal_features['hour'],
                temporal_features['day_of_week'],
                temporal_features['is_weekend'],
                temporal_features['is_peak_hour']
            ]
            
            return np.array(features).reshape(1, -1)
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return np.zeros((1, 8))
    
    def predict_traffic_situation(self, vehicle_counts: VehicleCounts) -> TrafficPrediction:
        """
        Predict traffic situation based on vehicle counts.
        
        Args:
            vehicle_counts: Current vehicle count data.
            
        Returns:
            TrafficPrediction object with prediction results.
        """
        try:
            # Prepare features
            features = self.prepare_features(vehicle_counts)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Make prediction
            if self.model is not None:
                prediction = self.model.predict(features_scaled)[0]
                probabilities = self.model.predict_proba(features_scaled)[0]
                confidence = max(probabilities)
                
                # Get predicted situation name
                if self.label_encoder is not None:
                    predicted_situation = self.label_encoder.inverse_transform([prediction])[0]
                else:
                    predicted_situation = self.traffic_situations[prediction]
            else:
                # Fallback prediction based on total vehicle count
                total_count = vehicle_counts.total_count
                if total_count < 50:
                    predicted_situation = "low"
                    confidence = 0.8
                elif total_count < 100:
                    predicted_situation = "normal"
                    confidence = 0.7
                elif total_count < 200:
                    predicted_situation = "high"
                    confidence = 0.6
                else:
                    predicted_situation = "congested"
                    confidence = 0.9
            
            # Determine congestion level
            congestion_level = self._determine_congestion_level(predicted_situation, vehicle_counts)
            
            # Create predicted vehicle counts (simple projection)
            predicted_counts = self._project_vehicle_counts(vehicle_counts, predicted_situation)
            
            return TrafficPrediction(
                predicted_situation=predicted_situation,
                confidence=confidence,
                congestion_level=congestion_level,
                predicted_vehicle_counts=predicted_counts,
                prediction_time=datetime.now(),
                features_used=self.feature_columns or []
            )
        except Exception as e:
            self.logger.error(f"Error predicting traffic situation: {e}")
            return TrafficPrediction(
                predicted_situation="unknown",
                confidence=0.0,
                congestion_level="unknown",
                predicted_vehicle_counts=vehicle_counts,
                prediction_time=datetime.now(),
                features_used=[]
            )
    
    def _determine_congestion_level(self, situation: str, vehicle_counts: VehicleCounts) -> str:
        """
        Determine congestion level based on prediction and vehicle counts.
        
        Args:
            situation: Predicted traffic situation.
            vehicle_counts: Current vehicle counts.
            
        Returns:
            Congestion level string.
        """
        try:
            # Base congestion on predicted situation
            if situation == "low":
                return "light"
            elif situation == "normal":
                return "moderate"
            elif situation == "high":
                return "heavy"
            elif situation == "congested":
                return "severe"
            else:
                # Fallback based on vehicle density
                density = vehicle_counts.total_count / 100  # Normalize
                if density < 0.5:
                    return "light"
                elif density < 1.0:
                    return "moderate"
                elif density < 2.0:
                    return "heavy"
                else:
                    return "severe"
        except Exception as e:
            self.logger.error(f"Error determining congestion level: {e}")
            return "unknown"
    
    def _project_vehicle_counts(self, current_counts: VehicleCounts, situation: str) -> VehicleCounts:
        """
        Project future vehicle counts based on current situation.
        
        Args:
            current_counts: Current vehicle counts.
            situation: Predicted traffic situation.
            
        Returns:
            Projected vehicle counts.
        """
        try:
            # Simple projection based on situation trend
            if situation == "low":
                multiplier = 0.8
            elif situation == "normal":
                multiplier = 1.0
            elif situation == "high":
                multiplier = 1.2
            else:  # congested
                multiplier = 1.5
            
            projected_car = int(current_counts.car_count * multiplier)
            projected_bike = int(current_counts.bike_count * multiplier)
            projected_bus = int(current_counts.bus_count * multiplier)
            projected_truck = int(current_counts.truck_count * multiplier)
            projected_total = projected_car + projected_bike + projected_bus + projected_truck
            
            return VehicleCounts(
                car_count=projected_car,
                bike_count=projected_bike,
                bus_count=projected_bus,
                truck_count=projected_truck,
                total_count=projected_total,
                timestamp=current_counts.timestamp + timedelta(minutes=15)  # 15 minutes ahead
            )
        except Exception as e:
            self.logger.error(f"Error projecting vehicle counts: {e}")
            return current_counts
    
    def analyze_historical_trend(self) -> str:
        """
        Analyze historical data to determine traffic trend.
        
        Returns:
            Trend description string.
        """
        try:
            if len(self.historical_data) < 5:
                return "insufficient_data"
            
            # Get recent data
            recent_data = self.historical_data[-10:]
            older_data = self.historical_data[-20:-10] if len(self.historical_data) >= 20 else self.historical_data[:-10]
            
            if not older_data:
                return "insufficient_data"
            
            # Calculate average counts
            recent_avg = sum(data.total_count for data in recent_data) / len(recent_data)
            older_avg = sum(data.total_count for data in older_data) / len(older_data)
            
            # Determine trend
            if recent_avg > older_avg * 1.1:
                return "increasing"
            elif recent_avg < older_avg * 0.9:
                return "decreasing"
            else:
                return "stable"
        except Exception as e:
            self.logger.error(f"Error analyzing historical trend: {e}")
            return "unknown"
    
    def is_peak_hour(self, timestamp: datetime) -> bool:
        """
        Check if current time is peak hour.
        
        Args:
            timestamp: Current timestamp.
            
        Returns:
            True if peak hour, False otherwise.
        """
        try:
            hour = timestamp.hour
            return (7 <= hour <= 9) or (17 <= hour <= 19)
        except Exception as e:
            self.logger.error(f"Error checking peak hour: {e}")
            return False
    
    def predict_traffic_flow(self, vehicle_counts: VehicleCounts) -> TrafficFlowResult:
        """
        Perform complete traffic flow prediction and analysis.
        
        Args:
            vehicle_counts: Current vehicle count data.
            
        Returns:
            TrafficFlowResult object with complete analysis.
        """
        import time
        start_time = time.time()
        
        try:
            # Add to historical data
            self.historical_data.append(vehicle_counts)
            
            # Limit historical data size
            if len(self.historical_data) > self.max_history_size:
                self.historical_data = self.historical_data[-self.max_history_size:]
            
            # Predict traffic situation
            prediction = self.predict_traffic_situation(vehicle_counts)
            
            # Analyze historical trend
            trend = self.analyze_historical_trend()
            
            # Check if peak hour
            peak_hour = self.is_peak_hour(vehicle_counts.timestamp)
            
            processing_time = time.time() - start_time
            
            return TrafficFlowResult(
                current_counts=vehicle_counts,
                prediction=prediction,
                historical_trend=trend,
                peak_hour_indicator=peak_hour,
                processing_time=processing_time
            )
        except Exception as e:
            self.logger.error(f"Error predicting traffic flow: {e}")
            return TrafficFlowResult(
                current_counts=vehicle_counts,
                prediction=TrafficPrediction(
                    predicted_situation="error",
                    confidence=0.0,
                    congestion_level="unknown",
                    predicted_vehicle_counts=vehicle_counts,
                    prediction_time=datetime.now(),
                    features_used=[]
                ),
                historical_trend="error",
                peak_hour_indicator=False,
                processing_time=time.time() - start_time
            )
    
    def train_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the traffic flow prediction model.
        
        Args:
            data: Training data with vehicle counts and traffic situations.
            
        Returns:
            Training results dictionary.
        """
        try:
            # Prepare features and target
            feature_columns = ['car_count', 'bike_count', 'bus_count', 'truck_count', 
                             'hour', 'day_of_week', 'is_weekend', 'is_peak_hour']
            
            X = data[feature_columns]
            y = data['traffic_situation']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
            self.model.fit(X_train_scaled, y_train_encoded)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test_encoded, y_pred)
            
            # Save model
            self.feature_columns = feature_columns
            self._save_model()
            
            # Generate report
            report = classification_report(y_test_encoded, y_pred, output_dict=True)
            confusion_mat = confusion_matrix(y_test_encoded, y_pred)
            
            results = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': confusion_mat.tolist(),
                'feature_importance': dict(zip(feature_columns, self.model.feature_importances_))
            }
            
            self.logger.info(f"Model training completed with accuracy: {accuracy:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dictionary of feature importance scores.
        """
        try:
            if self.model is None or self.feature_columns is None:
                return {}
            
            return dict(zip(self.feature_columns, self.model.feature_importances_))
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}
