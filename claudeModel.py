'''
# 1. First, install required libraries
pip install pandas numpy scikit-learn matplotlib seaborn

# 2. Run the multi-source extractor to get your CSV data
python federal_spending_extractor.py

# 3. Update the csv_files list in the training script with your actual filenames

# 4. Run the training script
python federal_spending_ai_trainer.py


'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federal Spending Fraud/Waste Detection AI Training Script
Uses Hugging Face Transformers + Traditional ML
Trains on multi-source spending data to detect fraud, waste, and anomalies
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional: Hugging Face transformers (install with: pip install transformers torch)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not installed. Will use traditional ML only.")
    print("   Install with: pip install transformers torch")

class SpendingAITrainer:
    def __init__(self, csv_files):
        """
        Initialize trainer with CSV files from the multi-source extractor
        csv_files: list of CSV file paths or single CSV path
        """
        self.csv_files = csv_files if isinstance(csv_files, list) else [csv_files]
        self.df = None
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self, include_your_training_data=None):
        """Load CSV data and prepare for training"""
        
        print("="*70)
        print("LOADING AND PREPARING DATA")
        print("="*70)
        
        # Load all CSV files
        dfs = []
        for file in self.csv_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                print(f"✓ Loaded: {file} ({len(df)} records)")
            except Exception as e:
                print(f"❌ Failed to load {file}: {e}")
        
        # Optionally merge with your existing training data
        if include_your_training_data:
            try:
                training_df = pd.read_csv(include_your_training_data)
                dfs.append(training_df)
                print(f"✓ Loaded training data: {include_your_training_data} ({len(training_df)} records)")
            except Exception as e:
                print(f"❌ Failed to load training data: {e}")
        
        if not dfs:
            raise ValueError("No data loaded!")
        
        # Combine all dataframes
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"\n✓ Total records: {len(self.df)}")
        
        # Data cleaning
        print("\nCleaning data...")
        self._clean_data()
        
        # Feature engineering
        print("Engineering features...")
        self._engineer_features()
        
        print(f"✓ Final dataset: {len(self.df)} records, {len(self.df.columns)} features")
        
        return self.df
    
    def _clean_data(self):
        """Clean and standardize data"""
        
        # Fill missing numeric values
        numeric_cols = ['Budgetary Resources', 'Obligations', 'Outlays', 'Amount At Risk (USD)']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        # Fill missing categorical values
        categorical_cols = ['Waste/Issue Type', 'Category of Spending', 'Performance Impact']
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col].fillna('Unknown', inplace=True)
        
        # Convert boolean flags
        bool_cols = ['Fraud Risk Flag', 'Overspending Flag']
        for col in bool_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0})
                self.df[col].fillna(0, inplace=True)
        
        # Remove duplicates
        before = len(self.df)
        self.df.drop_duplicates(subset=['Department/Entity', 'Program/Project Name'], keep='first', inplace=True)
        after = len(self.df)
        if before != after:
            print(f"  Removed {before - after} duplicate records")
    
    def _engineer_features(self):
        """Create additional features for better predictions"""
        
        # Financial ratios
        if all(col in self.df.columns for col in ['Budgetary Resources', 'Obligations', 'Outlays']):
            self.df['Obligation_Rate'] = np.where(
                self.df['Budgetary Resources'] > 0,
                self.df['Obligations'] / self.df['Budgetary Resources'],
                0
            )
            
            self.df['Outlay_Rate'] = np.where(
                self.df['Obligations'] > 0,
                self.df['Outlays'] / self.df['Obligations'],
                0
            )
            
            self.df['Budget_Variance'] = self.df['Outlays'] - self.df['Budgetary Resources']
            
            self.df['Risk_Percentage'] = np.where(
                self.df['Budgetary Resources'] > 0,
                (self.df['Amount At Risk (USD)'] / self.df['Budgetary Resources']) * 100,
                0
            )
        
        # Complexity score (if not present)
        if 'Complexity Rating' not in self.df.columns:
            self.df['Complexity Rating'] = 3  # Default medium
        
        # Binary flags for high-risk agencies
        high_risk_keywords = ['Medicare', 'Medicaid', 'Veterans', 'Defense', 'Health']
        self.df['High_Risk_Agency'] = self.df['Department/Entity'].apply(
            lambda x: 1 if any(keyword.lower() in str(x).lower() for keyword in high_risk_keywords) else 0
        )
        
        # Create severity score
        self.df['Severity_Score'] = (
            self.df['Fraud Risk Flag'] * 3 +
            self.df['Overspending Flag'] * 2 +
            (self.df['Amount At Risk (USD)'] > 1_000_000_000).astype(int) * 2
        )
    
    def train_fraud_detection_model(self):
        """Train a fraud detection classifier"""
        
        print("\n" + "="*70)
        print("TRAINING FRAUD DETECTION MODEL")
        print("="*70)
        
        if 'Fraud Risk Flag' not in self.df.columns:
            print("❌ No 'Fraud Risk Flag' column found. Skipping fraud detection.")
            return None
        
        # Prepare features
        feature_cols = [
            'Budgetary Resources', 'Obligations', 'Outlays', 'Amount At Risk (USD)',
            'Obligation_Rate', 'Outlay_Rate', 'Budget_Variance', 'Risk_Percentage',
            'Complexity Rating', 'High_Risk_Agency', 'Overspending Flag'
        ]
        
        # Filter to only columns that exist
        feature_cols = [col for col in feature_cols if col in self.df.columns]
        
        X = self.df[feature_cols].copy()
        y = self.df['Fraud Risk Flag'].copy()
        
        # Handle missing values
        X.fillna(0, inplace=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Fraud cases: {y.sum()} ({y.mean()*100:.1f}%)")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        print("\nTraining Random Forest Classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = rf_model.predict(X_test_scaled)
        y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✓ Accuracy: {accuracy*100:.2f}%")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Fraud Risk', 'Fraud Risk']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 5 Important Features:")
        print(feature_importance.head(5).to_string(index=False))
        
        # Save model
        self.models['fraud_detection'] = {
            'model': rf_model,
            'features': feature_cols,
            'accuracy': accuracy,
            'feature_importance': feature_importance
        }
        
        # Visualization
        self._plot_confusion_matrix(y_test, y_pred, "Fraud Detection")
        self._plot_feature_importance(feature_importance, "Fraud Detection")
        
        return rf_model
    
    def train_waste_type_classifier(self):
        """Train a multi-class classifier for waste types"""
        
        print("\n" + "="*70)
        print("TRAINING WASTE TYPE CLASSIFIER")
        print("="*70)
        
        if 'Waste/Issue Type' not in self.df.columns:
            print("❌ No 'Waste/Issue Type' column found. Skipping.")
            return None
        
        # Prepare data
        feature_cols = [
            'Budgetary Resources', 'Obligations', 'Outlays', 'Amount At Risk (USD)',
            'Obligation_Rate', 'Outlay_Rate', 'Budget_Variance', 'Risk_Percentage',
            'Complexity Rating', 'High_Risk_Agency', 'Fraud Risk Flag', 'Overspending Flag'
        ]
        
        feature_cols = [col for col in feature_cols if col in self.df.columns]
        
        X = self.df[feature_cols].copy()
        y = self.df['Waste/Issue Type'].copy()
        
        # Remove rows with 'Unknown' or 'Normal Operations'
        mask = ~y.isin(['Unknown', 'Normal Operations'])
        X = X[mask]
        y = y[mask]
        
        if len(y) < 20:
            print("❌ Not enough labeled data for waste type classification")
            return None
        
        X.fillna(0, inplace=True)
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.encoders['waste_type'] = le
        
        print(f"\nWaste types: {list(le.classes_)}")
        print(f"Training samples: {len(X)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Gradient Boosting
        print("\nTraining Gradient Boosting Classifier...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = gb_model.predict(X_test_scaled)
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✓ Accuracy: {accuracy*100:.2f}%")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        
        # Save model
        self.models['waste_classifier'] = {
            'model': gb_model,
            'features': feature_cols,
            'accuracy': accuracy,
            'label_encoder': le
        }
        
        return gb_model
    
    def train_risk_amount_predictor(self):
        """Train a regression model to predict Amount At Risk"""
        
        print("\n" + "="*70)
        print("TRAINING RISK AMOUNT PREDICTOR")
        print("="*70)
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
        
        # Prepare data
        feature_cols = [
            'Budgetary Resources', 'Obligations', 'Outlays',
            'Obligation_Rate', 'Outlay_Rate', 'Budget_Variance',
            'Complexity Rating', 'High_Risk_Agency', 'Fraud Risk Flag', 'Overspending Flag'
        ]
        
        feature_cols = [col for col in feature_cols if col in self.df.columns]
        
        X = self.df[feature_cols].copy()
        y = self.df['Amount At Risk (USD)'].copy()
        
        # Remove zero/null risk amounts for training
        mask = y > 0
        X = X[mask]
        y = y[mask]
        
        if len(y) < 20:
            print("❌ Not enough data with non-zero risk amounts")
            return None
        
        X.fillna(0, inplace=True)
        
        print(f"Training samples: {len(X)}")
        print(f"Mean risk amount: ${y.mean():,.0f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest Regressor
        print("\nTraining Random Forest Regressor...")
        rf_reg = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        rf_reg.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf_reg.predict(X_test)
        
        # Evaluation
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✓ Mean Absolute Error: ${mae:,.0f}")
        print(f"✓ R² Score: {r2:.3f}")
        
        # Save model
        self.models['risk_predictor'] = {
            'model': rf_reg,
            'features': feature_cols,
            'mae': mae,
            'r2': r2
        }
        
        return rf_reg
    
    def _plot_confusion_matrix(self, y_true, y_pred, title):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        filename = f"confusion_matrix_{title.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename)
        print(f"\n✓ Confusion matrix saved: {filename}")
        plt.close()
    
    def _plot_feature_importance(self, feature_importance, title):
        """Plot feature importance"""
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature')
        plt.title(f'Top 10 Feature Importance - {title}')
        plt.tight_layout()
        
        filename = f"feature_importance_{title.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename)
        print(f"✓ Feature importance plot saved: {filename}")
        plt.close()
    
    def predict_new_data(self, new_data_csv):
        """Make predictions on new data"""
        
        print("\n" + "="*70)
        print("MAKING PREDICTIONS ON NEW DATA")
        print("="*70)
        
        if not self.models:
            print("❌ No trained models available. Train models first.")
            return None
        
        # Load new data
        new_df = pd.read_csv(new_data_csv)
        print(f"✓ Loaded {len(new_df)} records from {new_data_csv}")
        
        # Make predictions with each model
        results = []
        
        if 'fraud_detection' in self.models:
            model_info = self.models['fraud_detection']
            model = model_info['model']
            features = model_info['features']
            
            X_new = new_df[features].fillna(0)
            X_new_scaled = self.scaler.transform(X_new)
            
            fraud_pred = model.predict(X_new_scaled)
            fraud_proba = model.predict_proba(X_new_scaled)[:, 1]
            
            new_df['Predicted_Fraud_Risk'] = fraud_pred
            new_df['Fraud_Probability'] = fraud_proba
            
            print(f"\n✓ Fraud Detection: {fraud_pred.sum()} high-risk cases identified")
        
        # Export predictions
        output_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        new_df.to_csv(output_file, index=False)
        print(f"\n✓ Predictions saved: {output_file}")
        
        return new_df
    
    def save_models(self, output_dir='./models'):
        """Save trained models to disk"""
        import pickle
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("SAVING MODELS")
        print("="*70)
        
        for model_name, model_info in self.models.items():
            filename = os.path.join(output_dir, f"{model_name}.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(model_info, f)
            print(f"✓ Saved: {filename}")
        
        # Save scaler
        scaler_file = os.path.join(output_dir, "scaler.pkl")
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Saved: {scaler_file}")
        
        print(f"\n✓ All models saved to {output_dir}/")


def main():
    """Main training pipeline"""
    
    print("="*70)
    print("FEDERAL SPENDING AI TRAINER")
    print("="*70)
    
    # Configuration
    csv_files = [
        "multi_source_spending_fy2024_20251002_120000.csv",  # Replace with your actual file
        # Add more CSV files here
    ]
    
    # Optional: Include your existing training data
    your_training_data = None  # "your_existing_training_data.csv"
    
    # Initialize trainer
    trainer = SpendingAITrainer(csv_files)
    
    # Load and prepare data
    try:
        df = trainer.load_and_prepare_data(include_your_training_data=your_training_data)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure to run the multi-source extractor first to generate CSV files!")
        print("   Or update the csv_files list with your actual file names.")
        return
    
    # Train models
    print("\n" + "="*70)
    print("STARTING MODEL TRAINING")
    print("="*70)
    
    fraud_model = trainer.train_fraud_detection_model()
    waste_model = trainer.train_waste_type_classifier()
    risk_model = trainer.train_risk_amount_predictor()
    
    # Save models
    trainer.save_models()
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    if trainer.models:
        print("\nTrained Models:")
        for model_name, model_info in trainer.models.items():
            print(f"  • {model_name}")
            if 'accuracy' in model_info:
                print(f"    Accuracy: {model_info['accuracy']*100:.2f}%")
            elif 'r2' in model_info:
                print(f"    R² Score: {model_info['r2']:.3f}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review the confusion matrices and feature importance plots")
    print("2. Use predict_new_data() to make predictions on new datasets")
    print("3. Fine-tune models by adjusting hyperparameters")
    print("4. Deploy models to production for real-time fraud detection")
    print("="*70)


if __name__ == "__main__":
    main()
