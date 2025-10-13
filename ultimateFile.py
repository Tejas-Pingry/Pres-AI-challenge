"""

I think this is it! I think this is project complete!

================================================================================
FEDERAL SPENDING ANALYZER - COMPLETE SYSTEM
================================================================================
This comprehensive script:
1. Fetches federal agency budget data from USASpending.gov API
2. Cleans, processes, and engineers features from the data
3. Trains machine learning models (or loads existing ones)
4. Runs predictions on new fiscal year data
5. Generates visualizations and detailed analysis reports
6. Identifies agencies with overspending patterns

Includes full error handling, SSL fail-safes, and data validation.
================================================================================
"""

import os
import sys
import time
import warnings
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib
import urllib3

warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

pd.set_option("display.float_format", "{:,.2f}".format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

VERIFY_SSL = False
API_BASE = "https://api.usaspending.gov/api/v2"

# Set base output directory
BASE_OUTPUT_DIR = "/Users/tejaskashyap/Desktop/MODEL_OUTPUT"
MODELS_DIR = os.path.join(BASE_OUTPUT_DIR, "models")
REPORTS_DIR = os.path.join(BASE_OUTPUT_DIR, "reports")
FIGURES_DIR = os.path.join(BASE_OUTPUT_DIR, "figures")
DATA_DIR = os.path.join(BASE_OUTPUT_DIR, "data")


class FederalSpendingAnalyzer:
    """
    Main class for federal spending analysis.
    Handles data collection, processing, model training, and predictions.
    """
    
    def __init__(self, fiscal_year):
        """
        Initialize the analyzer for a specific fiscal year.
        
        Args:
            fiscal_year: The fiscal year to analyze (e.g., 2024)
        """
        self.fiscal_year = str(fiscal_year)
        self.spending_file = os.path.join(DATA_DIR, f"us_agencies_budget_{self.fiscal_year}.csv")
        self.revenue_file = os.path.join(DATA_DIR, f"us_revenue_{self.fiscal_year}.csv")
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.df = None
        self.features = []
        
        # Create all output directories
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)
    
    def collect_agency_data(self):
        """
        Fetch budget data from USASpending.gov API for all federal agencies.
        Saves results to CSV file with full error handling.
        """
        print("\n" + "=" * 80)
        print(f"🌐 COLLECTING FEDERAL AGENCY DATA FOR FISCAL YEAR {self.fiscal_year}")
        print("=" * 80)
        
        agencies_url = f"{API_BASE}/references/toptier_agencies/"
        
        try:
            response = requests.get(agencies_url, verify=VERIFY_SSL, timeout=30)
            response.raise_for_status()
            agencies = response.json().get("results", [])
            print(f"✓ Found {len(agencies)} federal agencies")
        except Exception as e:
            print(f"❌ Failed to fetch agencies list: {e}")
            return False
        
        agency_budget_list = []
        failed_agencies = []
        
        for idx, agency in enumerate(agencies, 1):
            agency_code = agency.get("toptier_code")
            agency_name = agency.get("agency_name", "Unknown")
            
            if not agency_code:
                continue
            
            print(f"  [{idx}/{len(agencies)}] Fetching {agency_name}...", end="\r")
            
            budget_url = f"{API_BASE}/agency/{agency_code}/budgetary_resources/?fiscal_year={self.fiscal_year}"
            
            try:
                budget_response = requests.get(budget_url, verify=VERIFY_SSL, timeout=30)
                
                if budget_response.status_code != 200:
                    failed_agencies.append(agency_name)
                    continue
                
                budget_data = budget_response.json()
                year_data = next(
                    (entry for entry in budget_data.get("agency_data_by_year", [])
                     if str(entry.get("fiscal_year")) == self.fiscal_year),
                    None
                )
                
                if year_data:
                    agency_budget_list.append({
                        "Agency Code": agency_code,
                        "Agency Name": agency_name,
                        "Fiscal Year": self.fiscal_year,
                        "Budgetary Resources": year_data.get("agency_budgetary_resources") or 0,
                        "Obligations": year_data.get("agency_total_obligated") or 0,
                        "Outlays": year_data.get("agency_total_outlayed") or 0
                    })
                
                time.sleep(0.15)
                
            except Exception as e:
                failed_agencies.append(agency_name)
                continue
        
        print("\n")
        
        if not agency_budget_list:
            print("❌ No data collected. API may be down or fiscal year unavailable.")
            return False
        
        df_agencies = pd.DataFrame(agency_budget_list)
        df_agencies.to_csv(self.spending_file, index=False)
        
        print(f"✓ Successfully collected data for {len(agency_budget_list)} agencies")
        if failed_agencies:
            print(f"⚠️  Failed to fetch data for {len(failed_agencies)} agencies")
        print(f"💾 Saved to: {self.spending_file}")
        
        return True
    
    def load_and_prepare_data(self):
        """
        Load spending and revenue data, merge, clean, and engineer features.
        Creates derived metrics for analysis.
        """
        print("\n" + "=" * 80)
        print("📊 LOADING AND PREPARING DATA")
        print("=" * 80)
        
        if not os.path.exists(self.spending_file):
            print(f"❌ Spending file not found: {self.spending_file}")
            return False
        
        if os.path.getsize(self.spending_file) == 0:
            print(f"❌ Spending file is empty: {self.spending_file}")
            return False
        
        try:
            spend_df = pd.read_csv(self.spending_file)
            print(f"✓ Loaded {len(spend_df)} spending records")
        except Exception as e:
            print(f"❌ Error loading spending data: {e}")
            return False
        
        if os.path.exists(self.revenue_file) and os.path.getsize(self.revenue_file) > 0:
            try:
                revenue_df = pd.read_csv(self.revenue_file)
                print(f"✓ Loaded {len(revenue_df)} revenue records")
            except:
                revenue_df = self._create_placeholder_revenue()
        else:
            revenue_df = self._create_placeholder_revenue()
        
        # Ensure Fiscal Year columns are the same type (string) before merging
        spend_df["Fiscal Year"] = spend_df["Fiscal Year"].astype(str)
        revenue_df["Fiscal Year"] = revenue_df["Fiscal Year"].astype(str)
        
        self.df = pd.merge(spend_df, revenue_df, on="Fiscal Year", how="left")
        self.df.drop_duplicates(inplace=True)
        
        numeric_cols = ["Budgetary Resources", "Obligations", "Outlays", "Total Receipts (Millions)"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)
        
        self.df["Obligation Rate"] = self.df["Obligations"] / self.df["Budgetary Resources"].replace(0, np.nan)
        self.df["Outlay Rate"] = self.df["Outlays"] / self.df["Budgetary Resources"].replace(0, np.nan)
        self.df["Spending Deviation"] = (
            (self.df["Outlays"] - self.df["Budgetary Resources"]) / 
            self.df["Budgetary Resources"].replace(0, np.nan)
        )
        
        self.df["Spending Risk (USD)"] = np.abs(self.df["Spending Deviation"]) * self.df["Budgetary Resources"]
        
        # Fill NaN values BEFORE creating categorical column
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_columns] = self.df[numeric_columns].fillna(0)
        
        # Now create categorical column after all numeric operations are done
        self.df["Spending Category"] = pd.cut(
            self.df["Spending Deviation"],
            bins=[-np.inf, -0.05, 0.05, np.inf],
            labels=["Underspending", "On Target", "Overspending"]
        )
        
        self.features = [
            "Budgetary Resources", "Obligations", "Outlays",
            "Total Receipts (Millions)", "Obligation Rate", "Outlay Rate"
        ]
        
        print(f"✓ Data prepared with {len(self.df)} records and {len(self.features)} features")
        print(f"✓ Feature engineering complete")
        
        self._display_data_summary()
        
        return True
    
    def _create_placeholder_revenue(self):
        """Create placeholder revenue data if not available."""
        print("⚠️  Revenue data not found - using placeholder")
        return pd.DataFrame({
            "Fiscal Year": [self.fiscal_year],
            "Total Receipts (Millions)": [0]
        })
    
    def _display_data_summary(self):
        """Display summary statistics of the prepared data."""
        print("\n📈 DATA SUMMARY:")
        print(f"   Total Agencies: {len(self.df)}")
        
        category_counts = self.df["Spending Category"].value_counts()
        print(f"   Underspending: {category_counts.get('Underspending', 0)}")
        print(f"   On Target: {category_counts.get('On Target', 0)}")
        print(f"   Overspending: {category_counts.get('Overspending', 0)}")
        
        print(f"\n   Total Budgetary Resources: ${self.df['Budgetary Resources'].sum():,.0f}")
        print(f"   Total Obligations: ${self.df['Obligations'].sum():,.0f}")
        print(f"   Total Outlays: ${self.df['Outlays'].sum():,.0f}")
    
    def train_models(self):
        """
        Train machine learning models:
        - RandomForest Classifier for spending category prediction
        - GradientBoosting Regressor for deviation prediction
        """
        print("\n" + "=" * 80)
        print("🤖 TRAINING MACHINE LEARNING MODELS")
        print("=" * 80)
        
        if self.df is None or len(self.df) == 0:
            print("❌ No data available for training")
            return False
        
        X = self.df[self.features].copy()
        y_category = self.df["Spending Category"].copy()
        y_deviation = self.df["Spending Deviation"].copy()
        
        if len(X) < 10:
            print("⚠️  Warning: Very small dataset. Model accuracy may be limited.")
        
        try:
            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
                X, y_category, test_size=0.2, random_state=42, stratify=y_category
            )
        except ValueError:
            print("⚠️  Cannot stratify with small dataset. Using random split.")
            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
                X, y_category, test_size=0.2, random_state=42
            )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_c)
        X_test_scaled = self.scaler.transform(X_test_c)
        
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train_c)
        y_test_encoded = self.label_encoder.transform(y_test_c)
        
        print("\n📊 Training Category Classifier (RandomForest)...")
        classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        classifier.fit(X_train_scaled, y_train_encoded)
        
        train_acc = classifier.score(X_train_scaled, y_train_encoded)
        test_acc = classifier.score(X_test_scaled, y_test_encoded)
        
        print(f"   ✓ Training Accuracy: {train_acc:.3f}")
        print(f"   ✓ Testing Accuracy: {test_acc:.3f}")
        
        self.models['classifier'] = classifier
        
        print("\n📊 Training Deviation Regressor (GradientBoosting)...")
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X, y_deviation, test_size=0.2, random_state=42
        )
        
        regressor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        regressor.fit(X_train_r, y_train_r)
        
        train_r2 = regressor.score(X_train_r, y_train_r)
        test_r2 = regressor.score(X_test_r, y_test_r)
        y_pred = regressor.predict(X_test_r)
        rmse = np.sqrt(mean_squared_error(y_test_r, y_pred))
        
        print(f"   ✓ Training R²: {train_r2:.3f}")
        print(f"   ✓ Testing R²: {test_r2:.3f}")
        print(f"   ✓ RMSE: {rmse:.6f}")
        
        self.models['regressor'] = regressor
        
        self._save_models()
        
        return True
    
    def _save_models(self):
        """Save trained models and preprocessing objects to disk."""
        print("\n💾 Saving models...")
        
        try:
            joblib.dump(self.models['classifier'], f"{MODELS_DIR}/classifier.pkl")
            joblib.dump(self.models['regressor'], f"{MODELS_DIR}/regressor.pkl")
            joblib.dump(self.scaler, f"{MODELS_DIR}/scaler.pkl")
            joblib.dump(self.label_encoder, f"{MODELS_DIR}/label_encoder.pkl")
            
            print(f"✓ Models saved to /{MODELS_DIR} directory")
        except Exception as e:
            print(f"❌ Error saving models: {e}")
    
    def load_models(self):
        """Load pre-trained models from disk."""
        print("\n" + "=" * 80)
        print("🔁 LOADING PRE-TRAINED MODELS")
        print("=" * 80)
        
        try:
            self.models['classifier'] = joblib.load(f"{MODELS_DIR}/classifier.pkl")
            self.models['regressor'] = joblib.load(f"{MODELS_DIR}/regressor.pkl")
            self.scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
            self.label_encoder = joblib.load(f"{MODELS_DIR}/label_encoder.pkl")
            
            print("✓ Models loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("   Please train models first using option 1")
            return False
    
    def predict_and_analyze(self):
        """
        Run predictions on current data and generate comprehensive analysis.
        Creates visualizations and exports results.
        """
        print("\n" + "=" * 80)
        print("🔮 RUNNING PREDICTIONS AND ANALYSIS")
        print("=" * 80)
        
        if self.df is None or len(self.df) == 0:
            print("❌ No data available for prediction")
            return False
        
        if not self.models or self.scaler is None:
            print("❌ Models not loaded")
            return False
        
        X = self.df[self.features]
        X_scaled = self.scaler.transform(X)
        
        predicted_encoded = self.models['classifier'].predict(X_scaled)
        self.df["Predicted Category"] = self.label_encoder.inverse_transform(predicted_encoded)
        
        self.df["Predicted Deviation"] = self.models['regressor'].predict(X)
        self.df["Predicted Risk (USD)"] = np.abs(self.df["Predicted Deviation"]) * self.df["Budgetary Resources"]
        
        predictions_file = os.path.join(DATA_DIR, f"predicted_spending_{self.fiscal_year}.csv")
        self.df.to_csv(predictions_file, index=False)
        print(f"✓ Predictions saved to: {predictions_file}")
        
        analysis_file = os.path.join(REPORTS_DIR, f"spending_analysis_{self.fiscal_year}.xlsx")
        with pd.ExcelWriter(analysis_file, engine='openpyxl') as writer:
            self.df.to_excel(writer, sheet_name='Full Data', index=False)
            
            summary = self.df.groupby('Predicted Category').agg({
                'Agency Name': 'count',
                'Budgetary Resources': 'sum',
                'Outlays': 'sum',
                'Predicted Risk (USD)': 'sum'
            }).reset_index()
            summary.columns = ['Category', 'Count', 'Total Budget', 'Total Outlays', 'Total Risk']
            summary.to_excel(writer, sheet_name='Summary', index=False)
            
            overspending = self.df[self.df['Predicted Category'] == 'Overspending'].sort_values(
                'Predicted Risk (USD)', ascending=False
            )[['Agency Name', 'Budgetary Resources', 'Outlays', 'Predicted Deviation', 'Predicted Risk (USD)']]
            overspending.to_excel(writer, sheet_name='Overspending Agencies', index=False)
        
        print(f"✓ Analysis report saved to: {analysis_file}")
        
        self._generate_visualizations()
        self._print_analysis_summary()
        
        return True
    
    def _generate_visualizations(self):
        """Generate comprehensive visualizations of spending analysis."""
        print("\n📊 Generating visualizations...")
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Federal Spending Analysis - Fiscal Year {self.fiscal_year}', 
                     fontsize=16, fontweight='bold')
        
        category_counts = self.df['Predicted Category'].value_counts()
        colors = {'Overspending': '#e74c3c', 'On Target': '#2ecc71', 'Underspending': '#3498db'}
        pie_colors = [colors.get(cat, '#95a5a6') for cat in category_counts.index]
        
        axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                       startangle=90, colors=pie_colors)
        axes[0, 0].set_title('Distribution of Spending Categories', fontweight='bold')
        
        overspending = self.df[self.df['Predicted Category'] == 'Overspending'].nlargest(10, 'Predicted Risk (USD)')
        if len(overspending) > 0:
            axes[0, 1].barh(range(len(overspending)), overspending['Predicted Risk (USD)'] / 1e6, 
                           color='#e74c3c')
            axes[0, 1].set_yticks(range(len(overspending)))
            axes[0, 1].set_yticklabels(overspending['Agency Name'], fontsize=8)
            axes[0, 1].set_xlabel('Risk (Millions USD)')
            axes[0, 1].set_title('Top 10 Overspending Agencies by Risk', fontweight='bold')
            axes[0, 1].invert_yaxis()
        else:
            axes[0, 1].text(0.5, 0.5, 'No overspending agencies found', 
                           ha='center', va='center', fontsize=12)
            axes[0, 1].set_title('Top 10 Overspending Agencies by Risk', fontweight='bold')
        
        axes[1, 0].hist(self.df['Predicted Deviation'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Deviation')
        axes[1, 0].set_xlabel('Predicted Deviation')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Predicted Deviations', fontweight='bold')
        axes[1, 0].legend()
        
        category_budgets = self.df.groupby('Predicted Category')['Budgetary Resources'].sum() / 1e9
        bars = axes[1, 1].bar(category_budgets.index, category_budgets.values, 
                              color=[colors.get(cat, '#95a5a6') for cat in category_budgets.index])
        axes[1, 1].set_ylabel('Total Budget (Billions USD)')
        axes[1, 1].set_title('Total Budgetary Resources by Category', fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=15)
        
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'${height:.1f}B', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        viz_file = os.path.join(FIGURES_DIR, f"spending_analysis_{self.fiscal_year}.png")
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"✓ Visualizations saved to: {viz_file}")
        
        plt.close()
    
    def _print_analysis_summary(self):
        """Print detailed analysis summary to console."""
        print("\n" + "=" * 80)
        print("📋 ANALYSIS SUMMARY")
        print("=" * 80)
        
        print(f"\n🏛️  Total Agencies Analyzed: {len(self.df)}")
        
        category_summary = self.df.groupby('Predicted Category').agg({
            'Agency Name': 'count',
            'Budgetary Resources': 'sum',
            'Outlays': 'sum',
            'Predicted Risk (USD)': 'sum'
        })
        
        print("\n📊 By Category:")
        for category in ['Overspending', 'On Target', 'Underspending']:
            if category in category_summary.index:
                row = category_summary.loc[category]
                print(f"\n   {category}:")
                print(f"      Agencies: {row['Agency Name']}")
                print(f"      Total Budget: ${row['Budgetary Resources']:,.0f}")
                print(f"      Total Outlays: ${row['Outlays']:,.0f}")
                print(f"      Total Risk: ${row['Predicted Risk (USD)']:,.0f}")
        
        overspending = self.df[self.df['Predicted Category'] == 'Overspending'].sort_values(
            'Predicted Risk (USD)', ascending=False
        )
        
        if len(overspending) > 0:
            print(f"\n🚨 TOP OVERSPENDING AGENCIES:")
            for idx, (_, row) in enumerate(overspending.head(5).iterrows(), 1):
                print(f"\n   {idx}. {row['Agency Name']}")
                print(f"      Budget: ${row['Budgetary Resources']:,.0f}")
                print(f"      Outlays: ${row['Outlays']:,.0f}")
                print(f"      Deviation: {row['Predicted Deviation']:.2%}")
                print(f"      Risk: ${row['Predicted Risk (USD)']:,.0f}")
        else:
            print("\n✅ No agencies predicted to overspend")
        
        print("\n" + "=" * 80)


def main():
    """Main execution function with menu-driven interface."""
    print("\n" + "=" * 80)
    print("🏛️  FEDERAL SPENDING ANALYZER - COMPLETE SYSTEM")
    print("=" * 80)
    print("\nSelect Mode:")
    print("  1️⃣  Train Mode - Collect data, train models, and analyze")
    print("  2️⃣  Predict Mode - Load existing models and analyze new data")
    print("=" * 80)
    
    mode = input("\nEnter mode (1 or 2): ").strip()
    
    if mode not in ['1', '2']:
        print("❌ Invalid mode. Please enter 1 or 2.")
        return
    
    fiscal_year = input("Enter fiscal year (e.g., 2024): ").strip()
    
    if not fiscal_year.isdigit() or len(fiscal_year) != 4:
        print("❌ Invalid fiscal year. Please enter a 4-digit year.")
        return
    
    analyzer = FederalSpendingAnalyzer(fiscal_year)
    
    if mode == '1':
        print("\n🚀 Starting TRAIN MODE...")
        
        if not analyzer.collect_agency_data():
            print("\n❌ Data collection failed. Exiting.")
            return
        
        if not analyzer.load_and_prepare_data():
            print("\n❌ Data preparation failed. Exiting.")
            return
        
        if not analyzer.train_models():
            print("\n❌ Model training failed. Exiting.")
            return
        
        if not analyzer.predict_and_analyze():
            print("\n❌ Analysis failed. Exiting.")
            return
        
        print("\n" + "=" * 80)
        print("✅ TRAIN MODE COMPLETE!")
        print("=" * 80)
        print("\n📁 Output Files:")
        print(f"   • Data: {analyzer.spending_file}")
        print(f"   • Predictions: {os.path.join(DATA_DIR, f'predicted_spending_{fiscal_year}.csv')}")
        print(f"   • Analysis: {os.path.join(REPORTS_DIR, f'spending_analysis_{fiscal_year}.xlsx')}")
        print(f"   • Visualizations: {os.path.join(FIGURES_DIR, f'spending_analysis_{fiscal_year}.png')}")
        print(f"   • Models: {MODELS_DIR}/")
        print(f"\n📂 All files saved to: {BASE_OUTPUT_DIR}")
        
    elif mode == '2':
        print("\n🚀 Starting PREDICT MODE...")
        
        if not analyzer.load_models():
            print("\n❌ Could not load models. Please run Train Mode (option 1) first.")
            return
        
        if os.path.exists(analyzer.spending_file):
            print(f"✓ Using existing data file: {analyzer.spending_file}")
        else:
            print(f"⚠️  Data file not found. Collecting new data...")
            if not analyzer.collect_agency_data():
                print("\n❌ Data collection failed. Exiting.")
                return
        
        if not analyzer.load_and_prepare_data():
            print("\n❌ Data preparation failed. Exiting.")
            return
        
        if not analyzer.predict_and_analyze():
            print("\n❌ Analysis failed. Exiting.")
            return
        
        print("\n" + "=" * 80)
        print("✅ PREDICT MODE COMPLETE!")
        print("=" * 80)
        print("\n📁 Output Files:")
        print(f"   • Predictions: {os.path.join(DATA_DIR, f'predicted_spending_{fiscal_year}.csv')}")
        print(f"   • Analysis: {os.path.join(REPORTS_DIR, f'spending_analysis_{fiscal_year}.xlsx')}")
        print(f"   • Visualizations: {os.path.join(FIGURES_DIR, f'spending_analysis_{fiscal_year}.png')}")
        print(f"\n📂 All files saved to: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
