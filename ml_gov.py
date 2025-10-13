import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import joblib
import os
from IPython.display import display

class GovSpendingAnalyzer:
    def __init__(self, spending_file, revenue_file, fiscal_year):
        self.spending_file = spending_file
        self.revenue_file = revenue_file
        self.fiscal_year = fiscal_year
        self.models = {}
        self.scaler = StandardScaler()

    # ---------------------- LOAD & MERGE DATA ----------------------
    def load_data(self):
        print("\n" + "=" * 70)
        print("📊  LOADING HISTORICAL DATA")
        print("=" * 70)

        spend = pd.read_csv(self.spending_file)
        rev = pd.read_csv(self.revenue_file)
        print(f"✓ Loaded {len(spend):,} spending records and {len(rev):,} revenue records")

        # Ensure standard column naming
        spend.columns = spend.columns.str.strip()
        rev.columns = rev.columns.str.strip()

        # If revenue data has no agency info, aggregate it by fiscal year only
        if "Agency Name" not in rev.columns:
            rev = rev.groupby("Fiscal Year", as_index=False)["Total Receipts (Millions)"].sum()

        # Merge on both Fiscal Year and Agency Name when possible
        if "Agency Name" in rev.columns:
            df = pd.merge(spend, rev, on=["Fiscal Year", "Agency Name"], how="left")
        else:
            df = pd.merge(spend, rev, on="Fiscal Year", how="left")

        df.drop_duplicates(subset=["Agency Name", "Fiscal Year"], inplace=True)

        # Clean numeric data
        numeric_cols = [
            "Budgetary Resources", "Obligations", "Outlays", "Total Receipts (Millions)"
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        # Derived metrics
        df["Obligation Rate"] = df["Obligations"] / df["Budgetary Resources"].replace(0, np.nan)
        df["Outlay Rate"] = df["Outlays"] / df["Obligations"].replace(0, np.nan)
        df["Efficiency"] = df["Outlays"] / df["Budgetary Resources"].replace(0, np.nan)
        df.fillna(0, inplace=True)

        # Add trend by agency (previous year efficiency change)
        df["Efficiency Trend"] = df.groupby("Agency Name")["Efficiency"].diff().fillna(0)

        self.features = [
            "Budgetary Resources", "Obligations", "Outlays",
            "Total Receipts (Millions)", "Obligation Rate",
            "Outlay Rate", "Efficiency", "Efficiency Trend"
        ]
        self.df = df
        print(f"✓ Data ready with {len(df):,} agencies and {len(self.features)} features")

    # ---------------------- FEATURE ENGINEERING ----------------------
    def engineer_targets(self):
        print("\n" + "=" * 70)
        print("⚙️  CREATING PERFORMANCE METRICS")
        print("=" * 70)

        df = self.df
        df["Spending Deviation"] = (
            (df["Outlays"] - df["Budgetary Resources"]) / df["Budgetary Resources"].replace(0, np.nan)
        ).fillna(0)

        df["Spending Category"] = pd.cut(
            df["Spending Deviation"],
            bins=[-np.inf, -0.05, 0.05, np.inf],
            labels=["Underspending", "On Target", "Overspending"]
        )

        df["Spending Risk (USD)"] = np.abs(df["Spending Deviation"]) * df["Budgetary Resources"]
        self.df = df

        print("✓ Metrics created — Deviation, Category, Risk")

        preview = df[["Agency Name", "Fiscal Year", "Spending Category",
                      "Spending Deviation", "Spending Risk (USD)"]].copy()
        preview["Spending Deviation"] = preview["Spending Deviation"].round(3)
        preview["Spending Risk (USD)"] = preview["Spending Risk (USD)"].round(2)
        display(preview.head(10))

        summary = df["Spending Category"].value_counts().reset_index()
        summary.columns = ["Spending Category", "Count"]
        display(summary)

        # Save to Excel
        output_file = f"federal_spending_analysis_{self.fiscal_year}.xlsx"
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            preview.to_excel(writer, index=False, sheet_name="Preview")
            summary.to_excel(writer, index=False, sheet_name="Summary")
            df.to_excel(writer, index=False, sheet_name="Full Data")

        print(f"\n💾 Analysis exported to {output_file}")

    # ---------------------- TRAIN MODELS ----------------------
    def train_models(self):
        print("\n" + "=" * 70)
        print("🤖  TRAINING ANALYTICAL MODELS")
        print("=" * 70)

        X = self.df[self.features]
        y_class = self.df["Spending Category"]
        y_reg = self.df["Spending Deviation"]

        X_train, X_test, y_train_c, y_test_c = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train_c)
        y_test_enc = le.transform(y_test_c)

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train_s, y_train_enc)
        acc = clf.score(X_test_s, y_test_enc)
        print(f"✓ Category Classifier trained — accuracy: {acc:.2f}")

        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        reg = GradientBoostingRegressor(random_state=42)
        reg.fit(X_train_r, y_train_r)
        r2 = reg.score(X_test_r, y_test_r)
        print(f"✓ Deviation Regressor trained — R²: {r2:.2f}")

        self.models = {"category_classifier": clf, "deviation_regressor": reg, "label_encoder": le}

    # ---------------------- SAVE MODELS ----------------------
    def save_models(self):
        os.makedirs("models", exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, f"models/{name}.pkl")
        joblib.dump(self.scaler, "models/scaler.pkl")
        print("✓ Models saved to /models")

    # ---------------------- MAIN RUN ----------------------
    def run(self):
        self.load_data()
        self.engineer_targets()
        self.train_models()
        self.save_models()
        print("\n" + "=" * 70)
        print("✅  TRAINING COMPLETE — ANALYSIS READY")
        print("=" * 70)

# ---------------------- ENTRY POINT ----------------------
def main():
    print("=" * 70)
    print("🏛️  FEDERAL SPENDING ANALYZER")
    print("=" * 70)
    fiscal_year = input("Enter fiscal year for analysis (e.g., 2024): ").strip()

    spending_file = f"us_agencies_budget_{fiscal_year}.csv"
    revenue_file = f"us_revenue_{fiscal_year}.csv"

    analyzer = GovSpendingAnalyzer(spending_file, revenue_file, fiscal_year)
    analyzer.run()

if __name__ == "__main__":
    main()
