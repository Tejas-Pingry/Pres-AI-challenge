#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federal Spending Data Extractor & Categorizer
Pulls budget, obligations, and outlays from USAspending.gov API
Categorizes data for AI model training
"""

import requests
import pandas as pd
from time import sleep
import json
from datetime import datetime

class SpendingDataExtractor:
    def __init__(self, fiscal_year=2024):
        self.fiscal_year = fiscal_year
        self.base_url = "https://api.usaspending.gov/api/v2"
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
    def get_agencies(self):
        """Fetch all top-tier agencies"""
        url = f"{self.base_url}/references/toptier_agencies/"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json().get("results", [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching agencies: {e}")
            return []
    
    def get_agency_budget(self, agency_code, agency_name, retry_count=3):
        """Fetch budget data for a specific agency with retry logic"""
        url = f"{self.base_url}/agency/{agency_code}/budgetary_resources/?fiscal_year={self.fiscal_year}"
        
        for attempt in range(retry_count):
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    print(f"No data found for {agency_name} (404)")
                    return None
                else:
                    print(f"Attempt {attempt + 1}: Status {response.status_code} for {agency_name}")
                    sleep(2)
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1}: Error for {agency_name}: {e}")
                sleep(2)
        
        return None
    
    def categorize_spending(self, agency_name, budgetary_resources, obligations, outlays):
        """Categorize spending patterns and flag potential issues"""
        
        # Calculate variance metrics
        budget_to_outlay_ratio = outlays / budgetary_resources if budgetary_resources > 0 else 0
        obligation_rate = obligations / budgetary_resources if budgetary_resources > 0 else 0
        outlay_rate = outlays / obligations if obligations > 0 else 0
        
        # Initialize flags
        overspending_flag = budget_to_outlay_ratio > 1.0
        underutilization_flag = obligation_rate < 0.5
        fraud_risk_flag = False
        complexity_rating = 3  # Default medium complexity
        
        # Determine category based on agency type
        category = self._determine_category(agency_name)
        
        # Assess waste/issue type based on spending patterns
        waste_type = self._assess_waste_type(
            obligation_rate, 
            outlay_rate, 
            budget_to_outlay_ratio
        )
        
        # Calculate potential risk amount
        amount_at_risk = self._calculate_risk_amount(
            budgetary_resources, 
            obligations, 
            outlays, 
            waste_type
        )
        
        # Determine fraud risk based on agency and patterns
        fraud_risk_flag = self._assess_fraud_risk(
            agency_name, 
            obligation_rate, 
            outlay_rate
        )
        
        # Assess performance impact
        performance_impact = self._assess_performance(
            obligation_rate, 
            outlay_rate, 
            waste_type
        )
        
        # Complexity rating (1-5)
        complexity_rating = self._calculate_complexity(
            budgetary_resources, 
            waste_type
        )
        
        return {
            'category': category,
            'waste_type': waste_type,
            'amount_at_risk': amount_at_risk,
            'fraud_risk_flag': fraud_risk_flag,
            'overspending_flag': overspending_flag,
            'underutilization_flag': underutilization_flag,
            'performance_impact': performance_impact,
            'budget_authority_vs_outlays': self._ba_vs_out(budget_to_outlay_ratio),
            'complexity_rating': complexity_rating,
            'obligation_rate': round(obligation_rate * 100, 2),
            'outlay_rate': round(outlay_rate * 100, 2)
        }
    
    def _determine_category(self, agency_name):
        """Determine spending category based on agency"""
        discretionary_keywords = ['Defense', 'DOD', 'State', 'Veterans', 'Transportation', 'Energy', 'NASA', 'Justice']
        mandatory_keywords = ['Social Security', 'Medicare', 'Health', 'HHS', 'Labor']
        
        agency_upper = agency_name.upper()
        
        for keyword in mandatory_keywords:
            if keyword.upper() in agency_upper:
                return 'Mandatory'
        
        for keyword in discretionary_keywords:
            if keyword.upper() in agency_upper:
                return 'Discretionary'
        
        return 'Discretionary/Mixed'
    
    def _assess_waste_type(self, obligation_rate, outlay_rate, ba_to_out_ratio):
        """Assess potential waste type based on spending patterns"""
        
        if ba_to_out_ratio > 1.1:
            return 'Overspending'
        elif obligation_rate < 0.3:
            return 'Underutilization'
        elif outlay_rate < 0.5 and obligation_rate > 0.7:
            return 'Inefficient Workflow'
        elif obligation_rate > 0.95 and outlay_rate > 0.95:
            return 'Potential Improper Payments'
        else:
            return 'Normal Operations'
    
    def _calculate_risk_amount(self, budgetary_resources, obligations, outlays, waste_type):
        """Calculate estimated amount at risk"""
        
        if waste_type == 'Overspending':
            return max(0, outlays - budgetary_resources)
        elif waste_type == 'Underutilization':
            return budgetary_resources - obligations
        elif waste_type == 'Inefficient Workflow':
            return (obligations - outlays) * 0.1  # Estimate 10% waste
        elif waste_type == 'Potential Improper Payments':
            return outlays * 0.05  # Estimate 5% improper payment rate
        else:
            return 0
    
    def _assess_fraud_risk(self, agency_name, obligation_rate, outlay_rate):
        """Assess fraud risk based on agency type and patterns"""
        
        high_fraud_risk_agencies = ['Health', 'Medicare', 'Medicaid', 'HHS', 'Veterans', 'Social Security']
        
        agency_upper = agency_name.upper()
        
        # High-risk agencies
        for keyword in high_fraud_risk_agencies:
            if keyword.upper() in agency_upper:
                return True
        
        # Suspicious spending patterns
        if outlay_rate > 0.98 and obligation_rate > 0.98:
            return True
        
        return False
    
    def _assess_performance(self, obligation_rate, outlay_rate, waste_type):
        """Assess performance impact"""
        
        if waste_type in ['Overspending', 'Potential Improper Payments']:
            return 'Fiscal loss; compliance issues'
        elif waste_type == 'Underutilization':
            return 'Ineffective resource allocation'
        elif waste_type == 'Inefficient Workflow':
            return 'Delayed service delivery; operational inefficiency'
        else:
            return 'Normal operations'
    
    def _ba_vs_out(self, ratio):
        """Determine budget authority vs outlays relationship"""
        if ratio > 1.05:
            return 'BA < Out'
        elif ratio < 0.95:
            return 'BA > Out'
        else:
            return 'BA = Out'
    
    def _calculate_complexity(self, budgetary_resources, waste_type):
        """Calculate complexity rating (1-5)"""
        
        # Based on budget size
        if budgetary_resources > 100_000_000_000:  # >$100B
            complexity = 5
        elif budgetary_resources > 10_000_000_000:  # >$10B
            complexity = 4
        elif budgetary_resources > 1_000_000_000:   # >$1B
            complexity = 3
        elif budgetary_resources > 100_000_000:     # >$100M
            complexity = 2
        else:
            complexity = 1
        
        # Adjust for waste type
        if waste_type in ['Potential Improper Payments', 'Overspending']:
            complexity = min(5, complexity + 1)
        
        return complexity
    
    def extract_all_data(self):
        """Main extraction function"""
        agencies = self.get_agencies()
        
        if not agencies:
            print("Failed to fetch agencies")
            return pd.DataFrame()
        
        print(f"Found {len(agencies)} agencies. Starting extraction...\n")
        
        data_list = []
        
        for idx, agency in enumerate(agencies, 1):
            agency_code = agency.get("toptier_code")
            agency_name = agency.get("agency_name", "N/A")
            
            if not agency_code:
                continue
            
            print(f"[{idx}/{len(agencies)}] Processing: {agency_name}...")
            
            budget_data = self.get_agency_budget(agency_code, agency_name)
            
            if not budget_data:
                print(f"  ⚠️  No budget data available\n")
                continue
            
            # Extract fiscal year data
            year_data = None
            for year_entry in budget_data.get("agency_data_by_year", []):
                if year_entry.get("fiscal_year") == self.fiscal_year:
                    year_data = year_entry
                    break
            
            if not year_data:
                print(f"  ⚠️  No FY{self.fiscal_year} data\n")
                continue
            
            # Get financial metrics
            budgetary_resources = year_data.get("agency_budgetary_resources") or 0
            obligations = year_data.get("agency_total_obligated") or 0
            outlays = year_data.get("agency_total_outlayed") or 0
            
            # Categorize and flag issues
            categorization = self.categorize_spending(
                agency_name, 
                budgetary_resources, 
                obligations, 
                outlays
            )
            
            # Build comprehensive record
            record = {
                'Department/Entity': agency_name,
                'Agency Code': agency_code,
                'Program/Project Name': f"{agency_name} (Aggregate)",
                'Category of Spending': categorization['category'],
                'Fiscal Year': self.fiscal_year,
                'Budgetary Resources': budgetary_resources,
                'Obligations': obligations,
                'Outlays': outlays,
                'Obligation Rate (%)': categorization['obligation_rate'],
                'Outlay Rate (%)': categorization['outlay_rate'],
                'Waste/Issue Type': categorization['waste_type'],
                'Amount At Risk (USD)': categorization['amount_at_risk'],
                'Fraud Risk Flag': categorization['fraud_risk_flag'],
                'Overspending Flag': categorization['overspending_flag'],
                'Underutilization Flag': categorization['underutilization_flag'],
                'Performance Impact': categorization['performance_impact'],
                'Budget Authority vs Outlays': categorization['budget_authority_vs_outlays'],
                'Complexity Rating': categorization['complexity_rating'],
                'Program Scale': 'Large' if budgetary_resources > 10_000_000_000 else 'Medium' if budgetary_resources > 1_000_000_000 else 'Small',
                'Outcome Status': 'Ongoing',
                'Political Sensitivity': 'Medium',
                'Data Source': 'USAspending.gov API',
                'Extraction Date': datetime.now().strftime('%Y-%m-%d')
            }
            
            data_list.append(record)
            
            print(f"  ✓ Budget: ${budgetary_resources:,.0f}")
            print(f"  ✓ Obligations: ${obligations:,.0f} ({categorization['obligation_rate']:.1f}%)")
            print(f"  ✓ Outlays: ${outlays:,.0f} ({categorization['outlay_rate']:.1f}%)")
            print(f"  ✓ Waste Type: {categorization['waste_type']}")
            if categorization['amount_at_risk'] > 0:
                print(f"  ⚠️  Amount at Risk: ${categorization['amount_at_risk']:,.0f}")
            print()
            
            # Rate limiting
            sleep(0.5)
        
        return pd.DataFrame(data_list)


def main():
    """Main execution function"""
    
    print("=" * 70)
    print("Federal Spending Data Extractor & Categorizer")
    print("=" * 70)
    print()
    
    fiscal_year = 2024
    
    extractor = SpendingDataExtractor(fiscal_year=fiscal_year)
    
    print(f"Extracting data for Fiscal Year {fiscal_year}...\n")
    
    df = extractor.extract_all_data()
    
    if df.empty:
        print("No data extracted. Exiting.")
        return
    
    # Display summary statistics
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE - SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\nTotal Agencies Processed: {len(df)}")
    print(f"Total Budgetary Resources: ${df['Budgetary Resources'].sum():,.0f}")
    print(f"Total Obligations: ${df['Obligations'].sum():,.0f}")
    print(f"Total Outlays: ${df['Outlays'].sum():,.0f}")
    print(f"Total Amount at Risk: ${df['Amount At Risk (USD)'].sum():,.0f}")
    
    print("\n--- Issue Type Breakdown ---")
    print(df['Waste/Issue Type'].value_counts())
    
    print("\n--- Fraud Risk Agencies ---")
    fraud_count = df['Fraud Risk Flag'].sum()
    print(f"Agencies flagged for fraud risk: {fraud_count}")
    
    print("\n--- Overspending Agencies ---")
    overspend_count = df['Overspending Flag'].sum()
    print(f"Agencies flagged for overspending: {overspend_count}")
    
    # Export to CSV
    filename = f"federal_spending_fy{fiscal_year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"\n✓ Data exported to: {filename}")
    
    # Export high-risk subset
    high_risk = df[
        (df['Fraud Risk Flag'] == True) | 
        (df['Overspending Flag'] == True) | 
        (df['Amount At Risk (USD)'] > 1_000_000_000)
    ]
    
    if not high_risk.empty:
        risk_filename = f"high_risk_spending_fy{fiscal_year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        high_risk.to_csv(risk_filename, index=False)
        print(f"✓ High-risk data exported to: {risk_filename}")
    
    print("\n" + "=" * 70)
    print("Done! Ready for AI model training.")
    print("=" * 70)


if __name__ == "__main__":
    main()
