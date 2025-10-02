import requests
import pandas as pd
from time import sleep

pd.set_option("display.float_format", "{:,.2f}".format)

fiscal_year = input("Enter fiscal year: ")

base_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/mts/mts_table_1"

#Gets totals
params = {"filter": f"record_date:eq:{fiscal_year}-09-30","page[size]": 100}

response = requests.get(base_url, params=params)
response.raise_for_status()
data = response.json()

revenue_data = data.get("data", [])

# Sets empty revenue list
revenue_list = []

# Loops to get data
for entry in revenue_data:
    category = entry.get("classification_desc", "N/A")
    year = fiscal_year
    
    #Gets the amounts/receipts
    amounts = (entry.get("current_month_rcpt_amt") or entry.get("fytd_rcpt_amt") or entry.get("current_month_gross_rcpt_amt") or entry.get("fytd_gross_rcpt_amt") or "0")
    
    try:
        amounts = float(amounts)
    except (ValueError, TypeError):
        amounts = 0.0
    if "FY" in category and category != f"FY {fiscal_year}":
        continue
    # prints data
    revenue_list.append({
        "Fiscal Year": year,
        "Revenue Category": category,
        "Total Receipts (Millions)": amounts
    })
    
    print(f"{category} ({year}) - Total Amounts: ${amounts:,.2f}")
    
    sleep(0.1)

df_revenue = pd.DataFrame(revenue_list)

output_file = f"us_revenue_{fiscal_year}.csv"
df_revenue.to_csv(output_file, index=False)

print(f"\nSaved revenue data for {len(df_revenue)} categories to {output_file}")
df_revenue.head(28)
