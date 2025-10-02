import requests
import pandas as pd
from time import sleep

#basicvariables
fiscal_year=2024
agencies_url="https://api.usaspending.gov/api/v2/references/toptier_agencies/"
response = requests.get(agencies_url)
response.raise_for_status()
agency_data = response.json().get("results", [])

#settingemptylistforbudget
agency_budget_list=[]

# looping through agencys to get data
for agency in agency_data:
    agency_code=agency.get("toptier_code")
    agency_name=agency.get("agency_name", "N/A")
    if not agency_code:
        continue

    budget_url=f"https://api.usaspending.gov/api/v2/agency/{agency_code}/budgetary_resources/?fiscal_year={fiscal_year}"
    budget_response = requests.get(budget_url)
    
    if budget_response.status_code != 200:
        print(f"Failed to get budget for {agency_name}")
        continue

    budget_data=budget_response.json()
    year_data=None
    for year_entry in budget_data.get("agency_data_by_year", []):
        if year_entry.get("fiscal_year") == fiscal_year:
            year_data = year_entry
            break
    
    if year_data:
        budgetary_resources = year_data.get("agency_budgetary_resources") or 0
        obligations = year_data.get("agency_total_obligated") or 0
        outlays = year_data.get("agency_total_outlayed") or 0
        
        #printingdata
        agency_budget_list.append({"Agency Code": agency_code,"Agency Name": agency_name,"Fiscal Year": fiscal_year,"Budgetary Resources": budgetary_resources,"Obligations": obligations,"Outlays": outlays})
        print(f"{agency_name} - Budget: ${budgetary_resources:,.2f}, Obligations: ${obligations:,.2f}, Outlays: ${outlays:,.2f}")
    else:
        print(f"No data for {agency_name} (Code {agency_code}) in Fiscal yeaar {fiscal_year}")

df_agencies_budget = pd.DataFrame(agency_budget_list)

df_agencies_budget.head()

# idrk what obligations or outlays are i just saw them on the gov spending website
