import requests
import pandas as pd
from time import sleep

pd.set_option("display.float_format", "{:,.2f}".format)

# basic variables
fiscal_year=input("Enter fiscal year: ")
agencies_url="https://api.usaspending.gov/api/v2/references/toptier_agencies/"
response = requests.get(agencies_url)
response.raise_for_status()
agency_data = response.json().get("results", [])

# setting empty list for budget
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

        # printing data
        agency_budget_list.append({"Agency Code": agency_code,"Agency Name": agency_name,"Fiscal Year": fiscal_year,"Budgetary Resources": budgetary_resources,"Obligations": obligations,"Outlays": outlays})
        print(f"{agency_name} - Budget: ${budgetary_resources:,.2f}, Obligations: ${obligations:,.2f}, Outlays: ${outlays:,.2f}")
    else:
        print(f"No data for {agency_name} (Code {agency_code}) in Fiscal yeaar {fiscal_year}")

df_agencies_budget = pd.DataFrame(agency_budget_list)

output_file = f"us_agencies_budget_{fiscal_year}.csv"
df_agencies_budget.to_csv(output_file, index=False)

df_agencies_budget.head(118)

# idrk what obligations or outlays are i just saw them on the gov spending website



'''

Here's what outlays and obligations are: 

In the federal budget, an obligation is a binding commitment to make a payment for goods, services, or other obligations, such as awarding a grant, signing a contract, or compensating federal workers. 
These are legal liabilities to disburse funds, either immediately or in the future, and occur when the government enters into a binding agreement. 
Obligations are distinct from "outlays," which are the actual payments made. 



What outlays represent:
Actual payments: Outlays show the money that has been paid out, rather than just promised. 
Liquidating obligations: They are the result of the government making a payment to settle a legal obligation, which can be funded by an appropriation or permanent law. 
A measure of spending: Outlays are used to determine the size of the actual federal spending and, therefore, the annual deficit or surplus, as they are the amount that is higher than the revenue collected. 


Examples of outlays:
Disbursing payments: The Social Security Administration making payments to beneficiaries. 
Making payments for contracts: When a federal agency makes a payment to a contractor. 
Issuing grants: Transferring funds to a recipient's bank account. 
Personnel compensation: Paying salaries to federal employees. 
'''
