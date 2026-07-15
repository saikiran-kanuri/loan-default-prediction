# German Credit Data — Column Mapping (UCI Statlog Version)

| # | Column | Meaning of Codes |
|---|--------|-------------------|
| 0 | Creditability | 1 = Good, 0 = Bad (**target variable**) |
| 1 | Account Balance | 1 = No account, 2 = None (<0 DM), 3 = 0–200 DM, 4 = ≥200 DM |
| 2 | Duration of Credit (month) | Numeric — loan duration in months |
| 3 | Payment Status of Previous Credit | 0 = Delay in past, 1 = Critical account, 2 = Other credits existing, 3 = Paid duly till now, 4 = No credits taken / all paid |
| 4 | Purpose | 0 = New car, 1 = Used car, 2 = Furniture, 3 = Radio/TV, 4 = Appliances, 5 = Repairs, 6 = Education, 7 = Vacation, 8 = Retraining, 9 = Business, 10 = Other |
| 5 | Credit Amount | Numeric — loan amount in DM (Deutsche Mark) |
| 6 | Value Savings/Stocks | 1 = None, 2 = <100 DM, 3 = 100–500 DM, 4 = 500–1000 DM, 5 = ≥1000 DM |
| 7 | Length of current employment | 1 = Unemployed, 2 = <1 year, 3 = 1–4 years, 4 = 4–7 years, 5 = ≥7 years |
| 8 | Instalment per cent | Numeric — instalment rate as % of disposable income (1–4 scale) |
| 9 | Sex & Marital Status | 1 = Male-divorced, 2 = Male-single, 3 = Male-married/widowed, 4 = Female |
| 10 | Guarantors | 1 = None, 2 = Co-applicant, 3 = Guarantor |
| 11 | Duration in Current address | Numeric — years at current address (1–4 scale) |
| 12 | Most valuable available asset | 1 = None, 2 = Car/other, 3 = Life insurance, 4 = Real estate |
| 13 | Age (years) | Numeric — applicant's age |
| 14 | Concurrent Credits | 1 = At other banks, 2 = At stores, 3 = None |
| 15 | Type of apartment | 1 = Free/rent-free, 2 = Rented, 3 = Owned |
| 16 | No of Credits at this Bank | Numeric — count of existing credits |
| 17 | Occupation | 1 = Unemployed/unskilled non-resident, 2 = Unskilled resident, 3 = Skilled employee, 4 = Management/self-employed |
| 18 | No of dependents | Numeric — number of dependents |
| 19 | Telephone | 1 = None, 2 = Yes, registered |
| 20 | Foreign Worker | 1 = Yes, 2 = No |

**Notes:**
- Numeric columns marked "Numeric" are continuous/count values, not categorical codes.
- All categorical columns should be treated as `category` dtype (not raw int) during feature engineering, to avoid the model treating code order as meaningful (e.g. Purpose 10 is NOT "more" than Purpose 1).
- This mapping is used consistently in `features.py` and `evaluate.py` (SHAP feature labels).