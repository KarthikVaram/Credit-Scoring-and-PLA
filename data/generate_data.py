"""
generate_data.py
----------------
Generates a synthetic Lending Club-style loan dataset (10,000 loans)
with the same schema, feature distributions, and default patterns as
the real 2007-2011 Lending Club cohort.

Usage:
    python3 generate_data.py

Output:
    data/lending_club.csv
"""

import csv
import random
import math
import os

random.seed(42)

N = 10000

GRADES     = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
GRADE_DR   = {'A': 0.03, 'B': 0.07, 'C': 0.13, 'D': 0.20,
              'E': 0.27, 'F': 0.35, 'G': 0.42}
GRADE_RATE = {'A': (5.5, 8.5),   'B': (8.5, 12.5),  'C': (12.5, 16.5),
              'D': (16.5, 21.0), 'E': (21.0, 26.0), 'F': (26.0, 30.0),
              'G': (30.0, 35.0)}
PURPOSES   = ['debt_consolidation', 'credit_card', 'home_improvement',
              'major_purchase', 'medical', 'small_business', 'vacation', 'other']
HOME_OWN   = ['RENT', 'OWN', 'MORTGAGE']
TERMS      = [36, 60]

rows = []
for i in range(N):
    g   = random.choices(GRADES, weights=[25, 22, 18, 14, 10, 7, 4])[0]
    dr  = GRADE_DR[g]
    trm = random.choices(TERMS, weights=[65, 35])[0]

    annual_inc  = max(15000, random.lognormvariate(math.log(65000), 0.55))
    loan_amnt   = max(1000, min(35000, random.lognormvariate(math.log(12000), 0.55)))
    dti         = max(0, min(40, random.normalvariate(16, 7)))
    fico        = int(max(580, min(850, random.normalvariate(700, 55))))
    open_acc    = max(1, int(random.normalvariate(10, 5)))
    revol_util  = max(0, min(100, random.normalvariate(52, 25)))
    delinq_2yrs = random.choices([0, 1, 2, 3, 4], weights=[72, 14, 7, 4, 3])[0]
    pub_rec     = random.choices([0, 1, 2], weights=[88, 9, 3])[0]
    inq_last_6m = random.choices([0, 1, 2, 3, 4, 5], weights=[38, 26, 17, 10, 6, 3])[0]
    emp_length  = random.choices(range(11), weights=[6, 5, 9, 8, 8, 7, 6, 6, 6, 5, 30])[0]
    lo, hi      = GRADE_RATE[g]
    int_rate    = round(random.uniform(lo, hi), 2)
    purpose     = random.choice(PURPOSES)
    home_own    = random.choices(HOME_OWN, weights=[45, 12, 43])[0]

    # Default probability influenced by multiple risk factors
    p_default = dr
    if dti > 25:                          p_default += 0.05
    if fico < 650:                        p_default += 0.08
    if fico > 750:                        p_default -= 0.05
    if delinq_2yrs > 0:                  p_default += 0.04 * delinq_2yrs
    if pub_rec > 0:                       p_default += 0.06
    if revol_util > 80:                   p_default += 0.04
    if loan_amnt / (annual_inc / 12) > 0.5: p_default += 0.03
    p_default = max(0.01, min(0.95, p_default))

    default     = 1 if random.random() < p_default else 0
    loan_status = 'Charged Off' if default else 'Fully Paid'

    rows.append({
        'id':              i + 1,
        'loan_amnt':       round(loan_amnt, 2),
        'term':            f'{trm} months',
        'int_rate':        int_rate,
        'grade':           g,
        'emp_length':      emp_length,
        'home_ownership':  home_own,
        'annual_inc':      round(annual_inc, 2),
        'purpose':         purpose,
        'dti':             round(dti, 2),
        'delinq_2yrs':     delinq_2yrs,
        'fico_range_low':  fico - 4,
        'fico_range_high': fico + 4,
        'inq_last_6mths':  inq_last_6m,
        'open_acc':        open_acc,
        'pub_rec':         pub_rec,
        'revol_util':      round(revol_util, 1),
        'loan_status':     loan_status,
        'default':         default,
    })

os.makedirs('data', exist_ok=True)
out_path = os.path.join('data', 'lending_club.csv')
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

defaults = sum(r['default'] for r in rows)
print(f"Generated {N:,} loans  ->  {out_path}")
print(f"Default rate: {defaults} / {N} = {100*defaults/N:.1f}%")
grade_counts = {}
for r in rows:
    grade_counts[r['grade']] = grade_counts.get(r['grade'], 0) + 1
print("Grade distribution:", dict(sorted(grade_counts.items())))
