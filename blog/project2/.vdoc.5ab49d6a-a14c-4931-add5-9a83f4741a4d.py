# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
df = pd.read_stata('/Users/bibibingya/Downloads/emw_marketing_analytics/blog/project2/karlan_list_2007.dta')
variables_to_test = ['mrm2', 'years', 'freq', 'female']

results = []

for var in variables_to_test:
    treat = df[df['treatment'] == 1][var].dropna()
    control = df[df['treatment'] == 0][var].dropna()

    diff = treat.mean() - control.mean()
    se = np.sqrt(treat.var(ddof=1)/len(treat) + control.var(ddof=1)/len(control))
    t_manual = diff / se
    df_t = len(treat) + len(control) - 2
    p_manual = 2 * (1 - stats.t.cdf(abs(t_manual), df_t))

    reg_df = df[[var, 'treatment']].dropna()
    X = sm.add_constant(reg_df['treatment'])
    y = reg_df[var]
    model = sm.OLS(y, X).fit()
    
    results.append({
        "Variable": var,
        "Mean Difference": round(diff, 4),
        "T-stat (manual)": round(t_manual, 4),
        "P-value (manual)": round(p_manual, 4),
        "Coef (regression)": round(model.params['treatment'], 4),
        "T-stat (regression)": round(model.tvalues['treatment'], 4),
        "P-value (regression)": round(model.pvalues['treatment'], 4)
    })

results_df = pd.DataFrame(results)

print("Baseline Balance Table (Treatment vs Control):")
display(results_df)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
from scipy.stats import ttest_ind
import pandas as pd

df = pd.read_stata('/Users/bibibingya/Downloads/emw_marketing_analytics/blog/project2/karlan_list_2007.dta')
treated_df = df[df['treatment'] == 1]

comparisons = [
    ('ratio', 'ratio2', '1:1 vs 2:1'),
    ('ratio', 'ratio3', '1:1 vs 3:1'),
    ('ratio2', 'ratio3', '2:1 vs 3:1')
]

t_test_results = []

for base_col, compare_col, label in comparisons:
    group_base = treated_df[treated_df[base_col] == 1]['gave'].dropna()
    group_compare = treated_df[treated_df[compare_col] == 1]['gave'].dropna()

    t_stat, p_val = ttest_ind(group_base, group_compare, equal_var=False)

    t_test_results.append({
        'Comparison': label,
        'Mean (Group A)': round(group_base.mean(), 4),
        'Mean (Group B)': round(group_compare.mean(), 4),
        'T-statistic': round(t_stat, 4),
        'P-value': round(p_val, 4)
    })

results_df = pd.DataFrame(t_test_results)
results_df
#
#
#
#
#
#
#
#
#
#
#

import statsmodels.formula.api as smf
import pandas as pd

df = pd.read_stata('/Users/bibibingya/Downloads/emw_marketing_analytics/blog/project2/karlan_list_2007.dta')

model = smf.logit("gave ~ C(ratio)", data=df).fit()
print(model.summary())
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_stata('/Users/bibibingya/Downloads/emw_marketing_analytics/blog/project2/karlan_list_2007.dta')

donated_df = df[df['gave'] == 1]

treat = donated_df[donated_df['treatment'] == 1]['amount'].dropna()
control = donated_df[donated_df['treatment'] == 0]['amount'].dropna()

mean_treat = treat.mean()
mean_control = control.mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Treatment plot
axes[0].hist(treat, bins=30, color='skyblue', edgecolor='black')
axes[0].axvline(mean_treat, color='red', linestyle='--', linewidth=2, label=f'Mean = ${mean_treat:.2f}')
axes[0].set_title('Treatment Group (Donors Only)')
axes[0].set_xlabel('Donation Amount ($)')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Control plot
axes[1].hist(control, bins=30, color='lightgreen', edgecolor='black')
axes[1].axvline(mean_control, color='red', linestyle='--', linewidth=2, label=f'Mean = ${mean_control:.2f}')
axes[1].set_title('Control Group (Donors Only)')
axes[1].set_xlabel('Donation Amount ($)')
axes[1].legend()

plt.tight_layout()
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

control_draws = np.random.binomial(1, 0.018, 100000)
treatment_draws = np.random.binomial(1, 0.022, 10000)

control_sample = np.random.choice(control_draws, size=10000, replace=False)

differences = treatment_draws - control_sample
cumulative_avg = np.cumsum(differences) / np.arange(1, len(differences) + 1)

plt.figure(figsize=(10, 5))
plt.plot(cumulative_avg, color="blue", linewidth=1)
plt.axhline(y=0.004, color='red', linestyle='--', label='True Difference (0.022 - 0.018)')
plt.title("Cumulative Average of Differences: Treatment vs. Control")
plt.xlabel("Number of Simulations")
plt.ylabel("Cumulative Average Difference")
plt.legend()
plt.grid(True)
```
#
#
#
#
#
#
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

control_p = 0.018
treatment_p = 0.022
sample_sizes = [50, 200, 500, 1000]
n_simulations = 1000

simulated_distributions = {}

for n in sample_sizes:
    avg_diffs = []
    for _ in range(n_simulations):
        control = np.random.binomial(1, control_p, n)
        treatment = np.random.binomial(1, treatment_p, n)
        avg_diffs.append(np.mean(treatment) - np.mean(control))
    simulated_distributions[n] = avg_diffs

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, n in enumerate(sample_sizes):
    diffs = simulated_distributions[n]
    axes[i].hist(diffs, bins=50, color="lightblue", edgecolor="black", density=True)
    axes[i].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[i].axvline(np.mean(diffs), color='green', linestyle='-', linewidth=2, label='Mean')
    axes[i].set_title(f"Sample Size = {n}")
    axes[i].set_xlabel("Avg Difference (Treatment - Control)")
    axes[i].set_ylabel("Density")
    axes[i].legend()

plt.tight_layout()
plt.show()
#
#
#
#
