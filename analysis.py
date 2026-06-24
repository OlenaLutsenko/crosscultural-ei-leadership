# ==========================================
# 1. LOAD DATA
# ==========================================

import pandas as pd
from scipy.stats import spearmanr, kruskal

pd.set_option("display.max_columns", None)

df = pd.read_excel("../Data/Crosscultural_EI_TL.xlsx")
df.columns = df.columns.str.strip()


# ==========================================
# 2. VARIABLES
# ==========================================

cols_ei = [
    "Self-Awareness",
    "Self-Management",
    "Social-Awareness",
    "Relationship Management",
    "EI"
]

variables = cols_ei + ["GTL"]


# ==========================================
# 3. DATA CHECKING
# ==========================================

print("\n" + "="*60)
print("DATA CHECKING")
print("="*60)

print("\nFirst rows of dataset:")
print(df.head())

print("\nTotal sample size:")
print(len(df))

print("\nMissing values in main variables:")
print(df[variables].isnull().sum())

df_clean = df.dropna(subset=variables)

print("\nSample size after removing missing values in EI/GTL variables:")
print(len(df_clean))

print("\nCountry of origin groups:")
print(df["Country of origin grouped"].value_counts(dropna=False))

print("\nCountry of business groups:")
print(df["Country of business grouped"].value_counts(dropna=False))


# ==========================================
# 4. DESCRIPTIVE STATISTICS
# ==========================================

desc = df[variables].agg([
    "count",
    "mean",
    "std",
    "median",
    "min",
    "max"
]).T

desc["missing"] = df[variables].isna().sum()
desc = desc.round(2)

print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)

print("\nDescriptive statistics for EI and GTL variables:")
print(desc)

print("\nSkewness:")
print(df[variables].skew().round(3))

print("\nKurtosis:")
print(df[variables].kurtosis().round(3))


# ==========================================
# 5. SPEARMAN CORRELATIONS
# ==========================================

results = []

for col in cols_ei:
    temp = df[[col, "GTL"]].dropna()
    rho, p = spearmanr(temp[col], temp["GTL"])

    results.append([
        col,
        len(temp),
        round(rho, 3),
        "< .001" if p < 0.001 else round(p, 3)
    ])

corr_table = pd.DataFrame(
    results,
    columns=["Variable", "n", "Spearman_rho", "p"]
)

print("\n" + "="*60)
print("SPEARMAN CORRELATIONS")
print("="*60)

print("\nCorrelations between EI dimensions and GTL:")
print(corr_table)


# ==========================================
# 6. CORRELATIONS WITHIN COUNTRY OF ORIGIN GROUPS
# ==========================================

origin_corr_results = []

for grp in df["Country of origin grouped"].dropna().unique():
    temp = df[df["Country of origin grouped"] == grp][["EI", "GTL"]].dropna()

    if len(temp) >= 3:
        rho, p = spearmanr(temp["EI"], temp["GTL"])
        origin_corr_results.append([
            grp,
            len(temp),
            round(rho, 3),
            round(p, 4)
        ])

origin_corr_table = pd.DataFrame(
    origin_corr_results,
    columns=["Country of origin grouped", "n", "Spearman_rho", "p"]
)

print("\n" + "="*60)
print("CORRELATIONS WITHIN COUNTRY OF ORIGIN GROUPS")
print("="*60)

print("\nCorrelation between EI and GTL by country of origin group:")
print(origin_corr_table)


# ==========================================
# 7. CORRELATIONS WITHIN COUNTRY OF BUSINESS GROUPS
# ==========================================

business_corr_results = []

for grp in df["Country of business grouped"].dropna().unique():
    temp = df[df["Country of business grouped"] == grp][["EI", "GTL"]].dropna()

    if len(temp) >= 3:
        rho, p = spearmanr(temp["EI"], temp["GTL"])
        business_corr_results.append([
            grp,
            len(temp),
            round(rho, 3),
            round(p, 4)
        ])

business_corr_table = pd.DataFrame(
    business_corr_results,
    columns=["Country of business grouped", "n", "Spearman_rho", "p"]
)

print("\n" + "="*60)
print("CORRELATIONS WITHIN COUNTRY OF BUSINESS GROUPS")
print("="*60)

print("\nCorrelation between EI and GTL by country of business group:")
print(business_corr_table)


# ==========================================
# 8. AGE GROUP CORRELATIONS
# ==========================================

age_corr_results = []

for var in variables:
    temp = df[["Age group", var]].dropna()

    if len(temp) >= 3:
        rho, p = spearmanr(temp["Age group"], temp[var])
        age_corr_results.append([
            var,
            len(temp),
            round(rho, 3),
            round(p, 4)
        ])

age_corr_table = pd.DataFrame(
    age_corr_results,
    columns=["Variable", "n", "Spearman_rho", "p"]
)

print("\n" + "="*60)
print("AGE GROUP CORRELATIONS")
print("="*60)

print("\nCorrelations between age group and EI/GTL variables:")
print(age_corr_table)


# ==========================================
# 9. KRUSKAL-WALLIS TESTS BY COUNTRY OF ORIGIN
# ==========================================

origin_kruskal_results = []

def interpret_eta2(x):
    if x < 0.01:
        return "negligible"
    elif x < 0.06:
        return "small"
    elif x < 0.14:
        return "medium"
    else:
        return "large"

for var in variables:
    groups = []

    for g in df["Country of origin grouped"].dropna().unique():
        group_data = df[df["Country of origin grouped"] == g][var].dropna()

        if len(group_data) > 0:
            groups.append(group_data)

    if len(groups) >= 2:
        stat, p = kruskal(*groups)

        k = len(groups)
        n = sum(len(g) for g in groups)

        eta2_H = (stat - k + 1) / (n - k)

        origin_kruskal_results.append([
            var,
            round(stat, 3),
            round(p, 4),
            round(eta2_H, 3),
            interpret_eta2(eta2_H)
        ])

origin_kruskal_table = pd.DataFrame(
    origin_kruskal_results,
    columns=["Variable", "Kruskal_H", "p", "eta2_H", "Effect_size"]
)

print("\n" + "="*60)
print("KRUSKAL-WALLIS TESTS BY COUNTRY OF ORIGIN")
print("="*60)

print("\nDifferences by country of origin group:")
print(origin_kruskal_table)

# ==========================================
# 10. KRUSKAL-WALLIS TESTS BY COUNTRY OF BUSINESS
# ==========================================

business_kruskal_results = []

for var in variables:
    groups = []

    for g in df["Country of business grouped"].dropna().unique():
        group_data = df[df["Country of business grouped"] == g][var].dropna()

        if len(group_data) > 0:
            groups.append(group_data)

    if len(groups) >= 2:
        stat, p = kruskal(*groups)

        k = len(groups)
        n = sum(len(g) for g in groups)

        eta2_H = (stat - k + 1) / (n - k)

        business_kruskal_results.append([
            var,
            round(stat, 3),
            round(p, 4),
            round(eta2_H, 3),
            interpret_eta2(eta2_H)
        ])

business_kruskal_table = pd.DataFrame(
    business_kruskal_results,
    columns=["Variable", "Kruskal_H", "p", "eta2_H", "Effect_size"]
)

print("\n" + "="*60)
print("KRUSKAL-WALLIS TESTS BY COUNTRY OF BUSINESS")
print("="*60)

print("\nDifferences by country of business group:")
print(business_kruskal_table)

# ==========================================
# 12. MEDIAN VALUES BY COUNTRY OF BUSINESS
# ==========================================

business_medians = df.groupby("Country of business grouped")[variables].median().round(2)

print("\n" + "="*60)
print("MEDIAN VALUES BY COUNTRY OF BUSINESS")
print("="*60)

print("\nMedians for EI and GTL variables by country of business group:")
print(business_medians)

import scikit_posthocs as sp

# ==========================================
# 13. DUNN-HOLM POST-HOC TESTS BY COUNTRY OF ORIGIN
# ==========================================

print("\n" + "="*60)
print("DUNN–HOLM POST-HOC TESTS")
print("="*60)

origin_posthoc = ["Self-Awareness", "EI"]

for var in origin_posthoc:

    print(f"\nCountry of origin grouped: {var}")

    temp = df[
        ["Country of origin grouped", var]
    ].dropna()

    dunn = sp.posthoc_dunn(
        temp,
        val_col=var,
        group_col="Country of origin grouped",
        p_adjust="holm"
    )

    print(dunn.round(4))

# ==========================================
# 14. DUNN-HOLM BY COUNTRY OF BUSINESS
# ==========================================

print("\n" + "="*60)
print("DUNN–HOLM BY COUNTRY OF BUSINESS")
print("="*60)

business_posthoc = [
    "Self-Awareness",
    "Relationship Management",
    "EI",
    "GTL"
]

for var in business_posthoc:

    print(f"\nCountry of business grouped: {var}")

    temp = df[
        ["Country of business grouped", var]
    ].dropna()

    dunn = sp.posthoc_dunn(
        temp,
        val_col=var,
        group_col="Country of business grouped",
        p_adjust="holm"
    )

    print(dunn.round(4))

# ==========================================
# 15. Does EI predict GTL?
# ==========================================

print("\n" + "="*60)
print("Does EI predict GTL?")
print("="*60)

import statsmodels.api as sm

temp = df[['EI','GTL']].dropna()

X = sm.add_constant(temp['EI'])
y = temp['GTL']

model = sm.OLS(y, X).fit()

print(model.summary())

# ==========================================
# 16. Which dimensions of EI are the strongest predictors of GTL?
# ==========================================

print("\n" + "="*60)
print("Which dimensions of EI are the strongest predictors of GTL?")
print("="*60)

X = df[
[
"Self-Awareness",
"Self-Management",
"Social-Awareness",
"Relationship Management"
]
]

X = sm.add_constant(X)

y = df["GTL"]

model = sm.OLS(y,X).fit()

print(model.summary())

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import pandas as pd

X = df[
[
    "Self-Awareness",
    "Self-Management",
    "Social-Awareness",
    "Relationship Management"
]
]

X = sm.add_constant(X)

vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print(vif)

# ==========================================
# 17. Model of GTL prediction with overall EI and culture control
# ==========================================

print("\n" + "="*60)
print("Overall EI with culture control")
print("="*60)

import statsmodels.formula.api as smf

model = smf.ols(
'GTL ~ EI + C(Q("Country of origin grouped"))',
data=df
).fit()

print(model.summary())

model = smf.ols(
'GTL ~ EI + C(Q("Country of business grouped"))',
data=df
).fit()

print(model.summary())

# ==========================================
# 18. Model with Age
# ==========================================

print("\n" + "="*60)
print("Model with Age")
print("="*60)

model = smf.ols(
    'GTL ~ EI + Q("Age group")',
    data=df
).fit()

print(model.summary())

# ==========================================
# 19. Figure 1. Scatterplot of EI and GTL
# ==========================================

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))

sns.regplot(
    data=df,
    x="EI",
    y="GTL",
    scatter_kws={"alpha":0.7},
    line_kws={"color":"red"}
)

plt.xlabel("Emotional Intelligence")
plt.ylabel("Global Transformational Leadership")
plt.title("Relationship between EI and GTL")

plt.tight_layout()
plt.show()

# ==========================================
# 20. Figure 2. Heatmap of Spearman correlations
# ==========================================

corr = df[[
    "Self-Awareness",
    "Self-Management",
    "Social-Awareness",
    "Relationship Management",
    "EI",
    "GTL"
]].corr(method='spearman')

plt.figure(figsize=(7,6))

sns.heatmap(
    corr,
    annot=True,
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1
)

plt.title("Spearman correlations")

plt.tight_layout()
plt.show()

# ==========================================
# 21. Figure 3. Boxplot EI by countries of origin
# ==========================================

plt.figure(figsize=(7,5))

sns.boxplot(
    data=df,
    x="Country of origin grouped",
    y="EI"
)

plt.xlabel("Country of origin")
plt.ylabel("EI")

plt.tight_layout()
plt.show()

# ==========================================
# 23. Figure 4. Boxplot of GTL by country of business
# ==========================================
plt.figure(figsize=(7,5))

sns.boxplot(
    data=df,
    x="Country of business grouped",
    y="GTL"
)

plt.xlabel("Country of business")
plt.ylabel("GTL")

plt.tight_layout()
plt.show()

# ==========================================
# 24. Figure 5. Forest plot of regression coefficients
# ==========================================
coef = [1.14,0.17,0.67,0.61]

variables = [
    'Self-Awareness',
    'Self-Management',
    'Social-Awareness',
    'Relationship Management'
]

plt.figure(figsize=(6,4))

sns.barplot(
    x=coef,
    y=variables
)

plt.xlabel("Regression coefficient")
plt.tight_layout()
plt.show()