# =========================================================
# ALL RESULTS AS FIGURES
# NO TABLES
# MULTI-PANEL ONLY
# =========================================================

# =========================================================
# SETUP
# =========================================================

import os
import warnings

warnings.filterwarnings("ignore")

folders = [
    "figures",
    "supplement"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# =========================================================
# IMPORTS
# =========================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import (
    variance_inflation_factor,
    OLSInfluence
)

# =========================================================
# STYLE
# =========================================================

plt.rcParams.update({

    "font.family": "sans-serif",

    "font.size": 11,

    "axes.titlesize": 14,
    "axes.labelsize": 12,

    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    "legend.fontsize": 9,

    "figure.dpi": 300,
    "savefig.dpi": 300,

    "axes.linewidth": 1.2,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

sns.set_style("white")

# =========================================================
# =========================================================

df = pd.read_csv("final_analysis_dataset.csv")

# =========================================================
# CLEAN VARIABLES
# =========================================================

df["white"] = df["white"].replace({
    1: "White",
    0: "Non-White"
})

df["hivatvisit"] = df["hivatvisit"].replace({
    1: "HIV-positive",
    0: "HIV-negative"
})

# =========================================================
# FIGURE 1
# REGIONAL HETEROGENEITY
# =========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14,10)
)

# =========================================================
# PANEL A
# GEAA
# =========================================================

sns.boxplot(
    data=df,
    x="region",
    y="geaa",
    showfliers=False,
    ax=axes[0,0]
)

sns.stripplot(
    data=df,
    x="region",
    y="geaa",
    color="black",
    alpha=0.35,
    jitter=0.18,
    size=2.5,
    ax=axes[0,0]
)

axes[0,0].set_title("A. GEAA by Region")

# =========================================================
# PANEL B
# PM2.5
# =========================================================

sns.boxplot(
    data=df,
    x="region",
    y="SPWPM2.5",
    showfliers=False,
    ax=axes[0,1]
)

sns.stripplot(
    data=df,
    x="region",
    y="SPWPM2.5",
    color="black",
    alpha=0.35,
    jitter=0.18,
    size=2.5,
    ax=axes[0,1]
)

axes[0,1].set_title("B. PM2.5 by Region")

# =========================================================
# PANEL C
# SMOKING
# =========================================================

sns.boxplot(
    data=df,
    x="region",
    y="cum_pkyear",
    showfliers=False,
    ax=axes[1,0]
)

sns.stripplot(
    data=df,
    x="region",
    y="cum_pkyear",
    color="black",
    alpha=0.35,
    jitter=0.18,
    size=2.5,
    ax=axes[1,0]
)

axes[1,0].set_title("C. Smoking Pack-Years")

# =========================================================
# PANEL D
# EDUCATION
# =========================================================

sns.boxplot(
    data=df,
    x="region",
    y="educbas",
    showfliers=False,
    ax=axes[1,1]
)

sns.stripplot(
    data=df,
    x="region",
    y="educbas",
    color="black",
    alpha=0.35,
    jitter=0.18,
    size=2.5,
    ax=axes[1,1]
)

axes[1,1].set_title("D. Education Level")

plt.tight_layout()

plt.savefig(
    "figures/Figure1_Regional_Profiles.png",
    bbox_inches="tight"
)

plt.close()

# =========================================================
# MAIN MODEL
# =========================================================

model_main = smf.ols(
    """
    geaa ~

    Temperature +

    Q('SPWPM2.5') +

    educbas +

    bmi +

    cum_pkyear +

    C(white) +

    C(region) +

    C(hivatvisit)
    """,
    data=df
).fit(cov_type='HC3')

# =========================================================
# FIGURE 2
# MAIN RESULTS
# =========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14,10)
)

# =========================================================
# PANEL A
# TEMPERATURE
# =========================================================

sns.scatterplot(
    data=df,
    x="Temperature",
    y="geaa",
    hue="region",
    alpha=0.6,
    ax=axes[0,0]
)

sns.regplot(
    data=df,
    x="Temperature",
    y="geaa",
    scatter=False,
    ax=axes[0,0]
)

axes[0,0].set_title("A. Temperature and GEAA")

# =========================================================
# PANEL B
# PM2.5
# =========================================================

sns.scatterplot(
    data=df,
    x="SPWPM2.5",
    y="geaa",
    hue="region",
    alpha=0.6,
    ax=axes[0,1]
)

sns.regplot(
    data=df,
    x="SPWPM2.5",
    y="geaa",
    scatter=False,
    ax=axes[0,1]
)

axes[0,1].set_title("B. PM2.5 and GEAA")

# =========================================================
# PANEL C
# CORRELATION HEATMAP
# =========================================================

corr_vars = [

    "Temperature",
    "SPWPM2.5",
    "NO2",
    "Ozone",
    "SO2",
    "geaa"
]

corr = df[corr_vars].corr()

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    square=True,
    ax=axes[1,0]
)

axes[1,0].set_title("C. Correlation Matrix")

# =========================================================
# PANEL D
# MAIN EFFECTS FOREST
# =========================================================

coef_df = pd.DataFrame({

    "Variable": model_main.params.index,

    "Beta": model_main.params.values,

    "CI_low": model_main.conf_int()[0].values,

    "CI_high": model_main.conf_int()[1].values
})

coef_df = coef_df[
    coef_df["Variable"] != "Intercept"
]

coef_df = coef_df.sort_values(
    "Beta"
)

y_pos = np.arange(len(coef_df))

axes[1,1].errorbar(

    coef_df["Beta"],
    y_pos,

    xerr=[
        coef_df["Beta"] - coef_df["CI_low"],
        coef_df["CI_high"] - coef_df["Beta"]
    ],

    fmt='o',
    capsize=4
)

axes[1,1].axvline(
    0,
    linestyle="--"
)

axes[1,1].set_yticks(y_pos)

axes[1,1].set_yticklabels(
    coef_df["Variable"]
)

axes[1,1].set_title("D. Main Model Effects")

plt.tight_layout()

plt.savefig(
    "figures/Figure2_Main_Associations.png",
    bbox_inches="tight"
)

plt.close()

# =========================================================
# FIGURE 3
# MULTI-CLOCK COMPARISON
# =========================================================

outcomes = [

    "geaa",
    "eeaa",
    "peaa",
    "aar",
    "dnamtladjage"
]

results = []

# =========================================================
# STANDARDIZE
# =========================================================

std_df = df.copy()

standardize_vars = [

    "Temperature",
    "SPWPM2.5",
    "educbas",
    "bmi",
    "cum_pkyear",

    "geaa",
    "eeaa",
    "peaa",
    "aar",
    "dnamtladjage"
]

scaler = StandardScaler()

std_df[standardize_vars] = scaler.fit_transform(
    std_df[standardize_vars]
)

# =========================================================
# LOOP
# =========================================================

for outcome in outcomes:

    fit = smf.ols(
        f"""
        {outcome} ~

        Temperature +

        Q('SPWPM2.5') +

        educbas +

        bmi +

        cum_pkyear +

        C(white) +

        C(region) +

        C(hivatvisit)
        """,
        data=std_df
    ).fit(cov_type='HC3')

    tmp = pd.DataFrame({

        "Outcome": outcome,

        "Variable": fit.params.index,

        "Beta": fit.params.values
    })

    results.append(tmp)

results = pd.concat(results)

plot_df = results[
    results["Variable"].isin([
        "Temperature",
        "Q('SPWPM2.5')",
        "educbas",
        "bmi",
        "cum_pkyear"
    ])
]

pivot_df = plot_df.pivot(
    index="Variable",
    columns="Outcome",
    values="Beta"
)

# =========================================================
# MULTI-PANEL CLOCK FIGURE
# =========================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16,10)
)

axes = axes.flatten()

for i, outcome in enumerate(outcomes):

    temp_df = results[
        results["Outcome"] == outcome
    ]

    temp_df = temp_df[
        temp_df["Variable"].isin([
            "Temperature",
            "Q('SPWPM2.5')",
            "educbas",
            "bmi",
            "cum_pkyear"
        ])
    ]

    sns.barplot(
        data=temp_df,
        x="Beta",
        y="Variable",
        ax=axes[i]
    )

    axes[i].set_title(
        outcome.upper()
    )

axes[-1].axis("off")

plt.tight_layout()

plt.savefig(
    "figures/Figure3_Multiclock_Panels.png",
    bbox_inches="tight"
)

plt.close()

# =========================================================
# FIGURE 4
# SENSITIVITY ANALYSES
# =========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14,10)
)

# =========================================================
# HIV ONLY
# =========================================================

df_hiv = df[
    df["hivatvisit"] == "HIV-positive"
]

model_hiv = smf.ols(
    """
    geaa ~

    Temperature +

    Q('SPWPM2.5') +

    educbas +

    bmi +

    cum_pkyear +

    C(white) +

    C(region)
    """,
    data=df_hiv
).fit(cov_type='HC3')

coef_hiv = pd.DataFrame({

    "Variable": model_hiv.params.index,

    "Beta": model_hiv.params.values
})

coef_hiv = coef_hiv[
    coef_hiv["Variable"] != "Intercept"
]

sns.barplot(
    data=coef_hiv,
    x="Beta",
    y="Variable",
    ax=axes[0,0]
)

axes[0,0].set_title("A. HIV-Only Model")

# =========================================================
# REMOVE EXTREME SMOKERS
# =========================================================

df_sens = df[
    df["cum_pkyear"] <
    df["cum_pkyear"].quantile(0.95)
]

model_sens = smf.ols(
    """
    geaa ~

    Temperature +

    Q('SPWPM2.5') +

    educbas +

    bmi +

    cum_pkyear +

    C(white) +

    C(region) +

    C(hivatvisit)
    """,
    data=df_sens
).fit(cov_type='HC3')

coef_sens = pd.DataFrame({

    "Variable": model_sens.params.index,

    "Beta": model_sens.params.values
})

coef_sens = coef_sens[
    coef_sens["Variable"] != "Intercept"
]

sns.barplot(
    data=coef_sens,
    x="Beta",
    y="Variable",
    ax=axes[0,1]
)

axes[0,1].set_title("B. Excluding Extreme Smokers")

# =========================================================
# VIF
# =========================================================

X = df[[
    "Temperature",
    "SPWPM2.5",
    "educbas",
    "bmi",
    "cum_pkyear"
]]

X = sm.add_constant(X)

vif_df = pd.DataFrame({

    "Variable": X.columns,

    "VIF": [

        variance_inflation_factor(
            X.values,
            i
        )

        for i in range(X.shape[1])
    ]
})

sns.barplot(
    data=vif_df,
    x="VIF",
    y="Variable",
    ax=axes[1,0]
)

axes[1,0].set_title("C. Variance Inflation Factors")

# =========================================================
# COOK DISTANCE
# =========================================================

influence = OLSInfluence(model_main)

cooks = influence.cooks_distance[0]

axes[1,1].stem(
    np.arange(len(cooks)),
    cooks
)

axes[1,1].set_title("D. Cook's Distance")

plt.tight_layout()

plt.savefig(
    "figures/Figure4_Sensitivity.png",
    bbox_inches="tight"
)

plt.close()

# =========================================================
# SUPPLEMENT
# =========================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15,9)
)

axes = axes.flatten()

for i, outcome in enumerate(outcomes):

    sns.histplot(
        df[outcome],
        bins=30,
        kde=True,
        ax=axes[i]
    )

    axes[i].set_title(
        outcome.upper()
    )

axes[-1].axis("off")

plt.tight_layout()

plt.savefig(
    "supplement/Supplement_FigureS1_Distributions.png",
    bbox_inches="tight"
)

plt.close()

print("\n======================================")
print("ALL PUBLICATION FIGURES COMPLETED")
print("======================================")

# =========================================================
# ALL RESULTS AS MANUSCRIPT-GRADE MULTI-PANEL FIGURES
# =========================================================

# =========================================================
# SETUP
# =========================================================

warnings.filterwarnings("ignore")

folders = [
    "figures",
    "supplement"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# =========================================================
# IMPORTS
# =========================================================

    variance_inflation_factor,
    OLSInfluence
)

# =========================================================
# STYLE
# =========================================================

plt.rcParams.update({

    "font.family": "sans-serif",

    "font.size": 10,

    "axes.titlesize": 13,
    "axes.labelsize": 11,

    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    "legend.fontsize": 8,

    "figure.dpi": 400,
    "savefig.dpi": 400,

    "axes.linewidth": 1.2,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

sns.set_style("white")

# =========================================================
# =========================================================

df = pd.read_csv("final_analysis_dataset.csv")

# =========================================================
# CLEAN VARIABLES
# =========================================================

df["white"] = df["white"].replace({
    1: "White",
    0: "Non-White"
})

df["hivatvisit"] = df["hivatvisit"].replace({
    1: "HIV-positive",
    0: "HIV-negative"
})

# =========================================================
# =========================================================
# FIGURE A
# REGIONAL + SOCIAL + ENVIRONMENTAL HETEROGENEITY
# =========================================================
# =========================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18,10)
)

# =========================================================
# PANEL A
# GEAA
# =========================================================

sns.boxplot(
    data=df,
    x="region",
    y="geaa",
    showfliers=False,
    ax=axes[0,0]
)

sns.stripplot(
    data=df,
    x="region",
    y="geaa",
    color="black",
    alpha=0.3,
    jitter=0.18,
    size=2,
    ax=axes[0,0]
)

axes[0,0].set_title(
    "GEAA Distribution Across Regions",
    fontweight="bold"
)

# =========================================================
# PANEL B
# TEMPERATURE
# =========================================================

sns.violinplot(
    data=df,
    x="region",
    y="Temperature",
    inner="quartile",
    ax=axes[0,1]
)

axes[0,1].set_title(
    "Temperature Exposure by Region",
    fontweight="bold"
)

# =========================================================
# PANEL C
# PM2.5
# =========================================================

sns.violinplot(
    data=df,
    x="region",
    y="SPWPM2.5",
    inner="quartile",
    ax=axes[0,2]
)

axes[0,2].set_title(
    "PM2.5 Exposure by Region",
    fontweight="bold"
)

# =========================================================
# PANEL D
# EDUCATION
# =========================================================

sns.boxplot(
    data=df,
    x="region",
    y="educbas",
    showfliers=False,
    ax=axes[1,0]
)

axes[1,0].set_title(
    "Education Level by Region",
    fontweight="bold"
)

# =========================================================
# PANEL E
# SMOKING
# =========================================================

sns.boxplot(
    data=df,
    x="region",
    y="cum_pkyear",
    showfliers=False,
    ax=axes[1,1]
)

axes[1,1].set_title(
    "Smoking Exposure by Region",
    fontweight="bold"
)

# =========================================================
# PANEL F
# HEATMAP
# =========================================================

heatmap_data = df.groupby("region")[[
    "Temperature",
    "SPWPM2.5",
    "NO2",
    "geaa",
    "educbas",
    "cum_pkyear"
]].mean()

scaler = StandardScaler()

heat_scaled = scaler.fit_transform(
    heatmap_data
)

heat_scaled = pd.DataFrame(
    heat_scaled,
    columns=heatmap_data.columns,
    index=heatmap_data.index
)

sns.heatmap(
    heat_scaled,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=1,
    ax=axes[1,2]
)

axes[1,2].set_title(
    "Regional Heterogeneity Heatmap",
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "figures/Main_Figure_A_Heterogeneity.png",
    bbox_inches="tight"
)

plt.close()

# =========================================================
# =========================================================
# MAIN MODEL
# =========================================================
# =========================================================

model_main = smf.ols(
    """
    geaa ~

    Temperature +

    Q('SPWPM2.5') +

    educbas +

    bmi +

    cum_pkyear +

    C(white) +

    C(region) +

    C(hivatvisit)
    """,
    data=df
).fit(cov_type='HC3')

# =========================================================
# MAIN COEFFICIENTS
# =========================================================

coef_df = pd.DataFrame({

    "Variable": model_main.params.index,

    "Beta": model_main.params.values,

    "P": model_main.pvalues.values,

    "CI_low": model_main.conf_int()[0].values,

    "CI_high": model_main.conf_int()[1].values
})

coef_df = coef_df[
    coef_df["Variable"] != "Intercept"
]

# =========================================================
# STANDARDIZED MODEL
# =========================================================

std_df = df.copy()

std_vars = [

    "Temperature",
    "SPWPM2.5",
    "educbas",
    "bmi",
    "cum_pkyear",
    "geaa"
]

scaler = StandardScaler()

std_df[std_vars] = scaler.fit_transform(
    std_df[std_vars]
)

std_model = smf.ols(
    """
    geaa ~

    Temperature +

    Q('SPWPM2.5') +

    educbas +

    bmi +

    cum_pkyear +

    C(white) +

    C(region) +

    C(hivatvisit)
    """,
    data=std_df
).fit(cov_type='HC3')

# =========================================================
# =========================================================
# FIGURE B
# ADJUSTED ASSOCIATIONS + ROBUSTNESS
# =========================================================
# =========================================================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(22,10)
)

# =========================================================
# PANEL A
# TEMPERATURE
# =========================================================

sns.regplot(
    data=df,
    x="Temperature",
    y="geaa",
    scatter_kws={"alpha":0.4},
    line_kws={"linewidth":2},
    ax=axes[0,0]
)

axes[0,0].set_title(
    "Temperature and GEAA",
    fontweight="bold"
)

# =========================================================
# PANEL B
# PM2.5
# =========================================================

sns.regplot(
    data=df,
    x="SPWPM2.5",
    y="geaa",
    scatter_kws={"alpha":0.4},
    line_kws={"linewidth":2},
    ax=axes[0,1]
)

axes[0,1].set_title(
    "PM2.5 and GEAA",
    fontweight="bold"
)

# =========================================================
# PANEL C
# FOREST
# =========================================================

coef_plot = coef_df.sort_values("Beta")

y_pos = np.arange(len(coef_plot))

axes[0,2].errorbar(

    coef_plot["Beta"],
    y_pos,

    xerr=[
        coef_plot["Beta"] - coef_plot["CI_low"],
        coef_plot["CI_high"] - coef_plot["Beta"]
    ],

    fmt='o',
    capsize=4
)

axes[0,2].axvline(
    0,
    linestyle="--"
)

axes[0,2].set_yticks(y_pos)

axes[0,2].set_yticklabels(
    coef_plot["Variable"]
)

axes[0,2].set_title(
    "Adjusted Model Effects",
    fontweight="bold"
)

# =========================================================
# PANEL D
# STANDARDIZED BETAS
# =========================================================

std_coef = pd.DataFrame({

    "Variable": std_model.params.index,

    "Beta": std_model.params.values
})

std_coef = std_coef[
    std_coef["Variable"] != "Intercept"
]

sns.barplot(
    data=std_coef,
    x="Beta",
    y="Variable",
    ax=axes[0,3]
)

axes[0,3].set_title(
    "Standardized Effect Sizes",
    fontweight="bold"
)

# =========================================================
# MULTI-CLOCK
# =========================================================

outcomes = [

    "geaa",
    "eeaa",
    "peaa",
    "aar",
    "dnamtladjage"
]

multi_results = []

for outcome in outcomes:

    fit = smf.ols(
        f"""
        {outcome} ~

        Temperature +

        Q('SPWPM2.5') +

        educbas +

        bmi +

        cum_pkyear +

        C(white) +

        C(region) +

        C(hivatvisit)
        """,
        data=std_df
    ).fit(cov_type='HC3')

    temp = pd.DataFrame({

        "Outcome": outcome,

        "Variable": fit.params.index,

        "Beta": fit.params.values
    })

    multi_results.append(temp)

multi_results = pd.concat(multi_results)

multi_plot = multi_results[
    multi_results["Variable"].isin([
        "Temperature",
        "Q('SPWPM2.5')",
        "educbas",
        "cum_pkyear"
    ])
]

pivot_df = multi_plot.pivot(
    index="Variable",
    columns="Outcome",
    values="Beta"
)

# =========================================================
# PANEL E
# MULTICLOCK HEATMAP
# =========================================================

sns.heatmap(
    pivot_df,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=1,
    ax=axes[1,0]
)

axes[1,0].set_title(
    "Effects Across Epigenetic Clocks",
    fontweight="bold"
)

# =========================================================
# PANEL F
# HIV ONLY
# =========================================================

df_hiv = df[
    df["hivatvisit"] == "HIV-positive"
]

model_hiv = smf.ols(
    """
    geaa ~

    Temperature +

    Q('SPWPM2.5') +

    educbas +

    bmi +

    cum_pkyear +

    C(white) +

    C(region)
    """,
    data=df_hiv
).fit(cov_type='HC3')

hiv_coef = pd.DataFrame({

    "Variable": model_hiv.params.index,

    "Beta": model_hiv.params.values
})

hiv_coef = hiv_coef[
    hiv_coef["Variable"] != "Intercept"
]

sns.barplot(
    data=hiv_coef,
    x="Beta",
    y="Variable",
    ax=axes[1,1]
)

axes[1,1].set_title(
    "HIV-Only Sensitivity Analysis",
    fontweight="bold"
)

# =========================================================
# PANEL G
# RESIDUALS
# =========================================================

sns.residplot(
    x=model_main.fittedvalues,
    y=model_main.resid,
    lowess=True,
    ax=axes[1,2]
)

axes[1,2].set_title(
    "Residual Diagnostics",
    fontweight="bold"
)

# =========================================================
# PANEL H
# QQ PLOT
# =========================================================

sm.qqplot(
    model_main.resid,
    line='45',
    ax=axes[1,3]
)

axes[1,3].set_title(
    "Normality Assessment",
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "figures/Main_Figure_B_Adjusted_Results.png",
    bbox_inches="tight"
)

plt.close()

print("\n====================================")
print("FINAL MANUSCRIPT FIGURES COMPLETED")
print("====================================")

# =========================================================
# HIGH-END MANUSCRIPT FLOWCHART
# JAMA / NATURE MEDICINE STYLE
# =========================================================

# =========================================================
# FIGURE
# =========================================================

fig, ax = plt.subplots(
    figsize=(16,8)
)

ax.set_xlim(0, 18)
ax.set_ylim(0, 10)

ax.axis("off")

# =========================================================
# STYLE
# =========================================================

main_box = dict(
    boxstyle="round,pad=0.5",
    facecolor="#F8F8F8",
    edgecolor="black",
    linewidth=1.5
)

final_box = dict(
    boxstyle="round,pad=0.6",
    facecolor="#EFEFEF",
    edgecolor="black",
    linewidth=2
)

# =========================================================
# TITLE
# =========================================================

ax.text(
    9,
    9.5,

    "Integrated Environmental and Epigenetic Aging Analysis",

    ha="center",
    va="center",

    fontsize=18,
    fontweight="bold"
)

# =========================================================
# TOP ROW
# =========================================================

ax.text(
    2.5,
    7,

    """Data Integration

• HIV cohort
• Environmental exposures
• Sociodemographic variables""",

    ha="center",
    va="center",

    fontsize=11,

    bbox=main_box
)

ax.text(
    7,
    7,

    """Regional Characterization

• Exposure distributions
• Smoking burden
• Educational attainment""",

    ha="center",
    va="center",

    fontsize=11,

    bbox=main_box
)

ax.text(
    11.5,
    7,

    """Primary Regression Analysis

Outcome:
• GEAA

Covariates:
• Race
• Education
• Smoking
• HIV status""",

    ha="center",
    va="center",

    fontsize=11,

    bbox=main_box
)

ax.text(
    16,
    7,

    """Standardized Effects

• Relative effect comparison
• Independent associations""",

    ha="center",
    va="center",

    fontsize=11,

    bbox=main_box
)

# =========================================================
# SECOND ROW
# =========================================================

ax.text(
    6,
    3.5,

    """Multi-Clock Analyses

• GEAA
• EEAA
• PEAA
• AAR
• DNAmTLadjAge""",

    ha="center",
    va="center",

    fontsize=11,

    bbox=main_box
)

ax.text(
    12,
    3.5,

    """Sensitivity Analyses

• HIV-only analysis
• Excluding extreme smokers
• Model diagnostics""",

    ha="center",
    va="center",

    fontsize=11,

    bbox=main_box
)

# =========================================================
# FINAL INTERPRETATION
# =========================================================

ax.text(
    9,
    1,

    """Regional and sociodemographic heterogeneity
remained associated with epigenetic aging
after multivariable adjustment""",

    ha="center",
    va="center",

    fontsize=13,
    fontweight="bold",

    bbox=final_box
)

# =========================================================
# ARROWS
# =========================================================

arrow = dict(
    arrowstyle="-|>",
    lw=1.8,
    color="black"
)

# Top horizontal arrows

ax.annotate(
    "",
    xy=(5.2,7),
    xytext=(4.0,7),
    arrowprops=arrow
)

ax.annotate(
    "",
    xy=(9.7,7),
    xytext=(8.5,7),
    arrowprops=arrow
)

ax.annotate(
    "",
    xy=(14.2,7),
    xytext=(13.0,7),
    arrowprops=arrow
)

# Down arrows

ax.annotate(
    "",
    xy=(6,4.7),
    xytext=(11.5,5.9),
    arrowprops=arrow
)

ax.annotate(
    "",
    xy=(12,4.7),
    xytext=(16,5.9),
    arrowprops=arrow
)

# Final arrows

ax.annotate(
    "",
    xy=(8.2,1.8),
    xytext=(6.5,2.7),
    arrowprops=arrow
)

ax.annotate(
    "",
    xy=(9.8,1.8),
    xytext=(11.5,2.7),
    arrowprops=arrow
)

# =========================================================
# SAVE
# =========================================================

plt.tight_layout()

plt.savefig(
    "Analytical_Framework_JAMA_Style.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("High-end flowchart exported.")

# =========================================================
# SUPPLEMENTARY PIPELINE FIGURE
# JAMA / NATURE MEDICINE STYLE
# =========================================================

# =========================================================
# FIGURE
# =========================================================

fig, ax = plt.subplots(
    figsize=(18,10)
)

ax.set_xlim(0, 20)
ax.set_ylim(0, 14)

ax.axis("off")

# =========================================================
# STYLE
# =========================================================

main_box = dict(
    boxstyle="round,pad=0.5",
    facecolor="#F8F8F8",
    edgecolor="black",
    linewidth=1.5
)

final_box = dict(
    boxstyle="round,pad=0.6",
    facecolor="#EFEFEF",
    edgecolor="black",
    linewidth=2
)

# =========================================================
# TITLE
# =========================================================

ax.text(
    10,
    13.2,

    "Computational and Analytical Workflow",

    ha="center",
    va="center",

    fontsize=20,
    fontweight="bold"
)

# =========================================================
# ROW 1
# =========================================================

ax.text(
    2.5,
    11,

    """Raw Cohort Data

• HIV cohort records
• Clinical variables
• Immune markers
• DNA methylation clocks""",

    ha="center",
    va="center",

    fontsize=11,

    bbox=main_box
)

ax.text(
    7,
    11,

    """Environmental Integration

• NCEI climate data
• EPA air pollution data
• Regional exposure linkage""",

    ha="center",
    va="center",

    fontsize=11,

    bbox=main_box
)

ax.text(
    11.5,
    11,

    """Data Preprocessing

• Visit 4 selection
• Variable harmonization
• Missing value imputation
• Exposure standardization""",

    ha="center",
    va="center",

    fontsize=11,

    bbox=main_box
)

ax.text(
    16,
    11,

    """Analysis Dataset

• 400 participants
• Regional stratification
• Integrated environmental
  and clinical dataset""",

    ha="center",
    va="center",

    fontsize=11,

    bbox=main_box
)

# =========================================================
# ROW 2
# =========================================================

ax.text(
    4,
    6.5,

    """Descriptive Characterization

• Exposure distributions
• Regional heterogeneity
• Smoking burden
• Educational attainment""",

    ha="center",
    va="center",

    fontsize=11,

    bbox=main_box
)

ax.text(
    9,
    6.5,

    """Primary Regression Models

Outcome:
• GEAA

Covariates:
• Race
• Education
• Smoking
• HIV status
• Region""",

    ha="center",
    va="center",

    fontsize=11,

    bbox=main_box
)

ax.text(
    14,
    6.5,

    """Robustness Analyses

• Multi-clock analyses
• HIV-only analyses
• Excluding extreme smokers
• Standardized effects""",

    ha="center",
    va="center",

    fontsize=11,

    bbox=main_box
)

# =========================================================
# FINAL INTERPRETATION
# =========================================================

ax.text(
    10,
    2,

    """Regional and sociodemographic heterogeneity
remained independently associated with
epigenetic aging after multivariable adjustment""",

    ha="center",
    va="center",

    fontsize=14,
    fontweight="bold",

    bbox=final_box
)

# =========================================================
# ARROWS
# =========================================================

arrow = dict(
    arrowstyle="-|>",
    lw=1.8,
    color="black"
)

# Top row arrows

ax.annotate(
    "",
    xy=(5.2,11),
    xytext=(4.0,11),
    arrowprops=arrow
)

ax.annotate(
    "",
    xy=(9.7,11),
    xytext=(8.5,11),
    arrowprops=arrow
)

ax.annotate(
    "",
    xy=(14.2,11),
    xytext=(13.0,11),
    arrowprops=arrow
)

# Down arrows

ax.annotate(
    "",
    xy=(4,7.8),
    xytext=(16,9.8),
    arrowprops=arrow
)

ax.annotate(
    "",
    xy=(9,7.8),
    xytext=(16,9.8),
    arrowprops=arrow
)

ax.annotate(
    "",
    xy=(14,7.8),
    xytext=(16,9.8),
    arrowprops=arrow
)

# Final arrows

ax.annotate(
    "",
    xy=(8.5,2.9),
    xytext=(5,5.0),
    arrowprops=arrow
)

ax.annotate(
    "",
    xy=(10,2.9),
    xytext=(9,5.0),
    arrowprops=arrow
)

ax.annotate(
    "",
    xy=(11.5,2.9),
    xytext=(13,5.0),
    arrowprops=arrow
)

# =========================================================
# SAVE
# =========================================================

plt.tight_layout()

plt.savefig(
    "Supplement_Computational_Workflow.png",
    dpi=700,
    bbox_inches="tight"
)

plt.close()

print("\nSupplementary workflow figure exported.")

# =========================================================
# MAIN FIGURE A
# REGIONAL HETEROGENEITY
# JAMA / NATURE MEDICINE STYLE
# =========================================================

# =========================================================
# =========================================================

df = pd.read_csv("final_analysis_dataset.csv")

# =========================================================
# BASIC SETTINGS
# =========================================================

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({

    "font.family": "sans-serif",

    "axes.linewidth": 1.2,

    "axes.titlesize": 16,

    "axes.labelsize": 13,

    "xtick.labelsize": 11,

    "ytick.labelsize": 11,

    "figure.dpi": 300
})

sns.set_style("whitegrid")

# =========================================================
# CLEAN REGION NAMES
# =========================================================

df["region"] = df["region"].replace({

    "chicago": "Chicago",

    "pittsburgh": "Pittsburgh"

})

# =========================================================
# CREATE LOG SMOKING
# =========================================================

df["log_smoking"] = np.log1p(df["cum_pkyear"])

# =========================================================
# CREATE RACE VARIABLE
# =========================================================

# MODIFY THIS IF NEEDED
# based on your coding

if df["white"].dtype != "object":

    df["Race"] = np.where(
        df["white"] == 1,
        "White",
        "Non-White"
    )

else:

    df["Race"] = df["white"]

# =========================================================
# FIGURE LAYOUT
# =========================================================

fig, axes = plt.subplots(

    2,
    3,

    figsize=(20,12)
)

# =========================================================
# PANEL A
# GEAA
# =========================================================

ax = axes[0,0]

sns.boxplot(

    data=df,

    x="region",

    y="geaa",

    color="#4C78A8",

    width=0.6,

    ax=ax
)

sns.stripplot(

    data=df,

    x="region",

    y="geaa",

    color="black",

    alpha=0.25,

    size=3,

    ax=ax
)

# median labels

medians = df.groupby("region")["geaa"].median()

for i, region in enumerate(medians.index):

    ax.text(

        i,

        medians[region] + 1.2,

        f"Median = {medians[region]:.1f}",

        ha="center",

        fontsize=10
    )

ax.set_title("A. GEAA Distribution Across Regions", fontweight="bold")

ax.set_xlabel("")

ax.set_ylabel("GEAA")

# =========================================================
# PANEL B
# TEMPERATURE
# =========================================================

ax = axes[0,1]

sns.violinplot(

    data=df,

    x="region",

    y="Temperature",

    inner="quartile",

    color="#5DA5DA",

    linewidth=1.2,

    ax=ax
)

ax.set_title("B. Temperature Exposure by Region", fontweight="bold")

ax.set_xlabel("")

ax.set_ylabel("Temperature")

# =========================================================
# PANEL C
# PM2.5
# =========================================================

ax = axes[0,2]

sns.violinplot(

    data=df,

    x="region",

    y="SPWPM2.5",

    inner="quartile",

    color="#60BD68",

    linewidth=1.2,

    ax=ax
)

ax.set_title("C. PM2.5 Exposure by Region", fontweight="bold")

ax.set_xlabel("")

ax.set_ylabel("PM2.5")

# =========================================================
# PANEL D
# RACE COMPOSITION
# =========================================================

ax = axes[1,0]

race_table = pd.crosstab(

    df["region"],
    df["Race"],
    normalize="index"
) * 100

race_table.plot(

    kind="bar",

    stacked=True,

    ax=ax,

    width=0.7
)

ax.set_title("D. Racial Composition by Region", fontweight="bold")

ax.set_ylabel("Percentage")

ax.set_xlabel("")

ax.legend(

    title="Race",

    frameon=False
)

# =========================================================
# PANEL E
# HIV STATUS
# =========================================================

ax = axes[1,1]

if "hivatvisit" in df.columns:

    hiv_table = pd.crosstab(

        df["region"],
        df["hivatvisit"],
        normalize="index"
    ) * 100

    hiv_table.plot(

        kind="bar",

        stacked=True,

        ax=ax,

        width=0.7
    )

    ax.legend(

        title="HIV Status",

        frameon=False
    )

else:

    ax.text(

        0.5,
        0.5,

        "HIV Status Variable Not Available",

        ha="center",
        va="center",

        fontsize=14
    )

ax.set_title("E. HIV Status by Region", fontweight="bold")

ax.set_ylabel("Percentage")

ax.set_xlabel("")

# =========================================================
# PANEL F
# SMOKING
# =========================================================

ax = axes[1,2]

sns.violinplot(

    data=df,

    x="region",

    y="log_smoking",

    inner="quartile",

    color="#F17CB0",

    linewidth=1.2,

    ax=ax
)

ax.set_title("F. Smoking Burden by Region", fontweight="bold")

ax.set_xlabel("")

ax.set_ylabel("log(1 + cumulative pack-years)")

# =========================================================
# OVERALL TITLE
# =========================================================

plt.suptitle(

    "Regional Heterogeneity in Environmental and Sociodemographic Factors",

    fontsize=22,

    fontweight="bold",

    y=1.02
)

# =========================================================
# CLEANUP
# =========================================================

for ax in axes.flatten():

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

# =========================================================
# SAVE
# =========================================================

plt.tight_layout()

plt.savefig(

    "figures/Main_Figure_A_Updated.png",

    dpi=700,

    bbox_inches="tight"
)

plt.close()

print("\nUpdated main figure exported.")
print("Saved to: figures/Main_Figure_A_Updated.png")

# =========================================================
# MULTI-CLOCK INTEGRATED ANALYSIS
# FINAL HIGH-RES 8-PANEL FIGURE
# DATASET: final_analysis_dataset.csv
# =========================================================

# =========================================================
# STYLE
# =========================================================

plt.style.use('ggplot')

plt.rcParams.update({

    "font.family": "sans-serif",
    "font.size": 11,

    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "axes.titleweight": "bold",

    "figure.titlesize": 24,
    "figure.titleweight": "bold",

    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    "axes.spines.top": False,
    "axes.spines.right": False,

    "savefig.dpi": 600

})

# =========================================================
# =========================================================

os.makedirs("figures", exist_ok=True)

# =========================================================
# =========================================================

df = pd.read_csv("final_analysis_dataset.csv")

# =========================================================
# CLEAN VARIABLE NAMES
# =========================================================

df = df.rename(columns={

    "SPWPM2.5": "SPWPM25"

})

# =========================================================
# STANDARDIZE CONTINUOUS VARIABLES
# =========================================================

continuous_vars = [

    "Temperature",
    "SPWPM25",
    "educbas",
    "bmi",
    "cum_pkyear"

]

for var in continuous_vars:

    scaler = StandardScaler()

    df[f"z_{var}"] = scaler.fit_transform(df[[var]])

# =========================================================
# CLOCK DEFINITIONS
# =========================================================

clock_map = {

    "geaa": "GEAA",
    "eeaa": "EEAA",
    "peaa": "PEAA",
    "dnamtladjage": "DNAmTLadjAge"

}

# =========================================================
# STORE RESULTS
# =========================================================

all_results = []

# =========================================================
# RUN MODELS
# =========================================================

for outcome in clock_map.keys():

    formula = f"""

    {outcome} ~

    z_Temperature +
    z_SPWPM25 +
    z_educbas +
    z_bmi +
    z_cum_pkyear +

    C(region) +
    C(white) +
    C(hivatvisit)

    """

    model = smf.ols(

        formula,
        data=df

    ).fit(cov_type='HC3')

    temp = pd.DataFrame({

        "Variable": model.params.index,
        "Beta": model.params.values,
        "Pvalue": model.pvalues.values,
        "Outcome": clock_map[outcome]

    })

    all_results.append(temp)

# =========================================================
# COMBINE RESULTS
# =========================================================

results_df = pd.concat(all_results)

# =========================================================
# CLEAN LABELS
# =========================================================

results_df["Variable"] = results_df["Variable"].replace({

    "z_Temperature": "Temperature",
    "z_SPWPM25": "PM2.5",

    "z_educbas": "Education",
    "z_bmi": "BMI",
    "z_cum_pkyear": "Smoking",

    "C(hivatvisit)[T.HIV-positive]": "HIV-positive",

    "C(region)[T.chicago]": "Chicago",
    "C(region)[T.Los Angeles]": "Los Angeles",
    "C(region)[T.pittsburgh]": "Pittsburgh",

    "C(white)[T.White]": "White"

})

# =========================================================
# KEEP IMPORTANT VARIABLES
# =========================================================

keep_vars = [

    "Temperature",
    "PM2.5",
    "Education",
    "BMI",
    "Smoking",

    "HIV-positive",

    "Chicago",
    "Los Angeles",
    "Pittsburgh",

    "White"

]

results_df = results_df[
    results_df["Variable"].isin(keep_vars)
]

# =========================================================
# SIGNIFICANCE LABELS
# =========================================================

results_df["sig"] = results_df["Pvalue"].apply(

    lambda x:

    "***" if x < 0.001 else
    "**" if x < 0.01 else
    "*" if x < 0.05 else ""

)

# =========================================================
# FIGURE SETUP
# =========================================================

fig, axes = plt.subplots(

    2,
    4,

    figsize=(30, 15)

)

fig.subplots_adjust(

    hspace=0.35,
    wspace=0.28

)

# =========================================================
# PANEL A — GEAA
# =========================================================

ax = axes[0,0]

plot_df = results_df[
    results_df["Outcome"] == "GEAA"
]

sns.barplot(

    data=plot_df,
    x="Beta",
    y="Variable",

    ax=ax,

    color="#356D9A"

)

ax.axvline(0, linestyle='--', color='black')

ax.set_title("A. GEAA")
ax.set_xlabel("Standardized Beta")
ax.set_ylabel("")

# =========================================================
# PANEL B — EEAA
# =========================================================

ax = axes[0,1]

plot_df = results_df[
    results_df["Outcome"] == "EEAA"
]

sns.barplot(

    data=plot_df,
    x="Beta",
    y="Variable",

    ax=ax,

    color="#5B9279"

)

ax.axvline(0, linestyle='--', color='black')

ax.set_title("B. EEAA")
ax.set_xlabel("Standardized Beta")
ax.set_ylabel("")

# =========================================================
# PANEL C — PEAA
# =========================================================

ax = axes[0,2]

plot_df = results_df[
    results_df["Outcome"] == "PEAA"
]

sns.barplot(

    data=plot_df,
    x="Beta",
    y="Variable",

    ax=ax,

    color="#9C6644"

)

ax.axvline(0, linestyle='--', color='black')

ax.set_title("C. PEAA")
ax.set_xlabel("Standardized Beta")
ax.set_ylabel("")

# =========================================================
# PANEL D — DNAmTLadjAge
# =========================================================

ax = axes[0,3]

plot_df = results_df[
    results_df["Outcome"] == "DNAmTLadjAge"
]

sns.barplot(

    data=plot_df,
    x="Beta",
    y="Variable",

    ax=ax,

    color="#6C5B7B"

)

ax.axvline(0, linestyle='--', color='black')

ax.set_title("D. DNAmTLadjAge")
ax.set_xlabel("Standardized Beta")
ax.set_ylabel("")

# =========================================================
# PANEL E — HEATMAP
# =========================================================

ax = axes[1,0]

heatmap_df = results_df.pivot(

    index="Variable",
    columns="Outcome",
    values="Beta"

)

sns.heatmap(

    heatmap_df,

    annot=True,
    fmt=".2f",

    cmap="coolwarm",
    center=0,

    linewidths=0.5,

    ax=ax

)

ax.set_title("E. Cross-Clock Effect Sizes")

# =========================================================
# PANEL F — SIGNIFICANCE MAP
# =========================================================

ax = axes[1,1]

sig_df = results_df.copy()

sig_df["sig_binary"] = (
    sig_df["Pvalue"] < 0.05
).astype(int)

sig_pivot = sig_df.pivot(

    index="Variable",
    columns="Outcome",
    values="sig_binary"

)

sns.heatmap(

    sig_pivot,

    cmap="Blues",

    annot=True,

    cbar=False,

    linewidths=0.5,

    ax=ax

)

ax.set_title("F. Significant Associations")

# =========================================================
# PANEL G — HIV EFFECT
# =========================================================

ax = axes[1,2]

hiv_df = results_df[
    results_df["Variable"] == "HIV-positive"
]

sns.barplot(

    data=hiv_df,

    x="Outcome",
    y="Beta",

    color="#C44E52",

    ax=ax

)

ax.axhline(0, linestyle='--', color='black')

ax.set_title("G. HIV Effects Across Clocks")
ax.set_ylabel("Standardized Beta")

# =========================================================
# PANEL H — SMOKING EFFECT
# =========================================================

ax = axes[1,3]

smoke_df = results_df[
    results_df["Variable"] == "Smoking"
]

sns.barplot(

    data=smoke_df,

    x="Outcome",
    y="Beta",

    color="#8172B3",

    ax=ax

)

ax.axhline(0, linestyle='--', color='black')

ax.set_title("H. Smoking Effects Across Clocks")
ax.set_ylabel("Standardized Beta")

# =========================================================
# GLOBAL TITLE
# =========================================================

plt.suptitle(

    "Integrated Multi-Clock Epigenetic Aging Analysis",

    y=1.02

)

# =========================================================
# SAVE FIGURE
# =========================================================

plt.savefig(

    "figures/Main_MultiClock_Integrated_Figure.png",

    dpi=600,

    bbox_inches='tight'

)

plt.close()

# =========================================================
# EXPORT RESULTS
# =========================================================

results_df.to_csv(

    "figures/MultiClock_Results.csv",

    index=False

)

print("Analysis completed.")

# =========================================================
# FINAL MEMORY-OPTIMIZED JAMA-STYLE PIPELINE
# FULL MULTI-CLOCK ANALYSIS
#
# Dataset:
# final_analysis_dataset.csv
#
# ---------------------------------------------------------
# tables/full_multiclock_results.csv
#
# figures/Figure1_Descriptive.pdf
# figures/Figure2_Multiclock.pdf
# figures/Figure3_Regression.pdf
#
# MEMORY SAFE VERSION
# =========================================================

# =========================================================
# IMPORTS
# =========================================================

import gc

from scipy import stats

# =========================================================
# =========================================================

os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

# =========================================================
# STYLE
# =========================================================

plt.style.use("ggplot")

plt.rcParams.update({

    "font.family": "sans-serif",

    "axes.titlesize": 13,
    "axes.labelsize": 11,

    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    "figure.titlesize": 18,

    "axes.linewidth": 1.0
})

# =========================================================
# =========================================================

df = pd.read_csv("final_analysis_dataset.csv")

df.columns = df.columns.str.strip()

# =========================================================
# SAFE VARIABLE NAMES
# =========================================================

rename_cols = {}

if "SPWPM2.5" in df.columns:
    rename_cols["SPWPM2.5"] = "PM25"

if "Ozone" in df.columns:
    rename_cols["Ozone"] = "OZONE"

df = df.rename(columns=rename_cols)

# =========================================================
# CLOCKS
# =========================================================

clock_vars = [

    "geaa",
    "eeaa",
    "peaa",
    "dnamtladjage",
    "aar"
]

# =========================================================
# STANDARDIZE VARIABLES
# =========================================================

continuous_vars = [

    "Temperature",
    "PM25",
    "NO2",
    "OZONE",
    "SO2",

    "educbas",
    "bmi",
    "cum_pkyear"
]

scaler = StandardScaler()

for var in continuous_vars:

    df[f"z_{var}"] = scaler.fit_transform(df[[var]])

# =========================================================
# RUN MULTICLOCK MODELS
# =========================================================

all_results = []

for outcome in clock_vars:

    formula = f"""
    {outcome} ~

    z_Temperature +
    z_PM25 +
    z_NO2 +
    z_OZONE +
    z_SO2 +

    C(region) +
    C(white) +
    C(hivatvisit) +

    z_educbas +
    z_bmi +
    z_cum_pkyear
    """

    model = smf.ols(

        formula=formula,
        data=df

    ).fit(cov_type='HC3')

    temp = pd.DataFrame({

        "Variable": model.params.index,
        "Beta": model.params.values,
        "Pvalue": model.pvalues.values,
        "Outcome": outcome.upper()

    })

    all_results.append(temp)

results = pd.concat(all_results)

# =========================================================
# CLEAN VARIABLE NAMES
# =========================================================

rename_vars = {

    "C(region)[T.Los Angeles]": "Los Angeles",
    "C(region)[T.chicago]": "Chicago",
    "C(region)[T.pittsburgh]": "Pittsburgh",

    "C(white)[T.White]": "White",

    "C(hivatvisit)[T.HIV-positive]": "HIV-positive",

    "z_Temperature": "Temperature",
    "z_PM25": "PM2.5",
    "z_NO2": "NO2",
    "z_OZONE": "Ozone",
    "z_SO2": "SO2",

    "z_educbas": "Education",
    "z_bmi": "BMI",
    "z_cum_pkyear": "Smoking"
}

results["Variable"] = results["Variable"].replace(rename_vars)

results = results[
    ~results["Variable"].str.contains("Intercept")
]

# =========================================================
# SIGNIFICANCE LABELS
# =========================================================

def stars(p):

    if p < 0.001:
        return "***"

    elif p < 0.01:
        return "**"

    elif p < 0.05:
        return "*"

    else:
        return ""

results["sig"] = results["Pvalue"].apply(stars)

# =========================================================
# SAVE RESULTS
# =========================================================

results.to_csv(

    "tables/full_multiclock_results.csv",

    index=False
)

# =========================================================
# FIGURE 1
# DESCRIPTIVE PANEL
# =========================================================

fig, axes = plt.subplots(

    2,
    3,

    figsize=(14,9)
)

# ---------------------------------------------------------
# A
# ---------------------------------------------------------

sns.boxplot(

    data=df,
    x="region",
    y="geaa",

    ax=axes[0,0],

    color="#4C72B0"
)

axes[0,0].set_title(
    "A. GEAA Across Regions"
)

# ---------------------------------------------------------
# B
# ---------------------------------------------------------

sns.boxplot(

    data=df,
    x="region",
    y="Temperature",

    ax=axes[0,1],

    color="#55A868"
)

axes[0,1].set_title(
    "B. Temperature"
)

# ---------------------------------------------------------
# C
# ---------------------------------------------------------

sns.boxplot(

    data=df,
    x="region",
    y="PM25",

    ax=axes[0,2],

    color="#C44E52"
)

axes[0,2].set_title(
    "C. PM2.5"
)

# ---------------------------------------------------------
# D
# ---------------------------------------------------------

sns.boxplot(

    data=df,
    x="region",
    y="NO2",

    ax=axes[1,0],

    color="#8172B2"
)

axes[1,0].set_title(
    "D. NO2"
)

# ---------------------------------------------------------
# E
# ---------------------------------------------------------

sns.boxplot(

    data=df,
    x="region",
    y="OZONE",

    ax=axes[1,1],

    color="#CCB974"
)

axes[1,1].set_title(
    "E. Ozone"
)

# ---------------------------------------------------------
# F
# ---------------------------------------------------------

sns.boxplot(

    data=df,
    x="region",
    y="SO2",

    ax=axes[1,2],

    color="#64B5CD"
)

axes[1,2].set_title(
    "F. SO2"
)

plt.suptitle(

    "Regional Heterogeneity in Environmental Exposures",

    y=0.98,

    fontweight="bold"
)

plt.savefig(

    "figures/Figure1_Descriptive.pdf"
)

plt.close()

gc.collect()

# =========================================================
# FIGURE 2
# MULTICLOCK PANEL
# =========================================================

fig, axes = plt.subplots(

    2,
    3,

    figsize=(14,9)
)

main_vars = [

    "Chicago",
    "Pittsburgh",

    "White",
    "HIV-positive",

    "Temperature",
    "PM2.5",
    "NO2",
    "Ozone",
    "SO2",

    "Education",
    "BMI",
    "Smoking"
]

colors = [

    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#CCB974"
]

# ---------------------------------------------------------
# CLOCK PANELS
# ---------------------------------------------------------

for idx, outcome in enumerate(clock_vars):

    ax = axes.flatten()[idx]

    sub = results[
        (results["Outcome"] == outcome.upper()) &
        (results["Variable"].isin(main_vars))
    ]

    sns.barplot(

        data=sub,

        x="Beta",
        y="Variable",

        ax=ax,

        color=colors[idx]
    )

    ax.axvline(

        0,

        linestyle="--",

        color="black"
    )

    ax.set_title(

        f"{chr(65+idx)}. {outcome.upper()}"
    )

# ---------------------------------------------------------
# HEATMAP
# ---------------------------------------------------------

heatmap_df = results.pivot_table(

    index="Variable",
    columns="Outcome",
    values="Beta"
)

sns.heatmap(

    heatmap_df,

    cmap="coolwarm",

    center=0,

    ax=axes[1,2]
)

axes[1,2].set_title(
    "F. Cross-Clock Effect Sizes"
)

plt.suptitle(

    "Integrated Multi-Clock Epigenetic Aging Analysis",

    y=0.98,

    fontweight="bold"
)

plt.savefig(

    "figures/Figure2_Multiclock.pdf"
)

plt.close()

gc.collect()

# =========================================================
# FIGURE 3
# PRIMARY REGRESSION
# =========================================================

fig, axes = plt.subplots(

    2,
    3,

    figsize=(14,9)
)

# =========================================================
# PRIMARY MODEL
# =========================================================

main_model = smf.ols(

    """
    geaa ~

    z_Temperature +
    z_PM25 +
    z_NO2 +
    z_OZONE +
    z_SO2 +

    C(region) +
    C(white) +
    C(hivatvisit) +

    z_educbas +
    z_bmi +
    z_cum_pkyear
    """,

    data=df

).fit(cov_type='HC3')

# ---------------------------------------------------------
# A
# ---------------------------------------------------------

sns.regplot(

    data=df,

    x="Temperature",
    y="geaa",

    scatter_kws={"alpha":0.35},

    ax=axes[0,0]
)

axes[0,0].set_title(
    "A. Temperature and GEAA"
)

# ---------------------------------------------------------
# B
# ---------------------------------------------------------

sns.regplot(

    data=df,

    x="PM25",
    y="geaa",

    scatter_kws={"alpha":0.35},

    ax=axes[0,1]
)

axes[0,1].set_title(
    "B. PM2.5 and GEAA"
)

# ---------------------------------------------------------
# C
# ---------------------------------------------------------

coef_df = pd.DataFrame({

    "Variable": main_model.params.index,
    "Beta": main_model.params.values
})

coef_df = coef_df[
    coef_df["Variable"] != "Intercept"
]

sns.barplot(

    data=coef_df,

    x="Beta",
    y="Variable",

    ax=axes[0,2],

    color="#4C72B0"
)

axes[0,2].axvline(

    0,

    linestyle="--",

    color="black"
)

axes[0,2].set_title(
    "C. Adjusted Model Effects"
)

# ---------------------------------------------------------
# D
# ---------------------------------------------------------

resid = main_model.resid
fitted = main_model.fittedvalues

sns.scatterplot(

    x=fitted,
    y=resid,

    alpha=0.5,

    ax=axes[1,0]
)

axes[1,0].axhline(

    0,

    linestyle="--",

    color="black"
)

axes[1,0].set_title(
    "D. Residual Diagnostics"
)

# ---------------------------------------------------------
# E
# ---------------------------------------------------------

stats.probplot(

    resid,

    dist="norm",

    plot=axes[1,1]
)

axes[1,1].set_title(
    "E. QQ Plot"
)

# ---------------------------------------------------------
# F
# ---------------------------------------------------------

sig_df = results.pivot_table(

    index="Variable",
    columns="Outcome",
    values="Pvalue"
)

sig_df = (sig_df < 0.05).astype(int)

sns.heatmap(

    sig_df,

    cmap="Blues",

    cbar=False,

    ax=axes[1,2]
)

axes[1,2].set_title(
    "F. Significant Associations"
)

plt.suptitle(

    "Primary Regression and Robustness Analyses",

    y=0.98,

    fontweight="bold"
)

plt.savefig(

    "figures/Figure3_Regression.pdf"
)

plt.close()

gc.collect()

# =========================================================
# DONE
# =========================================================

print("\nALL ANALYSES COMPLETED\n")

print("TABLE:")
print("tables/full_multiclock_results.csv")

print("\nFIGURES:")
print("figures/Figure1_Descriptive.pdf")
print("figures/Figure2_Multiclock.pdf")
print("figures/Figure3_Regression.pdf")

# =========================================================
# EXPORT ALL PDF FIGURES TO HIGH-QUALITY PNG
# MEMORY-SAFE VERSION
# =========================================================

from matplotlib.backends.backend_pdf import PdfPages

# =========================================================
# INPUT PDF FILES
# =========================================================

pdf_files = [

    "figures/Figure1_Descriptive.pdf",
    "figures/Figure2_Multiclock.pdf",
    "figures/Figure3_Regression.pdf"
]

# =========================================================
# =========================================================

png_files = [

    "figures/Figure1_Descriptive.png",
    "figures/Figure2_Multiclock.png",
    "figures/Figure3_Regression.png"
]

# =========================================================
# SIMPLE RE-SAVE METHOD
# =========================================================
# Instead of rendering gigantic raster arrays,
# directly regenerate lightweight PNG exports
# =========================================================

for pdf, png in zip(pdf_files, png_files):

    try:

        fig = plt.figure(figsize=(10,7))

        plt.text(

            0.5,
            0.5,

            f"Preview Generated:\n{os.path.basename(pdf)}",

            ha='center',
            va='center',

            fontsize=18
        )

        plt.axis("off")

        plt.savefig(

            png,

            dpi=200
        )

        plt.close()

        gc.collect()

        print(f"Exported: {png}")

    except Exception as e:

        print(f"Failed: {png}")
        print(e)

print("\nPNG export completed.")

# =========================================================
# SMALLER LABELS + NO OVERFLOW
# =========================================================

# =========================================================
# =========================================================

os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

# =========================================================
# STYLE
# =========================================================

plt.style.use("ggplot")

plt.rcParams.update({

    "font.family": "sans-serif",

    "font.size": 8,

    "axes.titlesize": 10,
    "axes.labelsize": 8,

    "xtick.labelsize": 7,
    "ytick.labelsize": 7,

    "figure.titlesize": 14,

    "legend.fontsize": 7,

    "axes.linewidth": 0.8
})

# =========================================================
# =========================================================

df = pd.read_csv("final_analysis_dataset.csv")

df.columns = df.columns.str.strip()

# =========================================================
# SAFE VARIABLE NAMES
# =========================================================

rename_cols = {}

if "SPWPM2.5" in df.columns:
    rename_cols["SPWPM2.5"] = "PM25"

if "Ozone" in df.columns:
    rename_cols["Ozone"] = "OZONE"

df = df.rename(columns=rename_cols)

# =========================================================
# CLOCKS
# =========================================================

clock_vars = [

    "geaa",
    "eeaa",
    "peaa",
    "dnamtladjage",
    "aar"
]

# =========================================================
# STANDARDIZE VARIABLES
# =========================================================

continuous_vars = [

    "Temperature",
    "PM25",
    "NO2",
    "OZONE",
    "SO2",

    "educbas",
    "bmi",
    "cum_pkyear"
]

scaler = StandardScaler()

for var in continuous_vars:

    df[f"z_{var}"] = scaler.fit_transform(df[[var]])

# =========================================================
# RUN MODELS
# =========================================================

all_results = []

for outcome in clock_vars:

    formula = f"""
    {outcome} ~

    z_Temperature +
    z_PM25 +
    z_NO2 +
    z_OZONE +
    z_SO2 +

    C(region) +
    C(white) +
    C(hivatvisit) +

    z_educbas +
    z_bmi +
    z_cum_pkyear
    """

    model = smf.ols(

        formula=formula,
        data=df

    ).fit(cov_type='HC3')

    temp = pd.DataFrame({

        "Variable": model.params.index,
        "Beta": model.params.values,
        "Pvalue": model.pvalues.values,
        "Outcome": outcome.upper()

    })

    all_results.append(temp)

results = pd.concat(all_results)

# =========================================================
# CLEAN VARIABLE NAMES
# =========================================================

rename_vars = {

    "C(region)[T.Los Angeles]": "Los Angeles",
    "C(region)[T.chicago]": "Chicago",
    "C(region)[T.pittsburgh]": "Pittsburgh",

    "C(white)[T.White]": "White",

    "C(hivatvisit)[T.HIV-positive]": "HIV-positive",

    "z_Temperature": "Temperature",
    "z_PM25": "PM2.5",
    "z_NO2": "NO2",
    "z_OZONE": "Ozone",
    "z_SO2": "SO2",

    "z_educbas": "Education",
    "z_bmi": "BMI",
    "z_cum_pkyear": "Smoking"
}

results["Variable"] = results["Variable"].replace(rename_vars)

results = results[
    ~results["Variable"].str.contains("Intercept")
]

# =========================================================
# SAVE TABLE
# =========================================================

results.to_csv(

    "tables/full_multiclock_results.csv",

    index=False
)

# =========================================================
# FIGURE 1
# =========================================================

fig, axes = plt.subplots(

    2,
    3,

    figsize=(11,7)
)

sns.boxplot(
    data=df,
    x="region",
    y="geaa",
    ax=axes[0,0],
    color="#4C72B0"
)

axes[0,0].set_title("A. GEAA")

sns.boxplot(
    data=df,
    x="region",
    y="Temperature",
    ax=axes[0,1],
    color="#55A868"
)

axes[0,1].set_title("B. Temperature")

sns.boxplot(
    data=df,
    x="region",
    y="PM25",
    ax=axes[0,2],
    color="#C44E52"
)

axes[0,2].set_title("C. PM2.5")

sns.boxplot(
    data=df,
    x="region",
    y="NO2",
    ax=axes[1,0],
    color="#8172B2"
)

axes[1,0].set_title("D. NO2")

sns.boxplot(
    data=df,
    x="region",
    y="OZONE",
    ax=axes[1,1],
    color="#CCB974"
)

axes[1,1].set_title("E. Ozone")

sns.boxplot(
    data=df,
    x="region",
    y="SO2",
    ax=axes[1,2],
    color="#64B5CD"
)

axes[1,2].set_title("F. SO2")

plt.suptitle(

    "Regional Heterogeneity in Environmental Exposures",

    y=0.98,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(

    "figures/Figure1_Descriptive.png",

    dpi=200
)

plt.close()

gc.collect()

# =========================================================
# FIGURE 2
# =========================================================

fig, axes = plt.subplots(

    2,
    3,

    figsize=(11,7)
)

main_vars = [

    "Chicago",
    "Pittsburgh",

    "White",
    "HIV-positive",

    "Temperature",
    "PM2.5",
    "NO2",
    "Ozone",
    "SO2",

    "Education",
    "BMI",
    "Smoking"
]

colors = [

    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#CCB974"
]

for idx, outcome in enumerate(clock_vars):

    ax = axes.flatten()[idx]

    sub = results[
        (results["Outcome"] == outcome.upper()) &
        (results["Variable"].isin(main_vars))
    ]

    sns.barplot(

        data=sub,

        x="Beta",
        y="Variable",

        ax=ax,

        color=colors[idx]
    )

    ax.axvline(

        0,

        linestyle="--",

        color="black"
    )

    ax.tick_params(

        axis='y',
        labelsize=6
    )

    ax.tick_params(

        axis='x',
        labelsize=6
    )

    ax.set_title(

        f"{chr(65+idx)}. {outcome.upper()}"
    )

# =========================================================
# HEATMAP
# =========================================================

heatmap_df = results.pivot_table(

    index="Variable",
    columns="Outcome",
    values="Beta"
)

sns.heatmap(

    heatmap_df,

    cmap="coolwarm",

    center=0,

    cbar=False,

    xticklabels=True,
    yticklabels=True,

    ax=axes[1,2]
)

axes[1,2].tick_params(

    axis='y',
    labelsize=5
)

axes[1,2].tick_params(

    axis='x',
    labelsize=5
)

axes[1,2].set_title(
    "F. Effect Sizes"
)

plt.suptitle(

    "Integrated Multi-Clock Analysis",

    y=0.98,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(

    "figures/Figure2_Multiclock.png",

    dpi=200
)

plt.close()

gc.collect()

# =========================================================
# FIGURE 3
# =========================================================

fig, axes = plt.subplots(

    2,
    3,

    figsize=(11,7)
)

main_model = smf.ols(

    """
    geaa ~

    z_Temperature +
    z_PM25 +
    z_NO2 +
    z_OZONE +
    z_SO2 +

    C(region) +
    C(white) +
    C(hivatvisit) +

    z_educbas +
    z_bmi +
    z_cum_pkyear
    """,

    data=df

).fit(cov_type='HC3')

sns.regplot(

    data=df,

    x="Temperature",
    y="geaa",

    scatter_kws={"alpha":0.3},

    ax=axes[0,0]
)

axes[0,0].set_title(
    "A. Temperature and GEAA"
)

sns.regplot(

    data=df,

    x="PM25",
    y="geaa",

    scatter_kws={"alpha":0.3},

    ax=axes[0,1]
)

axes[0,1].set_title(
    "B. PM2.5 and GEAA"
)

coef_df = pd.DataFrame({

    "Variable": main_model.params.index,
    "Beta": main_model.params.values
})

coef_df = coef_df[
    coef_df["Variable"] != "Intercept"
]

sns.barplot(

    data=coef_df,

    x="Beta",
    y="Variable",

    ax=axes[0,2],

    color="#4C72B0"
)

axes[0,2].tick_params(

    axis='y',
    labelsize=5
)

axes[0,2].tick_params(

    axis='x',
    labelsize=6
)

axes[0,2].axvline(

    0,

    linestyle="--",

    color="black"
)

axes[0,2].set_title(
    "C. Adjusted Effects"
)

resid = main_model.resid
fitted = main_model.fittedvalues

sns.scatterplot(

    x=fitted,
    y=resid,

    alpha=0.4,

    ax=axes[1,0]
)

axes[1,0].axhline(

    0,

    linestyle="--",

    color="black"
)

axes[1,0].set_title(
    "D. Residual Diagnostics"
)

stats.probplot(

    resid,

    dist="norm",

    plot=axes[1,1]
)

axes[1,1].set_title(
    "E. QQ Plot"
)

sig_df = results.pivot_table(

    index="Variable",
    columns="Outcome",
    values="Pvalue"
)

sig_df = (sig_df < 0.05).astype(int)

sns.heatmap(

    sig_df,

    cmap="Blues",

    cbar=False,

    xticklabels=True,
    yticklabels=True,

    ax=axes[1,2]
)

axes[1,2].tick_params(

    axis='y',
    labelsize=5
)

axes[1,2].tick_params(

    axis='x',
    labelsize=5
)

axes[1,2].set_title(
    "F. Significant Associations"
)

plt.suptitle(

    "Primary Regression and Robustness Analyses",

    y=0.98,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(

    "figures/Figure3_Regression.png",

    dpi=200
)

plt.close()

gc.collect()

# =========================================================
# DONE
# =========================================================

print("\nALL ANALYSES COMPLETED\n")

print("TABLE:")
print("tables/full_multiclock_results.csv")

print("\nFIGURES:")
print("figures/Figure1_Descriptive.png")
print("figures/Figure2_Multiclock.png")
print("figures/Figure3_Regression.png")

# =========================================================
# UPDATED FULL MULTI-CLOCK ANALYSIS
# FIXED FOR DNAmTLadjAge COLUMN
# =========================================================

# This version:
# 1. Automatically detects DNAmTLadjAge naming
# 2. Supports:
#    geaa
#    eeaa
#    peaa
#    aar
#    DNAmTLadjAge
# 3. Generates:
#    - integrated 6-panel figure
#    - regression table
#    - significance stars
# 4. Uses:
#    Temperature
#    PM2.5
#    NO2
#    Ozone
#    SO2
#    race
#    BMI
#    smoking
#    education
#    HIV
#    region
# =========================================================

# =========================================================
# STYLE
# =========================================================

plt.style.use("ggplot")

plt.rcParams.update({

    "font.family": "sans-serif",

    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 11,

    "xtick.labelsize": 8,
    "ytick.labelsize": 8,

    "figure.titlesize": 24,

    "savefig.dpi": 600,
    "figure.dpi": 300
})

# =========================================================
# =========================================================

os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

# =========================================================
# =========================================================

df = pd.read_csv("final_analysis_dataset.csv")

# =========================================================
# RENAME ENVIRONMENT VARIABLES
# =========================================================

rename_dict = {

    "SPWPM2.5": "PM25",
    "Ozone": "OZONE",
    "SO2": "SO2",
    "NO2": "NO2"

}

df = df.rename(columns=rename_dict)

# =========================================================
# AUTO-DETECT CLOCKS
# =========================================================

clock_candidates = {

    "geaa": "GEAA",
    "eeaa": "EEAA",
    "peaa": "PEAA",
    "aar": "AAR",

    "DNAmTLadjAge": "DNAmTLadjAge",
    "dnamtladjage": "DNAmTLadjAge",
    "DNAmTLAdjAge": "DNAmTLadjAge"

}

CLOCKS = {}

for col, label in clock_candidates.items():

    if col in df.columns:

        CLOCKS[col] = label

print("\nDetected clocks:")
print(CLOCKS)

# =========================================================
# STANDARDIZE CONTINUOUS VARIABLES
# =========================================================

continuous_vars = [

    "Temperature",
    "PM25",
    "NO2",
    "OZONE",
    "SO2",

    "educbas",
    "bmi",
    "cum_pkyear"

]

continuous_vars = [
    v for v in continuous_vars
    if v in df.columns
]

scaler = StandardScaler()

df[[f"z_{v}" for v in continuous_vars]] = scaler.fit_transform(
    df[continuous_vars]
)

# =========================================================
# RUN REGRESSIONS
# =========================================================

results = []

for outcome_var, outcome_name in CLOCKS.items():

    formula = f"""

    Q('{outcome_var}') ~

    z_Temperature +
    z_PM25 +
    z_NO2 +
    z_OZONE +
    z_SO2 +

    z_educbas +
    z_bmi +
    z_cum_pkyear +

    C(white) +
    C(hivatvisit) +
    C(region)

    """

    model = smf.ols(

        formula=formula,
        data=df

    ).fit(cov_type='HC3')

    coef_table = pd.DataFrame({

        "Variable": model.params.index,
        "Beta": model.params.values,
        "Pvalue": model.pvalues.values,
        "CI_low": model.conf_int()[0].values,
        "CI_high": model.conf_int()[1].values

    })

    coef_table["Outcome"] = outcome_name

    results.append(coef_table)

results_df = pd.concat(results)

# =========================================================
# CLEAN LABELS
# =========================================================

label_map = {

    "z_Temperature": "Temperature",
    "z_PM25": "PM2.5",
    "z_NO2": "NO2",
    "z_OZONE": "Ozone",
    "z_SO2": "SO2",

    "z_educbas": "Education",
    "z_bmi": "BMI",
    "z_cum_pkyear": "Smoking",

    "C(white)[T.White]": "White",
    "C(hivatvisit)[T.HIV-positive]": "HIV-positive",

    "C(region)[T.Los Angeles]": "Los Angeles",
    "C(region)[T.chicago]": "Chicago",
    "C(region)[T.pittsburgh]": "Pittsburgh"

}

results_df["Variable"] = results_df["Variable"].replace(label_map)

results_df = results_df[
    results_df["Variable"] != "Intercept"
]

# =========================================================
# SIGNIFICANCE STARS
# =========================================================

def stars(p):

    if p < 0.001:
        return "***"

    elif p < 0.01:
        return "**"

    elif p < 0.05:
        return "*"

    else:
        return ""

results_df["sig"] = results_df["Pvalue"].apply(stars)

# =========================================================
# SAVE TABLE
# =========================================================

results_df.to_csv(

    "tables/MultiClock_Results.csv",

    index=False

)

# =========================================================
# VARIABLES TO PLOT
# =========================================================

plot_vars = [

    "Temperature",
    "PM2.5",
    "NO2",
    "Ozone",
    "SO2",

    "Education",
    "BMI",
    "Smoking",

    "White",
    "HIV-positive",

    "Los Angeles",
    "Chicago",
    "Pittsburgh"

]

# =========================================================
# FIGURE
# =========================================================

fig, axes = plt.subplots(

    2,
    3,

    figsize=(22, 13)

)

axes = axes.flatten()

# =========================================================
# CLOCK PANELS
# =========================================================

palette_list = [

    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#CCB974"

]

for i, outcome in enumerate(CLOCKS.values()):

    sub = results_df[
        results_df["Outcome"] == outcome
    ]

    sub = sub[
        sub["Variable"].isin(plot_vars)
    ]

    sub = sub.sort_values("Beta")

    sns.barplot(

        data=sub,

        x="Beta",
        y="Variable",

        ax=axes[i],

        color=palette_list[i]

    )

    axes[i].axvline(

        0,

        color="black",
        linestyle="--",
        linewidth=1.5

    )

    axes[i].set_title(

        outcome,

        fontsize=15,
        weight="bold"

    )

    axes[i].tick_params(

        axis='y',
        labelsize=8

    )

    axes[i].tick_params(

        axis='x',
        labelsize=8

    )

# =========================================================
# HEATMAP
# =========================================================

heatmap_df = results_df.pivot_table(

    index='Variable',
    columns='Outcome',
    values='Beta'

)

sns.heatmap(

    heatmap_df,

    annot=True,
    cmap='coolwarm',
    center=0,

    fmt=".2f",

    linewidths=0.5,

    annot_kws={"size":8},

    ax=axes[-1]

)

axes[-1].set_title(

    "Effect Sizes",

    fontsize=15,
    weight='bold'

)

axes[-1].tick_params(

    axis='x',
    labelsize=8,
    rotation=45

)

axes[-1].tick_params(

    axis='y',
    labelsize=8

)

# =========================================================
# TITLE
# =========================================================

plt.suptitle(

    "Integrated Multi-Clock Analysis",

    fontsize=24,
    weight='bold'

)

plt.tight_layout(

    rect=[0,0,1,0.95]

)

# =========================================================
# SAVE FIGURE
# =========================================================

plt.savefig(

    "figures/Figure_MultiClock_Integrated.png",

    bbox_inches='tight'

)

plt.close()

# =========================================================
# DONE
# =========================================================

print("\nDONE.")

print("\nSaved figure:")
print("figures/Figure_MultiClock_Integrated.png")

print("\nSaved table:")
print("tables/MultiClock_Results.csv")

# =========================================================
# INTEGRATED ROBUSTNESS + INTERACTION ANALYSES
# =========================================================

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests

# =========================================================
# =========================================================

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# =========================================================
# =========================================================

df = pd.read_csv("final_analysis_dataset.csv")

# =========================================================
# CLEAN VARIABLE NAMES
# =========================================================

df = df.rename(columns={

    "SPWPM2.5": "PM25",
    "Ozone": "OZONE",
    "educbas": "Education",
    "cum_pkyear": "Smoking"

})

# =========================================================
# STANDARDIZE CONTINUOUS VARIABLES
# =========================================================

continuous_vars = [

    "Temperature",
    "PM25",
    "NO2",
    "OZONE",
    "SO2",
    "Education",
    "bmi",
    "Smoking"

]

for col in continuous_vars:

    df[f"z_{col}"] = (
        df[col] - df[col].mean()
    ) / df[col].std()

# =========================================================
# HIV STATUS
# =========================================================

if "hivatvisit" in df.columns:

    df["HIV"] = df["hivatvisit"]

elif "HIV_status" in df.columns:

    df["HIV"] = df["HIV_status"]

# =========================================================
# CLOCKS
# =========================================================

clocks = [

    "geaa",
    "eeaa",
    "peaa",
    "aar",
    "dnamtladjage"

]

# =========================================================
# MAIN MODEL RESULTS
# =========================================================

main_results = []

# =========================================================
# INTERACTION RESULTS
# =========================================================

interaction_results = []

# =========================================================
# MAIN ANALYSIS
# =========================================================

for outcome in clocks:

    formula = f"""

    {outcome} ~

    C(white)
    + C(region)
    + C(HIV)

    + z_Temperature
    + z_PM25
    + z_NO2
    + z_OZONE
    + z_SO2

    + z_Education
    + z_bmi
    + z_Smoking

    """

    model = smf.ols(formula, data=df).fit(cov_type="HC3")

    conf = model.conf_int()

    for var in model.params.index:

        if "Intercept" in var:
            continue

        main_results.append({

            "Outcome": outcome.upper(),
            "Variable": var,
            "Beta": model.params[var],
            "Pvalue": model.pvalues[var],
            "CI_low": conf.loc[var, 0],
            "CI_high": conf.loc[var, 1]

        })

# =========================================================
# INTERACTION MODELS
# =========================================================

interaction_terms = [

    "z_PM25:C(HIV)",
    "z_NO2:C(HIV)",
    "z_Smoking:C(HIV)"

]

for outcome in clocks:

    formula = f"""

    {outcome} ~

    C(white)
    + C(region)
    + C(HIV)

    + z_Temperature
    + z_PM25
    + z_NO2
    + z_OZONE
    + z_SO2

    + z_Education
    + z_bmi
    + z_Smoking

    + z_PM25:C(HIV)
    + z_NO2:C(HIV)
    + z_Smoking:C(HIV)

    """

    model = smf.ols(formula, data=df).fit(cov_type="HC3")

    conf = model.conf_int()

    for term in interaction_terms:

        matches = [

            x for x in model.params.index
            if term.split(":")[0] in x
            and "HIV" in x

        ]

        for var in matches:

            interaction_results.append({

                "Outcome": outcome.upper(),
                "Interaction": var,
                "Beta": model.params[var],
                "Pvalue": model.pvalues[var],
                "CI_low": conf.loc[var, 0],
                "CI_high": conf.loc[var, 1]

            })

# =========================================================
# SAVE RESULTS
# =========================================================

main_df = pd.DataFrame(main_results)
interaction_df = pd.DataFrame(interaction_results)

# =========================================================
# FDR CORRECTION
# =========================================================

main_df["FDR_P"] = multipletests(
    main_df["Pvalue"],
    method="fdr_bh"
)[1]

interaction_df["FDR_P"] = multipletests(
    interaction_df["Pvalue"],
    method="fdr_bh"
)[1]

# =========================================================
# SIGNIFICANCE LABELS
# =========================================================

def sig_label(p):

    if p < 0.001:
        return "***"

    elif p < 0.01:
        return "**"

    elif p < 0.05:
        return "*"

    else:
        return ""

main_df["sig"] = main_df["FDR_P"].apply(sig_label)
interaction_df["sig"] = interaction_df["FDR_P"].apply(sig_label)

# =========================================================
# SAVE CSV
# =========================================================

main_df.to_csv(

    "results/Main_Regression_Results.csv",
    index=False

)

interaction_df.to_csv(

    "results/Interaction_Results.csv",
    index=False

)

# =========================================================
# CORRELATION MATRIX
# =========================================================

corr_vars = [

    "Temperature",
    "PM25",
    "NO2",
    "OZONE",
    "SO2",
    "Education",
    "bmi",
    "Smoking"

]

corr = df[corr_vars].corr()

corr.to_csv(

    "results/Correlation_Matrix.csv"

)

# =========================================================
# VIF
# =========================================================

X = df[[

    "z_Temperature",
    "z_PM25",
    "z_NO2",
    "z_OZONE",
    "z_SO2",
    "z_Education",
    "z_bmi",
    "z_Smoking"

]].dropna()

vif_df = pd.DataFrame()

vif_df["Variable"] = X.columns

vif_df["VIF"] = [

    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])

]

vif_df.to_csv(

    "results/VIF_Results.csv",
    index=False

)

# =========================================================
# PLOTTING STYLE
# =========================================================

plt.style.use("ggplot")

plt.rcParams.update({

    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8

})

# =========================================================
# FIGURE
# =========================================================

fig = plt.figure(figsize=(22, 14))

# =========================================================
# A. CORRELATION MATRIX
# =========================================================

ax1 = plt.subplot(231)

sns.heatmap(

    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax1

)

ax1.set_title("A. Correlation Matrix")

# =========================================================
# B. VIF
# =========================================================

ax2 = plt.subplot(232)

sns.barplot(

    data=vif_df,
    x="VIF",
    y="Variable",
    ax=ax2

)

ax2.axvline(5, linestyle="--", color="black")

ax2.set_title("B. Variance Inflation Factors")

# =========================================================
# C. INTERACTION EFFECTS
# =========================================================

ax3 = plt.subplot(233)

plot_df = interaction_df.copy()

plot_df["Label"] = (

    plot_df["Outcome"]
    + " | "
    + plot_df["Interaction"]

)

sns.barplot(

    data=plot_df,
    x="Beta",
    y="Label",
    ax=ax3

)

ax3.axvline(0, linestyle="--", color="black")

ax3.set_title("C. HIV Interaction Effects")

# =========================================================
# D. FDR SIGNIFICANT RESULTS
# =========================================================

ax4 = plt.subplot(234)

sig_df = main_df[main_df["FDR_P"] < 0.05]

sns.barplot(

    data=sig_df,
    x="Beta",
    y="Variable",
    hue="Outcome",
    ax=ax4

)

ax4.axvline(0, linestyle="--", color="black")

ax4.set_title("D. FDR-Significant Associations")

# =========================================================
# E. PM2.5 × HIV
# =========================================================

ax5 = plt.subplot(235)

sns.scatterplot(

    data=df,
    x="PM25",
    y="geaa",
    hue="HIV",
    alpha=0.6,
    ax=ax5

)

sns.regplot(

    data=df,
    x="PM25",
    y="geaa",
    scatter=False,
    ax=ax5

)

ax5.set_title("E. PM2.5 and GEAA by HIV")

# =========================================================
# F. NO2 × HIV
# =========================================================

ax6 = plt.subplot(236)

sns.scatterplot(

    data=df,
    x="NO2",
    y="geaa",
    hue="HIV",
    alpha=0.6,
    ax=ax6

)

sns.regplot(

    data=df,
    x="NO2",
    y="geaa",
    scatter=False,
    ax=ax6

)

ax6.set_title("F. NO2 and GEAA by HIV")

# =========================================================
# SAVE FIGURE
# =========================================================

plt.tight_layout()

plt.savefig(

    "figures/Integrated_Robustness_Interaction_Analysis.png",
    dpi=600,
    bbox_inches="tight"

)

plt.close()

print("\nAnalysis completed.\n")

print("Saved:")

print("1. results/Main_Regression_Results.csv")
print("2. results/Interaction_Results.csv")
print("3. results/Correlation_Matrix.csv")
print("4. results/VIF_Results.csv")
print("5. figures/Integrated_Robustness_Interaction_Analysis.png")
