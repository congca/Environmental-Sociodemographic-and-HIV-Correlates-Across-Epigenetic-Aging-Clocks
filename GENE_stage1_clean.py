import numpy
import scipy
import statsmodels

print(numpy.__version__)
print(scipy.__version__)
print(statsmodels.__version__)
import pandas as pd

df = pd.read_csv('final_analysis_dataset.csv')

# 查看数据的基本信息
print(df.info())
print(df.describe())
print(df.isnull().sum())  # 检查缺失值
# =========================================================
# 02_analysis_figures.py
# =========================================================

# =========================================================
# LOAD LIBRARIES
# =========================================================

import os

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf

# =========================================================
# =========================================================

os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

# =========================================================
# GLOBAL FIGURE STYLE
# =========================================================

plt.rcParams.update({

    "font.family": "sans-serif",

    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,

    "xtick.labelsize": 11,
    "ytick.labelsize": 11,

    "legend.fontsize": 11,

    "figure.dpi": 300,
    "savefig.dpi": 300,

    "axes.linewidth": 1.2,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

sns.set_style("white")

# =========================================================
# LOAD CLEAN DATASET
# =========================================================

df = pd.read_csv("final_analysis_dataset.csv")

print("\nDataset Loaded")
print(df.shape)

# =========================================================
# CLEAN TYPES
# =========================================================

df["region"] = df["region"].astype(str)

# =========================================================
# MAIN OUTCOME
# =========================================================

OUTCOME = "geaa"

# =========================================================
# TABLE 1
# REGIONAL SUMMARY
# =========================================================

summary_table = df.groupby("region")[[
    "Temperature",
    "SPWPM2.5",
    "NO2",
    "Ozone",
    "SO2",
    "geaa",
    "eeaa",
    "educbas",
    "bmi",
    "cum_pkyear"
]].agg([
    "mean",
    "std",
    "median"
])

summary_table.to_csv(
    "tables/Table1_Regional_Summary.csv"
)

print("\nTable 1 exported")

# =========================================================
# MAIN ADJUSTED MODEL
# =========================================================

model = smf.ols(
    """
    geaa ~
    Temperature * C(region) +
    Q('SPWPM2.5') +
    NO2 +
    Ozone +
    SO2 +
    educbas +
    C(white) +
    bmi +
    cum_pkyear +
    C(hivatvisit)
    """,
    data=df
).fit()

print(model.summary())

# =========================================================
# EXPORT MODEL RESULTS
# =========================================================

results_df = pd.DataFrame({

    "Variable": model.params.index,
    "Beta": model.params.values,
    "Pvalue": model.pvalues.values,
    "CI_lower": model.conf_int()[0].values,
    "CI_upper": model.conf_int()[1].values

})

results_df.to_csv(
    "tables/Adjusted_Model_Results.csv",
    index=False
)

print("\nAdjusted model exported")

# =========================================================
# FIGURE 1
# GEAA DISTRIBUTION BY REGION
# =========================================================

fig, ax = plt.subplots(figsize=(8,6))

sns.boxplot(
    data=df,
    x="region",
    y=OUTCOME,
    width=0.6,
    showcaps=True,
    showfliers=False,
    linewidth=1.5,
    ax=ax
)

sns.stripplot(
    data=df,
    x="region",
    y=OUTCOME,
    color="black",
    alpha=0.35,
    jitter=0.18,
    size=4,
    ax=ax
)

ax.set_title(
    "Regional Differences in GEAA",
    fontweight="bold",
    pad=15
)

ax.set_xlabel("")
ax.set_ylabel("GEAA")

sns.despine()

plt.tight_layout()

plt.savefig(
    "figures/Figure1_GEAA_by_Region.png",
    bbox_inches="tight"
)

plt.close()

print("Figure 1 exported")

# =========================================================
# FIGURE 2
# TEMPERATURE vs GEAA
# =========================================================

g = sns.lmplot(
    data=df,
    x="Temperature",
    y=OUTCOME,
    hue="region",
    height=6,
    aspect=1.3,

    scatter_kws={
        "alpha":0.5,
        "s":50
    },

    line_kws={
        "linewidth":2
    }
)

g.fig.subplots_adjust(top=0.90)

g.fig.suptitle(
    "Temperature and GEAA Across Regions",
    fontsize=16,
    fontweight="bold"
)

plt.savefig(
    "figures/Figure2_Temperature_GEAA.png",
    bbox_inches="tight"
)

plt.close()

print("Figure 2 exported")

# =========================================================
# REGION-SPECIFIC MODELS
# =========================================================

regions = sorted(df["region"].unique())

region_effects = []

for region in regions:

    temp_df = df[df["region"] == region]

    fit = smf.ols(
        """
        geaa ~
        Temperature +
        educbas +
        bmi +
        cum_pkyear +
        C(white) +
        C(hivatvisit)
        """,
        data=temp_df
    ).fit()

    beta = fit.params["Temperature"]
    pval = fit.pvalues["Temperature"]

    ci_low = fit.conf_int().loc["Temperature",0]
    ci_high = fit.conf_int().loc["Temperature",1]

    region_effects.append({

        "Region": region,
        "Beta": beta,
        "CI_low": ci_low,
        "CI_high": ci_high,
        "Pvalue": pval
    })

forest_df = pd.DataFrame(region_effects)

forest_df.to_csv(
    "tables/Region_Specific_Effects.csv",
    index=False
)

print("\nRegion-specific results exported")

# =========================================================
# FIGURE 3
# FOREST PLOT
# =========================================================

fig, ax = plt.subplots(figsize=(7,5))

y_pos = np.arange(len(forest_df))

ax.errorbar(
    forest_df["Beta"],
    y_pos,

    xerr=[
        forest_df["Beta"] - forest_df["CI_low"],
        forest_df["CI_high"] - forest_df["Beta"]
    ],

    fmt='o',
    capsize=4,
    linewidth=2
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1.2
)

ax.set_yticks(y_pos)
ax.set_yticklabels(forest_df["Region"])

ax.set_xlabel("Beta Coefficient for Temperature")

ax.set_title(
    "Region-Specific Temperature Effects on GEAA",
    fontweight="bold",
    pad=15
)

sns.despine()

plt.tight_layout()

plt.savefig(
    "figures/Figure3_ForestPlot.png",
    bbox_inches="tight"
)

plt.close()

print("Figure 3 exported")

# =========================================================
# FIGURE 4
# =========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14,10)
)

# =========================================================
# PANEL A
# =========================================================

sns.boxplot(
    data=df,
    x="region",
    y=OUTCOME,
    showfliers=False,
    ax=axes[0,0]
)

axes[0,0].set_title("A. GEAA by Region")

# =========================================================
# PANEL B
# =========================================================

sns.scatterplot(
    data=df,
    x="Temperature",
    y=OUTCOME,
    hue="region",
    alpha=0.6,
    ax=axes[0,1]
)

axes[0,1].set_title("B. Temperature and GEAA")

# =========================================================
# PANEL C
# =========================================================

sns.scatterplot(
    data=df,
    x="SPWPM2.5",
    y=OUTCOME,
    hue="region",
    alpha=0.6,
    ax=axes[1,0]
)

axes[1,0].set_title("C. PM2.5 and GEAA")

# =========================================================
# PANEL D
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
    ax=axes[1,1],
    cbar_kws={"shrink":0.8}
)

axes[1,1].set_title("D. Correlation Matrix")

# =========================================================
# CLEANUP
# =========================================================

for ax in axes.flatten():

    ax.grid(False)

plt.tight_layout()

plt.savefig(
    "figures/Figure4_Multipanel.png",
    bbox_inches="tight"
)

plt.close()

print("Figure 4 exported")

# =========================================================
# SUPPLEMENTARY FIGURE
# =========================================================

fig, ax = plt.subplots(figsize=(8,6))

sns.boxplot(
    data=df,
    x="region",
    y="eeaa",
    showfliers=False,
    ax=ax
)

sns.stripplot(
    data=df,
    x="region",
    y="eeaa",
    color="black",
    alpha=0.3,
    jitter=0.15,
    ax=ax
)

ax.set_title(
    "Supplementary Figure: EEAA by Region",
    fontweight="bold"
)

sns.despine()

plt.tight_layout()

plt.savefig(
    "figures/Supplementary_EEAA.png",
    bbox_inches="tight"
)

plt.close()

print("Supplementary figure exported")

# =========================================================
# CORRELATION MATRIX EXPORT
# =========================================================

corr.to_csv(
    "tables/Correlation_Matrix.csv"
)

# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n======================================")
print("ANALYSIS PIPELINE COMPLETED")
print("======================================")

print("""

FIGURES
--------
figures/Figure1_GEAA_by_Region.png
figures/Figure2_Temperature_GEAA.png
figures/Figure3_ForestPlot.png
figures/Figure4_Multipanel.png
figures/Supplementary_EEAA.png

TABLES
--------
tables/Table1_Regional_Summary.csv
tables/Adjusted_Model_Results.csv
tables/Region_Specific_Effects.csv
tables/Correlation_Matrix.csv

""")
# =========================================================
# INTERACTION MODEL
# =========================================================

model_interaction = smf.ols(
    """
    geaa ~
    Temperature * C(region) +
    Q('SPWPM2.5') +
    educbas +
    bmi +
    cum_pkyear +
    C(white) +
    C(hivatvisit)
    """,
    data=df
).fit()

print(model_interaction.summary())
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
).fit()

print(model_main.summary())
# =========================================================
# =========================================================

from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.preprocessing import StandardScaler

# =========================================================
# STANDARDIZED VARIABLES
# =========================================================

standardize_vars = [
    "Temperature",
    "SPWPM2.5",
    "educbas",
    "bmi",
    "cum_pkyear",
    "geaa"
]

scaler = StandardScaler()

df_std = df.copy()

df_std[standardize_vars] = scaler.fit_transform(
    df_std[standardize_vars]
)

# =========================================================
# STANDARDIZED MODEL
# =========================================================

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
    data=df_std
).fit()

std_results = pd.DataFrame({

    "Variable": std_model.params.index,
    "Standardized_Beta": std_model.params.values,
    "Pvalue": std_model.pvalues.values

})

std_results.to_csv(
    "tables/Standardized_Betas.csv",
    index=False
)

# =========================================================
# FDR CORRECTION
# =========================================================

results_df["FDR_Pvalue"] = multipletests(
    results_df["Pvalue"],
    method="fdr_bh"
)[1]

results_df.to_csv(
    "tables/Main_Model_Results_FDR.csv",
    index=False
)

# =========================================================
# ADJUSTED PREDICTION GRID
# =========================================================

temp_seq = np.linspace(
    df["Temperature"].min(),
    df["Temperature"].max(),
    100
)

prediction_frames = []

for region in df["region"].unique():

    pred_df = pd.DataFrame({

        "Temperature": temp_seq,

        "SPWPM2.5": df["SPWPM2.5"].median(),
        "educbas": df["educbas"].median(),
        "bmi": df["bmi"].median(),
        "cum_pkyear": df["cum_pkyear"].median(),

        "white": "White",
        "region": region,
        "hivatvisit": "HIV-positive"
    })

    pred_df["Predicted_GEAA"] = model_interaction.predict(pred_df)

    prediction_frames.append(pred_df)

predictions = pd.concat(prediction_frames)

# =========================================================
# FIGURE
# ADJUSTED MARGINAL EFFECTS
# =========================================================

plt.figure(figsize=(8,6))

sns.lineplot(
    data=predictions,
    x="Temperature",
    y="Predicted_GEAA",
    hue="region",
    linewidth=3
)

plt.title(
    "Adjusted Association Between Temperature and GEAA"
)

plt.xlabel("Temperature")
plt.ylabel("Predicted GEAA")

sns.despine()

plt.tight_layout()

plt.savefig(
    "figures/Figure5_Adjusted_Marginal_Effects.png"
)

plt.close()

# =========================================================
# INFLUENCE DIAGNOSTICS
# =========================================================

influence = OLSInfluence(model_main)

cooks = influence.cooks_distance[0]

plt.figure(figsize=(8,5))

plt.stem(
    np.arange(len(cooks)),
    cooks
)

plt.title("Cook's Distance")

plt.xlabel("Observation")
plt.ylabel("Cook's Distance")

plt.tight_layout()

plt.savefig(
    "figures/Supplement_CooksDistance.png"
)

plt.close()

# =========================================================
# HIGH INFLUENCE OBSERVATIONS
# =========================================================

high_influence = np.where(
    cooks > (4 / len(df))
)[0]

pd.DataFrame({

    "Observation": high_influence,
    "CooksDistance": cooks[high_influence]

}).to_csv(
    "tables/High_Influence_Observations.csv",
    index=False
)

# =========================================================
# SENSITIVITY ANALYSIS
# REMOVE EXTREME SMOKERS
# =========================================================

df_sensitivity = df[
    df["cum_pkyear"] < df["cum_pkyear"].quantile(0.95)
]

sens_model = smf.ols(
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
    data=df_sensitivity
).fit()

sens_results = pd.DataFrame({

    "Variable": sens_model.params.index,
    "Beta": sens_model.params.values,
    "Pvalue": sens_model.pvalues.values

})

sens_results.to_csv(
    "tables/Sensitivity_NoExtremeSmokers.csv",
    index=False
)

print("\nAdvanced analyses completed.")
# =========================================================
# CONTINUATION ANALYSIS PIPELINE
# Robustness + Heterogeneity + Diagnostics
# =========================================================

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import (
    variance_inflation_factor,
    OLSInfluence
)

# =========================================================
# ROBUST INTERACTION MODEL
# =========================================================

model_interaction = smf.ols(
    """
    geaa ~
    Temperature * C(region) +
    Q('SPWPM2.5') +
    educbas +
    bmi +
    cum_pkyear +
    C(white) +
    C(hivatvisit)
    """,
    data=df
).fit(cov_type='HC3')

print(model_interaction.summary())

# =========================================================
# EXPORT INTERACTION RESULTS
# =========================================================

interaction_results = pd.DataFrame({

    "Variable": model_interaction.params.index,
    "Beta": model_interaction.params.values,
    "Pvalue": model_interaction.pvalues.values,
    "CI_lower": model_interaction.conf_int()[0].values,
    "CI_upper": model_interaction.conf_int()[1].values

})

# =========================================================
# FDR CORRECTION
# =========================================================

interaction_results["FDR_Pvalue"] = multipletests(
    interaction_results["Pvalue"],
    method="fdr_bh"
)[1]

interaction_results.to_csv(
    "tables/Interaction_Model_Results.csv",
    index=False
)

# =========================================================
# ADJUSTED PREDICTIONS
# =========================================================

temp_seq = np.linspace(
    df["Temperature"].min(),
    df["Temperature"].max(),
    100
)

prediction_frames = []

for region in df["region"].unique():

    pred_df = pd.DataFrame({

        "Temperature": temp_seq,

        "SPWPM2.5": df["SPWPM2.5"].median(),
        "educbas": df["educbas"].median(),
        "bmi": df["bmi"].median(),
        "cum_pkyear": df["cum_pkyear"].median(),

        "white": "White",
        "region": region,
        "hivatvisit": "HIV-positive"
    })

    pred_df["Predicted_GEAA"] = model_interaction.predict(pred_df)

    prediction_frames.append(pred_df)

predictions = pd.concat(prediction_frames)

# =========================================================
# FIGURE
# ADJUSTED MARGINAL EFFECTS
# =========================================================

plt.figure(figsize=(8,6))

sns.lineplot(
    data=predictions,
    x="Temperature",
    y="Predicted_GEAA",
    hue="region",
    linewidth=3
)

plt.title(
    "Adjusted Association Between Temperature and GEAA",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Temperature")
plt.ylabel("Predicted GEAA")

sns.despine()

plt.tight_layout()

plt.savefig(
    "figures/Figure5_Adjusted_Marginal_Effects.png",
    dpi=300
)

plt.close()

# =========================================================
# REGION-SPECIFIC MODELS
# =========================================================

regions = sorted(df["region"].unique())

region_effects = []

for region in regions:

    temp_df = df[
        df["region"] == region
    ]

    fit = smf.ols(
        """
        geaa ~
        Temperature +
        educbas +
        bmi +
        cum_pkyear +
        C(white) +
        C(hivatvisit)
        """,
        data=temp_df
    ).fit(cov_type='HC3')

    beta = fit.params["Temperature"]

    ci_low = fit.conf_int().loc["Temperature",0]
    ci_high = fit.conf_int().loc["Temperature",1]

    pval = fit.pvalues["Temperature"]

    region_effects.append({

        "Region": region,
        "Beta": beta,
        "CI_low": ci_low,
        "CI_high": ci_high,
        "Pvalue": pval
    })

forest_df = pd.DataFrame(region_effects)

forest_df.to_csv(
    "tables/ForestPlot_Results.csv",
    index=False
)

# =========================================================
# FOREST PLOT
# =========================================================

fig, ax = plt.subplots(figsize=(7,5))

y_pos = np.arange(len(forest_df))

ax.errorbar(

    forest_df["Beta"],
    y_pos,

    xerr=[
        forest_df["Beta"] - forest_df["CI_low"],
        forest_df["CI_high"] - forest_df["Beta"]
    ],

    fmt='o',
    capsize=4,
    linewidth=2
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1.2
)

ax.set_yticks(y_pos)
ax.set_yticklabels(forest_df["Region"])

ax.set_xlabel("Temperature Effect on GEAA")

ax.set_title(
    "Region-Specific Temperature Associations",
    fontsize=15,
    fontweight="bold"
)

sns.despine()

plt.tight_layout()

plt.savefig(
    "figures/Figure6_ForestPlot.png",
    dpi=300
)

plt.close()

# =========================================================
# VIF ANALYSIS
# =========================================================

X = df[[
    "Temperature",
    "SPWPM2.5",
    "educbas",
    "bmi",
    "cum_pkyear"
]]

X = sm.add_constant(X)

vif_df = pd.DataFrame()

vif_df["Variable"] = X.columns

vif_df["VIF"] = [

    variance_inflation_factor(
        X.values,
        i
    )

    for i in range(X.shape[1])

]

print(vif_df)

vif_df.to_csv(
    "tables/VIF_Table.csv",
    index=False
)

# =========================================================
# STANDARDIZED BETAS
# =========================================================

standardize_vars = [
    "Temperature",
    "SPWPM2.5",
    "educbas",
    "bmi",
    "cum_pkyear",
    "geaa"
]

scaler = StandardScaler()

df_std = df.copy()

df_std[standardize_vars] = scaler.fit_transform(
    df_std[standardize_vars]
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
    data=df_std
).fit(cov_type='HC3')

std_results = pd.DataFrame({

    "Variable": std_model.params.index,
    "Standardized_Beta": std_model.params.values,
    "Pvalue": std_model.pvalues.values

})

std_results.to_csv(
    "tables/Standardized_Betas.csv",
    index=False
)

# =========================================================
# RESIDUAL DIAGNOSTICS
# =========================================================

plt.figure(figsize=(7,5))

sns.residplot(
    x=model_interaction.fittedvalues,
    y=model_interaction.resid,
    lowess=True
)

plt.xlabel("Fitted Values")
plt.ylabel("Residuals")

plt.title(
    "Residual Diagnostics"
)

plt.tight_layout()

plt.savefig(
    "figures/Supplement_Residuals.png",
    dpi=300
)

plt.close()

# =========================================================
# QQ PLOT
# =========================================================

fig = sm.qqplot(
    model_interaction.resid,
    line='45'
)

plt.title("QQ Plot")

plt.savefig(
    "figures/Supplement_QQPlot.png",
    dpi=300
)

plt.close()

# =========================================================
# COOK'S DISTANCE
# =========================================================

influence = OLSInfluence(model_interaction)

cooks = influence.cooks_distance[0]

plt.figure(figsize=(8,5))

plt.stem(
    np.arange(len(cooks)),
    cooks
)

plt.title("Cook's Distance")

plt.xlabel("Observation")
plt.ylabel("Cook's Distance")

plt.tight_layout()

plt.savefig(
    "figures/Supplement_CooksDistance.png",
    dpi=300
)

plt.close()

# =========================================================
# HIV-ONLY SENSITIVITY ANALYSIS
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

hiv_results = pd.DataFrame({

    "Variable": model_hiv.params.index,
    "Beta": model_hiv.params.values,
    "Pvalue": model_hiv.pvalues.values

})

hiv_results.to_csv(
    "tables/HIV_Only_Results.csv",
    index=False
)

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

sens_results = pd.DataFrame({

    "Variable": model_sens.params.index,
    "Beta": model_sens.params.values,
    "Pvalue": model_sens.pvalues.values

})

sens_results.to_csv(
    "tables/Sensitivity_NoExtremeSmokers.csv",
    index=False
)

print("\n===================================")
print("CONTINUATION ANALYSES COMPLETE")
print("===================================")
# =========================================================
# REGIONAL HETEROGENEITY HEATMAP
# =========================================================

# =========================================================
# CREATE REGIONAL SUMMARY DATA
# =========================================================

heatmap_df = pd.DataFrame({

    "Region": [
        "Baltimore",
        "Los Angeles",
        "Chicago",
        "Pittsburgh"
    ],

    "Temperature": [
        17.88,
        21.15,
        14.63,
        14.80
    ],

    "PM2.5": [
        12.59,
        12.16,
        13.69,
        12.61
    ],

    "NO2": [
        35.64,
        50.64,
        39.43,
        26.86
    ],

    "GEAA": [
        -1.53,
        -0.33,
        1.41,
        0.37
    ],

    "Education": [
        5.24,
        4.79,
        5.22,
        4.37
    ],

    "Smoking": [
        13.65,
        10.07,
        11.15,
        15.21
    ]
})

# =========================================================
# STANDARDIZE VALUES
# =========================================================

features = [
    "Temperature",
    "PM2.5",
    "NO2",
    "GEAA",
    "Education",
    "Smoking"
]

scaler = StandardScaler()

scaled_values = scaler.fit_transform(
    heatmap_df[features]
)

scaled_df = pd.DataFrame(
    scaled_values,
    columns=features
)

scaled_df.index = heatmap_df["Region"]

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(8,5))

sns.heatmap(
    scaled_df,

    cmap="coolwarm",

    center=0,

    linewidths=1,

    annot=True,
    fmt=".2f",

    cbar_kws={
        "label":"Standardized Value"
    }
)

plt.title(
    "Regional Heterogeneity in Environmental and Aging Profiles",
    fontsize=15,
    fontweight="bold",
    pad=15
)

plt.ylabel("")
plt.xlabel("")

plt.tight_layout()

plt.savefig(
    "Figure_Regional_Heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Heatmap exported.")
# =========================================================
# MULTI-CLOCK REGRESSION ANALYSIS
# =========================================================

# =========================================================
# =========================================================

df = pd.read_csv("cleaned_visit4_imputed.csv")

# =========================================================
# CLEAN VARIABLES
# =========================================================

df["white"] = df["white"].map({
    1: "White",
    0: "Non-White"
})

df["hivatvisit"] = df["hivatvisit"].map({
    1: "HIV-positive",
    0: "HIV-negative"
})

# =========================================================
# OUTCOMES
# =========================================================

outcomes = [
    "geaa",
    "eeaa",
    "peaa",
    "dnamtladjage",
    "aar"
]

# =========================================================
# RESULTS STORAGE
# =========================================================

results = []

# =========================================================
# LOOP THROUGH CLOCKS
# =========================================================

for outcome in outcomes:

    formula = f"""
    {outcome} ~
    Temperature +
    Q('SPWPM2.5') +
    C(region) +
    C(white) +
    educbas +
    bmi +
    cum_pkyear +
    C(hivatvisit)
    """

    model = smf.ols(
        formula=formula,
        data=df
    ).fit()

    temp = pd.DataFrame({

        "Outcome": outcome,

        "Variable": model.params.index,

        "Beta": model.params.values,

        "Pvalue": model.pvalues.values,

        "CI_lower": model.conf_int()[0].values,

        "CI_upper": model.conf_int()[1].values,

        "R_squared": model.rsquared
    })

    results.append(temp)

# =========================================================
# COMBINE
# =========================================================

results_df = pd.concat(results)

# =========================================================
# FDR CORRECTION
# =========================================================

results_df["FDR_p"] = multipletests(
    results_df["Pvalue"],
    method="fdr_bh"
)[1]

# =========================================================
# ROUND
# =========================================================

results_df = results_df.round(4)

# =========================================================
# SAVE
# =========================================================

results_df.to_csv(
    "MultiClock_Regression_Results.csv",
    index=False
)

print(results_df.head())

print("\nResults exported.")

# 可视化因变量的分布
plt.figure(figsize=(10, 5))
sns.histplot(df['aar'], bins=30, kde=True)
plt.title('Distribution of AAR')
plt.xlabel('AAR')
plt.ylabel('Frequency')
plt.show()

# 可视化因变量的分布
plt.figure(figsize=(10, 5))
sns.histplot(df['eeaa'], bins=30, kde=True)
plt.title('Distribution of EEAA')
plt.xlabel('EEAA')
plt.ylabel('Frequency')
plt.show()

# 可视化因变量的分布
plt.figure(figsize=(10, 5))
sns.histplot(df['peaa'], bins=30, kde=True)
plt.title('Distribution of peaa')
plt.xlabel('peaa')
plt.ylabel('Frequency')
plt.show()

# 可视化因变量的分布
plt.figure(figsize=(10, 5))
sns.histplot(df['geaa'], bins=30, kde=True)
plt.title('Distribution of geaa')
plt.xlabel('geaa')
plt.ylabel('Frequency')
plt.show()

# 可视化因变量的分布
plt.figure(figsize=(10, 5))
sns.histplot(df['dnamtladjage'], bins=30, kde=True)
plt.title('Distribution of dnamtladjage')
plt.xlabel('dnamtladjage')
plt.ylabel('Frequency')
plt.show()
import scipy.stats as stats

# 定义因变量列表
dependent_vars = ['aar', 'eeaa', 'peaa', 'geaa', 'dnamtladjage']

# 创建子图
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

# 绘制散点图 + 拟合线 + p值 + 图例
for i, var in enumerate(dependent_vars):
    x = df['Ozone']
    y = df[var]

    # 绘制散点
    axes[i].scatter(x, y, alpha=0.6, label='Data')

    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * x + intercept
    axes[i].plot(x, line, color='red', label=f'Fit line')

    # 添加标题和标签
    axes[i].set_title(f'Ozone vs {var}')
    axes[i].set_xlabel('Ozone')
    axes[i].set_ylabel(var)
    axes[i].grid(True)

    # 添加 p 值注释
    axes[i].text(0.05, 0.95, f'p = {p_value:.3e}', transform=axes[i].transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

    # 添加图例
    axes[i].legend()

# 最后一个子图保留为 Ozone 分布或留空
sns.histplot(df['Ozone'], bins=30, kde=True, ax=axes[-1], color='red', alpha=0.5)
axes[-1].set_title('Ozone Distribution')
axes[-1].set_xlabel('Ozone')
axes[-1].set_ylabel('Frequency')
axes[-1].grid(True)

plt.tight_layout()
plt.show()

# 设置Seaborn风格
sns.set(style="whitegrid")

# 定义因变量列表
dependent_vars = ['aar', 'eeaa', 'peaa', 'geaa', 'dnamtladjage']

# 创建子图
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

# 绘制每个因变量与 Temperature 的散点图 + 拟合线 + p 值
for i, var in enumerate(dependent_vars):
    x = df['Temperature']
    y = df[var]

    # Seaborn散点图
    sns.scatterplot(x=x, y=y, ax=axes[i], alpha=0.6, label='Data')

    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * x + intercept
    axes[i].plot(x, line, color='red', label='Fit line')

    # 设置标题与轴
    axes[i].set_title(f'Temperature vs {var}')
    axes[i].set_xlabel('Temperature')
    axes[i].set_ylabel(var)
    axes[i].grid(True)

    # 添加 p 值注释
    axes[i].text(0.05, 0.95, f'p = {p_value:.3e}', transform=axes[i].transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

    # 添加图例
    axes[i].legend()

# 最后一幅图：Temperature 分布图
sns.histplot(df['Temperature'], bins=30, kde=True, ax=axes[-1], color='red', alpha=0.5)
axes[-1].set_title('Temperature Distribution')
axes[-1].set_xlabel('Temperature')
axes[-1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('Temperature_Plot.png', dpi=300)
plt.show()

from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 替换地理位置
location_mapping = {
    0: 'Baltimore',
    0.333333333: 'Chicago',
    1: 'Pittsburgh',
    0.666666667: 'LA'
}
df[df.columns[0]] = df[df.columns[0]].replace(location_mapping)
group_col = df.columns[0]

# 设置风格
sns.set(style="whitegrid")

# 因变量
dependent_vars = ['aar', 'eeaa', 'peaa', 'geaa', 'dnamtladjage']

# 创建图
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(18, 22))
axes = axes.flatten()

# 遍历变量画图 + 添加p值 + Tukey分组差异
for i, var in enumerate(dependent_vars):
    ax_box = axes[i]
    ax_violin = axes[i + len(dependent_vars)]

    # 1. 箱线图 + 均值 ± SE
    sns.boxplot(data=df, x=group_col, y=var, ax=ax_box, palette="Set2", showfliers=False)
    means = df.groupby(group_col)[var].mean()
    sems = df.groupby(group_col)[var].sem()
    x_pos = range(len(means))
    ax_box.errorbar(x=x_pos, y=means, yerr=sems, fmt='o', color='black', capsize=5, label='Mean ± SE')

    # 2. ANOVA 总体 p 值
    model = ols(f'{var} ~ C({group_col})', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    pval = anova_table['PR(>F)'][0]
    ax_box.text(0.05, 0.95, f'ANOVA p = {pval:.3e}', transform=ax_box.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))
    ax_box.set_title(f'{var} (Boxplot)')
    ax_box.set_xlabel('Geographic Location')
    ax_box.set_ylabel(var)
    ax_box.legend()
    ax_box.grid(True)

    # 3. Violin plot + 同样标注
    sns.violinplot(data=df, x=group_col, y=var, ax=ax_violin, palette="Set2", inner="quartile")
    ax_violin.text(0.05, 0.95, f'ANOVA p = {pval:.3e}', transform=ax_violin.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))
    ax_violin.set_title(f'{var} (Violin)')
    ax_violin.set_xlabel('Geographic Location')
    ax_violin.set_ylabel(var)
    ax_violin.grid(True)

    # 4. 可选：Tukey HSD 显著性分组结果打印
    tukey = pairwise_tukeyhsd(endog=df[var], groups=df[group_col], alpha=0.05)
    print(f'\nTukey HSD results for {var}:\n{tukey.summary()}')

plt.tight_layout()
plt.savefig('Location_with_Pvalues.png', dpi=300)
plt.show()
from scipy import stats

# 假设 df 是你的数据框
# 计算 95th 百分位数
precipitation_threshold = df['Precipitation'].quantile(0.95)
temperature_threshold = df['Temperature'].quantile(0.95)

# 添加极端天气标签
df['Extreme_Precipitation'] = df['Precipitation'] > precipitation_threshold
df['Extreme_Temperature'] = df['Temperature'] > temperature_threshold

# 因变量列表
dependent_vars = ['aar', 'eeaa', 'peaa', 'geaa', 'dnamtladjage']

# 创建图
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(18, 22))
axes = axes.flatten()

# 可视化极端降水影响
for i, var in enumerate(dependent_vars):
    ax = axes[i]
    sns.boxplot(data=df, x='Extreme_Precipitation', y=var, ax=ax, palette="Set2", showfliers=False)

    # 计算每组均值 ± 标准误
    means = df.groupby('Extreme_Precipitation')[var].mean()
    sems = df.groupby('Extreme_Precipitation')[var].sem()
    x_pos = [0, 1]
    ax.errorbar(x=x_pos, y=means, yerr=sems, fmt='o', color='black', capsize=5, label='Mean ± SE')

    # t检验
    group1 = df[df['Extreme_Precipitation'] == False][var]
    group2 = df[df['Extreme_Precipitation'] == True][var]
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
    ax.text(0.05, 0.95, f't-test p = {p_val:.3e}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

    ax.set_title(f'Extreme Precipitation vs {var}')
    ax.set_xlabel('Extreme Precipitation')
    ax.set_ylabel(var)
    ax.grid(True)
    ax.legend()

# 可视化极端温度影响
for i, var in enumerate(dependent_vars):
    ax = axes[i + len(dependent_vars)]
    sns.boxplot(data=df, x='Extreme_Temperature', y=var, ax=ax, palette="Set2", showfliers=False)

    # 计算均值 ± 标准误
    means = df.groupby('Extreme_Temperature')[var].mean()
    sems = df.groupby('Extreme_Temperature')[var].sem()
    x_pos = [0, 1]
    ax.errorbar(x=x_pos, y=means, yerr=sems, fmt='o', color='black', capsize=5, label='Mean ± SE')

    # t检验
    group1 = df[df['Extreme_Temperature'] == False][var]
    group2 = df[df['Extreme_Temperature'] == True][var]
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
    ax.text(0.05, 0.95, f't-test p = {p_val:.3e}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

    ax.set_title(f'Extreme Temperature vs {var}')
    ax.set_xlabel('Extreme Temperature')
    ax.set_ylabel(var)
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig('Extreme_Weather_Impact_with_Pvalues.png', dpi=300)
plt.show()

# 计算极端值阈值
precipitation_threshold = df['Precipitation'].quantile(0.95)
temperature_threshold = df['Temperature'].quantile(0.95)

# 添加极端天气标签
df['Extreme_Precipitation'] = df['Precipitation'] > precipitation_threshold
df['Extreme_Temperature'] = df['Temperature'] > temperature_threshold

# 提取城市列表
cities = df[df.columns[0]].unique()

# 创建图
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(18, 22))
axes = axes.flatten()

# --- 1. 每城市 极端降水 vs GEAA ---
for i, city in enumerate(cities):
    city_data = df[df[df.columns[0]] == city]
    ax = axes[i]

    sns.boxplot(data=city_data, x='Extreme_Precipitation', y='geaa', ax=ax, palette="Set2", showfliers=False)

    # 添加误差条
    means = city_data.groupby('Extreme_Precipitation')['geaa'].mean()
    sems = city_data.groupby('Extreme_Precipitation')['geaa'].sem()
    x_pos = [0, 1]
    ax.errorbar(x=x_pos, y=means, yerr=sems, fmt='o', color='black', capsize=5, label='Mean ± SE')

    # 添加 t 检验结果
    group1 = city_data[city_data['Extreme_Precipitation'] == False]['geaa']
    group2 = city_data[city_data['Extreme_Precipitation'] == True]['geaa']
    if len(group1) > 1 and len(group2) > 1:
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        ax.text(0.05, 0.95, f'p = {p_val:.3e}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

    ax.set_title(f'{city} - Extreme Precipitation vs GEAA')
    ax.set_xlabel('Extreme Precipitation')
    ax.set_ylabel('GEAA')
    ax.grid(True)
    ax.legend()

# --- 2. 每城市 极端温度 vs GEAA ---
for i, city in enumerate(cities):
    city_data = df[df[df.columns[0]] == city]
    ax = axes[i + len(cities)]

    sns.boxplot(data=city_data, x='Extreme_Temperature', y='geaa', ax=ax, palette="Set2", showfliers=False)

    # 添加误差条
    means = city_data.groupby('Extreme_Temperature')['geaa'].mean()
    sems = city_data.groupby('Extreme_Temperature')['geaa'].sem()
    x_pos = [0, 1]
    ax.errorbar(x=x_pos, y=means, yerr=sems, fmt='o', color='black', capsize=5, label='Mean ± SE')

    # 添加 t 检验结果
    group1 = city_data[city_data['Extreme_Temperature'] == False]['geaa']
    group2 = city_data[city_data['Extreme_Temperature'] == True]['geaa']
    if len(group1) > 1 and len(group2) > 1:
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        ax.text(0.05, 0.95, f'p = {p_val:.3e}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

    ax.set_title(f'{city} - Extreme Temperature vs GEAA')
    ax.set_xlabel('Extreme Temperature')
    ax.set_ylabel('GEAA')
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig('Impact_of_extreme_weather_on_GEAA_by_city_with_pvalues.png', dpi=300)
plt.show()
# =========================================================
# Environmental Heterogeneity and GEAA
# JAMx / High-Level Epidemiology Style
# =========================================================

# =========================================================
# LOAD LIBRARIES
# =========================================================

# =========================================================
# GLOBAL STYLE
# =========================================================

plt.rcParams.update({

    "font.family": "sans-serif",
    "font.size": 12,

    "axes.titlesize": 16,
    "axes.labelsize": 14,

    "xtick.labelsize": 11,
    "ytick.labelsize": 11,

    "legend.fontsize": 11,

    "figure.dpi": 300,
    "savefig.dpi": 300,

    "axes.linewidth": 1.2,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

sns.set_style("white")

# =========================================================
# LOAD CLEAN DATASET
# =========================================================

df = pd.read_csv("final_analysis_dataset.csv")

# =========================================================
# CLEAN CATEGORICAL VARIABLES
# =========================================================

df["region"] = df["region"].astype(str)

# =========================================================
# MAIN OUTCOME
# =========================================================

OUTCOME = "geaa"

# =========================================================
# FIGURE 1
# GEAA DISTRIBUTION BY REGION
# =========================================================

fig, ax = plt.subplots(figsize=(8,6))

sns.boxplot(
    data=df,
    x="region",
    y=OUTCOME,
    width=0.6,
    showcaps=True,
    showfliers=False,
    linewidth=1.5,
    ax=ax
)

sns.stripplot(
    data=df,
    x="region",
    y=OUTCOME,
    color="black",
    alpha=0.35,
    jitter=0.18,
    size=4,
    ax=ax
)

ax.set_title(
    "Regional Differences in GEAA",
    fontweight="bold",
    pad=15
)

ax.set_xlabel("")
ax.set_ylabel("GEAA")

sns.despine()

plt.tight_layout()

plt.savefig(
    "Figure1_GEAA_by_Region.png",
    bbox_inches="tight"
)

plt.close()

# =========================================================
# FIGURE 2
# TEMPERATURE vs GEAA BY REGION
# =========================================================

fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Temperature",
    y=OUTCOME,
    hue="region",
    alpha=0.6,
    s=50,
    ax=ax
)

sns.regplot(
    data=df,
    x="Temperature",
    y=OUTCOME,
    scatter=False,
    ci=None,
    line_kws={"linewidth":2},
    ax=ax
)

ax.set_title(
    "Temperature and GEAA Across Regions",
    fontweight="bold",
    pad=15
)

ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("GEAA")

sns.despine()

plt.tight_layout()

plt.savefig(
    "Figure2_Temperature_GEAA.png",
    bbox_inches="tight"
)

plt.close()

# =========================================================
# ADJUSTED MAIN MODEL
# =========================================================

model = smf.ols(
    """
    geaa ~
    Temperature * C(region) +
    Q('SPWPM2.5') +
    NO2 +
    Ozone +
    SO2 +
    educbas +
    C(white) +
    bmi +
    cum_pkyear +
    C(hivatvisit)
    """,
    data=df
).fit()

print(model.summary())

# =========================================================
# SAVE REGRESSION RESULTS
# =========================================================

results_df = pd.DataFrame({

    "Variable": model.params.index,
    "Beta": model.params.values,
    "Pvalue": model.pvalues.values,
    "CI_lower": model.conf_int()[0].values,
    "CI_upper": model.conf_int()[1].values

})

results_df.to_csv(
    "Adjusted_Model_Results.csv",
    index=False
)

# =========================================================
# REGION-SPECIFIC MODELS
# =========================================================

regions = sorted(df["region"].unique())

region_effects = []

for region in regions:

    temp_df = df[df["region"] == region]

    fit = smf.ols(
        """
        geaa ~
        Temperature +
        educbas +
        bmi +
        cum_pkyear +
        C(white) +
        C(hivatvisit)
        """,
        data=temp_df
    ).fit()

    beta = fit.params["Temperature"]
    pval = fit.pvalues["Temperature"]

    ci_low = fit.conf_int().loc["Temperature",0]
    ci_high = fit.conf_int().loc["Temperature",1]

    region_effects.append({

        "Region": region,
        "Beta": beta,
        "CI_low": ci_low,
        "CI_high": ci_high,
        "Pvalue": pval
    })

forest_df = pd.DataFrame(region_effects)

forest_df.to_csv(
    "Region_Specific_Effects.csv",
    index=False
)

print(forest_df)

# =========================================================
# FIGURE 3
# FOREST PLOT
# =========================================================

fig, ax = plt.subplots(figsize=(7,5))

y_pos = np.arange(len(forest_df))

ax.errorbar(
    forest_df["Beta"],
    y_pos,
    xerr=[
        forest_df["Beta"] - forest_df["CI_low"],
        forest_df["CI_high"] - forest_df["Beta"]
    ],
    fmt='o',
    capsize=4,
    linewidth=2
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1.2
)

ax.set_yticks(y_pos)
ax.set_yticklabels(forest_df["Region"])

ax.set_xlabel("Beta Coefficient for Temperature")
ax.set_title(
    "Region-Specific Temperature Effects on GEAA",
    fontweight="bold",
    pad=15
)

sns.despine()

plt.tight_layout()

plt.savefig(
    "Figure3_ForestPlot_Temperature_GEAA.png",
    bbox_inches="tight"
)

plt.close()

# =========================================================
# FIGURE 4
# =========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14,10)
)

# ---------------------------------
# PANEL A
# ---------------------------------

sns.boxplot(
    data=df,
    x="region",
    y=OUTCOME,
    showfliers=False,
    ax=axes[0,0]
)

axes[0,0].set_title("A. GEAA by Region")

# ---------------------------------
# PANEL B
# ---------------------------------

sns.scatterplot(
    data=df,
    x="Temperature",
    y=OUTCOME,
    hue="region",
    alpha=0.6,
    ax=axes[0,1]
)

axes[0,1].set_title("B. Temperature and GEAA")

# ---------------------------------
# PANEL C
# ---------------------------------

sns.scatterplot(
    data=df,
    x="SPWPM2.5",
    y=OUTCOME,
    hue="region",
    alpha=0.6,
    ax=axes[1,0]
)

axes[1,0].set_title("C. PM2.5 and GEAA")

# ---------------------------------
# PANEL D
# ---------------------------------

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
    ax=axes[1,1],
    cbar_kws={"shrink":0.8}
)

axes[1,1].set_title("D. Correlation Matrix")

# ---------------------------------
# CLEANUP
# ---------------------------------

for ax in axes.flatten():

    ax.grid(False)

plt.tight_layout()

plt.savefig(
    "Figure4_Multipanel_Publication.png",
    bbox_inches="tight"
)

plt.close()

# =========================================================
# SUPPLEMENTARY FIGURE
# EEAA DISTRIBUTION
# =========================================================

fig, ax = plt.subplots(figsize=(8,6))

sns.boxplot(
    data=df,
    x="region",
    y="eeaa",
    showfliers=False,
    ax=ax
)

sns.stripplot(
    data=df,
    x="region",
    y="eeaa",
    color="black",
    alpha=0.3,
    jitter=0.15,
    ax=ax
)

ax.set_title(
    "Supplementary Figure: EEAA by Region",
    fontweight="bold"
)

sns.despine()

plt.tight_layout()

plt.savefig(
    "Supplementary_EEAA_by_Region.png",
    bbox_inches="tight"
)

plt.close()

# =========================================================
# SUMMARY TABLE
# =========================================================

summary_table = df.groupby("region")[[
    "Temperature",
    "SPWPM2.5",
    "NO2",
    "Ozone",
    "SO2",
    "geaa"
]].agg([
    "mean",
    "std",
    "median"
])

summary_table.to_csv(
    "Regional_Summary_Table.csv"
)

# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n====================================")
print("PUBLICATION FIGURE PIPELINE COMPLETE")
print("====================================")

print("""
Generated files:

MAIN FIGURES
-------------
1. Figure1_GEAA_by_Region.png
2. Figure2_Temperature_GEAA.png
3. Figure3_ForestPlot_Temperature_GEAA.png
4. Figure4_Multipanel_Publication.png

SUPPLEMENTARY
-------------
5. Supplementary_EEAA_by_Region.png

TABLES
-------------
6. Adjusted_Model_Results.csv
7. Region_Specific_Effects.csv
8. Regional_Summary_Table.csv
""")