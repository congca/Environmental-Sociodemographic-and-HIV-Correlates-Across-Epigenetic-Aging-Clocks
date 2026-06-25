# =========================================================
# TABLE 1 GENERATION
# JAMA-STYLE
# =========================================================
 
library(tidyverse)
library(gtsummary)
library(flextable)
library(officer)
# =========================================================
# FINAL STABLE TABLE 1
# =========================================================

library(tidyverse)

library(tableone)

library(flextable)

library(officer)

# =========================================================
# LOAD DATA
# =========================================================

df <- read_csv(
  "final_analysis_dataset.csv"
)

# =========================================================
# FACTORS
# =========================================================

df$region <- factor(df$region)

df$white <- factor(df$white)

df$hivatvisit <- factor(df$hivatvisit)

# =========================================================
# VARIABLES
# =========================================================

vars <- c(
  
  "white",
  "educbas",
  "bmi",
  "cum_pkyear",
  "hivatvisit",
  
  "Temperature",
  "SPWPM2.5",
  "NO2",
  "Ozone",
  "SO2",
  
  "aar",
  "eeaa",
  "peaa",
  "dnamtladjage"
  
)

# optional immune markers
immune_vars <- c(
  
  "lncd4",
  "lncd8",
  "lnsencd8",
  "lnactcd8"
  
)

immune_vars <- immune_vars[
  immune_vars %in% names(df)
]

vars <- c(vars, immune_vars)

# =========================================================
# FACTOR VARIABLES
# =========================================================

factorVars <- c(
  
  "white",
  "hivatvisit"
  
)

# =========================================================
# CREATE TABLE
# =========================================================

tab1 <- CreateTableOne(
  
  vars = vars,
  
  strata = "region",
  
  data = df,
  
  factorVars = factorVars,
  
  addOverall = TRUE
  
)

# =========================================================
# PRINT TABLE
# =========================================================

tab1_df <- print(
  
  tab1,
  
  quote = FALSE,
  
  noSpaces = TRUE,
  
  printToggle = FALSE
  
)

tab1_df <- data.frame(
  
  Characteristic = rownames(tab1_df),
  
  tab1_df,
  
  row.names = NULL
  
)

# =========================================================
# RENAME VARIABLES
# =========================================================

tab1_df$Characteristic <- recode(
  
  tab1_df$Characteristic,
  
  "white = 1" = "White race, n (%)",
  
  "educbas" = "Education level (0–7)",
  
  "bmi" = "BMI (kg/m²)",
  
  "cum_pkyear" = "Smoking (pack-years)",
  
  "hivatvisit = 1" = "HIV-positive, n (%)",
  
  "Temperature" = "Temperature (°F)",
  
  "SPWPM2.5" = "PM2.5 (μg/m³)",
  
  "NO2" = "NO₂ (ppb)",
  
  "Ozone" = "Ozone (ppm)",
  
  "SO2" = "SO₂ (ppb)",
  
  "aar" = "AAR",
  
  "eeaa" = "EEAA",
  
  "peaa" = "PEAA",
  
  "dnamtladjage" = "DNAmTLadjAge",
  
  "lncd4" = "CD4 count",
  
  "lncd8" = "CD8 count",
  
  "lnsencd8" = "Senescent CD8",
  
  "lnactcd8" = "Activated CD8"
  
)

# =========================================================
# FLEXTABLE
# =========================================================

ft <- flextable(tab1_df)

ft <- autofit(ft)

ft <- fontsize(
  
  ft,
  
  size = 9,
  
  part = "all"
  
)

ft <- bold(
  
  ft,
  
  part = "header"
  
)

# =========================================================
# EXPORT WORD
# =========================================================

doc <- read_docx()

doc <- body_add_par(
  
  doc,
  
  "Table 1. Sociodemographic, Environmental, and Epigenetic Aging Characteristics by Study Region",
  
  style = "heading 1"
  
)

doc <- body_add_flextable(
  
  doc,
  
  value = ft
  
)

print(
  
  doc,
  
  target = "Table1_Region.docx"
  
)

# =========================================================
# DONE
# =========================================================

cat("\nDONE\n")

cat(
  "\nExported: Table1_Region.docx\n"
)