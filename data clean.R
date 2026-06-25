

library(tidyverse)



df <- read_csv("original.csv")



df <- df %>%
  filter(visit == 4)

cat("N after visit filtering:\n")
print(nrow(df))


df <- df %>%
  rename(region = macsidnumber)


df <- df %>%
  select(-Precipitation)


df$Temperature <- gsub("\\+", "", df$Temperature)
df$Temperature <- gsub(",", ".", df$Temperature)

df$Temperature <- as.numeric(df$Temperature)

# scaling factor correction
df$Temperature <- df$Temperature / 10



env_vars <- c(
  "SPWPM2.5",
  "NO2",
  "Ozone",
  "SO2"
)

df <- df %>%
  mutate(
    across(
      all_of(env_vars),
      as.numeric
    )
  )



aging_vars <- c(
  "aar",
  "eeaa",
  "peaa",
  "geaa",
  "dnamtladjage"
)

df <- df %>%
  mutate(
    across(
      all_of(aging_vars),
      as.numeric
    )
  )



covars <- c(
  "educbas",
  "bmi",
  "cum_pkyear"
)

df <- df %>%
  mutate(
    across(
      all_of(covars),
      as.numeric
    )
  )



df$white <- factor(
  df$white,
  levels = c(0,1),
  labels = c("Non-White", "White")
)

df$hivatvisit <- factor(
  df$hivatvisit,
  levels = c(0,1),
  labels = c("HIV-negative", "HIV-positive")
)

df$region <- as.factor(df$region)



analysis_df <- df %>%
  select(
    region,
    white,
    educbas,
    bmi,
    cum_pkyear,
    hivatvisit,
    Temperature,
    SPWPM2.5,
    NO2,
    Ozone,
    SO2,
    aar,
    eeaa,
    peaa,
    geaa,
    dnamtladjage
  )



numeric_vars <- names(analysis_df)[
  sapply(analysis_df, is.numeric)
]

for(v in numeric_vars){
  
  median_value <- median(
    analysis_df[[v]],
    na.rm = TRUE
  )
  
  analysis_df[[v]][
    is.na(analysis_df[[v]])
  ] <- median_value
}



get_mode <- function(x){
  
  ux <- unique(x)
  
  ux[
    which.max(
      tabulate(match(x, ux))
    )
  ]
}

factor_vars <- names(analysis_df)[
  sapply(analysis_df, is.factor)
]

for(v in factor_vars){
  
  mode_value <- get_mode(
    analysis_df[[v]]
  )
  
  analysis_df[[v]][
    is.na(analysis_df[[v]])
  ] <- mode_value
}



print(nrow(analysis_df))


print(colSums(is.na(analysis_df)))

print(table(analysis_df$region))

print(summary(analysis_df$Temperature))



write_csv(
  analysis_df,
  "final_analysis_dataset.csv"
)

cat("\nDataset exported:\n")
cat("final_analysis_dataset.csv\n")
