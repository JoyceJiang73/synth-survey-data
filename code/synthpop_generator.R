library(synthpop)
library(dplyr)

df <- read.csv('../training/training_20k.csv')

df <- df %>%
  mutate(across(c(state,
                  voted,
                  Voters_Gender,
                  Parties_Description,
                  Residence_HHParties_Description, 
                  EthnicGroups_EthnicGroup1Desc,
                  Ethnic_Description,
                  nonpartisan_donation,
                  CommercialData_PropertyType),
                as.factor)) %>%
  dplyr::select(-Ethnic_Description, -row_index)

start_time <- proc.time()

synth_training_20k <- syn(df,
                          method = "parametric",
                          # visit.sequence = c(1,3,4,5,7,9,10,8,11,2,6),
                          m = 1,
                          k = nrow(df),
                          polyreg.maxit = 3000,
                          cont.na = list(calculated_age = NA,
                                         CommercialData_EstimatedIncomeAmount = NA,
                                         CommercialData_EstHomeValue = NA))

end_time <- proc.time()

execution_time <- end_time - start_time
print(execution_time)

for (i in 1:5) {
  filename <- paste0("../output/synthpop/synth_training_20k_", i, ".csv")
  write.csv(synth_training_20k$syn[[i]], filename, row.names = FALSE)
}
# write.csv(synth_training_20k$syn, "../02_data/synthpop/synth_training_20k_5.csv")