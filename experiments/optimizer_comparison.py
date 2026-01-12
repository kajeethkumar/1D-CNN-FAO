import pandas as pd

adam = pd.read_csv("results/adam_metrics.csv")
fao  = pd.read_csv("results/fao_metrics.csv")

comparison = pd.concat([adam, fao])
comparison.to_csv("results/optimizer_comparison.csv", index=False)
