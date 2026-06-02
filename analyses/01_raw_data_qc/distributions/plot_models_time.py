"""Plot proportion-missing per participant per day (heatmap over the first 90 days)."""

import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

df = pd.read_csv("data/raw/hormones_and_selfreport.csv")

df = df[df["day_in_study"] <= 90]

meta_cols = ["id", "study_interval", "day_in_study"]
vars_only = df.drop(columns=meta_cols)

df["missing_prop"] = vars_only.isna().mean(axis=1)

missing_map = df.pivot_table(
    index="id",
    columns="day_in_study",
    values="missing_prop",
    aggfunc="mean"
).reindex(columns=range(1, 91))

plt.figure(figsize=(14,6))
sns.heatmap(
    missing_map,
    cmap="magma",
    cbar_kws={"label": "Proportion missing"}
)
plt.title("Missingness over time per participant")
plt.xlabel("Day in study")
plt.ylabel("Participant")
plt.show()

# timeline overall missingness patterns
daily_missing = df.groupby("day_in_study")["missing_prop"].mean()

plt.figure(figsize=(10,4))
daily_missing.plot()
plt.title("Average missingness over time")
plt.xlabel("Day in study")
plt.ylabel("Mean proportion missing")
plt.grid(True)
plt.show()

# binning by month
df["time_bin"] = pd.cut(df["day_in_study"], bins=[0,30,60,90], labels=["Month 1","Month 2","Month 3"])

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="time_bin", y="missing_prop")
plt.title("Missingness by study period")
plt.show()

