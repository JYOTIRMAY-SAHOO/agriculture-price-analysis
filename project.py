# 🏆 FINAL PROJECT WITH TIME TREND (COMPLETE)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 🎨 STYLE

plt.style.use('dark_background')
sns.set_style("white")

plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#0E1117",
    "axes.labelcolor": "#EAEAEA",
    "xtick.color": "#AAAAAA",
    "ytick.color": "#AAAAAA",
    "text.color": "#FFFFFF"
})

plt.close('all')

# 📂 LOAD DATA

file_path = "/Users/jyotirmaysahoo/Library/CloudStorage/OneDrive-Personal/projects/python/data.csv"
df = pd.read_csv(file_path)

df.columns = df.columns.str.replace('_x0020_', '_')
df.columns = df.columns.str.strip()

df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], errors='coerce')
df = df.dropna()

# 📊 BASIC INFO

print("\n===== DATA SUMMARY =====")
print("Records:", len(df))
print("States:", df['State'].nunique())
print("Commodities:", df['Commodity'].nunique())

# 🌈 1. TOP COMMODITIES

plt.figure(figsize=(10,5))
top_com = df['Commodity'].value_counts().head(10)

plt.bar(top_com.index, top_com.values,
        color=sns.color_palette("flare", len(top_com)))

plt.title("Top Commodities", fontsize=14, weight='bold')
plt.xticks(rotation=40)
plt.tight_layout()
plt.show()

# 📊 2. PRICE DISTRIBUTION

plt.figure(figsize=(10,5))
sns.histplot(df['Modal_Price'], bins=40, kde=True, color="#3DDC97")

plt.axvline(df['Modal_Price'].mean(),
            color="red", linestyle="--")

plt.title("Price Distribution", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

# 📊 3. STATE-WISE PRICE

plt.figure(figsize=(10,6))
state_avg = df.groupby('State')['Modal_Price'].mean().sort_values().tail(10)

sns.barplot(x=state_avg.values,
            y=state_avg.index,
            palette="crest")

plt.title("Top States by Price", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

# 🔴 4. HEATMAP

plt.figure(figsize=(7,5))
corr = df[['Min_Price','Max_Price','Modal_Price']].corr()

sns.heatmap(corr, annot=True, cmap="rocket")

plt.title("Correlation Matrix", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

# 🎭 5. BOXPLOT

plt.figure(figsize=(10,5))
top5 = df['Commodity'].value_counts().head(5).index

sns.boxplot(x='Commodity', y='Modal_Price',
            data=df[df['Commodity'].isin(top5)],
            palette="Set3")

plt.title("Price Spread", fontsize=14, weight='bold')
plt.xticks(rotation=40)
plt.tight_layout()
plt.show()

# 📈 6. TIME TREND (FIXED + WORKING ⭐)

plt.figure(figsize=(12,5))

df_time = df.groupby('Arrival_Date')['Modal_Price'].mean().sort_index()

# reduce data points for better visualization
if len(df_time) > 50:
    df_time = df_time.iloc[::10]

plt.plot(df_time.index,
         df_time.values,
         color="#00D4FF",
         linewidth=2.5)

plt.fill_between(df_time.index,
                 df_time.values,
                 color="#00D4FF",
                 alpha=0.2)

plt.title("Price Trend Over Time", fontsize=14, weight='bold')
plt.xlabel("Date")
plt.ylabel("Price")

plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# 🤖 MACHINE LEARNING

df_ml = df.copy()

df_ml['Commodity'] = df_ml['Commodity'].astype('category').cat.codes
df_ml['State'] = df_ml['State'].astype('category').cat.codes

X = df_ml[['Commodity','State','Min_Price','Max_Price']]
y = df_ml['Modal_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\n===== MODEL PERFORMANCE =====")
print("R2:", model.score(X_test, y_test))
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))

# 📊 FEATURE IMPORTANCE

importance = pd.Series(model.feature_importances_, index=X.columns)

plt.figure(figsize=(8,4))
sns.barplot(x=importance.values,
            y=importance.index,
            palette="magma")

plt.title("Feature Importance", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

# 🧠 FINAL INSIGHTS

print("\n===== FINAL INSIGHTS =====")

print("Top Commodity:", top_com.index[0])
print("Most Expensive State:", state_avg.index[-1])

trend = "Increasing 📈" if df_time.iloc[-1] > df_time.iloc[0] else "Decreasing 📉"
print("Trend:", trend)

print("Conclusion: Prices depend on max price and regional variation.")