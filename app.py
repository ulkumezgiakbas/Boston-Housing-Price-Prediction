import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv("boston.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

sns.pairplot(df)
plt.savefig("pairplot.png")
plt.close()

X = df.drop("PRICE", axis=1)
y = df["PRICE"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lm = LinearRegression()
lm.fit(X_train, y_train)

coeff_df = pd.DataFrame(lm.coef_, index=X.columns, columns=["Coefficient"])
print(coeff_df)

pred = lm.predict(X_test)
print("MAE:", metrics.mean_absolute_error(y_test, pred))
print("MSE:", metrics.mean_squared_error(y_test, pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, pred)))
print("R2:", metrics.r2_score(y_test, pred))

plt.scatter(pred, y_test - pred)
plt.axhline(0, color="k", linewidth=1)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals Plot")
plt.savefig("residuals.png")
plt.close()

y_log = np.log(y)
Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y_log, test_size=0.3, random_state=42)
lm2 = LinearRegression()
lm2.fit(Xtr2, ytr2)
pred2 = lm2.predict(Xte2)
print("Log-RMSE:", np.sqrt(metrics.mean_squared_error(yte2, pred2)))

sample = X_test.iloc[[0]]
print("Sample prediction:", lm.predict(sample)[0], "Actual:", y_test.iloc[0])
