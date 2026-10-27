import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("carprices.csv")
ohe = OneHotEncoder(drop="first", sparse_output=False)
model = LinearRegression()
# print(df)
encoded_car = ohe.fit_transform(df[["Car Model"]])

car_df = pd.DataFrame(encoded_car, columns=ohe.get_feature_names_out(), index=df.index)

x = pd.concat([df.Mileage, df["Age(yrs)"], car_df], axis="columns")
y = df["Sell Price($)"]
print(x)
model.fit(x, y)

print("4 years old benz with 45000 Mileage: ", model.predict([[45000,4,0,1]]))
print("7 years old BMW X5 with 86000 Mileage: ", model.predict([[86000,7,1,0]]))
