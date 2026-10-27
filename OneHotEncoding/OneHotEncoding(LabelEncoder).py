import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv("homeprices.csv")

##Don't need the label encoder anymore
# le=LabelEncoder()
# dfle=df
# dfle.town=le.fit_transform(dfle.town)
# # print(dfle)

ohe = OneHotEncoder(drop="first", sparse_output=False)
model = LinearRegression()

town_encoded = ohe.fit_transform(df[["town"]])

town_df = pd.DataFrame(
    town_encoded, columns=ohe.get_feature_names_out(), index=df.index
)

x = pd.concat([df.area, town_df], axis="columns")
y = df.price
print(x)

model.fit(x, y)
print("Robinsville (2800 sqft): ", model.predict([[2800, 1, 0]]))
print("West Windsor (3400 sqft): ",model.predict([[3400,0,1]]))
