import pandas as pd
from sklearn.linear_model import LinearRegression

df=pd.read_csv("homeprices.csv")

merged=pd.concat([df,pd.get_dummies(df.town,dtype=int)], axis="columns" )
final=merged.drop(['town','west windsor'], axis="columns")
# print(final)

x=final.drop(['price'], axis="columns")
y=final.price

reg=LinearRegression()

reg.fit(x,y)

##For 2800 sq ft  in robinsville
print("Robinsville (2800 sqft): ",reg.predict([[2800,0,1]]))

##for 3400 sqft in west windsor
print("West Windsor (3400 sqft): ",reg.predict([[3400,0,0]]))
