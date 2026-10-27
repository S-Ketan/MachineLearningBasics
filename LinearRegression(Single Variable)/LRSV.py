import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv('homeprices.csv')
# print(df)
plt.xlabel('Area(sq ft)')
plt.ylabel('Price($)')
plt.title('Area-Price Chart')
plt.scatter(df.area, df.price, color='red', marker='+')

reg=linear_model.LinearRegression()
reg.fit(df[['area']],df.price)

print(reg.predict([[3300]]))
# print(reg.predict([[5000]]))

d=pd.read_csv('areas.csv')
# print(reg.predict(d))

d['prices']=reg.predict(d)
# print(d)
# d.to_csv('prediction.csv',index=False)

plt.plot(df.area,reg.predict(df[['area']]), color='blue')

plt.show()
