import pandas as pd
from sklearn import linear_model
from word2number import w2n

df = pd.read_csv("hiring.csv")
# print(df['interview_score(out of 10)'].median())

df = df.fillna(
    {
        "experience": "zero",
        "test_score(out of 10)": df["test_score(out of 10)"].median(),
    }
)


def convert_experience(value):
    return w2n.word_to_num(str(value))


df["experience"] = df["experience"].apply(convert_experience)
print(df)
reg = linear_model.LinearRegression()


reg.fit(
    df[["experience", "test_score(out of 10)", "interview_score(out of 10)"]],
    df["salary($)"],
)

print(reg.predict([[2, 9, 6]]))
print(reg.predict([[12, 10, 10]]))
