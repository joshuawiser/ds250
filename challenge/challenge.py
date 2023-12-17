import altair as alt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from datetime import datetime
import datetime as dt
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

url = "https://github.com/byuidatascience/data4names/raw/master/data-raw/names_year/names_year.csv"
names = pd.read_csv(url)

current_year = dt.datetime.now().year
names['age'] = current_year - names['year']

# Filter data for the name 'Russell'
russell_df = names[names['name'] == 'Russell']

# Create Altair chart
chart = (
    alt.Chart(russell_df, title="Russell Age Chart")
    .mark_line()
    .encode(
        x=alt.X("age:O", title="Age"),
        y=alt.Y("Total:Q", title="Frequency"),
        tooltip=["age:N", "Total:Q"],
    )
    .properties(width=600, height=400)
)

chart




# question 2
my_df = pd.DataFrame({
    "column1": ['N/A', 52, 22, 45, 31, -999, 21, 2, 0, 0, 'broken', 19, 6, 27, 0, np.nan, 0, 33, 42, -999],
    "column2": [25.7, 6.6, 42.5, 72.0, 4.8, 4.0, 81.2, 654.5, 42.0, 5.7, 54.2, 4.2, 6.3, 76.5, 7.2, 42.5, 76.8, 46.2, 11.9, 94.6]
    })

my_df['column1'] = pd.to_numeric(my_df['column1'], errors='coerce')

# Replace missing values in column1 with the mean of column2
mean_column2 = my_df['column2'].mean()
my_df['column1'].fillna(mean_column2, inplace=True)

# Report the mean and standard deviation of column1
mean_column1 = my_df['column1'].mean()
std_column1 = my_df['column1'].std()

print("Mean of column1:", mean_column1)
print("Standard deviation of column1:", std_column1)


#q3
people = pd.DataFrame({
    "name": ['Joseph', 'Maria', 'Edmond', 'Fay', 'Rachel', 'George', 'Hector', 'Wesley', 'Silas', 'Leslie', 'Janet', 'Norman'],
    "age": ["10-19", "10-19", "20-29", "50-59", "10-19", "40-49", "50-59", "30-39", "60-69", "50-59", "10-19", "20-29"]
})

# Split the "age" category into two columns and use the minimum age
people[['min_age', 'max_age']] = people['age'].str.split('-', expand=True)
people['min_age'] = people['min_age'].astype(int)

# Create a histogram using Altair
chart = alt.Chart(people).mark_bar().encode(
    alt.X("min_age:O", bin=True, title='Minimum Age'),
    alt.Y('count():Q', title='Count'),
    tooltip=['min_age:O', 'count():Q']
).properties(
    title='Histogram of Minimum Age'
)

# Show the chart
chart

#q4

dwellings_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")
X = pd.get_dummies(dwellings_ml.drop(['numbaths', 'parcel'], axis=1), drop_first=True)
y = (dwellings_ml['numbaths'] > 2).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(random_state=1776)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")