# %%
# import libraries
import pandas as pd
import altair as alt
import numpy as np
import seaborn as sns
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from prettytable import PrettyTable

# %%
# gather data and change column names
df = pd.read_csv(
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/star-wars-survey/StarWars.csv",
    encoding="latin1",
)
df = df.rename(
    columns={
        "RespondentID": "id",
        "Have you seen any of the 6 films in the Star Wars franchise?": "anyfilm",
        "Do you consider yourself to be a fan of the Star Wars film franchise?": "fan",
        "Which of the following Star Wars films have you seen? Please select all that apply.": "seen1",
        "Unnamed: 4": "seen2",
        "Unnamed: 5": "seen3",
        "Unnamed: 6": "seen4",
        "Unnamed: 7": "seen5",
        "Unnamed: 8": "seen6",
        "Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.": "rank1",
        "Unnamed: 10": "rank2",
        "Unnamed: 11": "rank3",
        "Unnamed: 12": "rank4",
        "Unnamed: 13": "rank5",
        "Unnamed: 14": "rank6",
        "Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.": "hansolo",
        "Unnamed: 16": "lukeskywalker",
        "Unnamed: 17": "leia",
        "Unnamed: 18": "anakin",
        "Unnamed: 19": "obiwan",
        "Unnamed: 20": "palpatine",
        "Unnamed: 21": "vader",
        "Unnamed: 22": "lando",
        "Unnamed: 23": "boba",
        "Unnamed: 24": "c3p0",
        "Unnamed: 25": "r2d2",
        "Unnamed: 26": "jarjar",
        "Unnamed: 27": "padme",
        "Unnamed: 28": "yoda",
        "Which character shot first?": "shotfirst",
        "Are you familiar with the Expanded Universe?": "familiarexpu",
        "Do you consider yourself to be a fan of the Expanded Universe?æ": "fanexpu",
        "Do you consider yourself to be a fan of the Star Trek franchise?": "fanstartrek",
        "Gender": "gender",
        "Age": "age",
        "Household Income": "income",
        "Education": "education",
        "Location (Census Region)": "location",
    }
)
df = df.drop_duplicates()
# %%
# print a table for grand question 1
table = PrettyTable()
table.field_names = df.columns
print(table)
# %%
# part a of grand question 2
df = df[
    (df["anyfilm"] == "Yes")
    & (
        (df["seen1"] == "Star Wars: Episode I  The Phantom Menace")
        | (df["seen2"] == "Star Wars: Episode II  Attack of the Clones")
        | (df["seen3"] == "Star Wars: Episode III  Revenge of the Sith")
        | (df["seen4"] == "Star Wars: Episode IV  A New Hope")
        | (df["seen5"] == "Star Wars: Episode V The Empire Strikes Back")
        | (df["seen6"] == "Star Wars: Episode VI Return of the Jedi")
    )
]


# %%
# part b of grand question 2
def calculate_midpoint(age):
    if pd.notna(age):
        if "-" in age:
            start, end = map(int, age.split("-"))
            return (start + end) / 2
        elif ">" in age:
            return int(age.split(">")[1])
        else:
            return int(age)
    else:
        return np.nan


df["age"] = df["age"].apply(calculate_midpoint)


# %%
def calculate_midpoint_income(income):
    if pd.notna(income):
        income_numeric = income.replace("$", "").replace(",", "").replace("+", "")

        if "-" in income_numeric:
            start, end = map(int, income_numeric.split("-"))
            return (start + end) / 2
        elif ">" in income_numeric:
            return int(income_numeric.split(">")[1])
        else:
            return int(income_numeric)
    else:
        return np.nan


df["income"] = df["income"].apply(calculate_midpoint_income)

# %%
# %%
df = df.replace(
    {
        "Star Wars: Episode I  The Phantom Menace": 1,
        "Star Wars: Episode II  Attack of the Clones": 1,
        "Star Wars: Episode III  Revenge of the Sith": 1,
        "Star Wars: Episode IV  A New Hope": 1,
        "Star Wars: Episode V The Empire Strikes Back": 1,
        "Star Wars: Episode VI Return of the Jedi": 1,
        "Yes": 1,
        "No": 0,
        "Unfamiliar (N/A)": 0,
        "Very unfavorably": 1,
        "Somewhat unfavorably": 2,
        "Neither favorably nor unfavorably (neutral)": 3,
        "Somewhat favorably": 4,
        "Very favorably": 5,
        "Female": 0,
        "Male": 1,
        "I don't understand this question": 0,
        "Greedo": 1,
        "Han": 2,
        "Less than high school degree": 1,
        "High school degree": 2,
        "Some college or Associate degree": 3,
        "Bachelor degree": 4,
        "Graduate degree": 5,
        "South Atlantic": 1,
        "West South Central": 2,
        "West North Central": 3,
        "Middle Atlantic": 4,
        "East North Central": 5,
        "Pacific": 6,
        "Mountain": 7,
        "New England": 8,
        "East South Central": 9,
        23.5: 1,
        37.0: 2,
        52.5: 3,
        60.0: 4,
        12499.5: 1,
        37499.5: 2,
        74999.5: 3,
        124999.5: 4,
        150000.0: 5,
    }
)
# %%

# %%
# create a chart showing which movies people have seen
alt.data_transformers.enable("default", max_rows=None)

movies_seen_df = df[["seen1", "seen2", "seen3", "seen4", "seen5", "seen6"]]
movies_seen_df = movies_seen_df.rename(
    columns={
        "seen1": "The Phantom Menace",
        "seen2": "Attack of the Clones",
        "seen3": "Revenge of the Sith",
        "seen4": "A New Hope",
        "seen5": "The Empire Strikes Back",
        "seen6": "Return of the Jedi",
    }
)

df_melted = movies_seen_df.melt()
df_melted = df_melted[df_melted["value"] == 1]
total_viewers = df["anyfilm"].value_counts()

df_melted["percentage"] = (
    (df_melted.groupby("variable")["value"].transform("sum") / total_viewers[1] * 100)
    .dropna()
    .round(0)
)

df_melted["percentage_str"] = df_melted["percentage"].astype(int).astype(str) + "%"

chart = (
    alt.Chart(df_melted)
    .mark_bar()
    .encode(
        y=alt.Y(
            "variable:N",
            title="",
            sort=[
                "The Phantom Menace",
                "Attack of the Clones",
                "Revenge of the Sith",
                "A New Hope",
                "The Empire Strikes Back",
                "Return of the Jedi",
            ],
        ),
        x=alt.X("sum(value):Q", title="", axis=None),
        tooltip=["variable:N", alt.Tooltip("percentage_str:N", title="Percentage")],
    )
    .properties(title="Which 'Star Wars' Movies Have You Seen?")
)

chart_text = chart.mark_text(align="left", baseline="middle", dx=3).encode(
    text=alt.Text("percentage_str:N")
)

chart_final = (chart + chart_text).properties(
    title="Which 'Star Wars' Movies Have You Seen?"
)

chart_final

# %%
# recreate the chart showing who shot first
df_filtered_shotfirst = df[df["shotfirst"] != ""]
df_filtered_shotfirst = df_filtered_shotfirst.replace(
    {0: "I don't understand this question", 1: "Greedo", 2: "Han"}
)

percentage_data = (
    df_filtered_shotfirst["shotfirst"].value_counts(normalize=True).reset_index()
)
percentage_data.columns = ["shotfirst", "percentage"]
percentage_data["percentage"] = percentage_data["percentage"].round(
    2
)  # Round to two decimal places

chart_shotfirst_percentage = (
    alt.Chart(percentage_data)
    .mark_bar()
    .encode(
        y=alt.Y(
            "shotfirst:N",
            title="",
            sort=["Han", "Greedo", "I don't understand this question"],
        ),
        x=alt.X("percentage:Q", title="", axis=None),
        tooltip=[
            "shotfirst:N",
            alt.Tooltip("percentage:Q", title="Percentage", format=".0%"),
        ],
    )
)

chart_shotfirst_percentage_text = chart_shotfirst_percentage.mark_text(
    align="left", baseline="middle", dx=3
).encode(text=alt.Text("percentage:Q", format=".0%"))

chart_shotfirst_percentage_final = (
    chart_shotfirst_percentage + chart_shotfirst_percentage_text
).properties(title="Who Shot First?")

chart_shotfirst_percentage_final
# %%
# Create the 'target' column
df["target"] = df["income"].replace({1: 0, 2: 0, 3: 1, 4: 1, 5: 1})
df = df.dropna(subset=["income"])
X = df.drop(columns={"income", "target"})
y = df["target"]
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.2f}")
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# %%
dt_model.fit(X_train, y_train)
feature_importances = dt_model.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": feature_importances}
)
feature_importance_df = feature_importance_df.sort_values(
    by="Importance", ascending=False
)
alt.Chart(feature_importance_df).mark_bar().encode(
    x=alt.X("Importance:Q", title="Importance"),
    y=alt.Y("Feature:N", title="Feature"),
).properties(title="Feature Importance Chart")
# %%
importance_threshold = 0.005
important_features = feature_importance_df[
    feature_importance_df["Importance"] > importance_threshold
]
print("Important Features:")
print(important_features["Feature"].tolist())
# %%
selected_features = important_features["Feature"].tolist()
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
dt_model_selected = DecisionTreeClassifier(random_state=42)
dt_model_selected.fit(X_train_selected, y_train)
y_pred_selected = dt_model_selected.predict(X_test_selected)
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print(f"Decision Tree Accuracy with Important Features: {accuracy_selected:.2f}")
class_report_selected = classification_report(y_test, y_pred_selected)
print("Classification Report with Important Features:\n", class_report_selected)
# %%
