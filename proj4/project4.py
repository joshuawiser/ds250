# %%
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

# %%
dwellings_ml = pd.read_csv(
    "https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv"
)

# %%
h_subset = dwellings_ml.filter(
    [
        "livearea",
        "finbsmnt",
        "basement",
        "yearbuilt",
        "nocars",
        "numbdrm",
        "numbaths",
        "before1980",
        "stories",
        "yrbuilt",
    ]
).sample(500)
# %%
h_subset_2 = dwellings_ml.filter(
    [
        "totunits",
        "sprice",
        "deduct",
        "netprice",
        "tasp",
        "before1980",
        "smonth",
        "yrbuilt",
        "syear",
    ]
).sample(500)

# %%
h_subset_3 = dwellings_ml.filter(
    [
        "condition_AVG",
        "condition_Excel",
        "condition_Fair",
        "condition_Good",
        "condition_VGood",
        "before1980",
        "quality_A",
        "yrbuilt",
        "quality_B",
        "quality_C",
        "quality_D",
        "quality_X",
    ]
).sample(500)
# %%
h_subset_4 = dwellings_ml.filter(
    [
        "gartype_Att",
        "gartype_Att/Det",
        "gartype_CP",
        "gartype_Det",
        "gartype_None",
        "gartype_att/CP",
        "gartype_det/CP",
        "arcstyle_BI-LEVEL",
        "arcstyle_CONVERSIONS",
        "arcstyle_END UNIT",
        "arcstyle_MIDDLE UNIT",
        "yrbuilt",
        "before1980",
    ]
).sample(500)
# %%
h_subset_5 = dwellings_ml.filter(
    [
        "arcstyle_ONE AND HALF-STORY",
        "arcstyle_ONE-STORY",
        "arcstyle_SPLIT LEVEL",
        "arcstyle_THREE-STORY",
        "arcstyle_TRI-LEVEL",
        "arcstyle_TRI-LEVEL WITH BASEMENT",
        "arcstyle_TWO AND HALF-STORY",
        "arcstyle_TWO-STORY",
        "qualified_Q",
        "qualified_U",
        "status_I",
        "status_V",
        "yrbuilt",
        "before1980",
    ]
).sample(500)


# %%
sns.pairplot(h_subset, hue="before1980")
# %%
sns.pairplot(h_subset_2, hue="before1980")
# %%
sns.pairplot(h_subset_3, hue="before1980")
# %%
sns.pairplot(h_subset_4, hue="before1980")
# %%
sns.pairplot(h_subset_5, hue="before1980")


# %%
corr1 = h_subset.drop(columns="before1980").corr()
sns.heatmap(corr1)
# %%
corr2 = h_subset_2.drop(columns="before1980").corr()
sns.heatmap(corr2)
# %%
corr3 = h_subset_3.drop(columns="before1980").corr()
sns.heatmap(corr3)
# %%
corr4 = h_subset_4.drop(columns="before1980").corr()
sns.heatmap(corr4)
# %%
corr5 = h_subset_5.drop(columns="before1980").corr()
sns.heatmap(corr5)

# %%
X_pred = dwellings_ml.drop(
    dwellings_ml.filter(regex="before1980|yrbuilt").columns, axis=1
)
# %%
y_pred = dwellings_ml.filter(regex="before1980")
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_pred, y_pred, test_size=0.34, random_state=76
)
# %%
y_test
# %%
X_train
# %%
X_test
# %%
features = [
    "livearea",
    "finbsmnt",
    "basement",
    "nocars",
    "numbdrm",
    "numbaths",
    "before1980",
    "stories",
    "yrbuilt",
    "totunits",
    "sprice",
    "deduct",
    "netprice",
    "tasp",
    "smonth",
    "yrbuilt",
    "syear",
    "condition_AVG",
    "condition_Excel",
    "condition_Fair",
    "condition_Good",
    "condition_VGood",
    "quality_A",
    "yrbuilt",
    "quality_B",
    "quality_C",
    "quality_D",
    "quality_X",
    "gartype_Att",
    "gartype_Att/Det",
    "gartype_CP",
    "gartype_Det",
    "gartype_None",
    "gartype_att/CP",
    "gartype_det/CP",
    "arcstyle_BI-LEVEL",
    "arcstyle_CONVERSIONS",
    "arcstyle_END UNIT",
    "arcstyle_MIDDLE UNIT",
    "yrbuilt",
    "arcstyle_ONE AND HALF-STORY",
    "arcstyle_ONE-STORY",
    "arcstyle_SPLIT LEVEL",
    "arcstyle_THREE-STORY",
    "arcstyle_TRI-LEVEL",
    "arcstyle_TRI-LEVEL WITH BASEMENT",
    "arcstyle_TWO AND HALF-STORY",
    "arcstyle_TWO-STORY",
    "qualified_Q",
    "qualified_U",
    "status_I",
    "status_V",
    "yrbuilt",
]
subset_dt = dwellings_ml[features]
subset_dt = subset_dt.dropna()
le_dt = LabelEncoder()
subset_dt["before1980"] = le_dt.fit_transform(subset_dt["before1980"])
X_dt = subset_dt.drop(["before1980", "yrbuilt"], axis=1)
y_dt = subset_dt["before1980"]
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(
    X_dt, y_dt, test_size=0.2, random_state=42
)
# %%
clf_dt = DecisionTreeClassifier(random_state=42)

clf_dt.fit(X_train_dt, y_train_dt)

y_pred_dt = clf_dt.predict(X_test_dt)

accuracy_dt = accuracy_score(y_test_dt, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt * 100:.2f}%")
print("Decision Tree Classification Report:")
print(classification_report(y_test_dt, y_pred_dt))

# %%
feature_importances_dt = clf_dt.feature_importances_

feature_importance_df_dt = pd.DataFrame(
    {"Feature": X_dt.columns, "Importance": feature_importances_dt}
)

feature_importance_df_dt = feature_importance_df_dt.sort_values(
    by="Importance", ascending=False
)

chart_dt = (
    alt.Chart(feature_importance_df_dt)
    .mark_bar()
    .encode(
        y=alt.Y("Feature:N", title="Feature"),
        x=alt.X("Importance:Q", title="Importance"),
        color=alt.Color("Importance:Q", scale=alt.Scale(scheme="viridis")),
        tooltip=["Feature", "Importance"],
    )
    .properties(title="Feature Importance in Decision Tree Classifier")
)

chart_dt

# %%
subset_rf = dwellings_ml[features]
subset_rf = subset_rf.dropna()

le_rf = LabelEncoder()
subset_rf["before1980"] = le_rf.fit_transform(subset_rf["before1980"])

X_rf = subset_rf.drop(["before1980", "yrbuilt"], axis=1)
y_rf = subset_rf["before1980"]

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=42
)

clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)

clf_rf.fit(X_train_rf, y_train_rf)

y_pred_rf = clf_rf.predict(X_test_rf)

accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")

print("Random Forest Classification Report:")
print(classification_report(y_test_rf, y_pred_rf))

# %%

feature_importances_rf = clf_rf.feature_importances_

feature_importance_df_rf = pd.DataFrame(
    {"Feature": X_rf.columns, "Importance": feature_importances_rf}
)

feature_importance_df_rf = feature_importance_df_rf.sort_values(
    by="Importance", ascending=False
)

chart_rf = (
    alt.Chart(feature_importance_df_rf)
    .mark_bar()
    .encode(
        y=alt.Y("Feature:N", title="Feature"),
        x=alt.X("Importance:Q", title="Importance"),
        color=alt.Color("Importance:Q", scale=alt.Scale(scheme="viridis")),
        tooltip=["Feature", "Importance"],
    )
    .properties(title="Feature Importance in Random Forest Classifier")
)

chart_rf
# %%
important_features_dt = X_dt.columns[feature_importances_dt >= 0.015]
if "before1980" not in important_features_dt:
    important_features_dt = np.append(important_features_dt, "before1980")
if "yrbuilt" not in important_features_dt:
    important_features_dt = np.append(important_features_dt, "yrbuilt")
important_features_rf = X_rf.columns[feature_importances_rf >= 0.012]
if "before1980" not in important_features_rf:
    important_features_rf = np.append(important_features_rf, "before1980")
if "yrbuilt" not in important_features_rf:
    important_features_rf = np.append(important_features_rf, "yrbuilt")

print("Important Features for Decision Tree:")
print(important_features_dt)

print("Important Features for Random Forest:")
print(important_features_rf)

# %%
improved_dt = dwellings_ml[important_features_dt]
improved_dt = improved_dt.dropna()
le_dt = LabelEncoder()
improved_dt["before1980"] = le_dt.fit_transform(improved_dt["before1980"])
X_dt = improved_dt.drop(["before1980", "yrbuilt"], axis=1)
y_dt = improved_dt["before1980"]
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(
    X_dt, y_dt, test_size=0.2, random_state=42
)

clf_dt = DecisionTreeClassifier(random_state=42)

clf_dt.fit(X_train_dt, y_train_dt)

y_pred_dt = clf_dt.predict(X_test_dt)

accuracy_dt = accuracy_score(y_test_dt, y_pred_dt)
print(f"Improved Decision Tree Accuracy: {accuracy_dt * 100:.2f}%")
print("Decision Tree Classification Report:")
print(classification_report(y_test_dt, y_pred_dt))
# %%
improved_rf = dwellings_ml[important_features_rf]
improved_rf = improved_rf.dropna()

le_rf = LabelEncoder()
improved_rf["before1980"] = le_rf.fit_transform(improved_rf["before1980"])

X_rf = improved_rf.drop(["before1980", "yrbuilt"], axis=1)
y_rf = improved_rf["before1980"]

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=42
)
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)

clf_rf.fit(X_train_rf, y_train_rf)

y_pred_rf = clf_rf.predict(X_test_rf)

accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
print(f"Improved Random Forest Accuracy: {accuracy_rf * 100:.2f}%")

print("Random Forest Classification Report:")
print(classification_report(y_test_rf, y_pred_rf))
# %%
