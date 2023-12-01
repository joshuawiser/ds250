# %%
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the data
dwellings_ml = pd.read_csv(
    "https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv"
)

# Select relevant features
features = [
    "basement",
    "livearea",
    "numbdrm",
    "numbaths",
    "nocars",
    "sprice",
    "stories",
    "condition_AVG",
    "quality_C",
    "gartype_Att",
    "arcstyle_END UNIT",
    "arcstyle_MIDDLE UNIT",
    "arcstyle_ONE AND HALF-STORY",
    "arcstyle_ONE-STORY",
    "arcstyle_TRI-LEVEL WITH BASEMENT",
    "qualified_U",
    "status_V",
    "before1980",
    "yrbuilt",
    "smonth",
]

# Create a subset for Decision Tree
subset_dt = dwellings_ml[features]
subset_dt = subset_dt.dropna()

# Encode categorical variables for Decision Tree
le_dt = LabelEncoder()
subset_dt["basement"] = le_dt.fit_transform(subset_dt["basement"])
subset_dt["before1980"] = le_dt.fit_transform(subset_dt["before1980"])

# Split the data into features (X) and target variable (y) for Decision Tree
X_dt = subset_dt.drop(["before1980", "yrbuilt"], axis=1)
y_dt = subset_dt["before1980"]

# Split the data into training and testing sets for Decision Tree
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(
    X_dt, y_dt, test_size=0.2, random_state=42
)

# Initialize the Decision Tree classifier
clf_dt = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree classifier
clf_dt.fit(X_train_dt, y_train_dt)

# Make predictions on the test set for Decision Tree
y_pred_dt = clf_dt.predict(X_test_dt)

# Evaluate the Decision Tree model
accuracy_dt = accuracy_score(y_test_dt, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt * 100:.2f}%")
print("Decision Tree Classification Report:")
print(classification_report(y_test_dt, y_pred_dt))

# %%
# Extract feature importances for Decision Tree
feature_importances_dt = clf_dt.feature_importances_

# Create a DataFrame for plotting for Decision Tree
feature_importance_df_dt = pd.DataFrame(
    {"Feature": X_dt.columns, "Importance": feature_importances_dt}
)

# Sort the DataFrame by importance in descending order for Decision Tree
feature_importance_df_dt = feature_importance_df_dt.sort_values(
    by="Importance", ascending=False
)

# Create a horizontal bar chart using Altair for Decision Tree
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

# Display the chart for Decision Tree
chart_dt

# %%
# Create a subset for Random Forest
subset_rf = dwellings_ml[features]
subset_rf = subset_rf.dropna()

# Encode categorical variables for Random Forest
le_rf = LabelEncoder()
subset_rf["basement"] = le_rf.fit_transform(subset_rf["basement"])
subset_rf["before1980"] = le_rf.fit_transform(subset_rf["before1980"])

# Split the data into features (X) and target variable (y) for Random Forest
X_rf = subset_rf.drop(["before1980", "yrbuilt"], axis=1)
y_rf = subset_rf["before1980"]

# Split the data into training and testing sets for Random Forest
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=42
)

# Initialize the Random Forest classifier
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier
clf_rf.fit(X_train_rf, y_train_rf)

# Make predictions on the test set for Random Forest
y_pred_rf = clf_rf.predict(X_test_rf)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")

# Display Random Forest classification report
print("Random Forest Classification Report:")
print(classification_report(y_test_rf, y_pred_rf))

# %%
# Extract feature importances for Random Forest
feature_importances_rf = clf_rf.feature_importances_

# Create a DataFrame for plotting for Random Forest
feature_importance_df_rf = pd.DataFrame(
    {"Feature": X_rf.columns, "Importance": feature_importances_rf}
)

# Sort the DataFrame by importance in descending order for Random Forest
feature_importance_df_rf = feature_importance_df_rf.sort_values(
    by="Importance", ascending=False
)

# Create a horizontal bar chart using Altair for Random Forest
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

# Display the chart for Random Forest
chart_rf
# %%
# Use train_test_split() to create testing and training data
X_train, X_test, y_train, y_test = train_test_split(
    X_dt, y_dt, test_size=0.34, random_state=76
)

# Calculate the average of the first 10 values in testing y values
average_first_10_values = y_test[:10].mean()

print(f"Average of the first 10 values in testing y: {average_first_10_values}")

# %%
# Use train_test_split() to create testing and training data
X_train, X_test, y_train, y_test = train_test_split(
    X_dt, y_dt, test_size=0.34, random_state=76
)

# Calculate the average of the first 10 values in training X values for sprice
average_first_10_sprice_train = X_train["sprice"][:10].mean()

print(
    f"Average of the first 10 values in training X for sprice: {average_first_10_sprice_train}"
)

# %%
