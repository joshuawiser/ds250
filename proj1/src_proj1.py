import pandas as pd
import altair as alt
import numpy as np
import datetime as dt

url = "https://raw.githubusercontent.com/byuidatascience/data4names/master/data-raw/names_year/names_year.csv"

df = pd.read_csv(url)

name = "Joshua"
name_df = df[df["name"] == name]


chart = (
    alt.Chart(name_df, title="Popularity of the name Joshua")
    .encode(
        alt.X(
            "year:N",
            axis=alt.Axis(values=np.arange(1900, 2020, step=5).tolist()),
            scale=alt.Scale(zero=False),
        ),
        alt.Y("Total"),
        color=alt.condition(
            alt.datum.year == 2000, alt.value("black"), alt.value("#0077b6")
        ),
    )
    .mark_bar()
    .properties(width=600, height=400)
)


chart


filtered_df = df[(df["name"] == "Joshua") & (df["year"] >= 1995) & (df["year"] <= 2015)]

# Calculate the total number of people named Joshua for each year
total_by_year = filtered_df.groupby("year")["Total"].sum().reset_index()

# Print the resulting table
total_by_year


current_year = dt.datetime.now().year
name_2 = "Brittany"
name_2_df = df[df["name"] == name_2].copy()
name_2_df.loc[:, "age"] = (
    current_year - name_2_df["year"]
)  # Use .loc to create 'age' column

# Create a chart for the subset DataFrame
chart2 = (
    alt.Chart(name_2_df, title="Brittany age chart")
    .encode(
        alt.X(
            "age:N",
            axis=alt.Axis(values=np.arange(0, 100).tolist()),
            scale=alt.Scale(zero=False),
        ),
        alt.Y("Total"),
        color="Total",
    )
    .mark_bar()
    .properties(width=600, height=400)
)

# Display the chart for the subset DataFrame
chart2


name1 = "Peter"
name2 = "Mary"
name3 = "Martha"
name4 = "Paul"

names_df = df[df["name"].isin([name1, name2, name3, name4])]

names_df

names_df = names_df[names_df["year"] <= 2000]

chart3 = (
    alt.Chart(names_df, title="Mary, Martha, Peter and Paul")
    .encode(
        alt.X(
            "year:N",
            axis=alt.Axis(values=np.arange(1910, 2005, step=5).tolist()),
            scale=alt.Scale(zero=False),
        ),
        alt.Y("Total"),
        color="name",
    )
    .mark_line()
    .properties(width=600, height=400)
)
chart3

star_wars_name = "Luke"
sw_df = df[df["name"] == star_wars_name]
sw_df

chart4 = (
    alt.Chart(sw_df, title="Star Wars Effect on the Name Luke")
    .encode(
        alt.X(
            "year:N",
            axis=alt.Axis(values=np.arange(1900, 2020, step=5).tolist()),
            scale=alt.Scale(zero=False),
        ),
        alt.Y("Total"),
        color="Total",
    )
    .mark_bar()
    .properties(width=600, height=400)
)
chart4


name_3 = "Oliver"
name_3_df = df[df["name"] == name_3].copy()
utah_total_df = name_3_df[["UT", "Total"]]

total_olivers_in_utah = utah_total_df["UT"].sum()
total_olivers_in_utah
