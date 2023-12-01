import pandas as pd
import altair as alt
import numpy as np

url = "https://raw.githubusercontent.com/byuidatascience/data4names/master/data-raw/names_year/names_year.csv"

df = pd.read_csv(url)

name = "Joshua"
name_df = df[df['name'] == name]  # Correct the column name to 'Name'

chart = (alt.Chart(name_df,
                  title = "Popularity of the name Joshua")
                .encode(alt.X('year', axis = alt.Axis(
                    values = np.arange(1900,2020, step = 5).tolist()),
                    scale = alt.Scale(zero = False)),
                        alt.Y('Total'),
                        color = 'Total')
                .mark_bar()
                .properties(
                    width = 600,
                    height = 400
                ))
chart