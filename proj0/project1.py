import pandas as pd
import altair as alt

url = (
    "https://github.com/byuidatascience/data4python4ds/raw/master/data-raw/mpg/mpg.csv"
)
mpg = pd.read_csv(url)

chart = alt.Chart(mpg).encode(x="displ", y="hwy").mark_circle()
chart

print(
    mpg.head(5)
    .filter(["manufacturer", "model", "year", "hwy"])
    .to_markdown(index=False)
)
