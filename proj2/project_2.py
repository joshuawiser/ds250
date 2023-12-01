import pandas as pd
import altair as alt
import numpy as np
from scipy import stats
import json

data = "C:/Fall 23/ds250/proj2/flights_missing.json"
df = pd.read_json(data)
df_copy = df.copy()
# GQ 1
# used to figure out an issue I was having with a blank row pulling up.
df.query("airport_name.isna() or airport_name == 'NaN'")
df.query("airport_code.isna() or airport_code == 'NaN'")

airport_name_mapping = {
    "IAD": "Washington, DC: Washington Dulles International",
    "SLC": "Salt Lake City, UT: Salt Lake City International",
    "SAN": "San Diego, CA: San Diego International",
    "ORD": "Chicago, IL: Chicago O'Hare International",
    "DEN": "Denver, CO: Denver International",
    "ATL": "Atlanta, GA: Hartsfield-Jackson Atlanta International",
    "SFO": "San Francisco, CA: San Francisco International",
}


def replace_empty_strings(row):
    if row["airport_name"] == "":
        return airport_name_mapping.get(row["airport_code"], "")
    else:
        return row["airport_name"]


df["airport_name"] = df.apply(replace_empty_strings, axis=1)


filtered_df = df.filter(
    [
        "airport_name",
        "num_of_flights_total",
        "num_of_delays_total",
        "minutes_delayed_total",
    ]
)

aggregated_df = (
    filtered_df.groupby("airport_name")
    .agg(
        {
            "num_of_flights_total": "sum",
            "num_of_delays_total": "sum",
            "minutes_delayed_total": "sum",
        }
    )
    .reset_index()
)

aggregated_df


aggregated_df["percent_of_flights_delayed"] = (
    aggregated_df["num_of_delays_total"] / aggregated_df["num_of_flights_total"]
) * 100
aggregated_df["percent_of_flights_delayed"] = aggregated_df[
    "percent_of_flights_delayed"
].round(2)


aggregated_df["avg_delay_hours"] = (
    aggregated_df["minutes_delayed_total"] / aggregated_df["num_of_delays_total"]
) / 60
aggregated_df["avg_delay_hours"] = aggregated_df["avg_delay_hours"].round(2)

aggregated_df.sort_values("percent_of_flights_delayed")


# GQ 2

filtered_month_df = df.filter(
    [
        "airport_name",
        "month",
        "num_of_flights_total",
        "num_of_delays_total",
        "minutes_delayed_total",
    ]
)
filtered_month_df


aggregated_month_df = (
    filtered_month_df.groupby("month")
    .agg(
        {
            "num_of_flights_total": "sum",
            "num_of_delays_total": "sum",
            "minutes_delayed_total": "sum",
        }
    )
    .reset_index()
)

aggregated_month_df = aggregated_month_df.query('month != "n/a"')


aggregated_month_df["percent_of_flights_delayed"] = (
    aggregated_month_df["num_of_delays_total"]
    / aggregated_month_df["num_of_flights_total"]
) * 100
aggregated_month_df["percent_of_flights_delayed"] = aggregated_month_df[
    "percent_of_flights_delayed"
].round(2)

aggregated_month_df["avg_delay_hours"] = (
    aggregated_month_df["minutes_delayed_total"]
    / aggregated_month_df["num_of_delays_total"]
) / 60
aggregated_month_df["avg_delay_hours"] = aggregated_month_df["avg_delay_hours"].round(2)
aggregated_month_df.sort_values("percent_of_flights_delayed")

# GQ 3

df_without_missing = df.query("num_of_delays_late_aircraft != -999")

mean_late_aircraft = df_without_missing["num_of_delays_late_aircraft"].mean()
mean_late_aircraft = mean_late_aircraft.round(2)

df_total_weather = df.filter(
    [
        "airport_code",
        "month",
        "num_of_delays_total",
        "num_of_delays_late_aircraft",
        "num_of_delays_nas",
        "num_of_delays_weather",
    ]
)
df_total_weather["num_of_delays_late_aircraft"] = df_total_weather[
    "num_of_delays_late_aircraft"
].replace(-999, mean_late_aircraft)

df_total_weather["num_of_delays_late_new"] = (
    30 / 100 * df_total_weather["num_of_delays_late_aircraft"].round(0)
)


def calculate_num_of_delays_nas_new(row):
    if row["month"] in ["April", "May", "June", "July", "August"]:
        return 0.4 * row["num_of_delays_nas"]
    else:
        return 0.65 * row["num_of_delays_nas"]


df_total_weather["num_of_delays_nas_new"] = df_total_weather.apply(
    calculate_num_of_delays_nas_new, axis=1
)
df_total_weather["num_of_weather_delays_total"] = (
    df_total_weather["num_of_delays_late_new"]
    + df_total_weather["num_of_delays_nas_new"]
    + df_total_weather["num_of_delays_weather"]
)
df_total_weather["num_of_weather_delays_total"] = df_total_weather[
    "num_of_weather_delays_total"
].astype(int)
df_total_weather.head(5)

# GQ 4
filtered_total_weather = df_total_weather.filter(
    ["airport_code", "num_of_delays_total", "num_of_weather_delays_total"]
)

filtered_total_weather

agg_weather_df = (
    df_total_weather.groupby("airport_code")
    .agg({"num_of_delays_total": "sum", "num_of_weather_delays_total": "sum"})
    .reset_index()
)

agg_weather_df


weather_chart = (
    alt.Chart(agg_weather_df)
    .mark_bar()
    .encode(
        x=alt.X("airport_code:N"),
        y=alt.Y(alt.repeat("layer"))
        .aggregate("mean")
        .title("Mean of US and Worldwide Gross"),
        color=alt.ColorDatum(alt.repeat("layer")),
    )
    .repeat(layer=["num_of_delays_total", "num_of_weather_delays_total"])
)

weather_chart


# GQ 5


df_copy.replace("", "NaN", inplace=True)

df_copy.to_json("modified_file.json", orient="records", lines=True)


row_to_print = df_copy.iloc[25].to_dict()

json_data = json.dumps(row_to_print)
print(json_data)
