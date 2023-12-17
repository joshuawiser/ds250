import pandas as pd
import altair as alt
import streamlit as st

file = "C:/Fall 23/ds250/agent_state_project/Agent State Summary by State 231130_192802.csv"
df = pd.read_csv(file)

df.fillna("00:00:00", inplace=True)
df = df.drop(columns={"AGENT GROUP"})
df["AGENT"] = df["AGENT FIRST NAME"] + " " + df["AGENT LAST NAME"]
df = df.drop(columns={"AGENT FIRST NAME", "AGENT LAST NAME"})
df = df.rename(
    columns={
        "AGENT": "agent",
        "After Call Work / AGENT STATE TIME": "acw",
        "Not Ready / AGENT STATE TIME": "not ready",
        "On Call / AGENT STATE TIME": "on call",
        "On Preview / AGENT STATE TIME": "on preview",
        "Ready / AGENT STATE TIME": "ready",
        "Ringing / AGENT STATE TIME": "ringing",
    }
)

df["acw"] = pd.to_timedelta(df["acw"]).dt.total_seconds() / 60  # Convert to minutes
df["acw"] = df["acw"].astype(float)  # Explicitly convert to float

st.set_page_config(layout="wide")
st.header("Agent Reason Code Summary")

agent_list = df["agent"].tolist()
agent_list.insert(0, "All")
agent_list.insert(1, "Team Josh")

agent_selection = st.multiselect(
    "select",
    agent_list,
    placeholder="Select Agent(s) or Team(s)",
    label_visibility="hidden",
)
team_josh = ["Joshua Wiser", "Kyle Rasmussen", "Josh Smith"]
if "All" in agent_selection:
    select_df = df.copy()
elif "Team Josh" in agent_selection:
    select_df = df[df["agent"].isin(team_josh)]
else:
    select_df = df[df["agent"].isin(agent_selection)]

st.dataframe(select_df, use_container_width=True, hide_index=True)
st.bar_chart(
    select_df, x="agent", y="acw", use_container_width=False, width=750, height=500
)
