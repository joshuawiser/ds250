import pandas as pd
import altair as alt
import numpy as np
import sqlite3
import datarobot as dr

sqlite_file = "C:\Fall 23\ds250\proj3\lahmansbaseballdb.sqlite"
con = sqlite3.connect(sqlite_file)

results = pd.read_sql_query("SELECT * FROM teams", con)
results

table = pd.read_sql_query("SELECT * FROM sqlite_master WHERE type='table'", con)
print(table.filter(["name"]))
print("\n\n")
# 8 is collegeplaying
print(table.sql[24])

# GQ 1

question_one_query = """
    SELECT
        p.playerID,
        cp.schoolID,
        s.salary,
        s.yearID,
        s.teamID
    FROM
        people p
        JOIN collegeplaying cp ON p.playerID = cp.playerID
        JOIN salaries s ON p.playerID = s.playerID
    WHERE
        cp.schoolID = 'byu'
    ORDER BY
        s.salary DESC;
"""
df = pd.read_sql_query(question_one_query, con)
print(df)

# GQ 2

part_one_query = """
    SELECT
        b.playerID,
        b.yearID,
        CAST(SUM(b.H) AS REAL) / SUM(b.AB) AS batting_average
    FROM
        batting b
    GROUP BY
        b.playerID, b.yearID
    HAVING
        SUM(b.AB) >= 1
    ORDER BY
        batting_average DESC, b.playerID
    LIMIT 5;
"""

df1 = pd.read_sql_query(part_one_query, con)
df1


part_two_query = """
    SELECT
        b.playerID,
        b.yearID,
        CAST(SUM(b.H) AS REAL) / SUM(b.AB) AS batting_average
    FROM
        batting b
    GROUP BY
        b.playerID, b.yearID
    HAVING
        SUM(b.AB) >= 10
    ORDER BY
        batting_average DESC, b.playerID
    LIMIT 5;
"""

df2 = pd.read_sql_query(part_two_query, con)
df2


part_three_query = """
    SELECT
        b.playerID,
        CAST(SUM(b.H) AS REAL) / SUM(b.AB) AS career_batting_average
    FROM
        batting b
    GROUP BY
        b.playerID
    HAVING
        SUM(b.AB) >= 100
    ORDER BY
        career_batting_average DESC
    LIMIT 5;
"""

df3 = pd.read_sql_query(part_three_query, con)
df3

# GQ 3

q_three_query = """
SELECT
    t.teamID,
    t.name AS team_name,
    COUNT(*) AS num_top_5_rank
FROM
    teams t
WHERE
    t.teamID IN ('BOS', 'NYA') 
    AND (t.name = 'Boston Red Sox' OR t.name = 'New York Yankees')
    AND t.teamRank <= 5
GROUP BY
    t.teamID, team_name;


"""

df4 = pd.read_sql_query(q_three_query, con)
df4

chart = (
    alt.Chart(df4)
    .mark_bar()
    .encode(x="team_name", y="num_top_5_rank", color="team_name")
    .properties(title="Frequency of Top 5 Rankings Comparison")
)


chart


quiz_query = """SELECT
    b.playerID,
    b.yearID,
    b.AB AS at_bats,
    b.H AS hits
FROM
    batting b
WHERE
    b.playerID = 'addybo01';
"""

df5 = pd.read_sql_query(quiz_query, con)
df5
