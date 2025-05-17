from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load and preprocess data once at startup
matches = pd.read_csv("matches.csv", index_col=0)
matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype("int")

# Rolling average columns
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0])

# Final feature columns
predictors = ["venue_code", "opp_code", "hour", "day_code"] + new_cols

# Train model
train = matches_rolling[matches_rolling["date"] < '2022-01-01']
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
rf.fit(train[predictors], train["target"])

# Handle inconsistent team names
class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)

# Get unique sorted team list for dropdowns
team_list = sorted(matches["team"].unique())

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        team = request.form['team']
        opponent = request.form['opponent']
        hour = int(request.form['hour'])
        venue = request.form['venue']

        team = mapping[team]
        opponent = mapping[opponent]

        try:
            latest_match = matches_rolling[matches_rolling["team"] == team].sort_values("date").iloc[-1]
            input_data = latest_match[new_cols].copy()

            venue_code = 1 if venue == 'home' else 0
            opp_code = matches["opponent"].astype("category").cat.categories.get_loc(opponent)
            day_code = 5  # Assume Saturday

            row = pd.DataFrame([{
                "venue_code": venue_code,
                "opp_code": opp_code,
                "hour": hour,
                "day_code": day_code,
                **input_data.to_dict()
            }])

            prediction = rf.predict(row)[0]
            result = "Win" if prediction == 1 else "Not Win"

        except Exception as e:
            result = "Insufficient data for prediction."

    return render_template('index.html', result=result, teams=team_list)

if __name__ == '__main__':
    app.run(debug=True)
