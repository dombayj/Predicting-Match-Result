import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
matches = pd.read_csv("matches.csv", index_col=0)

# print(matches.head())


#Cleaning data types to be number not objects

matches["date"] = pd.to_datetime(matches["date"])

#converting "home" "away" venues to 1 or 0 type numbers to use it in ml
matches["venue"] = matches["venue"].astype("category").cat.codes

#doing samething but now we are doing for opponents
matches["opp_code"] = matches["opponent"].astype("category").cat.codes

#extracting hours from the data set for each match and changing its type to int
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")

#now we are encoding week days to numbers
matches["day_code"] = matches["date"].dt.day_of_week

#setting a target for our alghoritm in 2 axis
matches["target"] = (matches["result"] == "W").astype("int")


## Creating the machine learning model using random forest method
rf = RandomForestClassifier(n_estimators= 50, min_samples_split= 10, random_state= 1)

train = matches[matches["date"] < "2022-01-01"]
test = matches[matches["date"] >= "2022-01-01"]

#setting our predictors for our machine learning to understand data and then setting the target data
predictors = ["venue", "opp_code", "day_code", "hour"]
rf.fit(train[predictors], train["target"])

preds = rf.predict(test[predictors])

#playing with our parameters to increase accuracy
acc = accuracy_score(test["target"], preds)
# print(acc)

combined = pd.DataFrame(dict(actual=test["target"], prediction = preds))
# print(pd.crosstab(index=combined["actual"], columns=combined["prediction"]))

precision = precision_score(test["target"], preds)
# print(precision)

#creating a function to increase our alghoritm to predict, by the teams current condition related the previous weeks
grouped_mathces = matches.groupby("team")

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed = "left").mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset= new_cols)
    return group

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel("team")

#because some indicies are repeating themselves we need to make them unique
matches_rolling.index = range(matches_rolling.shape[0])

# print(matches_rolling)

## Now we can retrain our ml model with our new columns (yey)

def make_predictions(data, predictors):
    train = data[data["date"] < "2022-01-01"]
    test = data[data["date"] >= "2022-01-01"]
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], prediction = preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision

combined , precision = make_predictions(matches_rolling, predictors + new_cols) 

# print(precision)
# print(combined)

#because that we can not see clearly if we are misinterpreting a particular teams win in "combined" or not, we can make it more clear by merging(like inner join in sql table)
combined = combined.merge(matches_rolling[["date", "team", "result", "opponent"]], left_index=True, right_index=True)
# print(combined)

## In one match we actually can see 2 teams result like on side has won and other side has lost so we can join these two (with carefully checking if same team's name is same in opp column and team column)
# dont forget to consider missing values!
class MissingDict(dict):
    __missing__ = lambda self, key:key

map_values = {
    "Brighton and Hove Albion" : "Brighton",
    "Manchester United" : "Manchester Utd",
    "Newcastle United" : "Newcastle Utd",
    "Tottenham Hotspur" : "Tottenham",
    "West Ham United" : "West Ham",
    "Wolverhampton Wanderers" : "Wolves"
}
mapping = MissingDict(**map_values)

#And now we are updating our combined data set with our new and proper team names
combined["new_team"] = combined["team"].map(mapping)

merged = combined.merge(combined, left_on=["date", "new_team"] , right_on=["date", "opponent"])












