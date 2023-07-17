import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

JSON_FILE = "../resources/data.json"

TRANSLATIONS = {
    "dt": "Decision Trees Classifier",
    "rf": "Random Forest Classifier",
    "et": "Extra Trees Classifier",
}


def translate_model(model: str) -> str:
    """
    Translates a two-letter string to the full classifier model name.

    Args:
        model (str): A two-letter representation of a classifier model.

    Returns:
        str: The full name of the classifier model.

    Raises:
        KeyError: If the two-letter model name is not recognized.
    """
    if len(model) != 2:
        return model

    return TRANSLATIONS[model]


def format_tick(value, tick_number):
    """
    Formats tick values for plotting large numbers in the 'k' (thousands) format.

    Args:
        value (int or float): The original tick value.
        tick_number (int): The tick number (not used in this function but included for completeness).

    Returns:
        str: A string representation of the tick value in 'k' format.
    """
    if value == 0:
        return f"1k"  # Since we start at 1000

    return f"{value * 5}k"


# This script reads a JSON file line by line, parses each line into a dictionary,
# translates short model names to full names, and appends dictionary entries to a list.
# The result is a list of dictionaries, each containing details about a model's performance on a given task.

# Assume the json_data variable is your loaded JSON data
df_list = []
with open(JSON_FILE) as json_file:
    for row in json_file.readlines():
        json_data = json.loads(row)

        for model, entries in json_data.items():
            for entry in entries:
                model = translate_model(model)
                df_list.append(
                    {
                        "Model": model,
                        "Sample amount": entry["sample_amount"],
                        "Secret length": entry["dim"],
                        "# Secrets Found": entry["n_success"],
                        "# Failures": entry["n_failure"],
                        "Error rate (%)": entry["error_rate"] * 100,
                        "Duration": entry["total_duration"],
                    }
                )

df = pd.DataFrame(df_list)
# sns.lmplot(x="Secret length", y="Duration", hue="Model", data=df)
# sns.lmplot(x="Secret length", y="# Secrets Found", hue="Model", data=df)
# sns.lmplot(x="Sample amount", y="Duration", hue="Model", data=df)
# sns.lmplot(x="Sample amount", y="# Secrets Found", hue="Model", data=df)
# sns.lmplot(x="Error rate (%)", y="# Secrets Found", hue="Model", data=df)

# Barplots
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Secret length", y="Duration", hue="Model", data=df, errorbar=None)
ax.set_xticks(range(0, 19))
ax.set_xticklabels(range(2, 21))
fig.show()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Secret length", y="# Secrets Found", hue="Model", data=df, errorbar=None)
ax.set_xticks(range(0, 19))
ax.set_xticklabels(range(2, 21))
fig.show()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Sample amount", y="Duration", hue="Model", data=df, errorbar=None)
ax.xaxis.set_major_formatter(FuncFormatter(format_tick))
fig.show()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Sample amount", y="# Secrets Found", hue="Model", data=df, errorbar=None)
ax.xaxis.set_major_formatter(FuncFormatter(format_tick))
fig.show()

plt.figure(figsize=(10, 6))
sns.barplot(
    x="Error rate (%)", y="# Secrets Found", hue="Model", data=df, errorbar=None
)
plt.show()
