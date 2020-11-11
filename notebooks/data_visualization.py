# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import matplotlib.pyplot as plt
import xlrd

# Loading the data:

# +
# full_data = pd.read_csv("../data/aphid_ladybeetle.csv")
full_data = pd.read_excel("../data/aphid_ladybeetle.xls")

full_data
# -

# Retrieving data for aphids and labebeetles separetely:

aphid_data = full_data[full_data['variable'] == 'aphid']
ladybeetle_data = full_data[full_data['variable'] == 'ladybeetle']

# Creating time labels:

time_labels_dict = {
    0: 0.0,
    1: 1/4,
    2: 1.0,
    3: 2.0
}

# Plotting data for aphids:

# +
fig, ax = plt.subplots(1, 4, sharey=True, gridspec_kw={'hspace': 0})
fig.suptitle("Aphids population distribution")
y_min = 0
y_max = 650
for label, time_value in time_labels_dict.items():
    ax[label].plot(
        aphid_data[aphid_data.time == label].x.values, 
        aphid_data[aphid_data.time == label].density.values,
        "o",
        label=f'Time = {time_value}'
    )
    ax[label].set_xlabel('x')
    if label == 0:
        ax[label].set_ylabel("Aphid per ten stems")
    ax[label].set_title(f"Time = {time_value}")
    ax[label].set_xlim(left=0)
    ax[label].set_ylim([y_min, y_max])

plt.tight_layout()
plt.savefig("aphid_data.png", dpi=300)
plt.show()
# -

# Plotting ladybeetles data:

# +
fig, ax = plt.subplots(1, 4, sharey=True, gridspec_kw={'hspace': 0})
fig.suptitle("Ladybeetles population distribution")
y_min = 0
y_max = 12
for label, time_value in time_labels_dict.items():
    ax[label].plot(
        ladybeetle_data[ladybeetle_data.time == label].x.values, 
        ladybeetle_data[ladybeetle_data.time == label].density.values,
        "o",
        label=f'Time = {time_value}'
    )
    ax[label].set_xlabel('x')
    if label == 0:
        ax[label].set_ylabel("Aphid per ten stems")
    ax[label].set_title(f"Time = {time_value}")
    ax[label].set_xlim(left=0)
    ax[label].set_ylim([y_min, y_max])

plt.tight_layout()
plt.savefig("ladybeetle_data.png", dpi=300)
plt.show()
