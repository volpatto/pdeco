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

# # Data preprocessing and visualization

# Import required packages:

import pandas as pd
import matplotlib.pyplot as plt

# ## Reading data

# Loading the raw data:

# +
full_data = pd.read_excel("../data/aphid_ladybeetle_raw.xls")

full_data
# -

# Retrieving raw data for aphids and labebeetles separetely:

aphid_data = full_data[full_data['variable'] == 'aphid'].copy()
ladybeetle_data = full_data[full_data['variable'] == 'ladybeetle'].copy()
ladybeetle_data.reset_index(drop=True, inplace=True)

aphid_data

ladybeetle_data

# Fix spatial space between measurement for aphids:

# +
aphid_data = aphid_data.round({'x': 0})

aphid_data
# -

# Fix spatial space between measurement for ladybird beetles:

# +
ladybeetle_data= ladybeetle_data.round({'x': 0})

ladybeetle_data
# -

# Note that some duplicated values are present, as well as the number of data for ladybird are 33 rows, while for aphids are 32. We should drop one of them. 
#
# After a looking in the original ladybird beetle data for time 1, the entry corresponding to line 10 is between 2 and 3 in `x`, so this datum looks like as an additional measurement.

# +
ladybeetle_data.drop(ladybeetle_data.index[10], inplace=True)
ladybeetle_data.reset_index(drop=True, inplace=True)

ladybeetle_data
# -

# Row 20 is wrongly truncated. Let's fix it.

ladybeetle_data.iloc[20, 2] = 5.0

ladybeetle_data

# Now everything looks OK. Now let's write our new data to use in the simulations.

aphid_data.to_csv("../data/aphid.csv", index=False)
ladybeetle_data.to_csv("../data/ladybeetle.csv", index=False)

# ## Data visualization

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
x_limits = [0, 9]
y_limits = [0, 650]
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
    ax[label].set_xlim(x_limits)
    ax[label].set_ylim(y_limits)

plt.tight_layout()
plt.savefig("aphid_data.png", dpi=300)
plt.show()
# -

# Plotting ladybeetles data:

# +
fig, ax = plt.subplots(1, 4, sharey=True, gridspec_kw={'hspace': 0})
fig.suptitle("Ladybeetles population distribution")
x_limits = [0, 9]
y_limits = [0, 13]
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
    ax[label].set_xlim(x_limits)
    ax[label].set_ylim(y_limits)

plt.tight_layout()
plt.savefig("ladybeetle_data.png", dpi=300)
plt.show()
