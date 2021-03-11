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
full_data = pd.read_csv(
    "../data/2017 Lin and Pennings/1_filtered_location/dike/INS-GCET-1608_Iva_2014_1_1.CSV",
    header=0,
    skiprows=[1, 2],
    parse_dates=True
)

full_data
# -

# ## Data visualization

# +
fig, ax = plt.subplots(figsize=(9, 6))

width = 0.8
full_data.plot(
    x="Date", 
    y=["Uroleucon", "Cycloneda_sanguinea"], 
    secondary_y= 'Cycloneda_sanguinea', 
    ax=ax,
    kind="bar", 
    width=width
)

ax.set_ylabel('Uroleucon')
ax.right_ax.set_ylabel('Cycloneda sanguinea')

plt.tight_layout()
plt.savefig("dike_2014", dpi=300)
plt.show()
