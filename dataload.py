# The dependencies
import numpy as np
import matplotlib.pyplot as plt
import requests
import pandas as pd
from IPython.display import display
from io import StringIO
from dstapi import DstApi # The helper class
import math
import numpy as np
import os
import io, urllib.request

# Kode til at sætte session verify = False for at bypass vpn SSL
old_request = requests.Session.request
def new_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return old_request(self, *args, **kwargs)
requests.Session.request = new_request

params = {
    'table': 'pris113',
    'format': 'BULK',
    'variables': [
        {'code': 'Tid', 'values': ['*']},
        {'code': 'TYPE', 'values': ['INDEKS']}
    ]
}
r = requests.post('https://api.statbank.dk/v1' + '/data', json=params)
print(r.text[:200])
pd.read_table(StringIO(r.text), sep=';').head()
org_dataset = pd.read_table(StringIO(r.text), sep=';')

cpi = org_dataset.copy()
cpi['TID'] = pd.to_datetime(cpi['TID'].str.replace('M',''), format='%Y%m').dt.to_period('M')
cpi = cpi.sort_values(by='TID', ascending=True).reset_index(drop=True)
cpi['INDHOLD'] = cpi['INDHOLD'].str.replace(',', '.')
cpi['Inflation (level)'] = pd.to_numeric(cpi['INDHOLD'])
cpijan2020 = cpi.loc[cpi['TID'] == pd.Period('2020-01', freq='M'), 'Inflation (level)'].values[0]
cpi['Inflation (level 2020=100)'] = cpi['INDHOLD'] / cpijan2020 * 100
cpi['Inflation (Monthly)'] = cpi['Inflation (level 2020=100)'].pct_change(periods=1)
cpi['Inflation (YoY)'] = cpi['nflation (level 2020=100)'].pct_change(periods=12)

cpi = cpi.melt(
    id_vars=['TID'],
    value_vars=['INDHOLD', 'Inflation (level)', 'Inflation(Monthly)', 'Inflation (YoY)'],
    var_name='Variable',
    value_name='Value'
).sort_values(['TID', 'Variable']).reset_index(drop=True)
display(cpi.head(12))

# Save cpi as a csv in the same folder as this script
csv_path = os.path.join(os.path.dirname(__file__), "cpi.csv")
cpi.to_csv(csv_path, index=False)




##################### Plotting (optional) #####################

# cpi['TID'] = cpi['TID'].dt.to_timestamp()
# år = cpi['TID'].dt.year
# unique_years = sorted(år.unique())
# num_years = len(unique_years)
# ticks = [cpi['TID'][år == y].iloc[0] for y in unique_years]

# plt.figure(figsize=(12,6))
# plt.plot(cpi['TID'], cpi['2020indeks'], label='CPI (2020=100)', color='blue')
# plt.title('Forbrugerprisindekts (2020=100)')
# plt.xlabel('Tid')
# plt.ylabel('Fobrugerprisindeks (2020=100)')
# plt.grid(True, axis='both', linestyle='--', alpha=0.7)
# plt.xticks(
#     ticks=ticks,
#     labels=[str(y) for y in unique_years],
#     rotation=45
# )
# plt.show()


# plt.figure(figsize=(12, 6))
# plt.plot(cpi['TID'], cpi['pi_t']*100, label='Month-to-Month Inflation', color='orange')
# plt.title('Inflationsrate Måned-til-Måned')
# plt.xlabel('Tid')
# plt.ylabel('Inflationsrate (%)')
# plt.grid(True, axis='both', linestyle='--', alpha=0.7)
# plt.xticks(
#     ticks=ticks,
#     labels=[str(y) for y in unique_years],
#     rotation=45
# )
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(cpi['TID'], cpi['pi^12_t']*100, label='Year-to-Year Inflation', color='green')
# plt.title('Inflationsrate År-Til-År')
# plt.xlabel('Tid')
# plt.ylabel('Inflationsrate YoY (%)')
# plt.grid(True, axis='both', linestyle='--', alpha=0.7)
# plt.xticks(
#     ticks=ticks,
#     labels=[str(y) for y in unique_years],
#     rotation=45
# )
# plt.show()