import streamlit as st
import json
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.formula.api import ols



@st.cache_data
def load_covid_modeldata():
    # Dataset opnieuw inladen en voorbereiden
    df = pd.read_csv('covid.csv')
    df['Datum'] = pd.to_datetime(df['Datum'], format='%Y-%m-%d')
    df['Vaccinatiegraad_volledig'] = df['Vaccinatiegraad_volledig'] / 100
    df = df.groupby('Datum').agg({
    'Besmettingen': 'sum',
    'Ziekenhuisopnames': 'sum',
    'Vaccinatiegraad_volledig': 'mean'
}).reset_index()

    # Toevoegen van indicatorvariabele voor lockdown
    lockdowns = [
        {"start": "2020-03-15", "end": "2020-06-01"},
        {"start": "2020-12-15", "end": "2021-04-28"},
        {"start": "2021-12-19", "end": "2022-01-14"}
    ]

    df['Lockdown'] = 0
    for lockdown in lockdowns:
        df.loc[(df['Datum'] >= pd.to_datetime(lockdown['start'])) & (df['Datum'] <= pd.to_datetime(lockdown['end'])), 'Lockdown'] = 1

    # Verplaatsen ziekenhuisopnames twee weken naar achteren
    df['Ziekenhuisopnames'] = df['Ziekenhuisopnames'].shift(-7)
    df.dropna(inplace=True)

    return df

@st.cache_data
def load_covid_mapdata():
    # Load COVID data
    mapdata = pd.read_csv('covid.csv')
    mapdata['Datum'] = pd.to_datetime(mapdata['Datum'])

    # Load GeoJSON data
    with open("gemeenten2023.geojson") as f:
        geojson = json.load(f)

    # Load population data
    inwonertal = pd.read_csv("inwonertal2023.csv")

    # Merge population data into COVID data
    mapdata = mapdata.merge(inwonertal, how='left', on='Gemeente')
    mapdata['Besmettingen_per_100000'] = (mapdata['Besmettingen'] / mapdata['Population']) * 100000
    mapdata['Ziekenhuisopnames_per_100000'] = (mapdata['Ziekenhuisopnames'] / mapdata['Population']) * 100000

    return mapdata, geojson

# Load data using the cached function
mapdata, geojson = load_covid_mapdata()

# Streamlit Sidebar UI for date selection
st.sidebar.header("Kaart opties")
start_date = st.sidebar.slider(
    "Selecteer een datum", 
    min_value=datetime(2020, 12, 28), 
    max_value=datetime(2023, 2, 28),  # Limit to ensure a full month can be selected
    value=datetime(2020, 12, 28)
)

# Define the end date as one month after the selected start date
end_date = start_date + timedelta(days=30)

# Filter the data for the selected month range
filtered_data = mapdata[(mapdata['Datum'] >= start_date) & (mapdata['Datum'] <= end_date)]

# Aggregate data for the selected month
monthly_data = filtered_data.groupby("Gemeente").agg({
    'Besmettingen': 'sum',
    'Ziekenhuisopnames': 'sum',
    'Vaccinatiegraad_deels': 'mean',
    'Vaccinatiegraad_volledig': 'mean',
    'Besmettingen_per_100000': 'sum',
    'Ziekenhuisopnames_per_100000': 'sum'
}).reset_index()

# Sidebar options for selecting color metric and log scale
st.sidebar.header("Variabele voor het kleuren van de kaart")
color_option = st.sidebar.radio(
    "Kies een variabele:",
    ('Besmettingen per 100,000', 'Ziekenhuisopnames per 100,000', 'Vaccinatiegraad (Volledig)')
)
log_scale = st.sidebar.checkbox("Logaritmische schaal")

# Set the color column and color scale based on the selected option
if color_option == 'Besmettingen per 100,000':
    color_column = 'Besmettingen_per_100000'
    color_scale = "YlOrRd"
elif color_option == 'Ziekenhuisopnames per 100,000':
    color_column = 'Ziekenhuisopnames_per_100000'
    color_scale = "Blues"
else:
    color_column = 'Vaccinatiegraad_volledig'
    color_scale = "Greens"

# Apply log scale if selected
if log_scale:
    monthly_data[color_column] = monthly_data[color_column].apply(lambda x: x if x <= 0 else np.log(x + 1))

# Generate choropleth map with the selected color metric and additional hover data
fig = px.choropleth(
    monthly_data,
    geojson=geojson,
    locations='Gemeente',
    color=color_column,
    featureidkey="properties.statnaam",
    title=f"COVID-19 Data by {color_option} (Month starting {start_date.strftime('%Y-%m-%d')})",
    hover_data={
        "Besmettingen": True,
        "Besmettingen_per_100000": True,
        "Ziekenhuisopnames": True,
        "Ziekenhuisopnames_per_100000": True,
        "Vaccinatiegraad_deels": True,
        "Vaccinatiegraad_volledig": True
    },
    color_continuous_scale=color_scale,
    projection="mercator"  # Set the projection type to equirectangular
)

# Update geos for style and set larger dimensions
fig.update_geos(
    fitbounds="locations",
    visible=False
)
fig.update_layout(
    width=1300,  # Increased width
    height=900,  # Increased height
    margin={"r":0,"t":40,"l":0,"b":0}  # Adjusted margins for better view
)

# Display in Streamlit with the specified dimensions
st.plotly_chart(fig, use_container_width=False)

#####################
# MODEL 
#####################

modeldata = load_covid_modeldata()

model = ols('Ziekenhuisopnames ~ Besmettingen + Vaccinatiegraad_volledig', data=modeldata).fit()


# Modelparameters en samenvatting bekijken
st.write("Model Parameters:")
st.write(model.params)
st.write("Model Samenvatting:")
st.write(model.summary())

# Voorspellingen maken op basis van het model
explanatory_data = pd.DataFrame({
    'Besmettingen': np.linspace(modeldata['Besmettingen'].min(), modeldata['Besmettingen'].max(), 100),
    'Vaccinatiegraad_volledig': [0] * 100,
    #'Lockdown': [1] * 100  # Bijvoorbeeld: tijdens een lockdown
})

# Voorspelde ziekenhuisopnames toevoegen aan explanatory_data
explanatory_data['Ziekenhuisopnames'] = model.predict(explanatory_data)

# Resultaten tonen in Streamlit
st.write("Voorspellingen op basis van het model:")
st.write(explanatory_data)

# Residuen visualiseren
st.write("Residuen Plot:")
residuals = model.resid
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(residuals, kde=True, ax=ax)
plt.xlabel('Residuen')
plt.title('Verdeling van de Residuen')
st.pyplot(fig)

# Visualisatie van voorspellingen
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=explanatory_data, x='Besmettingen', y='Ziekenhuisopnames', ax=ax, label='Voorspelde Ziekenhuisopnames', color='blue')
plt.xlabel('Aantal Besmettingen')
plt.ylabel('Aantal Ziekenhuisopnames')
plt.title('Voorspelde Ziekenhuisopnames op basis van Besmettingen, Vaccinatiegraad, en Lockdown')
st.pyplot(fig)
