import streamlit as st
import json
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from statsmodels.formula.api import ols

@st.cache_data
def load_variant_data():
    data = pd.read_csv('COVID-19_varianten.csv', delimiter=';')
    data['Date_of_statistics_week_start'] = pd.to_datetime(data['Date_of_statistics_week_start'])
    data = data[(data['Date_of_statistics_week_start'] >= "2020-12-28") & 
                (data['Date_of_statistics_week_start'] <= "2023-03-31")]

    # Aggregate variant prevalence data
    variant_prevalence = data.groupby(['Date_of_statistics_week_start', 'Variant_name'])['Variant_cases'].sum().reset_index()
    
    # Calculate relative prevalence by week
    total_cases_per_week = variant_prevalence.groupby('Date_of_statistics_week_start')['Variant_cases'].transform('sum')
    variant_prevalence['Relative_Prevalence'] = (variant_prevalence['Variant_cases'] / total_cases_per_week) * 100

    return data, variant_prevalence

@st.cache_data
def load_covid_modeldata():
    df = pd.read_csv('covid.csv')
    df['Datum'] = pd.to_datetime(df['Datum'], format='%Y-%m-%d')
    df = df.groupby('Datum').agg({
        'Besmettingen': 'sum',
        'Ziekenhuisopnames': 'sum',
        'Vaccinatiegraad_volledig': 'mean'
    }).reset_index()

    lockdowns = [
        {"start": "2020-03-15", "end": "2020-06-01"},
        {"start": "2020-12-15", "end": "2021-04-28"},
        {"start": "2021-12-19", "end": "2022-01-14"}
    ]

    df['Lockdown'] = 0
    for lockdown in lockdowns:
        df.loc[(df['Datum'] >= pd.to_datetime(lockdown['start'])) & (df['Datum'] <= pd.to_datetime(lockdown['end'])), 'Lockdown'] = 1

    df['Ziekenhuisopnames'] = df['Ziekenhuisopnames'].shift(-7)
    df.dropna(inplace=True)

    return df

@st.cache_data
def load_covid_mapdata():
    mapdata = pd.read_csv('covid.csv')
    mapdata['Datum'] = pd.to_datetime(mapdata['Datum'])

    with open("gemeenten2023.geojson") as f:
        geojson = json.load(f)

    inwonertal = pd.read_csv("inwonertal2023.csv")
    mapdata = mapdata.merge(inwonertal, how='left', on='Gemeente')
    mapdata['Besmettingen_per_100000'] = (mapdata['Besmettingen'] / mapdata['Population']) * 100000
    mapdata['Ziekenhuisopnames_per_100000'] = (mapdata['Ziekenhuisopnames'] / mapdata['Population']) * 100000

    return mapdata, geojson

# Load data using the cached function
mapdata, geojson = load_covid_mapdata()

# Streamlit Sidebar UI for date selection
start_date = st.sidebar.slider(
    "Selecteer een datum", 
    min_value=datetime(2020, 12, 28), 
    max_value=datetime(2023, 2, 28) - timedelta(days=6),  # Limit to ensure a full week can be selected
    value=datetime(2020, 12, 28),
    step=timedelta(days=7)
)
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

if color_option == 'Besmettingen per 100,000':
    color_column = 'Besmettingen_per_100000'
    color_scale = "YlOrRd"
elif color_option == 'Ziekenhuisopnames per 100,000':
    color_column = 'Ziekenhuisopnames_per_100000'
    color_scale = "Blues"
else:
    color_column = 'Vaccinatiegraad_volledig'
    color_scale = "Greens"

if log_scale:
    monthly_data[color_column] = monthly_data[color_column].apply(lambda x: x if x <= 0 else np.log(x + 1))

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
    projection="mercator"
)

fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(width=1300, height=900, margin={"r":0,"t":40,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=False)

#####################
# MODEL 
#####################

modeldata = load_covid_modeldata()
variant_data, variant_prevalence = load_variant_data()

# Variant prevalence periods
variant_periods = {
    "Alpha": ("2020-12-28", "2021-06-21"),
    "Delta": ("2021-06-28", "2021-12-20"),
    "Omicron": ("2021-12-27", "2023-03-27")
}

# Variant selection in sidebar
st.sidebar.header("Model Instellingen")
variant_option = st.sidebar.selectbox("Kies een variant", list(variant_periods.keys()))

# Filter 'modeldata' based on the selected variant's time period
start_date, end_date = variant_periods[variant_option]
modeldata = modeldata[(modeldata['Datum'] >= start_date) & 
                      (modeldata['Datum'] <= end_date)]

# Fit the model using the filtered 'modeldata'
model = ols('Ziekenhuisopnames ~ Besmettingen + Vaccinatiegraad_volledig', data=modeldata).fit()

# Show R-squared value
st.write("Model R-squared:", model.rsquared)

# Plotly graph for relative variant prevalence
fig = px.line(variant_prevalence, x='Date_of_statistics_week_start', y='Relative_Prevalence', color='Variant_name',
              title="Relative Prevalence of COVID-19 Variants Over Time")
fig.update_layout(xaxis_title="Date", yaxis_title="Relative Prevalence (%)", legend_title="Variant Name")

# Add shaded areas for dominance periods
for variant, (start, end) in variant_periods.items():
    fig.add_vrect(x0=start, x1=end, fillcolor="LightSalmon" if variant == variant_option else "LightGray", opacity=0.3, line_width=0)

st.plotly_chart(fig, use_container_width=True)

# Additional UI elements for predictions
st.sidebar.write("Set inputs for predictions")
vaccinatiegraad_input = st.sidebar.number_input("Vaccinatiegraad (Volledig):", min_value=0, max_value=100, value=50, step=1)
besmettingen_input = st.sidebar.slider("Aantal Besmettingen:", min_value=int(modeldata['Besmettingen'].min()), max_value=int(modeldata['Besmettingen'].max()), value=int(modeldata['Besmettingen'].median()), step=10)

# Prepare data for prediction
input_data = pd.DataFrame({'Besmettingen': [besmettingen_input], 'Vaccinatiegraad_volledig': [vaccinatiegraad_input]})
predicted_ziekenhuisopnames = model.predict(input_data)[0]

# Display the prediction result
st.write("Verwachte Aantal Ziekenhuisopnames:", round(predicted_ziekenhuisopnames))
