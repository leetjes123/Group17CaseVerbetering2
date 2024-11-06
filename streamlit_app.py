import streamlit as st
import json
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from statsmodels.formula.api import ols


##################
# DATA FUNCTIONS #
##################

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



# Define tabs
st.set_page_config(layout="wide")
tab1, tab2 = st.tabs(["COVID Analyse", "Verkeersdata Analyse"])

# Main Dashboard Code
with tab1:
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
    st.plotly_chart(fig, use_container_width=False)

    #####################
    # MODEL 
    #####################

    # Laad gegevens voor het model
    modeldata = load_covid_modeldata()
    variant_data, variant_prevalence = load_variant_data()

    # Periodes van variantprevalentie
    variant_periods = {
        "Alpha": ("2020-12-28", "2021-06-21"),
        "Delta": ("2021-06-28", "2021-12-20"),
        "Omicron": ("2021-12-27", "2023-03-27")
    }

    # Variantselectie in de zijbalk
    st.sidebar.header("Modelinstellingen")
    variant_option = st.sidebar.selectbox("Kies een variant", list(variant_periods.keys()))

    # Filter 'modeldata' op basis van de geselecteerde variantperiode
    start_date, end_date = variant_periods[variant_option]
    modeldata = modeldata[(modeldata['Datum'] >= start_date) & 
                        (modeldata['Datum'] <= end_date)]

    # Pas het model toe op de gefilterde 'modeldata'
    model = ols('Ziekenhuisopnames ~ Besmettingen + Vaccinatiegraad_volledig', data=modeldata).fit()

    # R-kwadraat waarde weergeven
    st.write("Model R-kwadraat:", model.rsquared)

    # Plotly grafiek voor relatieve variantprevalentie
    fig = px.line(variant_prevalence, x='Date_of_statistics_week_start', y='Relative_Prevalence', color='Variant_name',
                title="Relatieve prevalentie van COVID-19-varianten in de tijd")
    fig.update_layout(xaxis_title="Datum", yaxis_title="Relatieve prevalentie (%)", legend_title="Variantnaam")

    # Toevoegen van gearceerde gebieden voor dominantieperiodes
    for variant, (start, end) in variant_periods.items():
        fig.add_vrect(x0=start, x1=end, fillcolor="LightSalmon" if variant == variant_option else "LightGray", opacity=0.3, line_width=0)

    # De prevalentie grafiek weergeven
    st.plotly_chart(fig, use_container_width=True)

    # Zijbalk invoer voor voorspellingen
    st.sidebar.write("Stel invoer in voor voorspellingen")
    vaccinatiegraad_input = st.sidebar.number_input("Vaccinatiegraad (Volledig):", min_value=0, max_value=100, value=50, step=1)
    besmettingen_input = st.sidebar.slider("Aantal Besmettingen:", min_value=int(0), max_value=int(350000), value=int(20000), step=10)

    # Gegevens voorbereiden voor voorspellingen met geselecteerde en basisvaccinatiegraad
    input_data_selected = pd.DataFrame({'Besmettingen': [besmettingen_input], 'Vaccinatiegraad_volledig': [vaccinatiegraad_input]})
    input_data_baseline = pd.DataFrame({'Besmettingen': [besmettingen_input], 'Vaccinatiegraad_volledig': [5]})

    # Voorspel 'Ziekenhuisopnames' met de geselecteerde en basisvaccinatiegraad
    predicted_ziekenhuisopnames_selected = model.predict(input_data_selected)[0]
    predicted_ziekenhuisopnames_baseline = model.predict(input_data_baseline)[0]

    # Bereken de effectiviteit van vaccinatie
    reduction = predicted_ziekenhuisopnames_baseline - predicted_ziekenhuisopnames_selected
    effectivity_percentage = (reduction / predicted_ziekenhuisopnames_baseline) * 100 if predicted_ziekenhuisopnames_baseline != 0 else 0

    # Weergeven van voorspellingresultaten onder de prevalentie grafiek
    st.markdown("### Analyse van Vaccinatie-effectiviteit")

    # Verticaal weergegeven informatie als grote, horizontale kolommen
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Voorspelde ziekenhuisopnames (geselecteerde vaccinatiegraad)", f"{round(predicted_ziekenhuisopnames_selected, 2)}", "bij vaccinatiegraad: " + str(vaccinatiegraad_input) + "%")
    col2.metric("Voorspelde ziekenhuisopnames (vaccinatiegraad 5%)", f"{round(predicted_ziekenhuisopnames_baseline, 2)}")
    col3.metric("Vermindering in ziekenhuisopnames", f"{round(reduction, 2)}")
    col4.metric("Effectiviteit van vaccin", f"{round(effectivity_percentage, 2)}%")


###################################
# COVID API Tab ###################
###################################

with tab2:
    
    @st.cache_data
    def load_daily_data():

        df = pd.read_csv('intensiteit_daily_average.csv')

        return df

    df_daily = load_daily_data()

    @st.cache_data
    def load_weekly_data(year):
        df = pd.read_csv(f'intensiteit{year}_weekly.csv')
        return df

    df_grouped = pd.read_csv('intensiteit_daily_average.csv')

    #data per jaar filteren
    data = {
        2019: df_grouped[df_grouped['jaar'] == 2019],
        2020: df_grouped[df_grouped['jaar'] == 2020],
        2021: df_grouped[df_grouped['jaar'] == 2021],
        2022: df_grouped[df_grouped['jaar'] == 2022],
        2023: df_grouped[df_grouped['jaar'] == 2023],
        2024: df_grouped[df_grouped['jaar'] == 2024]}
    
    st.write('''Met behulp van open datasets van NDW kan er inzicht verkregen worden in de intensiteit van verkeer op de A10 in Amsterdam. 
            In de onderstaande box kan gekozen worden tussen verschillende jaren. ''')
    
    #selectbox maken om het jaar te selecteren
    year = st.selectbox("Selecteer een jaar", range(2019,2025))
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    
    data_aggregated = data[year].groupby('dag', as_index=False).agg({'gem_intensiteit': 'sum'})
    weekFig = px.bar(data_aggregated, 
                     x='dag', 
                     y='gem_intensiteit',
                     title=f"Intensiteit verkeerstromen in {year} (per week)", 
                     labels={'dag': 'Dag van de week', 'gem_intensiteit': 'Aantal'}, 
                     color='dag',
                     color_discrete_sequence=['steelblue'],
                     template='plotly_white', 
                     category_orders={'dag': day_order})
    
    weekFig.update_traces(marker=dict(line=dict(color='black', width=0.5), opacity=0.75))
    weekFig.update_xaxes(showgrid=True, gridcolor='lightgrey')
    weekFig.update_yaxes(showgrid=True, gridcolor='lightgrey')
    weekFig.update_layout(xaxis_title='Dag van de week', yaxis_title='Aantal', hovermode='x unified')
    
    #figuur laten zien
    st.plotly_chart(weekFig, use_container_width=True)
    ###################################
    # PLOT VAN WEEKDAG PER JAAR #######
    ###################################

    #Dagelijkse data laden
    @st.cache_data
    def load_daily_data():

        df = pd.read_csv('intensiteit_daily_average.csv')

        return df

    df_daily = load_daily_data()

    #Dropdown box maken en filteren op dag
    st.write('''Wat was het effect van de COVID-19 pandemie op de verdeling van verkeersintensiteit binnen een dag?
            Daarvoor kijken we naar onderstaande grafiek. Te zien is de berekende gemiddelde verkeersintensiteit per weekdag per jaar.
            Wat op valt is dat de oude vetrouwde spitsuren niet zijn opgeschoven of uitgespreid, 
            wat het geval zou zijn als er een stijging was in het aannemen van flexibele werktijden. 
            Wel is te zien dat over het algemeen de verkeerintensiteit sterk daalde na de start van de COVID-19 pandemie. 
            Kijkende naar de lijnen van 2023 en 2024, blijkt ook dat deze daling in verkeersintensiteit nog niet genihileerd is.
            Mogelijk door het aanhouden van de thuiswerkcultuur.''')

    weekday = st.selectbox('Selecteer een dag', df_daily['dag'].unique())

    dailyData = df_daily[df_daily['dag'] == weekday]

    #Plot
    dayFig = px.line(dailyData, x='tijd', y='gem_intensiteit', color='jaar',
                title=f'Intensiteit verkeersstromen op {weekday} - Vergelijking 2019-2024',
                labels={'tijd': 'Time', 'gem_intensiteit': 'Gemiddelde Intensiteit ()', 'year': 'Year'})

    dayFig.update_xaxes(rangeslider_visible=True)
    dayFig.update_layout(xaxis_title='Time of Day', yaxis_title='Average Intensity', hovermode='x')

    st.plotly_chart(dayFig, use_container_width=True)

# Continue with other sections if needed
