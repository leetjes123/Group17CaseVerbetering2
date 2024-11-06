import pandas as pd

df_2019 = pd.read_csv('intensiteit2019.csv', usecols=['start_meetperiode', 'gem_intensiteit', 'id_meetlocatie'])
df_2020 = pd.read_csv('intensiteit2020.csv', usecols=['start_meetperiode', 'gem_intensiteit', 'id_meetlocatie'])
df_2021 = pd.read_csv('intensiteit2021.csv', usecols=['start_meetperiode', 'gem_intensiteit', 'id_meetlocatie'])
df_2022 = pd.read_csv('intensiteit2022.csv', usecols=['start_meetperiode', 'gem_intensiteit', 'id_meetlocatie'])
df_2023 = pd.read_csv('intensiteit2023.csv', usecols=['start_meetperiode', 'gem_intensiteit', 'id_meetlocatie'])
df_2024 = pd.read_csv('intensiteit2024.csv', usecols=['start_meetperiode', 'gem_intensiteit', 'id_meetlocatie'])

df = pd.concat([df_2019,df_2020,df_2021,df_2022,df_2023,df_2024], ignore_index=True)

df['start_meetperiode'] = pd.to_datetime(df['start_meetperiode'])
df['jaar'] = df['start_meetperiode'].dt.year
df['dag'] = df['start_meetperiode'].dt.day_name()
df['tijd'] = df['start_meetperiode'].dt.strftime('%H:%M')

df_grouped = df.groupby(['dag', 'tijd','jaar'])['gem_intensiteit'].mean().reset_index()

df_grouped.to_csv('intensiteit_daily_average.csv', index=False)