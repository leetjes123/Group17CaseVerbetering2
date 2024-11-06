import pandas as pd

def aggregate_and_save(year):
    # Read the CSV file for the given year
    df = pd.read_csv(f'intensiteit{year}.csv', usecols=['start_meetperiode', 'gem_intensiteit'])
    
    # Convert start_meetperiode to datetime
    df['start_meetperiode'] = pd.to_datetime(df['start_meetperiode'])
    
    # Extract the week number
    df['week'] = df['start_meetperiode'].dt.isocalendar().week
    
    # Group by week number and sum the gem_intensiteit
    weekly_data = df.groupby('week')['gem_intensiteit'].sum().reset_index()
    
    # Sort by week number
    weekly_data = weekly_data.sort_values('week')
    
    # Save to a new CSV file
    output_file = f'intensiteit{year}_weekly.csv'
    weekly_data.to_csv(output_file, index=False)
    print(f"Saved aggregated data for {year} to {output_file}")

# Process data for each year from 2019 to 2024
for year in range(2019, 2025):
    aggregate_and_save(year)

print("Data aggregation complete for all years.")