import requests
import csv
from datetime import datetime, timedelta
import time  # For delay between requests

# Function to fetch historical weather data for a specific date
def fetch_historical_data(api_url, api_key, lat, lon, date):
    try:
        # Convert the date to a Unix timestamp
        timestamp = int(datetime.strptime(date, "%Y-%m-%d").timestamp())
        
        # Construct the API URL with parameters
        params = {
            'lat': lat,
            'lon': lon,
            'dt': timestamp,
            'appid': api_key,
            'units': 'metric'
        }
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the JSON response
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {date}: {e}")
        return None

# Function to save data to a CSV file
def save_data_to_csv(data, filename):
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["max_temp","min_temp","pressure","humidity"])  # Header
            writer.writerows(data)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

# Main script
if __name__ == "__main__":
    # API endpoint and your API key
    API_URL = "https://api.openweathermap.org/data/2.5/weather"
    API_KEY = "be944f759ab69e047faeb4168b132f92" 
    
    # Location coordinates for Hyderabad
    LAT = 17.3850  # Latitude for Hyderabad
    LON = 78.4867  # Longitude for Hyderabad
    
    # Start and end dates for the past 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)  # Approx. 5 years
    
    # Prepare a list to store the results
    weather_data = []
    
    # Loop through each date in the range
    current_date = start_date
    calls_made = 0
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        data = fetch_historical_data(API_URL, API_KEY, LAT, LON, date_str)
        if data and 'main' in data:
            max_temp = data['main']['temp_max']  
            min_temp = data['main']['temp_min']  
            pressure = data['main']['pressure']  
            humidity = data['main']['humidity']  
            weather_data.append([max_temp,min_temp,pressure,humidity])
        current_date += timedelta(days=1)  # Increment by one day
        
        # Increment the API call count
        calls_made += 1
        
        # If 60 calls are made, pause for a minute
        if calls_made >= 60:
            print("API limit reached, waiting for 60 seconds...")
            time.sleep(60)  # Wait for 60 seconds
            calls_made = 0  # Reset the counter
    
    # Save the collected data to a CSV file
    filename = "historical_weather_data_hyderabad.csv"
    save_data_to_csv(weather_data, filename)
