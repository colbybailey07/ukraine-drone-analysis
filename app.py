import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from math import radians, cos, sin, asin, sqrt

# ==========================================
# SETUP
# ==========================================
st.set_page_config(page_title="Ukraine Drone Analysis", layout="wide")
st.title("The War in Ukraine: Civilian Fatalities Due to Drone Strikes")

# ==========================================
# 1. DEFINE THE 15 HUB CITIES
# ==========================================
hub_coords = {
    'Kyiv': {'lat': 50.4501, 'lon': 30.5234},
    'Kharkiv': {'lat': 49.9935, 'lon': 36.2304},
    'Odesa': {'lat': 46.4825, 'lon': 30.7233},
    'Dnipro': {'lat': 48.4647, 'lon': 35.0462},
    'Donetsk': {'lat': 48.0159, 'lon': 37.8028},
    'Lviv': {'lat': 49.8397, 'lon': 24.0297},
    'Zaporizhzhia': {'lat': 47.8388, 'lon': 35.1396},
    'Kryvyi Rih': {'lat': 47.9105, 'lon': 33.3918},
    'Mykolaiv': {'lat': 46.9750, 'lon': 31.9946},
    'Mariupol': {'lat': 47.0971, 'lon': 37.5434},
    'Sevastopol': {'lat': 44.6166, 'lon': 33.5254},
    'Luhansk': {'lat': 48.5740, 'lon': 39.3078},
    'Vinnytsia': {'lat': 49.2331, 'lon': 28.4682},
    'Makiivka': {'lat': 48.0556, 'lon': 37.9615},
    'Simferopol': {'lat': 44.9572, 'lon': 34.1108}
}

# ==========================================
# 2. DIGITAL ATLAS (Distance Logic)
# ==========================================
known_towns = {
    # Kyiv Region
    'bucha': {'lat': 50.5444, 'lon': 30.2105},
    'irpin': {'lat': 50.5167, 'lon': 30.2500},
    'boryspil': {'lat': 50.3425, 'lon': 30.9511},
    'bila tserkva': {'lat': 49.8044, 'lon': 30.1288},
    # Donetsk Region
    'bakhmut': {'lat': 48.5987, 'lon': 38.0000},
    'soledar': {'lat': 48.6953, 'lon': 38.0667},
    'avdiivka': {'lat': 48.1397, 'lon': 37.7497},
    'kramatorsk': {'lat': 48.7392, 'lon': 37.5839},
    'sloviansk': {'lat': 48.8500, 'lon': 37.6167},
    'volnovakha': {'lat': 47.5833, 'lon': 37.5000},
    'lyman': {'lat': 48.9833, 'lon': 37.8000},
    # Luhansk Region
    'sievierodonetsk': {'lat': 48.9481, 'lon': 38.4933},
    'lysychansk': {'lat': 48.9167, 'lon': 38.4333},
    'kreminna': {'lat': 49.0500, 'lon': 38.2167},
    # South Region
    'melitopol': {'lat': 46.8489, 'lon': 35.3675},
    'berdiansk': {'lat': 46.7556, 'lon': 36.7889},
    'enerhodar': {'lat': 47.4989, 'lon': 34.6558},
    'tokmak': {'lat': 47.2500, 'lon': 35.7000},
    'kherson': {'lat': 46.6354, 'lon': 32.6169},
    'nova kakhovka': {'lat': 46.7667, 'lon': 33.3667},
    'ochakiv': {'lat': 46.6167, 'lon': 31.5500},
    # Kharkiv/Sumy/Central
    'izium': {'lat': 49.2000, 'lon': 37.2833},
    'kupiansk': {'lat': 49.7167, 'lon': 37.6167},
    'chuhuiv': {'lat': 49.8333, 'lon': 36.6833},
    'sumy': {'lat': 50.9077, 'lon': 34.7981},
    'okhtyrka': {'lat': 50.3167, 'lon': 34.8833},
    'poltava': {'lat': 49.5883, 'lon': 34.5514},
    'zhytomyr': {'lat': 50.2547, 'lon': 28.6587},
    'cherkasy': {'lat': 49.4444, 'lon': 32.0597},
    'uman': {'lat': 48.7484, 'lon': 30.2218},
    'kremenchuk': {'lat': 49.0631, 'lon': 33.4040},
    # West
    'lutsk': {'lat': 50.7472, 'lon': 25.3253},
    'rivne': {'lat': 50.6199, 'lon': 26.2516},
    'ternopil': {'lat': 49.5535, 'lon': 25.5948},
    'ivano-frankivsk': {'lat': 48.9226, 'lon': 24.7111},
    'uzhhorod': {'lat': 48.6208, 'lon': 22.2879},
    'chernivtsi': {'lat': 48.2917, 'lon': 25.9356},
    'khmelnytskyi': {'lat': 49.4230, 'lon': 26.9871}
}

# ==========================================
# 3. MATH HELPER: HAVERSINE
# ==========================================
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def find_closest_hub(town_lat, town_lon):
    closest_hub = None
    min_dist = float('inf')
    
    for hub_name, coords in hub_coords.items():
        dist = calculate_distance(town_lat, town_lon, coords['lat'], coords['lon'])
        if dist < min_dist:
            min_dist = dist
            closest_hub = hub_name
            
    return closest_hub

# ==========================================
# 4. MAIN CATEGORIZATION
# ==========================================
def assign_location(location_name):
    name = str(location_name).lower().strip()
    
    # 1. Strict Hub Match
    for hub in hub_coords.keys():
        if hub.lower() in name:
            return hub
            
    # 2. Known Town Lookup
    for town, coords in known_towns.items():
        if town in name:
            return find_closest_hub(coords['lat'], coords['lon'])

    return 'Other'

# ==========================================
# DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("master_data.csv")
        if 'location' in df.columns:
            df = df.rename(columns={'location': 'Region'})
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        df['Drone_Count'] = pd.to_numeric(df['Drone_Count'], errors='coerce').fillna(0)
        df['Fatality_Count'] = pd.to_numeric(df['Fatality_Count'], errors='coerce').fillna(0)

        df['City_Region'] = df['Region'].apply(assign_location)

        grouped_df = df.groupby(['month_year', 'City_Region']).agg({
            'Drone_Count': 'sum',
            'Fatality_Count': 'sum'
        }).reset_index()

        return df, grouped_df
    except Exception as e:
        return None, None

raw_df, grouped_df = load_data()

if raw_df is None:
    st.error("Error loading master_data.csv")
    st.stop()

# ==========================================
# DASHBOARD LAYOUT
# ==========================================

# --- FILTER LOGIC ---
city_sums = grouped_df.groupby('City_Region')['Drone_Count'].sum()
active_cities = [city for city in hub_coords.keys() if city in city_sums.index and city_sums[city] > 0]

# Create Tabs (Overview + Only Active Cities)
tabs = st.tabs(["Overview"] + active_cities)

# --- TAB 1: OVERVIEW ---
with tabs[0]:
    st.header("National Overview")
    
    total_strikes = int(grouped_df['Drone_Count'].sum())
    total_deaths = int(grouped_df['Fatality_Count'].sum())
    
    col_a, col_b = st.columns(2)
    col_a.metric("Total Drone Strikes", f"{total_strikes:,}")
    col_b.metric("Total Fatalities", f"{total_deaths:,}")
    
    st.markdown("---")

    # Map Data
    map_data = grouped_df.groupby('City_Region')[['Drone_Count', 'Fatality_Count']].sum().reset_index()
    map_data['lat'] = map_data['City_Region'].map(lambda x: hub_coords.get(x, {}).get('lat'))
    map_data['lon'] = map_data['City_Region'].map(lambda x: hub_coords.get(x, {}).get('lon'))
    map_data = map_data.dropna(subset=['lat', 'lon'])
    
    map_data = map_data[map_data['Drone_Count'] > 0]

    st.subheader("Geographic Heatmap: Active Hubs")
    fig_map = px.scatter_mapbox(
        map_data, lat="lat", lon="lon",
        size="Drone_Count", 
        color="Fatality_Count",
        hover_name="City_Region",
        zoom=5, center={"lat": 48.3794, "lon": 31.1656},
        mapbox_style="open-street-map",
        title="Size = Drone Volume | Color = Fatalities (Darker = Higher)",
        size_max=50, 
        color_continuous_scale=px.colors.sequential.Magma
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption("Note: Cities with 0 recorded strikes are hidden from this view.")

# --- ACTIVE CITY TABS ---
for i, city_name in enumerate(active_cities):
    with tabs[i + 1]:
        st.subheader(f"Analysis: {city_name}")
        
        city_data = grouped_df[grouped_df['City_Region'] == city_name].copy()
        
        c1, c2 = st.columns(2)
        c1.metric("Total Drones", int(city_data['Drone_Count'].sum()))
        c2.metric("Total Fatalities", int(city_data['Fatality_Count'].sum()))
        
        st.markdown("#### Monthly Drone Activity")
        fig_time = px.bar(
            city_data, x='month_year', y='Drone_Count',
            title=f"Timeline: {city_name}",
            template="plotly_white"
        )
        fig_time.update_layout(xaxis={'categoryorder':'category ascending'})
        st.plotly_chart(fig_time, use_container_width=True)

# ==========================================
# DISCLAIMER / FOOTER
# ==========================================
st.markdown("---")
st.caption(
    "Data from 'Massive Missile Attacks on Ukraine' and 'Civilian Harm in Ukraine' from Kaggle. "
    "Data is not representative of all civilian casualties nor all drone strikes in Ukraine. "
    "Data represents the general trend that can be applied to the entire population."
)
