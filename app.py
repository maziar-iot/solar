import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import ee

# --------------------------------------------------------------------
# Init GEE
# --------------------------------------------------------------------
PROJECT_ID = 'ee-maziarasmani'
try:
    ee.Initialize(project=PROJECT_ID)
except:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

# --------------------------------------------------------------------
# Analysis Functions (adapted from your original code)
# --------------------------------------------------------------------

def get_gee_data_10_criteria(locations):
    user_points = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point(loc['lon'], loc['lat']), {'name': loc['name'], 'lat': loc['lat'], 'lon': loc['lon']}) 
        for loc in locations
    ])
    
    srtm = ee.Image('USGS/SRTMGL1_003')
    gldas_collection = ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H').filterDate('2023-01-01', '2023-12-31').select(
        ['Tair_f_inst', 'SWdown_f_tavg', 'Wind_f_inst', 'Qair_f_inst', 'Psurf_f_inst', 'Rainf_f_tavg']
    )
    modis_aod = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES').filterDate('2023-01-01', '2023-12-31').select('Optical_Depth_055')

    def extract_values(point_feature):
        point_geometry = point_feature.geometry()
        
        slope_dict = ee.Terrain.slope(srtm).reduceRegion(reducer=ee.Reducer.mean(), geometry=point_geometry, scale=90)
        elevation_dict = srtm.select('elevation').reduceRegion(reducer=ee.Reducer.mean(), geometry=point_geometry, scale=90)
        gldas_dict = gldas_collection.mean().reduceRegion(reducer=ee.Reducer.mean(), geometry=point_geometry, scale=27830)
        aod_dict = modis_aod.mean().reduceRegion(reducer=ee.Reducer.mean(), geometry=point_geometry, scale=1000)
        
        slope = ee.Algorithms.If(slope_dict.contains('slope'), slope_dict.get('slope'), None)
        elevation = ee.Algorithms.If(elevation_dict.contains('elevation'), elevation_dict.get('elevation'), None)
        temp_k = ee.Algorithms.If(gldas_dict.contains('Tair_f_inst'), gldas_dict.get('Tair_f_inst'), None)
        radiation_w_m2 = ee.Algorithms.If(gldas_dict.contains('SWdown_f_tavg'), gldas_dict.get('SWdown_f_tavg'), None)
        wind_speed = ee.Algorithms.If(gldas_dict.contains('Wind_f_inst'), gldas_dict.get('Wind_f_inst'), None)
        humidity_specific = ee.Algorithms.If(gldas_dict.contains('Qair_f_inst'), gldas_dict.get('Qair_f_inst'), None)
        pressure_pa = ee.Algorithms.If(gldas_dict.contains('Psurf_f_inst'), gldas_dict.get('Psurf_f_inst'), None)
        precipitation_flux = ee.Algorithms.If(gldas_dict.contains('Rainf_f_tavg'), gldas_dict.get('Rainf_f_tavg'), None)
        aod = ee.Algorithms.If(aod_dict.contains('Optical_Depth_055'), aod_dict.get('Optical_Depth_055'), None)
        
        return point_feature.set({
            'Land Slope': slope, 'Elevation': elevation, 'Ambient Temperature_K': temp_k, 'Solar Irradiance_W_m2': radiation_w_m2,
            'Wind Speed': wind_speed, 'Specific Humidity': humidity_specific, 'Air Pressure_Pa': pressure_pa,
            'Precipitation_Flux': precipitation_flux, 'Dust Level (AOD)': aod
        })

    results_collection = user_points.map(extract_values).getInfo()

    criteria_data = []
    for feature in results_collection['features']:
        props = feature['properties']
        
        temp_c = (props.get('Ambient Temperature_K') - 273.15) if props.get('Ambient Temperature_K') is not None else None
        radiation_w = props.get('Solar Irradiance_W_m2')
        radiation_kwh_day = (radiation_w * 24 / 1000) if radiation_w is not None else None
        aod_scaled = props.get('Dust Level (AOD)') * 1000 if props.get('Dust Level (AOD)') is not None else None
        pressure_pa = props.get('Air Pressure_Pa')
        precip_flux = props.get('Precipitation_Flux')
        precip_annual_mm = (precip_flux * 3600 * 24 * 365) if precip_flux is not None else None
        
        relative_humidity = None
        q = props.get('Specific Humidity')
        if q is not None and temp_c is not None and pressure_pa is not None:
            es = 610.94 * np.exp((17.625 * temp_c) / (243.04 + temp_c))
            e_vapor_pressure = (q * pressure_pa) / (0.622 + (0.378 * q))
            rh = 100 * (e_vapor_pressure / es)
            relative_humidity = min(rh, 100.0)

        criteria_data.append({
            'Location': props['name'], 'Latitude': props['lat'], 'Longitude': props['lon'],
            'Solar Irradiance (kWh/m²/day)': radiation_kwh_day,
            'Ambient Temperature (°C)': temp_c,
            'Land Slope (°)': props.get('Land Slope'),
            'Dust Level (AOD)': aod_scaled,
            'Relative Humidity (%)': relative_humidity,
            'Wind Speed (m/s)': props.get('Wind Speed'),
            'Elevation (m)': props.get('Elevation'),
            'Air Pressure (hPa)': pressure_pa / 100 if pressure_pa is not None else None,
            'Precipitation (mm/year)': precip_annual_mm,
            'Latitude (°)': abs(props['lat'])
        })
        
    return pd.DataFrame(criteria_data).set_index('Location')

def calculate_shannon_weights(dataframe):
    dataframe = dataframe.dropna()
    if len(dataframe) < 2: raise ValueError("Not enough valid data for analysis.")
    p_matrix = dataframe / dataframe.sum(axis=0); num_alternatives = len(dataframe)
    k = 1 / np.log(num_alternatives); entropy_values = -k * (p_matrix * np.log(p_matrix.replace(0, 1e-12))).sum(axis=0)
    divergence = 1 - entropy_values; weights = divergence / divergence.sum()
    return weights, pd.DataFrame({'Entropy': entropy_values, 'Divergence': divergence, 'Weight': weights}).sort_values(by='Weight', ascending=False)

def run_topsis_ranking(dataframe, weights, criteria_types):
    dataframe = dataframe.dropna()
    r_matrix = dataframe / np.sqrt((dataframe**2).sum(axis=0)); v_matrix = r_matrix * weights
    ideal_positive = pd.Series(index=v_matrix.columns, dtype=float); ideal_negative = pd.Series(index=v_matrix.columns, dtype=float)
    for i, col in enumerate(v_matrix.columns):
        if criteria_types[i] == 'benefit': ideal_positive[col] = v_matrix[col].max(); ideal_negative[col] = v_matrix[col].min()
        else: ideal_positive[col] = v_matrix[col].min(); ideal_negative[col] = v_matrix[col].max()
    s_positive = np.sqrt(((v_matrix - ideal_positive)**2).sum(axis=1)); s_negative = np.sqrt(((v_matrix - ideal_negative)**2).sum(axis=1))
    closeness = s_negative / (s_positive + s_negative)
    results_df = pd.DataFrame({'S+': s_positive, 'S-': s_negative, 'Ci': closeness})
    results_df['Rank'] = results_df['Ci'].rank(ascending=False, method='first').astype(int)
    return results_df.sort_values(by='Rank')

# --------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------

st.title("🌍 تحلیل مکانی چندمعیاره با داده‌های ماهواره‌ای")

st.markdown("**راهنما:** روی نقشه کلیک کنید تا نقاط موردنظر انتخاب شوند، سپس دکمه‌ی تحلیل را بزنید.")

if 'points' not in st.session_state:
    st.session_state.points = []

m = folium.Map(location=[30, 0], zoom_start=2)

for p in st.session_state.points:
    folium.Marker([p['lat'], p['lon']], tooltip=p['name']).add_to(m)

map_data = st_folium(m, width=700, height=500)

if map_data and map_data['last_clicked']:
    latlng = map_data['last_clicked']
    st.session_state.points.append({
        'name': f"Site {len(st.session_state.points)+1}",
        'lat': latlng['lat'],
        'lon': latlng['lng']
    })
    st.experimental_rerun()

st.write(f"📍 تعداد نقاط انتخاب شده: {len(st.session_state.points)}")

if st.session_state.points:
    st.table(pd.DataFrame(st.session_state.points))

if st.button("🚀 شروع تحلیل"):
    with st.spinner("در حال دریافت داده‌ها و تحلیل..."):
        df = get_gee_data_10_criteria(st.session_state.points)
        st.subheader("📊 داده‌های استخراج‌شده:")
        st.dataframe(df)

        cleaned = df.dropna()
        criteria_df = cleaned.drop(columns=['Latitude', 'Longitude'])

        criteria_df['Latitude (°)'] = cleaned['Latitude']

        CRITERIA_TYPES = [
            'benefit', 'cost', 'cost', 'cost', 'cost',
            'benefit', 'cost', 'benefit', 'cost', 'cost'
        ]
        weights, weights_df = calculate_shannon_weights(criteria_df)
        ranking = run_topsis_ranking(criteria_df, weights, CRITERIA_TYPES)

        st.subheader("⚖️ وزن معیارها (روش شانون):")
        st.dataframe(weights_df)

        st.subheader("🏁 نتایج رتبه‌بندی TOPSIS:")
        st.dataframe(ranking)

        st.subheader("🗺️ نمایش نقشه‌ی نتایج:")
        result_map = folium.Map(location=[cleaned['Latitude'].mean(), cleaned['Longitude'].mean()], zoom_start=4)
        for idx, row in ranking.merge(cleaned[['Latitude', 'Longitude']], left_index=True, right_index=True).iterrows():
            popup = f"{idx}<br>Rank: {row['Rank']}<br>Ci: {row['Ci']:.4f}"
            icon_color = 'green' if row['Rank'] == 1 else 'blue'
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=popup,
                icon=folium.Icon(color=icon_color)
            ).add_to(result_map)
        st_folium(result_map, width=700)

if st.button("🔄 ریست"):
    st.session_state.points = []
    st.experimental_rerun()
