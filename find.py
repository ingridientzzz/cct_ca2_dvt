# imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# set streamlit to Wide mode by default
st.set_page_config(layout='wide')

# Configs
FONT_FAMILY = '"Helvetica Neue", Helvetica, Arial, sans-serif'
COLOR_BG = '#FFFFFF'
COLOR_TEXT = '#000000'
COLOR_PRIMARY = '#56B4E9' # sky blue (Used for Male 2011 in dist graph, and Density 2022)
COLOR_SECONDARY = '#999999' # grey (Used for Density 2011 in comparison)
COLOR_MALE = '#0072B2' # blue (Used for Male 2022 in dist graph)
COLOR_FEMALE = '#E69F00' # orange (Used for Female 2022 in dist graph)
COLOR_FEMALE_2011_DIST = '#CC79A7' # Reddish Purple (For Female 2011 in dist graph)


# load data from CSV files
@st.cache_data
def load_data(age_gender_csv_path, density_csv_path):
    df_age_gender_raw = pd.DataFrame()
    df_density_raw = pd.DataFrame()

    try:
        df_age_gender_raw = pd.read_csv(age_gender_csv_path)
        expected_cols_age = ['code', 'name', 'geography', 'sex', 'age', 'population_2011', 'population_2022']
        if not all(col in df_age_gender_raw.columns for col in expected_cols_age):
            st.error(f"Age/Gender CSV ({age_gender_csv_path}) is missing expected columns. Found: {df_age_gender_raw.columns.tolist()}. Expected: {expected_cols_age}")
            df_age_gender_raw = pd.DataFrame()
    except FileNotFoundError:
        st.error(f"Age/Gender CSV file not found at {age_gender_csv_path}")
    except Exception as e:
        st.error(f"Error loading Age/Gender CSV ({age_gender_csv_path}): {e}")

    try:
        df_density_raw = pd.read_csv(density_csv_path)
        expected_cols_density = ['Code', 'Name', 'Geography', 'Area (sq km)',
                                 'Estimated Population mid-2022', '2022 people per sq. km',
                                 'Estimated Population mid-2011', '2011 people per sq. km']
        if not all(col in df_density_raw.columns for col in expected_cols_density):
            st.error(f"Density CSV ({density_csv_path}) is missing expected columns. Found: {df_density_raw.columns.tolist()}. Expected: {expected_cols_density}")
            df_density_raw = pd.DataFrame()
    except FileNotFoundError:
        st.error(f"Density CSV file not found at {density_csv_path}")
    except Exception as e:
        st.error(f"Error loading Density CSV ({density_csv_path}): {e}")

    return df_age_gender_raw, df_density_raw

@st.cache_data
def preprocess_density_data_wide(df_density_raw): # Renamed to signify it returns wide format initially
    if df_density_raw.empty:
        return pd.DataFrame()
    df_density = df_density_raw.copy()
    rename_map = {
        'Code': 'code', 'Name': 'name', 'Geography': 'geography',
        'Area (sq km)': 'area_sq_km',
        'Estimated Population mid-2022': 'population_2022_abs',
        '2022 people per sq. km': 'density_2022',
        'Estimated Population mid-2011': 'population_2011_abs',
        '2011 people per sq. km': 'density_2011'
    }
    df_density = df_density.rename(columns=rename_map)
    density_cols = ['density_2011', 'density_2022']
    for col in density_cols:
        if col in df_density.columns:
            df_density[col] = pd.to_numeric(df_density[col], errors='coerce')
        else:
            df_density[col] = pd.NA
    # Keep relevant columns for wide format, including area if needed for other calcs later
    final_cols = ['code', 'name', 'geography', 'density_2011', 'density_2022', 'area_sq_km']
    existing_final_cols = [col for col in final_cols if col in df_density.columns]
    return df_density[existing_final_cols]

@st.cache_data
def melt_density_data(df_density_wide):
    if df_density_wide.empty or not all(col in df_density_wide.columns for col in ['density_2011', 'density_2022']):
        return pd.DataFrame()

    df_density_melted = df_density_wide.melt(
        id_vars=['code', 'name', 'geography'],
        value_vars=['density_2011', 'density_2022'],
        var_name='Year_Col_Density',
        value_name='Density'
    )
    df_density_melted['Year'] = df_density_melted['Year_Col_Density'].str.extract(r'(\d+)').astype(int)
    df_density_melted = df_density_melted.drop(columns=['Year_Col_Density'])
    return df_density_melted[['code', 'name', 'geography', 'Year', 'Density']].dropna(subset=['Density'])


@st.cache_data
def preprocess_age_gender_data(df_age_gender_raw, bins, labels):
    if df_age_gender_raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    df_age_gender_detail = df_age_gender_raw.copy()
    df_age_gender_detail['age_numeric'] = df_age_gender_detail['age'].astype(str).replace('90+', '90').astype(int)
    df_age_gender_detail['age_band'] = pd.cut(
        df_age_gender_detail['age_numeric'],
        bins=bins, labels=labels, right=True, include_lowest=True
    )
    df_age_gender_melted = df_age_gender_detail.melt(
        id_vars=['code', 'name', 'geography', 'sex', 'age', 'age_numeric', 'age_band'],
        value_vars=['population_2011', 'population_2022'],
        var_name='Year_Col', value_name='Population'
    )
    df_age_gender_melted['Year'] = df_age_gender_melted['Year_Col'].str.extract(r'(\d+)').astype(int)
    df_age_gender_melted = df_age_gender_melted.drop(columns=['Year_Col'])
    df_age_gender_melted['Population'] = pd.to_numeric(df_age_gender_melted['Population'], errors='coerce').fillna(0)
    df_age_gender_melted = df_age_gender_melted.sort_values(by=['name', 'Year', 'sex', 'age_numeric'])
    df_age_gender_detail['population_2011'] = pd.to_numeric(df_age_gender_detail['population_2011'], errors='coerce').fillna(0)
    df_age_gender_detail['population_2022'] = pd.to_numeric(df_age_gender_detail['population_2022'], errors='coerce').fillna(0)
    return df_age_gender_detail, df_age_gender_melted

@st.cache_data
def create_merged_df(df_age_gender_melted, df_density_melted):
    if df_age_gender_melted.empty or df_density_melted.empty:
        return pd.DataFrame()
    merged_df = pd.merge(
        df_age_gender_melted,
        df_density_melted,
        on=['code', 'name', 'geography', 'Year'],
        how='left'
    )
    return merged_df

# build other details
genders_map = {'M': 'Male', 'F': 'Female'}
age_group_order = ['0-17', '18-24', '25-39', '40-59', '60-74', '75+'] # For pyramid sorting
bins = [-1, 17, 24, 39, 59, 74, np.inf]
labels = age_group_order
age_bands_options = ['All Ages'] + labels

# prep base dfs
age_gender_csv_file_path = 'MYEB1_Table9.csv'
density_csv_file_path = 'MYE5_Table8.csv'

df_age_gender_raw, df_density_raw = load_data(age_gender_csv_file_path, density_csv_file_path)

df_density_wide = preprocess_density_data_wide(df_density_raw) # Used for original density chart & scatter plot baseline
df_density_melted = melt_density_data(df_density_wide)       # Used for merging for pyramids

all_locations_master = []
all_geographies_master = []

df_age_gender_detail = pd.DataFrame()
df_age_gender_melted = pd.DataFrame()
merged_df = pd.DataFrame() # For new charts

if not df_age_gender_raw.empty:
    df_age_gender_detail, df_age_gender_melted = preprocess_age_gender_data(df_age_gender_raw, bins, labels)
    if 'name' in df_age_gender_melted.columns: all_locations_master.extend(df_age_gender_melted['name'].unique())
    if 'geography' in df_age_gender_melted.columns: all_geographies_master.extend(df_age_gender_melted['geography'].unique())

if not df_density_wide.empty:
    if 'name' in df_density_wide.columns: all_locations_master.extend(df_density_wide['name'].unique())
    if 'geography' in df_density_wide.columns: all_geographies_master.extend(df_density_wide['geography'].unique())

if not df_age_gender_melted.empty and not df_density_melted.empty:
    merged_df = create_merged_df(df_age_gender_melted, df_density_melted)

all_locations = sorted(list(set(all_locations_master)))
all_geographies = sorted(list(set(all_geographies_master)))

if df_density_wide.empty and not df_density_raw.empty: st.warning("Failed to process density data (wide). Some charts may be unavailable.")
if df_density_melted.empty and not df_density_wide.empty : st.warning("Failed to process density data (melted). Some charts may be unavailable.")
if merged_df.empty and not (df_age_gender_melted.empty or df_density_melted.empty): st.warning("Failed to create merged dataset for advanced charts.")


# helper functions
def create_empty_figure(title_text):
    fig = go.Figure()
    fig.update_layout(title=title_text, xaxis={'visible': False}, yaxis={'visible': False}, font={'family': FONT_FAMILY, 'color': COLOR_TEXT})
    return fig

def get_vrect_coords_from_age_band(age_band_str):
    if age_band_str == 'All Ages': return None, None
    if age_band_str == '75+': return '75', '90+'
    parts = age_band_str.split('-')
    if len(parts) == 2: return str(parts[0]), str(parts[1])
    return None, None

# Global selection sidebar
st.sidebar.header("Global Filters")
selected_geographies = st.sidebar.multiselect("Select Geography Type(s):", options=all_geographies, default=all_geographies)

locations_for_filter_options = all_locations
if selected_geographies:
    temp_locations = []
    # Consider all data sources for populating location filter based on geography
    if not df_age_gender_melted.empty: temp_locations.extend(df_age_gender_melted[df_age_gender_melted['geography'].isin(selected_geographies)]['name'].unique())
    if not df_density_wide.empty: temp_locations.extend(df_density_wide[df_density_wide['geography'].isin(selected_geographies)]['name'].unique())
    locations_for_filter_options = sorted(list(set(temp_locations)))

default_locations = [loc for loc in ['ENGLAND', 'SCOTLAND', 'WALES', 'NORTHERN IRELAND'] if loc in locations_for_filter_options]
if not default_locations and locations_for_filter_options: default_locations = locations_for_filter_options[:min(4, len(locations_for_filter_options))]
global_selected_locations = st.sidebar.multiselect("Select Location(s):", options=locations_for_filter_options, default=default_locations)

# Filter main dataframes
df_display_melted_age_gender = pd.DataFrame()
df_display_detail_age_gender = pd.DataFrame()
if not df_age_gender_melted.empty:
    master_filter = df_age_gender_melted['name'].isin(global_selected_locations) & df_age_gender_melted['geography'].isin(selected_geographies)
    df_display_melted_age_gender = df_age_gender_melted[master_filter]
if not df_age_gender_detail.empty:
    master_filter = df_age_gender_detail['name'].isin(global_selected_locations) & df_age_gender_detail['geography'].isin(selected_geographies)
    df_display_detail_age_gender = df_age_gender_detail[master_filter]

df_display_density_wide = pd.DataFrame()
if not df_density_wide.empty:
    master_filter = df_density_wide['name'].isin(global_selected_locations) & df_density_wide['geography'].isin(selected_geographies)
    df_display_density_wide = df_density_wide[master_filter]

df_display_merged = pd.DataFrame()
if not merged_df.empty:
    master_filter = merged_df['name'].isin(global_selected_locations) & merged_df['geography'].isin(selected_geographies)
    df_display_merged = merged_df[master_filter]


disable_year_filter = df_display_melted_age_gender.empty and df_display_density_wide.empty and df_display_merged.empty
global_selected_year_display = st.sidebar.selectbox("Select Year(s) to Display:", options=["Comparison (2011 & 2022)", "2011 Only", "2022 Only"], index=0, disabled=disable_year_filter)

sex_options_map = {"Both": ["M", "F"], "Male": ["M"], "Female": ["F"]}
global_selected_gender_display_label = st.sidebar.selectbox("Select Sex:", options=["Both", "Male", "Female"], index=0, disabled=df_display_melted_age_gender.empty)
global_selected_sex_codes = sex_options_map[global_selected_gender_display_label]

global_selected_age_band_filter = st.sidebar.selectbox("Select Age Band (for filtering applicable charts):", options=age_bands_options, index=0, disabled=df_display_melted_age_gender.empty)

# Main Body
st.title("UK Population Dashboard: 2011 vs 2022")
st.markdown("---")

# Chart 1: Population Density Comparison (Original)
# ... (Code for this chart remains the same, using df_display_density_wide) ...
st.header("1. Population Density Comparison")
if df_display_density_wide.empty:
    # ... (empty chart logic) ...
    st.plotly_chart(create_empty_figure("No density data for current filter selection (Location/Geography)."), use_container_width=True)
else:
    filtered_df_density_chart = df_display_density_wide.sort_values('name')
    fig_density = go.Figure()
    plot_data_exists_density = False
    if global_selected_year_display == "2011 Only" or global_selected_year_display == "Comparison (2011 & 2022)":
        if 'density_2011' in filtered_df_density_chart.columns and not filtered_df_density_chart['density_2011'].dropna().empty:
            fig_density.add_trace(go.Bar(
                x=filtered_df_density_chart['name'], y=filtered_df_density_chart['density_2011'], name='Density 2011',
                marker_color=COLOR_SECONDARY if global_selected_year_display == "Comparison (2011 & 2022)" else COLOR_PRIMARY,
                hovertemplate=("**%{x}**<br>2011 Density: %{y:.1f} per sq km<extra></extra>")
            ))
            plot_data_exists_density = True
    if global_selected_year_display == "2022 Only" or global_selected_year_display == "Comparison (2011 & 2022)":
        if 'density_2022' in filtered_df_density_chart.columns and not filtered_df_density_chart['density_2022'].dropna().empty:
            fig_density.add_trace(go.Bar(
                x=filtered_df_density_chart['name'], y=filtered_df_density_chart['density_2022'], name='Density 2022',
                marker_color=COLOR_PRIMARY,
                hovertemplate=("**%{x}**<br>2022 Density: %{y:.1f} per sq km<extra></extra>")
            ))
            plot_data_exists_density = True
    if not plot_data_exists_density:
        st.plotly_chart(create_empty_figure("No density data available for the selected year(s) and locations."), use_container_width=True)
    else:
        title_density = 'Population Density'
        if global_selected_year_display != "Comparison (2011 & 2022)": title_density += f' ({global_selected_year_display.replace(" Only", "")})'
        else: title_density += ': 2011 vs 2022'
        fig_density.update_layout(
            title=title_density, xaxis_title='Location', yaxis_title='People per Square Kilometer',
            barmode='group' if global_selected_year_display == "Comparison (2011 & 2022)" else 'overlay',
            hovermode='x unified', legend_title_text='Year',
            xaxis={'categoryorder': 'array', 'categoryarray': sorted(filtered_df_density_chart['name'].unique())},
            font={'family': FONT_FAMILY}, margin=dict(l=40, r=20, t=60, b=40)
        )
        st.plotly_chart(fig_density, use_container_width=True)
st.markdown("---")

# Chart 2: Population Age Distribution (Original)
# ... (Code for this chart remains the same, using df_display_detail_age_gender) ...
st.header("2. Population Age Distribution")
loc_for_age_dist = None
if not df_display_detail_age_gender.empty:
    if global_selected_locations and global_selected_locations[0] in df_display_detail_age_gender['name'].unique():
        loc_for_age_dist = global_selected_locations[0]
        if len(global_selected_locations) > 1: st.info(f"Displaying age distribution for {loc_for_age_dist} (first selected).")
    elif not df_display_detail_age_gender['name'].empty:
        loc_for_age_dist = df_display_detail_age_gender['name'].unique()[0]
        st.info(f"Displaying age distribution for {loc_for_age_dist} (first available).")
    else: st.plotly_chart(create_empty_figure("No locations in filtered age data."), use_container_width=True)
elif not df_age_gender_raw.empty: st.plotly_chart(create_empty_figure("No age data for current filters."), use_container_width=True)
else: st.plotly_chart(create_empty_figure("Age data not loaded."), use_container_width=True)

if loc_for_age_dist and not df_display_detail_age_gender.empty:
    fig_age_gender = go.Figure()
    data_found_for_age_dist = False
    chart_df_age_detail = df_display_detail_age_gender[
        (df_display_detail_age_gender['name'] == loc_for_age_dist) &
        (df_display_detail_age_gender['sex'].isin(global_selected_sex_codes))
    ].sort_values('age_numeric')
    if not chart_df_age_detail.empty:
        for gender_code in global_selected_sex_codes:
            gender_specific_df = chart_df_age_detail[chart_df_age_detail['sex'] == gender_code]
            if not gender_specific_df.empty:
                data_found_for_age_dist = True
                gender_label = genders_map[gender_code]
                color_2022 = COLOR_MALE if gender_code == 'M' else COLOR_FEMALE
                color_2011 = COLOR_PRIMARY if gender_code == 'M' else COLOR_FEMALE_2011_DIST
                if global_selected_year_display == "2011 Only" or global_selected_year_display == "Comparison (2011 & 2022)":
                    if 'population_2011' in gender_specific_df.columns:
                        fig_age_gender.add_trace(go.Scatter(x=gender_specific_df['age'], y=gender_specific_df['population_2011'], name=f'{gender_label} 2011', mode='lines', marker_color=color_2011, line=dict(width=2, dash='dash'), hovertemplate=(f"<b>{gender_label} - Age: %{{x}}</b><br>2011 Pop: %{{y:,}}<extra></extra>")))
                if global_selected_year_display == "2022 Only" or global_selected_year_display == "Comparison (2011 & 2022)":
                    if 'population_2022' in gender_specific_df.columns:
                        fig_age_gender.add_trace(go.Scatter(x=gender_specific_df['age'], y=gender_specific_df['population_2022'], name=f'{gender_label} 2022', mode='lines', marker_color=color_2022, line=dict(width=2), hovertemplate=(f"<b>{gender_label} - Age: %{{x}}</b><br>2022 Pop: %{{y:,}}<extra></extra>")))
    if not data_found_for_age_dist: st.plotly_chart(create_empty_figure(f"No age data for {loc_for_age_dist} with filters."), use_container_width=True)
    else:
        title_age_dist = f'Pop Age Distribution in {loc_for_age_dist}'
        if global_selected_year_display != "Comparison (2011 & 2022)": title_age_dist += f' ({global_selected_year_display.replace(" Only", "")})'
        fig_age_gender.update_layout(title=title_age_dist, xaxis_title='Age', yaxis_title='Est. Population', xaxis={'type': 'category'}, hovermode='x unified', legend_title_text='Sex & Year', font={'family': FONT_FAMILY}, margin=dict(l=40, r=20, t=60, b=40))
        if global_selected_age_band_filter != 'All Ages': # Use the specific age band filter here
            x0_h, x1_h = get_vrect_coords_from_age_band(global_selected_age_band_filter)
            if x0_h is not None and x1_h is not None: fig_age_gender.add_vrect(x0=x0_h, x1=x1_h, fillcolor="rgba(128,128,128,0.2)", layer="below", line_width=0)
        st.plotly_chart(fig_age_gender, use_container_width=True)
st.markdown("---")

# Chart 3: Gender Population Comparison (Original)
# ... (Code for this chart remains the same, using df_display_melted_age_gender) ...
st.header("3. Gender Population Comparison")
year_for_gender_comp_actual = None
if global_selected_year_display == "2011 Only": year_for_gender_comp_actual = 2011
elif global_selected_year_display == "2022 Only": year_for_gender_comp_actual = 2022
elif global_selected_year_display == "Comparison (2011 & 2022)":
    year_for_gender_comp_actual = 2022
    st.info("Gender Pop Comparison chart defaults to 2022 data for 'Comparison' year selection.")

if df_display_melted_age_gender.empty:
    if not df_age_gender_raw.empty: st.plotly_chart(create_empty_figure("No gender data for current Location/Geo filters."), use_container_width=True)
elif not year_for_gender_comp_actual: st.plotly_chart(create_empty_figure("Select year for gender comparison."), use_container_width=True)
else:
    chart_df_gender_melted = df_display_melted_age_gender[df_display_melted_age_gender['Year'] == year_for_gender_comp_actual].copy()
    age_title_part = f"({global_selected_age_band_filter})" # Use the specific age band filter
    if global_selected_age_band_filter != 'All Ages':
        chart_df_gender_melted = chart_df_gender_melted[chart_df_gender_melted['age_band'] == global_selected_age_band_filter]
    chart_df_gender_melted = chart_df_gender_melted[chart_df_gender_melted['sex'].isin(global_selected_sex_codes)]
    if chart_df_gender_melted.empty: st.plotly_chart(create_empty_figure(f"No data for gender comparison in {year_for_gender_comp_actual} {age_title_part} with filters."), use_container_width=True)
    else:
        grouped_df_gender = chart_df_gender_melted.groupby(['name', 'sex'])['Population'].sum().reset_index().sort_values(by=['name', 'sex'])
        fig_gender_comp = go.Figure()
        plot_data_exists_gender = False
        for sex_code in global_selected_sex_codes:
            gender_label = genders_map[sex_code]
            df_sex_specific = grouped_df_gender[grouped_df_gender['sex'] == sex_code]
            if not df_sex_specific.empty:
                plot_data_exists_gender = True
                color = COLOR_MALE if sex_code == 'M' else COLOR_FEMALE
                fig_gender_comp.add_trace(go.Bar(x=df_sex_specific['name'], y=df_sex_specific['Population'], name=gender_label, marker_color=color, hovertemplate=(f"<b>%{{x}}</b><br>{gender_label}s: %{{y:,}}<br>Year: {year_for_gender_comp_actual}<br>Age Band: {global_selected_age_band_filter}<extra></extra>")))
        if not plot_data_exists_gender: st.plotly_chart(create_empty_figure(f"No data for selected sex(es) in {year_for_gender_comp_actual} {age_title_part}."), use_container_width=True)
        else:
            fig_gender_comp.update_layout(title=f'Pop by Sex in {year_for_gender_comp_actual} {age_title_part}', xaxis_title='Location', yaxis_title='Population', barmode='group', hovermode='x unified', legend_title_text='Sex', xaxis={'categoryorder': 'array', 'categoryarray': sorted(grouped_df_gender['name'].unique())}, font={'family': FONT_FAMILY}, margin=dict(l=40, r=20, t=60, b=40))
            st.plotly_chart(fig_gender_comp, use_container_width=True)
st.markdown("---")


# --- NEW CHARTS ---

# 4. Age Pyramids per Density Band
st.header("4. Age Pyramids by Population Density Band")
num_density_bands = st.sidebar.slider("Number of Density Bands (Quantiles):", min_value=2, max_value=5, value=4, disabled=df_display_merged.empty)

if df_display_merged.empty:
    st.plotly_chart(create_empty_figure("Merged data for density band analysis is not available. Check filters or data files."), use_container_width=True)
else:
    years_to_plot_pyramid = []
    if global_selected_year_display == "2011 Only": years_to_plot_pyramid = [2011]
    elif global_selected_year_display == "2022 Only": years_to_plot_pyramid = [2022]
    elif global_selected_year_display == "Comparison (2011 & 2022)": years_to_plot_pyramid = [2011, 2022]

    if not years_to_plot_pyramid:
        st.warning("Please select a year or 'Comparison' to display age pyramids.")
    else:
        for year_pyramid in years_to_plot_pyramid:
            st.subheader(f"Age Pyramids for {year_pyramid}")
            data_for_year_pyramid = df_display_merged[df_display_merged['Year'] == year_pyramid].copy()

            if data_for_year_pyramid.empty or data_for_year_pyramid['Density'].dropna().empty:
                st.write(f"No data with density information available for {year_pyramid} after applying global Location/Geography filters.")
                continue

            try:
                # Create density bands using qcut. Ensure unique bin edges.
                # Drop NA densities before qcut, handle cases with too few unique densities for qcut
                unique_densities = data_for_year_pyramid['Density'].dropna().unique()
                if len(unique_densities) < num_density_bands:
                     # If not enough unique values for qcut, use pd.cut with fewer bins or a different strategy
                    data_for_year_pyramid['density_band'] = pd.cut(data_for_year_pyramid['Density'], bins=max(1, len(unique_densities)-1) if len(unique_densities)>1 else 1, include_lowest=True, precision=0)
                    actual_num_bands = data_for_year_pyramid['density_band'].nunique()
                    st.info(f"Reduced to {actual_num_bands} density bands for {year_pyramid} due to limited unique density values.")
                else:
                    data_for_year_pyramid['density_band'] = pd.qcut(data_for_year_pyramid['Density'].dropna(), q=num_density_bands, labels=False, duplicates='drop', precision=0)

                # Convert interval to string for display
                if pd.api.types.is_categorical_dtype(data_for_year_pyramid['density_band']) or pd.api.types.is_interval_dtype(data_for_year_pyramid['density_band'].dtype):
                     data_for_year_pyramid['density_band_str'] = data_for_year_pyramid['density_band'].astype(str)
                else: # if labels=False, it's numeric
                     band_labels = [f"Band {i+1}" for i in range(data_for_year_pyramid['density_band'].nunique())]
                     data_for_year_pyramid['density_band_str'] = pd.cut(data_for_year_pyramid['density_band'], bins=data_for_year_pyramid['density_band'].nunique(), labels=band_labels, include_lowest=True, duplicates='drop').astype(str)


                density_bands_unique = sorted(data_for_year_pyramid['density_band_str'].dropna().unique())

                if not density_bands_unique:
                    st.write(f"Could not create density bands for {year_pyramid}.")
                    continue

                # Determine number of columns for subplots
                cols_subplot = min(len(density_bands_unique), 2) # Max 2 columns
                rows_subplot = (len(density_bands_unique) + cols_subplot - 1) // cols_subplot

                fig_pyramids = make_subplots(
                    rows=rows_subplot, cols=cols_subplot,
                    subplot_titles=[f"Density: {band}" for band in density_bands_unique],
                    shared_yaxes=True
                )

                plot_row = 1
                plot_col = 1
                pyramid_data_found = False

                for i, band_label_str in enumerate(density_bands_unique):
                    band_data = data_for_year_pyramid[data_for_year_pyramid['density_band_str'] == band_label_str]

                    if band_data.empty: continue

                    pyramid_agg = band_data.groupby(['age_band', 'sex'])['Population'].sum().unstack(fill_value=0).reset_index()
                    pyramid_agg['age_band'] = pd.Categorical(pyramid_agg['age_band'], categories=age_group_order, ordered=True)
                    pyramid_agg = pyramid_agg.sort_values('age_band')

                    if 'M' not in pyramid_agg.columns: pyramid_agg['M'] = 0
                    if 'F' not in pyramid_agg.columns: pyramid_agg['F'] = 0

                    # For pyramid, male population is negative
                    fig_pyramids.add_trace(go.Bar(
                        y=pyramid_agg['age_band'], x=-pyramid_agg['M'], name='Male',
                        orientation='h', marker_color=COLOR_MALE, hovertemplate="Age: %{y}<br>Male Pop: %{customdata:,}<extra></extra>", customdata=pyramid_agg['M']
                    ), row=plot_row, col=plot_col)
                    fig_pyramids.add_trace(go.Bar(
                        y=pyramid_agg['age_band'], x=pyramid_agg['F'], name='Female',
                        orientation='h', marker_color=COLOR_FEMALE, hovertemplate="Age: %{y}<br>Female Pop: %{x:,}<extra></extra>"
                    ), row=plot_row, col=plot_col)
                    pyramid_data_found = True

                    plot_col += 1
                    if plot_col > cols_subplot:
                        plot_col = 1
                        plot_row += 1

                if pyramid_data_found:
                    fig_pyramids.update_layout(
                        title_text=f"Age Pyramids by Density Band for {year_pyramid} (Filtered Locations/Geographies)",
                        barmode='relative',
                        bargap=0.1,
                        height=max(400, rows_subplot * 300), # Adjust height based on rows
                        #xaxis_title="Population", # Each subplot will have its own
                        yaxis_title="Age Band",
                        legend_title_text="Sex",
                        font=dict(family=FONT_FAMILY)
                    )
                    # Update x-axis for pyramids to handle negative values nicely
                    for i in range(1, len(density_bands_unique) + 1):
                        fig_pyramids.update_xaxes(
                            #title_text="Population",
                            # automargin=True,
                            # tickformat makes numbers more readable
                            tickformat=',.0s', # e.g. 10k, 1M
                            row=(i-1)//cols_subplot + 1, col=(i-1)%cols_subplot + 1
                        )

                    st.plotly_chart(fig_pyramids, use_container_width=True)
                else:
                    st.write(f"No population data to display pyramids for {year_pyramid} after creating density bands.")

            except Exception as e:
                st.error(f"Error creating age pyramids for {year_pyramid}: {e}")
st.markdown("---")

# 5. Population Change (2011 vs. 2022) by Density Scatter Plot
st.header("5. Population Change vs. Density (2011-2022)")

if df_display_melted_age_gender.empty or df_density_wide.empty:
    st.plotly_chart(create_empty_figure("Required data for population change analysis is not available. Check filters or data files."), use_container_width=True)
else:
    # Filter df_display_melted_age_gender by selected sex and age band (if not 'All Ages')
    pop_change_data = df_display_melted_age_gender.copy()
    if global_selected_gender_display_label != "Both":
        pop_change_data = pop_change_data[pop_change_data['sex'].isin(global_selected_sex_codes)]
    if global_selected_age_band_filter != "All Ages":
        pop_change_data = pop_change_data[pop_change_data['age_band'] == global_selected_age_band_filter]

    # Aggregate population by location and year for the filtered demographic
    pop_agg_change = pop_change_data.groupby(['code', 'name', 'geography', 'Year'])['Population'].sum().reset_index()

    # Pivot to get 2011 and 2022 populations side-by-side
    pop_pivot = pop_agg_change.pivot_table(
        index=['code', 'name', 'geography'],
        columns='Year',
        values='Population',
        fill_value=0
    ).reset_index()

    if 2011 not in pop_pivot.columns: pop_pivot[2011] = 0
    if 2022 not in pop_pivot.columns: pop_pivot[2022] = 0

    pop_pivot.rename(columns={2011: 'population_2011', 2022: 'population_2022'}, inplace=True)

    # Calculate population change
    pop_pivot['pop_change_abs'] = pop_pivot['population_2022'] - pop_pivot['population_2011']
    pop_pivot['pop_change_pct'] = ((pop_pivot['population_2022'] - pop_pivot['population_2011']) / pop_pivot['population_2011'].replace(0, pd.NA) * 100) # Avoid div by zero

    # Merge with baseline density (e.g., 2011 density from df_density_wide)
    # df_density_wide already filtered by global location/geography as df_display_density_wide
    scatter_data = pd.merge(
        pop_pivot,
        df_display_density_wide[['code', 'density_2011', 'density_2022']], # Use already filtered density data
        on='code',
        how='left'
    )
    scatter_data.dropna(subset=['density_2011', 'pop_change_pct'], inplace=True) # Ensure essential data is present

    if scatter_data.empty:
        st.plotly_chart(create_empty_figure("No data available for scatter plot after processing and filtering."), use_container_width=True)
    else:
        change_type = st.radio("Select Change Type for Y-axis:", ("Percentage Change", "Absolute Change"), index=0, key="scatter_change_type")
        y_col_scatter = 'pop_change_pct' if change_type == "Percentage Change" else 'pop_change_abs'
        y_axis_title = "Population Change (%)" if y_col_scatter == 'pop_change_pct' else "Absolute Population Change"

        density_year_for_x = st.radio("Select Density Year for X-axis:", ("2011 Density", "2022 Density"), index=0, key="scatter_density_year")
        x_col_scatter = 'density_2011' if density_year_for_x == "2011 Density" else 'density_2022'
        x_axis_title = "Population Density in 2011 (people/sq km)" if x_col_scatter == 'density_2011' else "Population Density in 2022 (people/sq km)"


        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=scatter_data[x_col_scatter],
            y=scatter_data[y_col_scatter],
            mode='markers',
            marker=dict(color=COLOR_PRIMARY, size=8, opacity=0.7),
            text=scatter_data['name'], # For hover
            hovertemplate=(
                "<b>%{text}</b><br>" +
                f"{x_axis_title}: %{{x:.1f}}<br>" +
                f"{y_axis_title}: %{{y:.2f}}" + ("%" if y_col_scatter == 'pop_change_pct' else "") +
                "<br>Pop 2011: %{customdata[0]:,}<br>Pop 2022: %{customdata[1]:,}"+
                "<extra></extra>"
            ),
            customdata=scatter_data[['population_2011', 'population_2022']]
        ))

        title_scatter = f'{change_type} vs. {density_year_for_x}'
        if global_selected_gender_display_label != "Both": title_scatter += f' for {global_selected_gender_display_label}s'
        if global_selected_age_band_filter != "All Ages": title_scatter += f' (Age: {global_selected_age_band_filter})'

        fig_scatter.update_layout(
            title=title_scatter,
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            font=dict(family=FONT_FAMILY),
            hovermode='closest'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")
# footer
url = "https://github.com/ingridientzzz"
st.markdown(f"[App based on an original by: ingridientzzz]({url})")
