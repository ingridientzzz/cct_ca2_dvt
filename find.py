# imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# import openpyxl # No longer needed
import os

# set streamlit to Wide mode by default
st.set_page_config(layout='wide')

# Configs
FONT_FAMILY = '"Helvetica Neue", Helvetica, Arial, sans-serif'
COLOR_BG = '#FFFFFF' # Corrected from #FFFF
COLOR_TEXT = '#000000' # Corrected from #0000
COLOR_PRIMARY = '#56B4E9' # sky blue (Used for Male 2011 in dist graph, and Density 2022)
COLOR_SECONDARY = '#999999' # grey (Used for Density 2011 in comparison) # Corrected from #9999
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
def preprocess_density_data_wide(df_density_raw): # Renamed as it returns wide format
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
    final_cols = ['code', 'name', 'geography', 'density_2011', 'density_2022', 'area_sq_km']
    existing_final_cols = [col for col in final_cols if col in df_density.columns]
    return df_density[existing_final_cols]

@st.cache_data
def melt_density_data(df_density_wide):
    if df_density_wide.empty or not all(col in df_density_wide.columns for col in ['code', 'name', 'geography', 'density_2011', 'density_2022']):
        # Ensure essential ID columns and data columns are present
        st.warning("Cannot melt density data: essential columns missing from wide format density data.")
        return pd.DataFrame()

    df_density_melted = df_density_wide.melt(
        id_vars=['code', 'name', 'geography'], # Removed 'area_sq_km' as it's not year-specific for density values
        value_vars=['density_2011', 'density_2022'],
        var_name='Year_Col_Density',
        value_name='Density'
    )
    df_density_melted['Year'] = df_density_melted['Year_Col_Density'].str.extract(r'(\d+)').astype(int)
    df_density_melted = df_density_melted.drop(columns=['Year_Col_Density'])
    # Keep only relevant columns and drop rows where Density might be NaN after melting
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
        st.warning("Cannot create merged data: one or both input dataframes are empty.")
        return pd.DataFrame()
    # Ensure 'Year' column is int for merging
    df_age_gender_melted['Year'] = df_age_gender_melted['Year'].astype(int)
    df_density_melted['Year'] = df_density_melted['Year'].astype(int)

    merged_df = pd.merge(
        df_age_gender_melted,
        df_density_melted,
        on=['code', 'name', 'geography', 'Year'],
        how='left' # Keep all age/gender data, add density where available
    )
    # Drop rows where Density is NaN after merge if they are critical for an analysis
    # For sex ratio vs density, we need Density, so drop if NaN
    # merged_df.dropna(subset=['Density'], inplace=True) # Reconsider if this is too aggressive
    return merged_df

# build other details
genders_map = {'M': 'Male', 'F': 'Female'}
bins = [-1, 17, 24, 39, 59, 74, np.inf]
labels = ['0-17', '18-24', '25-39', '40-59', '60-74', '75+']
age_bands_options = ['All Ages'] + labels

# prep base dfs
age_gender_csv_file_path = 'MYEB1_Table9.csv'
density_csv_file_path = 'MYE5_Table8.csv'

df_age_gender_raw, df_density_raw = load_data(age_gender_csv_file_path, density_csv_file_path)

df_density_wide = preprocess_density_data_wide(df_density_raw) # For original density chart
df_density_melted = melt_density_data(df_density_wide)       # For merging

all_locations_master = []
all_geographies_master = []

df_age_gender_detail = pd.DataFrame()
df_age_gender_melted = pd.DataFrame()
merged_df = pd.DataFrame()

if not df_age_gender_raw.empty:
    df_age_gender_detail, df_age_gender_melted = preprocess_age_gender_data(df_age_gender_raw, bins, labels)
    if 'name' in df_age_gender_melted.columns: all_locations_master.extend(df_age_gender_melted['name'].unique())
    if 'geography' in df_age_gender_melted.columns: all_geographies_master.extend(df_age_gender_melted['geography'].unique())

if not df_density_wide.empty: # Use df_density_wide for populating filters from density source
    if 'name' in df_density_wide.columns: all_locations_master.extend(df_density_wide['name'].unique())
    if 'geography' in df_density_wide.columns: all_geographies_master.extend(df_density_wide['geography'].unique())

if not df_age_gender_melted.empty and not df_density_melted.empty:
    merged_df = create_merged_df(df_age_gender_melted, df_density_melted)

all_locations = sorted(list(set(all_locations_master)))
all_geographies = sorted(list(set(all_geographies_master)))

if df_density_wide.empty and not df_density_raw.empty: st.warning("Failed to process density data (wide). Original density chart may be unavailable.")
if df_density_melted.empty and not df_density_wide.empty : st.warning("Failed to process density data (melted). Merged data for new charts may be affected.")
if merged_df.empty and not (df_age_gender_melted.empty or df_density_melted.empty): st.warning("Failed to create merged dataset for advanced charts.")


# helper functions
def create_empty_figure(title_text):
    fig = go.Figure()
    fig.update_layout(title=title_text, xaxis={'visible': False}, yaxis={'visible': False}, font={'family': FONT_FAMILY, 'color': COLOR_TEXT})
    return fig

def get_vrect_coords_from_age_band(age_band_str):
    if age_band_str == 'All Ages': return None, None
    if age_band_str == '75+': return '75', '90+' # '90+' is the max in 'age' column
    parts = age_band_str.split('-')
    if len(parts) == 2: return str(parts[0]), str(parts[1])
    return None, None

# Global selection sidebar
st.sidebar.header("Global Filters")
selected_geographies = st.sidebar.multiselect("Select Geography Type(s):", options=all_geographies, default=all_geographies)

locations_for_filter_options = all_locations
if selected_geographies:
    temp_locations = []
    if not df_age_gender_melted.empty: temp_locations.extend(df_age_gender_melted[df_age_gender_melted['geography'].isin(selected_geographies)]['name'].unique())
    if not df_density_wide.empty: temp_locations.extend(df_density_wide[df_density_wide['geography'].isin(selected_geographies)]['name'].unique())
    locations_for_filter_options = sorted(list(set(temp_locations)))

default_locations = [loc for loc in ['ENGLAND', 'SCOTLAND', 'WALES', 'NORTHERN IRELAND'] if loc in locations_for_filter_options]
if not default_locations and locations_for_filter_options: default_locations = locations_for_filter_options[:min(4, len(locations_for_filter_options))]
global_selected_locations = st.sidebar.multiselect("Select Location(s):", options=locations_for_filter_options, default=default_locations)

# Filter main dataframes based on selected locations and geographies
df_display_melted = pd.DataFrame() # For age/gender charts
df_display_detail = pd.DataFrame() # For age distribution
df_display_density = pd.DataFrame() # For original density chart (wide)
df_display_merged = pd.DataFrame() # For new merged data charts

if not df_age_gender_melted.empty:
    master_filter_age_gender = df_age_gender_melted['name'].isin(global_selected_locations) & df_age_gender_melted['geography'].isin(selected_geographies)
    df_display_melted = df_age_gender_melted[master_filter_age_gender]
if not df_age_gender_detail.empty:
    master_filter_detail = df_age_gender_detail['name'].isin(global_selected_locations) & df_age_gender_detail['geography'].isin(selected_geographies)
    df_display_detail = df_age_gender_detail[master_filter_detail]
if not df_density_wide.empty:
    master_filter_density = df_density_wide['name'].isin(global_selected_locations) & df_density_wide['geography'].isin(selected_geographies)
    df_display_density = df_density_wide[master_filter_density]
if not merged_df.empty:
    master_filter_merged = merged_df['name'].isin(global_selected_locations) & merged_df['geography'].isin(selected_geographies)
    df_display_merged = merged_df[master_filter_merged]


disable_year_filter = df_display_melted.empty and df_display_density.empty and df_display_merged.empty
global_selected_year_display = st.sidebar.selectbox("Select Year(s) to Display:", options=["Comparison (2011 & 2022)", "2011 Only", "2022 Only"], index=0, disabled=disable_year_filter)

sex_options_map = {"Both": ["M", "F"], "Male": ["M"], "Female": ["F"]}
global_selected_gender_display_label = st.sidebar.selectbox("Select Sex (for applicable charts):", options=["Both", "Male", "Female"], index=0, disabled=df_display_melted.empty)
global_selected_sex_codes = sex_options_map[global_selected_gender_display_label]

global_selected_age_band = st.sidebar.selectbox("Select Age Band (for applicable charts):", options=age_bands_options, index=0, disabled=df_display_melted.empty)


# Main Body
st.title("UK Population Dashboard: 2011 vs 2022")
st.markdown("---")

# 1: Population Density Comparison (Original - uses df_display_density)
st.header("1. Population Density Comparison")
if df_display_density.empty:
    if df_density_wide.empty and not df_density_raw.empty: st.plotly_chart(create_empty_figure("Density data could not be processed."), use_container_width=True)
    elif df_density_raw.empty: st.plotly_chart(create_empty_figure("Density data not loaded."), use_container_width=True)
    else: st.plotly_chart(create_empty_figure("No density data for current filter selection."), use_container_width=True)
else:
    filtered_df_density_chart = df_display_density.sort_values('name')
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
    if not plot_data_exists_density: st.plotly_chart(create_empty_figure("No density data for selected year(s)/locations."), use_container_width=True)
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


# 2: Population Age Distribution (Original - uses df_display_detail)
st.header("2. Population Age Distribution")
loc_for_age_dist = None
if not df_display_detail.empty:
    if global_selected_locations and global_selected_locations[0] in df_display_detail['name'].unique():
        loc_for_age_dist = global_selected_locations[0]
        if len(global_selected_locations) > 1: st.info(f"Displaying age distribution for {loc_for_age_dist} (first selected).")
    elif not df_display_detail['name'].empty:
        loc_for_age_dist = df_display_detail['name'].unique()[0]
        st.info(f"Displaying age distribution for {loc_for_age_dist} (first available).")
    else: st.plotly_chart(create_empty_figure("No locations in filtered age data."), use_container_width=True)
elif not df_age_gender_raw.empty: st.plotly_chart(create_empty_figure("No age data for current filters."), use_container_width=True)
else: st.plotly_chart(create_empty_figure("Age data not loaded."), use_container_width=True)

if loc_for_age_dist and not df_display_detail.empty:
    fig_age_gender = go.Figure()
    data_found_for_age_dist = False
    chart_df_age_detail = df_display_detail[
        (df_display_detail['name'] == loc_for_age_dist) &
        (df_display_detail['sex'].isin(global_selected_sex_codes)) # Global sex filter applies here
    ].sort_values('age_numeric')
    if not chart_df_age_detail.empty:
        for gender_code_loop in global_selected_sex_codes: # Loop through selected sexes
            gender_specific_df = chart_df_age_detail[chart_df_age_detail['sex'] == gender_code_loop]
            if not gender_specific_df.empty:
                data_found_for_age_dist = True
                gender_label_loop = genders_map[gender_code_loop]
                color_2022 = COLOR_MALE if gender_code_loop == 'M' else COLOR_FEMALE
                color_2011 = COLOR_PRIMARY if gender_code_loop == 'M' else COLOR_FEMALE_2011_DIST
                if global_selected_year_display == "2011 Only" or global_selected_year_display == "Comparison (2011 & 2022)":
                    if 'population_2011' in gender_specific_df.columns:
                        fig_age_gender.add_trace(go.Scatter(x=gender_specific_df['age'], y=gender_specific_df['population_2011'], name=f'{gender_label_loop} 2011', mode='lines', marker_color=color_2011, line=dict(width=2, dash='dash'), hovertemplate=(f"<b>{gender_label_loop} - Age: %{{x}}</b><br>2011 Pop: %{{y:,}}<extra></extra>")))
                if global_selected_year_display == "2022 Only" or global_selected_year_display == "Comparison (2011 & 2022)":
                    if 'population_2022' in gender_specific_df.columns:
                        fig_age_gender.add_trace(go.Scatter(x=gender_specific_df['age'], y=gender_specific_df['population_2022'], name=f'{gender_label_loop} 2022', mode='lines', marker_color=color_2022, line=dict(width=2), hovertemplate=(f"<b>{gender_label_loop} - Age: %{{x}}</b><br>2022 Pop: %{{y:,}}<extra></extra>")))
    if not data_found_for_age_dist: st.plotly_chart(create_empty_figure(f"No age data for {loc_for_age_dist} with filters."), use_container_width=True)
    else:
        title_age_dist = f'Pop Age Distribution in {loc_for_age_dist}'
        if global_selected_year_display != "Comparison (2011 & 2022)": title_age_dist += f' ({global_selected_year_display.replace(" Only", "")})'
        fig_age_gender.update_layout(title=title_age_dist, xaxis_title='Age', yaxis_title='Est. Population', xaxis={'type': 'category'}, hovermode='x unified', legend_title_text='Sex & Year', font={'family': FONT_FAMILY}, margin=dict(l=40, r=20, t=60, b=40))
        if global_selected_age_band != 'All Ages':
            x0_h, x1_h = get_vrect_coords_from_age_band(global_selected_age_band)
            if x0_h is not None and x1_h is not None: fig_age_gender.add_vrect(x0=x0_h, x1=x1_h, fillcolor="rgba(128,128,128,0.2)", layer="below", line_width=0)
        st.plotly_chart(fig_age_gender, use_container_width=True)
st.markdown("---")

# 3: Gender Population Comparison (Original - uses df_display_melted)
st.header("3. Gender Population Comparison")
year_for_gender_comp_actual = None
if global_selected_year_display == "2011 Only": year_for_gender_comp_actual = 2011
elif global_selected_year_display == "2022 Only": year_for_gender_comp_actual = 2022
elif global_selected_year_display == "Comparison (2011 & 2022)":
    year_for_gender_comp_actual = 2022
    st.info("Gender Pop Comparison chart defaults to 2022 data for 'Comparison' year selection.")

if df_display_melted.empty:
    if not df_age_gender_raw.empty: st.plotly_chart(create_empty_figure("No gender data for current Location/Geo filters."), use_container_width=True)
elif not year_for_gender_comp_actual: st.plotly_chart(create_empty_figure("Select year for gender comparison."), use_container_width=True)
else:
    chart_df_gender_melted = df_display_melted[df_display_melted['Year'] == year_for_gender_comp_actual].copy()
    age_title_part = f"({global_selected_age_band})"
    if global_selected_age_band != 'All Ages':
        chart_df_gender_melted = chart_df_gender_melted[chart_df_gender_melted['age_band'] == global_selected_age_band]
    chart_df_gender_melted = chart_df_gender_melted[chart_df_gender_melted['sex'].isin(global_selected_sex_codes)] # Global sex filter applies
    if chart_df_gender_melted.empty: st.plotly_chart(create_empty_figure(f"No data for gender comparison in {year_for_gender_comp_actual} {age_title_part} with filters."), use_container_width=True)
    else:
        grouped_df_gender = chart_df_gender_melted.groupby(['name', 'sex'])['Population'].sum().reset_index().sort_values(by=['name', 'sex'])
        fig_gender_comp = go.Figure()
        plot_data_exists_gender = False
        for sex_code_loop in global_selected_sex_codes: # Loop through selected sexes
            gender_label_loop = genders_map[sex_code_loop]
            df_sex_specific = grouped_df_gender[grouped_df_gender['sex'] == sex_code_loop]
            if not df_sex_specific.empty:
                plot_data_exists_gender = True
                color = COLOR_MALE if sex_code_loop == 'M' else COLOR_FEMALE
                fig_gender_comp.add_trace(go.Bar(x=df_sex_specific['name'], y=df_sex_specific['Population'], name=gender_label_loop, marker_color=color, hovertemplate=(f"<b>%{{x}}</b><br>{gender_label_loop}s: %{{y:,}}<br>Year: {year_for_gender_comp_actual}<br>Age Band: {global_selected_age_band}<extra></extra>")))
        if not plot_data_exists_gender: st.plotly_chart(create_empty_figure(f"No data for selected sex(es) in {year_for_gender_comp_actual} {age_title_part}."), use_container_width=True)
        else:
            fig_gender_comp.update_layout(title=f'Pop by Sex in {year_for_gender_comp_actual} {age_title_part}', xaxis_title='Location', yaxis_title='Population', barmode='group', hovermode='x unified', legend_title_text='Sex', xaxis={'categoryorder': 'array', 'categoryarray': sorted(grouped_df_gender['name'].unique())}, font={'family': FONT_FAMILY}, margin=dict(l=40, r=20, t=60, b=40))
            st.plotly_chart(fig_gender_comp, use_container_width=True)
st.markdown("---")


# 4. Sex Ratio by Density (NEW - uses df_display_merged)
st.header("4. Sex Ratio by Population Density")
st.info("Sex ratio is calculated as Males per 100 Females. This chart ignores the global 'Select Sex' filter.")

if df_display_merged.empty or 'Density' not in df_display_merged.columns:
    st.plotly_chart(create_empty_figure("Merged data with density is not available for sex ratio analysis. Check filters or data files."), use_container_width=True)
else:
    sex_ratio_data = df_display_merged.copy()

    # Apply Year Filter
    years_to_plot_sex_ratio = []
    if global_selected_year_display == "2011 Only": years_to_plot_sex_ratio = [2011]
    elif global_selected_year_display == "2022 Only": years_to_plot_sex_ratio = [2022]
    elif global_selected_year_display == "Comparison (2011 & 2022)": years_to_plot_sex_ratio = [2011, 2022]

    if not years_to_plot_sex_ratio:
        st.warning("Please select a year or 'Comparison' to display sex ratio by density.")
    else:
        sex_ratio_data = sex_ratio_data[sex_ratio_data['Year'].isin(years_to_plot_sex_ratio)]

        # Apply Age Band Filter
        age_band_title_part_sr = f"(Age Band: {global_selected_age_band})"
        if global_selected_age_band != 'All Ages':
            sex_ratio_data = sex_ratio_data[sex_ratio_data['age_band'] == global_selected_age_band]
        else:
            age_band_title_part_sr = "(All Ages)"


        if sex_ratio_data.empty or sex_ratio_data['Density'].dropna().empty:
            st.plotly_chart(create_empty_figure(f"No data for sex ratio analysis with current filters {age_band_title_part_sr}."), use_container_width=True)
        else:
            # Aggregate Male and Female populations
            # Group by all identifying columns including Density and Year, then unstack sex
            pop_by_sex_for_ratio = sex_ratio_data.groupby(
                ['code', 'name', 'geography', 'Year', 'Density', 'sex']
            )['Population'].sum().unstack(fill_value=0).reset_index()

            if 'M' not in pop_by_sex_for_ratio.columns: pop_by_sex_for_ratio['M'] = 0
            if 'F' not in pop_by_sex_for_ratio.columns: pop_by_sex_for_ratio['F'] = 0

            # Calculate Sex Ratio (Males per 100 Females)
            # Avoid division by zero by replacing F=0 with NaN, which results in NaN for sex_ratio
            pop_by_sex_for_ratio['sex_ratio'] = (pop_by_sex_for_ratio['M'] / pop_by_sex_for_ratio['F'].replace(0, pd.NA)) * 100

            # Drop rows where sex_ratio could not be calculated (e.g., F=0 or Density=NaN)
            final_plot_data_sr = pop_by_sex_for_ratio.dropna(subset=['Density', 'sex_ratio'])

            if final_plot_data_sr.empty:
                st.plotly_chart(create_empty_figure(f"Not enough data to calculate sex ratios for the scatter plot {age_band_title_part_sr}."), use_container_width=True)
            else:
                fig_sex_ratio_density = go.Figure()
                plot_data_exists_sr = False

                for year_val in years_to_plot_sex_ratio:
                    year_data_sr = final_plot_data_sr[final_plot_data_sr['Year'] == year_val]
                    if not year_data_sr.empty:
                        plot_data_exists_sr = True
                        fig_sex_ratio_density.add_trace(go.Scatter(
                            x=year_data_sr['Density'],
                            y=year_data_sr['sex_ratio'],
                            mode='markers',
                            name=f'Sex Ratio {year_val}',
                            marker=dict(
                                color=COLOR_PRIMARY if year_val == 2022 else COLOR_SECONDARY,
                                size=8, opacity=0.7
                            ),
                            text=year_data_sr['name'], # For hover
                            hovertemplate=(
                                "<b>%{text} (%{customdata[0]})</b><br>" +
                                "Density: %{x:.1f} per sq km<br>" +
                                "Sex Ratio: %{y:.1f} (Males per 100 Females)<br>" +
                                "Males: %{customdata[1]:,}<br>Females: %{customdata[2]:,}" +
                                "<extra></extra>"
                            ),
                            customdata=year_data_sr[['Year', 'M', 'F']]
                        ))

                if not plot_data_exists_sr:
                    st.plotly_chart(create_empty_figure(f"No valid sex ratio data to plot for selected year(s) {age_band_title_part_sr}."), use_container_width=True)
                else:
                    title_sr_density = f'Sex Ratio vs. Population Density {age_band_title_part_sr}'
                    if len(years_to_plot_sex_ratio) == 1:
                         title_sr_density += f' - {years_to_plot_sex_ratio[0]}'

                    fig_sex_ratio_density.update_layout(
                        title=title_sr_density,
                        xaxis_title='Population Density (people per sq km)',
                        yaxis_title='Sex Ratio (Males per 100 Females)',
                        font=dict(family=FONT_FAMILY),
                        hovermode='closest',
                        legend_title_text='Year'
                    )
                    st.plotly_chart(fig_sex_ratio_density, use_container_width=True)

st.markdown("---")
# footer
url = "https://github.com/ingridientzzz"
st.markdown(f"[App based on an original by: ingridientzzz]({url})")
