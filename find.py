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
        # Adjust expected columns based on MYE5_Table8.csv structure
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
def preprocess_density_data(df_density_raw):
    if df_density_raw.empty:
        return pd.DataFrame()

    df_density = df_density_raw.copy()
    # Rename columns for consistency and ease of use
    rename_map = {
        'Code': 'code',
        'Name': 'name',
        'Geography': 'geography',
        'Area (sq km)': 'area_sq_km',
        'Estimated Population mid-2022': 'population_2022_abs', # Absolute population
        '2022 people per sq. km': 'density_2022',
        'Estimated Population mid-2011': 'population_2011_abs', # Absolute population
        '2011 people per sq. km': 'density_2011'
    }
    df_density = df_density.rename(columns=rename_map)

    # Ensure density columns are numeric
    density_cols = ['density_2011', 'density_2022']
    for col in density_cols:
        if col in df_density.columns:
            df_density[col] = pd.to_numeric(df_density[col], errors='coerce') # Coerce will turn non-numeric to NaN
        else:
            st.warning(f"Column {col} not found in density data after renaming.")
            df_density[col] = pd.NA # Use pandas NA for missing numeric data

    # Select only necessary columns to prevent issues with non-numeric data in other columns if any
    final_cols = ['code', 'name', 'geography', 'density_2011', 'density_2022']
    # Add area and absolute population if they exist and are needed later, for now stick to density
    # final_cols.extend(['area_sq_km', 'population_2011_abs', 'population_2022_abs'])

    # Filter to keep only columns that actually exist in the dataframe after renaming
    existing_final_cols = [col for col in final_cols if col in df_density.columns]
    df_density = df_density[existing_final_cols]

    return df_density

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
        var_name='Year_Col',
        value_name='Population'
    )
    df_age_gender_melted['Year'] = df_age_gender_melted['Year_Col'].str.extract(r'(\d+)').astype(int)
    df_age_gender_melted = df_age_gender_melted.drop(columns=['Year_Col'])
    df_age_gender_melted['Population'] = pd.to_numeric(df_age_gender_melted['Population'], errors='coerce').fillna(0)
    df_age_gender_melted = df_age_gender_melted.sort_values(by=['name', 'Year', 'sex', 'age_numeric'])

    df_age_gender_detail['population_2011'] = pd.to_numeric(df_age_gender_detail['population_2011'], errors='coerce').fillna(0)
    df_age_gender_detail['population_2022'] = pd.to_numeric(df_age_gender_detail['population_2022'], errors='coerce').fillna(0)

    return df_age_gender_detail, df_age_gender_melted


# build other details
genders_map = {'M': 'Male', 'F': 'Female'}
bins = [-1, 17, 24, 39, 59, 74, np.inf]
labels = ['0-17', '18-24', '25-39', '40-59', '60-74', '75+']
age_bands_options = ['All Ages'] + labels

# prep base dfs
age_gender_csv_file_path = 'MYEB1_Table9.csv'
density_csv_file_path = 'MYE5_Table8.csv' # New CSV for density

df_age_gender_raw, df_density_raw = load_data(age_gender_csv_file_path, density_csv_file_path)

df_density = preprocess_density_data(df_density_raw)

all_locations_master = []
all_geographies_master = []

if not df_age_gender_raw.empty:
    df_age_gender_detail, df_age_gender_melted = preprocess_age_gender_data(df_age_gender_raw, bins, labels)
    if 'name' in df_age_gender_melted.columns:
        all_locations_master.extend(df_age_gender_melted['name'].unique())
    if 'geography' in df_age_gender_melted.columns:
        all_geographies_master.extend(df_age_gender_melted['geography'].unique())
else:
    df_age_gender_detail = pd.DataFrame()
    df_age_gender_melted = pd.DataFrame()
    # Error for age/gender data already handled in load_data

if not df_density.empty:
    if 'name' in df_density.columns:
        all_locations_master.extend(df_density['name'].unique())
    if 'geography' in df_density.columns:
        all_geographies_master.extend(df_density['geography'].unique())
# else: Error for density data already handled in load_data or preprocess

all_locations = sorted(list(set(all_locations_master)))
all_geographies = sorted(list(set(all_geographies_master)))


if df_density.empty and not df_density_raw.empty: # If raw had data but processing failed
    st.warning("Failed to process density data. The density chart may be unavailable.")


# helper functions
def create_empty_figure(title_text):
    fig = go.Figure()
    fig.update_layout(
        title=title_text,
        xaxis={'visible': False},
        yaxis={'visible': False},
        font={'family': FONT_FAMILY, 'color': COLOR_TEXT}
    )
    return fig

def get_vrect_coords_from_age_band(age_band_str):
    if age_band_str == 'All Ages':
        return None, None
    if age_band_str == '75+':
        return '75', '90+'
    parts = age_band_str.split('-')
    if len(parts) == 2:
        return str(parts[0]), str(parts[1])
    return None, None

# Global selection sidebar
st.sidebar.header("Global Filters")

# 1. Geography Filter
selected_geographies = st.sidebar.multiselect(
    "Select Geography Type(s):",
    options=all_geographies,
    default=all_geographies # Default to all initially
)

# Filter available locations based on selected geographies
locations_for_filter_options = all_locations # Start with all
if selected_geographies:
    temp_locations = []
    if not df_age_gender_melted.empty and 'geography' in df_age_gender_melted.columns and 'name' in df_age_gender_melted.columns:
        temp_locations.extend(df_age_gender_melted[df_age_gender_melted['geography'].isin(selected_geographies)]['name'].unique())
    if not df_density.empty and 'geography' in df_density.columns and 'name' in df_density.columns:
        temp_locations.extend(df_density[df_density['geography'].isin(selected_geographies)]['name'].unique())
    locations_for_filter_options = sorted(list(set(temp_locations)))


# 2. Location Filter
default_locations = [loc for loc in ['ENGLAND', 'SCOTLAND', 'WALES', 'NORTHERN IRELAND'] if loc in locations_for_filter_options]
if not default_locations and locations_for_filter_options:
    default_locations = locations_for_filter_options[:min(4, len(locations_for_filter_options))]

global_selected_locations = st.sidebar.multiselect(
    "Select Location(s):",
    options=locations_for_filter_options,
    default=default_locations
)

# Filter main dataframes based on selected locations and geographies
# For age/gender data:
df_display_melted = pd.DataFrame()
df_display_detail = pd.DataFrame()
if not df_age_gender_melted.empty:
    master_filter_age_gender = pd.Series([True] * len(df_age_gender_melted))
    if global_selected_locations:
        master_filter_age_gender &= df_age_gender_melted['name'].isin(global_selected_locations)
    if selected_geographies:
         master_filter_age_gender &= df_age_gender_melted['geography'].isin(selected_geographies)
    df_display_melted = df_age_gender_melted[master_filter_age_gender]

    master_filter_detail = pd.Series([True] * len(df_age_gender_detail))
    if global_selected_locations:
        master_filter_detail &= df_age_gender_detail['name'].isin(global_selected_locations)
    if selected_geographies:
        master_filter_detail &= df_age_gender_detail['geography'].isin(selected_geographies)
    df_display_detail = df_age_gender_detail[master_filter_detail]


# For density data:
df_display_density = pd.DataFrame()
if not df_density.empty:
    master_filter_density = pd.Series([True] * len(df_density))
    if global_selected_locations:
         master_filter_density &= df_density['name'].isin(global_selected_locations)
    if selected_geographies:
        if 'geography' in df_density.columns:
            master_filter_density &= df_density['geography'].isin(selected_geographies)
        # else: Warning about missing geography column in density data handled during preprocessing/loading
    df_display_density = df_density[master_filter_density]


# 3. Year Filter
disable_year_filter = df_display_melted.empty and df_display_density.empty
global_selected_year_display = st.sidebar.selectbox(
    "Select Year(s) to Display:",
    options=["Comparison (2011 & 2022)", "2011 Only", "2022 Only"],
    index=0,
    disabled=disable_year_filter
)

# 4. Sex Filter
sex_options_map = {"Both": ["M", "F"], "Male": ["M"], "Female": ["F"]}
global_selected_gender_display_label = st.sidebar.selectbox(
    "Select Sex:",
    options=["Both", "Male", "Female"],
    index=0,
    disabled=df_display_melted.empty # Sex filter only relevant for age/gender charts
)
global_selected_sex_codes = sex_options_map[global_selected_gender_display_label]

# 5. Age Band Filter
global_selected_age_band = st.sidebar.selectbox(
    "Select Age Band:",
    options=age_bands_options,
    index=0,
    disabled=df_display_melted.empty # Age band only relevant for age/gender charts
)


# Main Body
st.title("UK Population Dashboard: 2011 vs 2022")
st.markdown("---")

# 1: Population Density Comparison
st.header("Population Density Comparison")
if df_display_density.empty:
    if df_density.empty and not df_density_raw.empty and not df_density_raw.empty:
         st.plotly_chart(create_empty_figure("Density data could not be processed. Please check data file."), use_container_width=True)
    elif df_density_raw.empty:
         st.plotly_chart(create_empty_figure("Density data not loaded. Please check file MYE5_Table8.csv."), use_container_width=True)
    else:
        st.plotly_chart(create_empty_figure("No density data for current filter selection (Location/Geography)."), use_container_width=True)
else:
    filtered_df_density_chart = df_display_density.sort_values('name')
    fig_density = go.Figure()
    plot_data_exists_density = False

    if global_selected_year_display == "2011 Only" or global_selected_year_display == "Comparison (2011 & 2022)":
        if 'density_2011' in filtered_df_density_chart.columns and not filtered_df_density_chart['density_2011'].dropna().empty:
            fig_density.add_trace(go.Bar(
                x=filtered_df_density_chart['name'],
                y=filtered_df_density_chart['density_2011'],
                name='Density 2011',
                marker_color=COLOR_SECONDARY if global_selected_year_display == "Comparison (2011 & 2022)" else COLOR_PRIMARY,
                hovertemplate=("**%{x}**<br>" +
                               "2011 Density: %{y:.1f} per sq km<extra></extra>")
            ))
            plot_data_exists_density = True
    if global_selected_year_display == "2022 Only" or global_selected_year_display == "Comparison (2011 & 2022)":
        if 'density_2022' in filtered_df_density_chart.columns and not filtered_df_density_chart['density_2022'].dropna().empty:
            fig_density.add_trace(go.Bar(
                x=filtered_df_density_chart['name'],
                y=filtered_df_density_chart['density_2022'],
                name='Density 2022',
                marker_color=COLOR_PRIMARY,
                hovertemplate=("**%{x}**<br>" +
                               "2022 Density: %{y:.1f} per sq km<extra></extra>")
            ))
            plot_data_exists_density = True

    if not plot_data_exists_density:
        st.plotly_chart(create_empty_figure("No density data available for the selected year(s) and locations."), use_container_width=True)
    else:
        title_density = 'Population Density'
        if global_selected_year_display != "Comparison (2011 & 2022)":
            year_suffix = global_selected_year_display.replace(" Only", "")
            title_density += f' ({year_suffix})'
        else:
            title_density += ': 2011 vs 2022'

        fig_density.update_layout(
            title=title_density,
            xaxis_title='Location',
            yaxis_title='People per Square Kilometer',
            barmode='group' if global_selected_year_display == "Comparison (2011 & 2022)" else 'overlay',
            hovermode='x unified',
            legend_title_text='Year',
            xaxis={'categoryorder': 'array',
                   'categoryarray': sorted(filtered_df_density_chart['name'].unique())},
            font={'family': FONT_FAMILY},
            margin=dict(l=40, r=20, t=60, b=40)
        )
        st.plotly_chart(fig_density, use_container_width=True)
st.markdown("---")


# 2: Population Age Distribution
st.header("Population Age Distribution")
loc_for_age_dist = None

if not df_display_detail.empty:
    # Prefer globally selected location if it's in the filtered data
    if global_selected_locations and global_selected_locations[0] in df_display_detail['name'].unique():
        loc_for_age_dist = global_selected_locations[0]
        if len(global_selected_locations) > 1:
             st.info(f"Displaying age distribution for {loc_for_age_dist} (the first selected and available location).")
    # Otherwise, if any locations are in the filtered data, pick the first one
    elif not df_display_detail['name'].empty:
        loc_for_age_dist = df_display_detail['name'].unique()[0]
        st.info(f"Displaying age distribution for {loc_for_age_dist} (first available in filtered data).")
    else: # df_display_detail is not empty but has no names (e.g. all data filtered out by geography/location)
        st.plotly_chart(create_empty_figure("No locations available in the filtered age data for distribution chart."), use_container_width=True)
elif not df_age_gender_raw.empty:
    st.plotly_chart(create_empty_figure("No age data for current filter selection (Location/Geography)."), use_container_width=True)
else:
    st.plotly_chart(create_empty_figure("Age data not loaded. Please check file MYEB1_Table9.csv."), use_container_width=True)


if loc_for_age_dist and not df_display_detail.empty: # df_display_detail is already filtered by location/geography
    fig_age_gender = go.Figure()
    data_found_for_age_dist = False

    # Further filter df_display_detail for the specific location and sex for this chart
    chart_df_age_detail = df_display_detail[
        (df_display_detail['name'] == loc_for_age_dist) &
        (df_display_detail['sex'].isin(global_selected_sex_codes))
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
                        fig_age_gender.add_trace(go.Scatter(
                            x=gender_specific_df['age'],
                            y=gender_specific_df['population_2011'],
                            name=f'{gender_label} 2011',
                            mode='lines', marker_color=color_2011, line=dict(width=2, dash='dash'),
                            hovertemplate=(f"<b>{gender_label} - Age: %{{x}}</b><br>" + "2011 Population: %{y:,}<extra></extra>")
                        ))
                if global_selected_year_display == "2022 Only" or global_selected_year_display == "Comparison (2011 & 2022)":
                    if 'population_2022' in gender_specific_df.columns:
                        fig_age_gender.add_trace(go.Scatter(
                            x=gender_specific_df['age'],
                            y=gender_specific_df['population_2022'],
                            name=f'{gender_label} 2022',
                            mode='lines', marker_color=color_2022, line=dict(width=2),
                            hovertemplate=(f"<b>{gender_label} - Age: %{{x}}</b><br>" + "2022 Population: %{y:,}<extra></extra>")
                        ))

    if not data_found_for_age_dist:
        st.plotly_chart(create_empty_figure(f"No age distribution data for {loc_for_age_dist} with selected sex/year filters."), use_container_width=True)
    else:
        title_age_dist = f'Population Age Distribution in {loc_for_age_dist}'
        if global_selected_year_display != "Comparison (2011 & 2022)":
            year_suffix_age = global_selected_year_display.replace(" Only", "")
            title_age_dist += f' ({year_suffix_age})'
        fig_age_gender.update_layout(
            title=title_age_dist, xaxis_title='Age', yaxis_title='Estimated Population',
            xaxis={'type': 'category'}, hovermode='x unified', legend_title_text='Sex & Year',
            font={'family': FONT_FAMILY}, margin=dict(l=40, r=20, t=60, b=40)
        )
        if global_selected_age_band != 'All Ages':
            x0_highlight, x1_highlight = get_vrect_coords_from_age_band(global_selected_age_band)
            if x0_highlight is not None and x1_highlight is not None:
                fig_age_gender.add_vrect(
                    x0=x0_highlight, x1=x1_highlight, fillcolor="rgba(128,128,128,0.2)",
                    layer="below", line_width=0,
                )
        st.plotly_chart(fig_age_gender, use_container_width=True)
# No explicit else here if loc_for_age_dist is None, covered by initial checks for df_display_detail

st.markdown("---")

# 3: Gender Population Comparison
st.header("Gender Population Comparison")
year_for_gender_comp_actual = None
if global_selected_year_display == "2011 Only": year_for_gender_comp_actual = 2011
elif global_selected_year_display == "2022 Only": year_for_gender_comp_actual = 2022
elif global_selected_year_display == "Comparison (2011 & 2022)":
    year_for_gender_comp_actual = 2022 # Default to 2022 for this chart
    st.info("Gender Population Comparison chart defaults to 2022 data when 'Comparison (2011 & 2022)' is selected globally for Year.")

if df_display_melted.empty:
    if not df_age_gender_raw.empty: # If raw data was loaded but filtered to empty
        st.plotly_chart(create_empty_figure("No data for gender comparison with current Location/Geography filters."), use_container_width=True)
    # If df_age_gender_raw is empty, error handled at load time or age distribution section
elif not year_for_gender_comp_actual: # Should not happen due to default year selection
    st.plotly_chart(create_empty_figure("Select a year to display gender comparison."), use_container_width=True)
else:
    # df_display_melted is already filtered by location and geography
    chart_df_gender_melted = df_display_melted[df_display_melted['Year'] == year_for_gender_comp_actual].copy()

    age_title_part = f"({global_selected_age_band})"
    if global_selected_age_band != 'All Ages':
        chart_df_gender_melted = chart_df_gender_melted[chart_df_gender_melted['age_band'] == global_selected_age_band]

    chart_df_gender_melted = chart_df_gender_melted[chart_df_gender_melted['sex'].isin(global_selected_sex_codes)]

    if chart_df_gender_melted.empty: # Check after all filters for this chart
        st.plotly_chart(create_empty_figure(f"No data for gender comparison in {year_for_gender_comp_actual} {age_title_part} with selected filters."), use_container_width=True)
    else:
        grouped_df_gender = chart_df_gender_melted.groupby(['name', 'sex'])['Population'].sum().reset_index()
        grouped_df_gender = grouped_df_gender.sort_values(by=['name', 'sex'])
        fig_gender_comp = go.Figure()
        plot_data_exists_gender = False

        for sex_code in global_selected_sex_codes:
            gender_label = genders_map[sex_code]
            df_sex_specific = grouped_df_gender[grouped_df_gender['sex'] == sex_code]
            if not df_sex_specific.empty:
                plot_data_exists_gender = True
                color = COLOR_MALE if sex_code == 'M' else COLOR_FEMALE
                fig_gender_comp.add_trace(go.Bar(
                    x=df_sex_specific['name'], y=df_sex_specific['Population'], name=gender_label, marker_color=color,
                    hovertemplate=(f"<b>%{{x}}</b><br>{gender_label}s: %{{y:,}}<br>" +
                                   f"Year: {year_for_gender_comp_actual}<br>Age Band: {global_selected_age_band}<extra></extra>")
                ))

        if not plot_data_exists_gender:
            st.plotly_chart(create_empty_figure(f"No data for the selected sex(es) in {year_for_gender_comp_actual} {age_title_part}."), use_container_width=True)
        else:
            fig_gender_comp.update_layout(
                title=f'Population by Sex in {year_for_gender_comp_actual} {age_title_part}',
                xaxis_title='Location', yaxis_title='Population', barmode='group', hovermode='x unified',
                legend_title_text='Sex',
                xaxis={'categoryorder': 'array', 'categoryarray': sorted(grouped_df_gender['name'].unique())},
                font={'family': FONT_FAMILY}, margin=dict(l=40, r=20, t=60, b=40)
            )
            st.plotly_chart(fig_gender_comp, use_container_width=True)

# footer
st.markdown("---")
url = "https://github.com/ingridientzzz"
st.markdown(f"[App based on an original by: ingridientzzz]({url})")
