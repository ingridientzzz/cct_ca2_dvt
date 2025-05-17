# imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# set streamlit to Wide mode by default
st.set_page_config(layout='wide')

# Configs
FONT_FAMILY = '"Helvetica Neue", Helvetica, Arial, sans-serif'
# Okabe & Ito color-blind safe palette
COLOR_BG = '#FFFF' # This seems to be white, might be intended as #FFFFFF
COLOR_TEXT = '#0000' # This seems to be black, might be intended as #000000
COLOR_PRIMARY = '#56B4E9' # sky blue (Used for Male 2011 in dist graph)
COLOR_SECONDARY = '#9999' # grey # This seems to be an incomplete hex, assuming #999999
COLOR_MALE = '#0072B2' # blue (Used for Male 2022 in dist graph)
COLOR_FEMALE = '#E69F00' # orange (Used for Female 2022 in dist graph)
COLOR_FEMALE_2011_DIST = '#CC79A7' # Reddish Purple (For Female 2011 in dist graph)

# Corrected color codes if they were shorthand
COLOR_BG = '#FFFFFF'
COLOR_TEXT = '#000000'
COLOR_SECONDARY = '#999999'


# load data from CSV
@st.cache_data
def load_data(csv_file_path):
    # Load the dataframe from the provided CSV
    df_age_gender_raw = pd.read_csv(csv_file_path)
    # Basic check for expected columns
    expected_cols = ['code', 'name', 'geography', 'sex', 'age', 'population_2011', 'population_2022']
    if not all(col in df_age_gender_raw.columns for col in expected_cols):
        st.error(f"CSV file is missing expected columns. Found: {df_age_gender_raw.columns.tolist()}. Expected: {expected_cols}")
        return pd.DataFrame() # Return empty df on error
    return df_age_gender_raw


@st.cache_data
def preprocess_age_gender_data(df_age_gender_raw, bins, labels):
    if df_age_gender_raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_age_gender_detail = df_age_gender_raw.copy()
    # Ensure 'age' column is treated as string before replacement, then int
    df_age_gender_detail['age_numeric'] = df_age_gender_detail['age'].astype(str).replace('90+', '90').astype(int)
    df_age_gender_detail['age_band'] = pd.cut(
        df_age_gender_detail['age_numeric'],
        bins=bins, labels=labels, right=True, include_lowest=True
    )
    
    # Melt the dataframe to long format for population years
    df_age_gender_melted = df_age_gender_detail.melt(
        id_vars=['code', 'name', 'geography', 'sex', 'age', 'age_numeric', 'age_band'], # Added 'code', 'geography'
        value_vars=['population_2011', 'population_2022'],
        var_name='Year_Col',
        value_name='Population'
    )
    df_age_gender_melted['Year'] = df_age_gender_melted['Year_Col'].str.extract(r'(\d+)').astype(int)
    df_age_gender_melted = df_age_gender_melted.drop(columns=['Year_Col'])
    
    # Ensure population is numeric, coercing errors to NaN and then filling with 0
    df_age_gender_melted['Population'] = pd.to_numeric(df_age_gender_melted['Population'], errors='coerce').fillna(0)

    df_age_gender_melted = df_age_gender_melted.sort_values(by=['name', 'Year', 'sex', 'age_numeric'])
    
    # For df_age_gender_detail, ensure population columns are numeric
    df_age_gender_detail['population_2011'] = pd.to_numeric(df_age_gender_detail['population_2011'], errors='coerce').fillna(0)
    df_age_gender_detail['population_2022'] = pd.to_numeric(df_age_gender_detail['population_2022'], errors='coerce').fillna(0)

    return df_age_gender_detail, df_age_gender_melted


# build other details
genders_map = {'M': 'Male', 'F': 'Female'}
bins = [-1, 17, 24, 39, 59, 74, np.inf] # np.inf will include 90+ correctly
labels = ['0-17', '18-24', '25-39', '40-59', '60-74', '75+']    # age bands
age_bands_options = ['All Ages'] + labels

# prep base dfs, dropdowns
# Streamlit handles them differently
csv_file_path = 'MYEB1_Table9.csv'
df_age_gender_raw = load_data(csv_file_path)

if not df_age_gender_raw.empty:
    df_age_gender_detail, df_age_gender_melted = preprocess_age_gender_data(df_age_gender_raw, bins, labels)
    all_locations = sorted(list(df_age_gender_melted['name'].unique()))
    all_geographies = sorted(list(df_age_gender_melted['geography'].unique()))
else:
    # Initialize empty structures if data loading fails
    df_age_gender_detail = pd.DataFrame()
    df_age_gender_melted = pd.DataFrame()
    all_locations = []
    all_geographies = []
    st.error("Failed to load or process data. Please check the CSV file.")

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
        # In the data, '90+' is the max category for individual ages.
        # The age band '75+' covers ages 75 up to and including 90+.
        return '75', '90+' # '90+' is a category label
    parts = age_band_str.split('-')
    if len(parts) == 2:
        return str(parts[0]), str(parts[1])
    return None, None

# Global selection sidebar
st.sidebar.header("Global Filters")

# Initialize filtered_df for subsequent filters
filtered_df_for_cascading_filters = df_age_gender_melted.copy() if not df_age_gender_melted.empty else pd.DataFrame()

# 1. Geography Filter
if not filtered_df_for_cascading_filters.empty and 'geography' in filtered_df_for_cascading_filters.columns:
    unique_geographies = sorted(filtered_df_for_cascading_filters['geography'].unique())
    selected_geographies = st.sidebar.multiselect(
        "Select Geography Type(s):",
        options=unique_geographies,
        default=unique_geographies # Default to all initially or a sensible subset
    )
    if selected_geographies:
        filtered_df_for_cascading_filters = filtered_df_for_cascading_filters[filtered_df_for_cascading_filters['geography'].isin(selected_geographies)]
else:
    selected_geographies = []
    st.sidebar.multiselect("Select Geography Type(s):", options=[], disabled=True)


# 2. Location Filter (dependent on selected geographies)
if not filtered_df_for_cascading_filters.empty and 'name' in filtered_df_for_cascading_filters.columns:
    available_locations = sorted(filtered_df_for_cascading_filters['name'].unique())
    default_locations = [loc for loc in ['ENGLAND', 'SCOTLAND', 'WALES', 'NORTHERN IRELAND'] if loc in available_locations]
    if not default_locations and available_locations: # if preferred defaults not found, pick first few
        default_locations = available_locations[:min(4, len(available_locations))]

    global_selected_locations = st.sidebar.multiselect(
        "Select Location(s):",
        options=available_locations,
        default=default_locations
    )
else:
    global_selected_locations = []
    st.sidebar.multiselect("Select Location(s):", options=[], disabled=True)


# Filter main dataframes based on selected locations and geographies
if not df_age_gender_melted.empty:
    master_filter = (df_age_gender_melted['name'].isin(global_selected_locations))
    if selected_geographies: # Apply geography filter if any selected
         master_filter &= (df_age_gender_melted['geography'].isin(selected_geographies))
    
    df_display_melted = df_age_gender_melted[master_filter]
    
    # Also filter df_age_gender_detail for the age distribution chart
    master_filter_detail = (df_age_gender_detail['name'].isin(global_selected_locations))
    if selected_geographies:
        master_filter_detail &= (df_age_gender_detail['geography'].isin(selected_geographies))
    df_display_detail = df_age_gender_detail[master_filter_detail]

else:
    df_display_melted = pd.DataFrame()
    df_display_detail = pd.DataFrame()


# 3. Year Filter
global_selected_year_display = st.sidebar.selectbox(
    "Select Year(s) to Display:",
    options=["Comparison (2011 & 2022)", "2011 Only", "2022 Only"],
    index=0,
    disabled=df_display_melted.empty
)

# 4. Sex Filter
sex_options_map = {"Both": ["M", "F"], "Male": ["M"], "Female": ["F"]}
global_selected_gender_display_label = st.sidebar.selectbox(
    "Select Sex:",
    options=["Both", "Male", "Female"],
    index=0,
    disabled=df_display_melted.empty
)
global_selected_sex_codes = sex_options_map[global_selected_gender_display_label]

# 5. Age Band Filter
global_selected_age_band = st.sidebar.selectbox(
    "Select Age Band:",
    options=age_bands_options,
    index=0, # Default to "All Ages"
    disabled=df_display_melted.empty
)


# Main Body
st.title("UK Population Dashboard: 2011 vs 2022")
st.markdown("---")

# The "Population Density Comparison" chart section is removed as per requirements.

# 1: Population Age Distribution (was 2)
st.header("Population Age Distribution")
loc_for_age_dist = None

if global_selected_locations:
    # Use the first selected location for this chart, if multiple are selected.
    loc_for_age_dist = global_selected_locations[0]
    if len(global_selected_locations) > 1:
        st.info(f"Displaying age distribution for {loc_for_age_dist} (the first selected location). Others are included in aggregations if applicable in other charts.")
elif not df_display_detail.empty and 'name' in df_display_detail.columns:
    # If no locations are selected but data is available, pick the first available one
    available_names = df_display_detail['name'].unique()
    if len(available_names) > 0:
        loc_for_age_dist = available_names[0]
        st.info(f"No location selected. Displaying age distribution for {loc_for_age_dist} by default.")
    else:
        st.plotly_chart(create_empty_figure("No locations available in the filtered data."), use_container_width=True)
else:
    st.plotly_chart(create_empty_figure("Please select a location for age distribution."), use_container_width=True)


if loc_for_age_dist and not df_display_detail.empty:
    fig_age_gender = go.Figure()
    data_found_for_age_dist = False

    # Filter df_display_detail further for the specific location and sex for this chart
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
                    fig_age_gender.add_trace(go.Scatter(
                        x=gender_specific_df['age'], # 'age' column contains '0', '1'... '90+'
                        y=gender_specific_df['population_2011'],
                        name=f'{gender_label} 2011',
                        mode='lines',
                        marker_color=color_2011,
                        line=dict(width=2, dash='dash'),
                        hovertemplate=(f"<b>{gender_label} - Age: %{{x}}</b><br>" +
                                       "2011 Population: %{y:,}<extra></extra>")
                    ))
                if global_selected_year_display == "2022 Only" or global_selected_year_display == "Comparison (2011 & 2022)":
                    fig_age_gender.add_trace(go.Scatter(
                        x=gender_specific_df['age'],
                        y=gender_specific_df['population_2022'],
                        name=f'{gender_label} 2022',
                        mode='lines',
                        marker_color=color_2022,
                        line=dict(width=2),
                        hovertemplate=(f"<b>{gender_label} - Age: %{{x}}</b><br>" +
                                       "2022 Population: %{y:,}<extra></extra>")
                    ))
    
    if not data_found_for_age_dist:
        st.plotly_chart(create_empty_figure(f"No age distribution data for {loc_for_age_dist} with selected filters."), use_container_width=True)
    else:
        title_age_dist = f'Population Age Distribution in {loc_for_age_dist}'
        if global_selected_year_display != "Comparison (2011 & 2022)":
            year_suffix_age = global_selected_year_display.replace(" Only", "")
            title_age_dist += f' ({year_suffix_age})'
        
        fig_age_gender.update_layout(
            title=title_age_dist,
            xaxis_title='Age',
            yaxis_title='Estimated Population',
            xaxis={'type': 'category'}, # Keep as category for '0', '1', ..., '90+'
            hovermode='x unified',
            legend_title_text='Sex & Year',
            font={'family': FONT_FAMILY},
            margin=dict(l=40, r=20, t=60, b=40)
        )

        if global_selected_age_band != 'All Ages':
            x0_highlight, x1_highlight = get_vrect_coords_from_age_band(global_selected_age_band)
            if x0_highlight is not None and x1_highlight is not None:
                fig_age_gender.add_vrect(
                    x0=x0_highlight, x1=x1_highlight,
                    fillcolor="rgba(128,128,128,0.2)",
                    layer="below", 
                    line_width=0,
                )
        st.plotly_chart(fig_age_gender, use_container_width=True)
elif loc_for_age_dist and df_display_detail.empty and not df_age_gender_raw.empty :
     st.warning(f"No data available for location '{loc_for_age_dist}' after applying Geography filters. Try adjusting Geography selection.")
elif df_age_gender_raw.empty:
    pass # Error already shown during data loading
else: # No loc_for_age_dist and df_display_detail is empty
    st.plotly_chart(create_empty_figure("No data available for the selected filters."), use_container_width=True)


st.markdown("---")

# 2: Gender Population Comparison (was 3)
st.header("Gender Population Comparison")
year_for_gender_comp_actual = None
if global_selected_year_display == "2011 Only":
    year_for_gender_comp_actual = 2011
elif global_selected_year_display == "2022 Only":
    year_for_gender_comp_actual = 2022
elif global_selected_year_display == "Comparison (2011 & 2022)":
    # Default to 2022 for this chart if comparison is selected, or allow selection?
    # For now, let's make it explicit or use both if possible.
    # The original code defaulted to 2022. Let's stick to that for now.
    year_for_gender_comp_actual = 2022 
    st.info("Gender Population Comparison chart defaults to 2022 data when 'Comparison (2011 & 2022)' is selected globally.")

# Further filter df_display_melted based on year and age band for this specific chart
if not df_display_melted.empty and year_for_gender_comp_actual:
    chart_df_gender_melted = df_display_melted[df_display_melted['Year'] == year_for_gender_comp_actual].copy()

    age_title_part = f"({global_selected_age_band})"
    if global_selected_age_band != 'All Ages':
        chart_df_gender_melted = chart_df_gender_melted[chart_df_gender_melted['age_band'] == global_selected_age_band]
    else:
        age_title_part = "(All Ages)" # Already set

    # Filter by selected sex codes
    chart_df_gender_melted = chart_df_gender_melted[chart_df_gender_melted['sex'].isin(global_selected_sex_codes)]

    grouped_df_gender = chart_df_gender_melted.groupby(
        ['name', 'sex'] # Group by name and sex
    )['Population'].sum().reset_index()

    if grouped_df_gender.empty:
        st.plotly_chart(create_empty_figure(f"No data for gender comparison with current selections in {year_for_gender_comp_actual} {age_title_part}."), use_container_width=True)
    else:
        grouped_df_gender = grouped_df_gender.sort_values(by=['name', 'sex'])
        fig_gender_comp = go.Figure()

        # Iterate through selected sex codes to add traces
        for sex_code in global_selected_sex_codes:
            gender_label = genders_map[sex_code]
            df_sex_specific = grouped_df_gender[grouped_df_gender['sex'] == sex_code]
            
            if not df_sex_specific.empty:
                color = COLOR_MALE if sex_code == 'M' else COLOR_FEMALE
                fig_gender_comp.add_trace(go.Bar(
                    x=df_sex_specific['name'],
                    y=df_sex_specific['Population'],
                    name=gender_label,
                    marker_color=color,
                    hovertemplate=(
                        f"<b>%{{x}}</b><br>" +
                        f"{gender_label}s: %{{y:,}}<br>" + # Pluralize label
                        f"Year: {year_for_gender_comp_actual}<br>" +
                        f"Age Band: {global_selected_age_band}<extra></extra>"
                    )
                ))
        
        if not fig_gender_comp.data: # Check if any traces were actually added
            st.plotly_chart(create_empty_figure(f"No data for the selected sex(es) in {year_for_gender_comp_actual} {age_title_part}."), use_container_width=True)
        else:
            fig_gender_comp.update_layout(
                title=f'Population by Sex in {year_for_gender_comp_actual} {age_title_part}',
                xaxis_title='Location',
                yaxis_title='Population',
                barmode='group',
                hovermode='x unified',
                legend_title_text='Sex',
                xaxis={'categoryorder': 'array',
                       'categoryarray': sorted(global_selected_locations if global_selected_locations else grouped_df_gender['name'].unique())},
                font={'family': FONT_FAMILY},
                margin=dict(l=40, r=20, t=60, b=40)
            )
            st.plotly_chart(fig_gender_comp, use_container_width=True)
elif df_age_gender_raw.empty:
    pass # Error already shown
else:
    st.plotly_chart(create_empty_figure("Select year, location(s), sex, and age band for gender comparison."), use_container_width=True)
    
# footer
st.markdown("---")
url = "https://github.com/ingridientzzz"
st.markdown(f"[App based on an original by: ingridientzzz]({url})")
