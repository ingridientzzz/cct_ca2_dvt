# imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_js_eval import streamlit_js_eval # Added for theme detection

# --- Theme Detection and Plotly Template Setup ---
# Determine Plotly template based on Streamlit theme
# This JS attempts to get the theme set by Streamlit on the body's data-baseweb attribute.
# Falls back to prefers-color-scheme if the attribute isn't found.
js_to_get_theme = """
function getStreamlitTheme() {
    const streamlitDoc = window.parent.document;
    if (streamlitDoc && streamlitDoc.body) {
        const theme = streamlitDoc.body.getAttribute('data-baseweb');
        if (theme === 'dark' || theme === 'light') {
            return theme;
        }
    }
    // Fallback if data-baseweb is not available or not set yet (e.g., initial load might be too fast)
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        return 'dark';
    }
    return 'light'; // Default to light
}
getStreamlitTheme();
"""
try:
    current_streamlit_theme = streamlit_js_eval(js_to_get_theme)
except Exception:
    # In case of any JS evaluation error, default to light theme
    current_streamlit_theme = "light"

if current_streamlit_theme == "dark":
    plotly_template = "plotly_dark"
    highlight_fill_color = "rgba(255, 255, 255, 0.15)" # Lighter highlight for dark theme
else:
    plotly_template = "plotly_white"
    highlight_fill_color = "rgba(0, 0, 0, 0.1)"   # Darker highlight for light theme
# --- End Theme Detection ---


# load data - use st.cache_data for more responsive app
@st.cache_data
def load_data():
    # Assuming CSV files are always present in ./data/
    # Make sure these paths are correct for your environment
    try:
        df_density_raw = pd.read_csv("./data/MYE5_Table8.csv")
        df_age_gender_raw = pd.read_csv("./data/MYEB1_Table9.csv")
    except FileNotFoundError:
        st.error("Data files not found. Please ensure 'MYE5_Table8.csv' and 'MYEB1_Table9.csv' are in a 'data' subdirectory.")
        st.stop()
    return df_density_raw, df_age_gender_raw

@st.cache_data
def preprocess_density_data(df_density_raw):
    df_density = df_density_raw.rename(columns={
    'Area (sq km)': 'area_sq_km',
    'Estimated Population mid-2022': 'population_2022',
    '2022 people per sq. km': 'density_2022',
    'Estimated Population mid-2011': 'population_2011',
    '2011 people per sq. km': 'density_2011',
    'Name': 'name',
    'Code': 'code',
    'Geography': 'geography'
    })
    return df_density

@st.cache_data
def preprocess_age_gender_data(df_age_gender_raw):
    df_age_gender_detail = df_age_gender_raw.copy()
    df_age_gender_detail['age_numeric'] = df_age_gender_detail['age'].replace('90+', 90).astype(int)
    bins = [-1, 17, 24, 39, 59, 74, np.inf]
    labels = ['0-17', '18-24', '25-39', '40-59', '60-74', '75+']
    df_age_gender_detail['age_band'] = pd.cut(
    df_age_gender_detail['age_numeric'],
    bins=bins, labels=labels, right=True
    )
    df_age_gender_melted = df_age_gender_detail.melt(
    id_vars=['name', 'sex', 'age', 'age_numeric', 'age_band'],
    value_vars=['population_2011', 'population_2022'],
    var_name='Year_Col',
    value_name='Population'
    )
    df_age_gender_melted['Year'] = df_age_gender_melted['Year_Col'].str.extract(r'(\d+)').astype(int)
    df_age_gender_melted = df_age_gender_melted.drop(columns=['Year_Col'])
    df_age_gender_melted = df_age_gender_melted.sort_values(by=['name', 'Year', 'sex', 'age_numeric'])
    return df_age_gender_detail, df_age_gender_melted

# prep base dfs
df_density_raw, df_age_gender_raw = load_data()
df_density = preprocess_density_data(df_density_raw)
df_age_gender_detail, df_age_gender_melted = preprocess_age_gender_data(df_age_gender_raw)

# get values for dropdowns
all_locations = sorted(list(df_age_gender_melted['name'].unique()))
genders_map = {'M': 'Male', 'F': 'Female'}
age_bands_raw = ['0-17', '18-24', '25-39', '40-59', '60-74', '75+']
age_bands_options = ['All Ages'] + age_bands_raw

# helper functions
def create_empty_figure(title_text, template): # Added template argument
    fig = go.Figure()
    fig.update_layout(
        title=title_text,
        xaxis={'visible': False},
        yaxis={'visible': False},
        template=template # Use the dynamic template
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

url = "https://github.com/ingridientzzz" # Assuming this is intentional
st.markdown(f"[profile: ingridientzzz]({url})") # Assuming this is intentional
st.title("UK Population Dashboard: 2011 vs 2022")
st.markdown("---")

# --- Global Controls Section ---
st.sidebar.header("Global Filters")

global_selected_locations = st.sidebar.multiselect(
    "Select Geographic Location(s):",
    options=all_locations,
    default=['ENGLAND', 'SCOTLAND', 'WALES', 'NORTHERN IRELAND']
)

global_selected_year_display = st.sidebar.selectbox(
    "Select Year(s) to Display:",
    options=["Comparison (2011 & 2022)", "2011 Only", "2022 Only"],
    index=0
)

global_selected_gender_display = st.sidebar.selectbox(
    "Select Gender:",
    options=["Both", "Male", "Female"],
    index=0
)

global_selected_age_band = st.sidebar.selectbox(
    "Select Age Band:",
    options=age_bands_options,
    index=0
)
st.markdown("---")

# 1: Population Density Comparison
st.header("Population Density Comparison")
if not global_selected_locations:
    st.plotly_chart(create_empty_figure("Please select at least one location.", template=plotly_template), use_container_width=True)
else:
    filtered_df_density = df_density[
    df_density['name'].isin(global_selected_locations)
    ].sort_values('name')

    fig_density = go.Figure()

    if global_selected_year_display == "2011 Only" or global_selected_year_display == "Comparison (2011 & 2022)":
        fig_density.add_trace(go.Bar(
            x=filtered_df_density['name'],
            y=filtered_df_density['density_2011'],
            name='Density 2011',
            # marker_color removed, Plotly theme will assign color
            hovertemplate=("**%{x}**<br>" +
                           "2011 Density: %{y:.1f} per sq km<extra></extra>")
        ))
    if global_selected_year_display == "2022 Only" or global_selected_year_display == "Comparison (2011 & 2022)":
        fig_density.add_trace(go.Bar(
            x=filtered_df_density['name'],
            y=filtered_df_density['density_2022'],
            name='Density 2022',
            # marker_color removed, Plotly theme will assign color
            hovertemplate=("**%{x}**<br>" +
                           "2022 Density: %{y:.1f} per sq km<extra></extra>")
        ))

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
               'categoryarray': sorted(global_selected_locations)},
        template=plotly_template, # Use dynamic template
        margin=dict(l=40, r=20, t=60, b=40)
        # Removed paper_bgcolor, plot_bgcolor, font
    )
    st.plotly_chart(fig_density, use_container_width=True)
st.markdown("---")

# 2: Population Age Distribution
st.header("Population Age Distribution")
loc_for_age_dist = None
if global_selected_locations:
    loc_for_age_dist = global_selected_locations[0]
    if len(global_selected_locations) > 1:
        st.info(f"Displaying age distribution for {loc_for_age_dist} (the first selected location).")
else:
    st.plotly_chart(create_empty_figure("Please select a location for age distribution.", template=plotly_template), use_container_width=True)

if loc_for_age_dist:
    genders_to_plot = []
    if global_selected_gender_display == "Male":
        genders_to_plot.append('M')
    elif global_selected_gender_display == "Female":
        genders_to_plot.append('F')
    elif global_selected_gender_display == "Both":
        genders_to_plot.extend(['M', 'F'])

    if not genders_to_plot:
        st.plotly_chart(create_empty_figure("Please select a gender.", template=plotly_template), use_container_width=True)
    else:
        fig_age_gender = go.Figure()
        data_found_for_age_dist = False

        for gender_code in genders_to_plot:
            filtered_df_age_detail = df_age_gender_detail[
                (df_age_gender_detail['name'] == loc_for_age_dist) &
                (df_age_gender_detail['sex'] == gender_code)
            ].sort_values('age_numeric')

            if not filtered_df_age_detail.empty:
                data_found_for_age_dist = True
                gender_label = genders_map[gender_code]

                if global_selected_year_display == "2011 Only" or global_selected_year_display == "Comparison (2011 & 2022)":
                    fig_age_gender.add_trace(go.Scatter(
                        x=filtered_df_age_detail['age'],
                        y=filtered_df_age_detail['population_2011'],
                        name=f'{gender_label} 2011',
                        mode='lines+markers',
                        # marker_color removed
                        line=dict(width=2, dash='dash'),
                        hovertemplate=(f"<b>{gender_label} - Age: %{{x}}</b><br>" +
                                       "2011 Population: %{y:,}<extra></extra>")
                    ))
                if global_selected_year_display == "2022 Only" or global_selected_year_display == "Comparison (2011 & 2022)":
                    fig_age_gender.add_trace(go.Scatter(
                        x=filtered_df_age_detail['age'],
                        y=filtered_df_age_detail['population_2022'],
                        name=f'{gender_label} 2022',
                        mode='lines+markers',
                        # marker_color removed
                        line=dict(width=2),
                        hovertemplate=(f"<b>{gender_label} - Age: %{{x}}</b><br>" +
                                       "2022 Population: %{y:,}<extra></extra>")
                    ))

        if not data_found_for_age_dist:
            st.plotly_chart(create_empty_figure(f"No age distribution data for {loc_for_age_dist} with selected gender/year.", template=plotly_template), use_container_width=True)
        else:
            title_age_dist = f'Population Age Distribution in {loc_for_age_dist}'
            if global_selected_year_display != "Comparison (2011 & 2022)":
                year_suffix_age = global_selected_year_display.replace(" Only", "")
                title_age_dist += f' ({year_suffix_age})'
            
            fig_age_gender.update_layout(
                title=title_age_dist,
                xaxis_title='Age',
                yaxis_title='Estimated Population',
                xaxis={'type': 'category'},
                hovermode='x unified',
                legend_title_text='Gender & Year',
                template=plotly_template, # Use dynamic template
                margin=dict(l=40, r=20, t=60, b=40)
                # Removed paper_bgcolor, plot_bgcolor, font
            )

            if global_selected_age_band != 'All Ages':
                x0_highlight, x1_highlight = get_vrect_coords_from_age_band(global_selected_age_band)
                if x0_highlight is not None and x1_highlight is not None:
                    fig_age_gender.add_vrect(
                        x0=x0_highlight, x1=x1_highlight,
                        fillcolor=highlight_fill_color, # Use adaptive highlight color
                        layer="below", 
                        line_width=0,
                    )
            st.plotly_chart(fig_age_gender, use_container_width=True)
st.markdown("---")

# 3: Gender Population Comparison
st.header("Gender Population Comparison")
year_for_gender_comp = None
if global_selected_year_display == "2011 Only":
    year_for_gender_comp = 2011
elif global_selected_year_display == "2022 Only":
    year_for_gender_comp = 2022
elif global_selected_year_display == "Comparison (2011 & 2022)":
    year_for_gender_comp = 2022 # Defaulting to 2022 for comparison view
    st.info("Gender Population Comparison chart defaults to 2022 data when 'Comparison (2011 & 2022)' is selected globally.")

if not global_selected_locations or not year_for_gender_comp or not global_selected_age_band: # global_selected_age_band was missing in original condition
    st.plotly_chart(create_empty_figure("Select year, location(s), and age band for gender comparison.", template=plotly_template), use_container_width=True)
else:
    filtered_df_gender_melted = df_age_gender_melted[
        (df_age_gender_melted['Year'] == year_for_gender_comp) &
        (df_age_gender_melted['name'].isin(global_selected_locations))
    ].copy()

    age_title_part = f"({global_selected_age_band})"
    if global_selected_age_band != 'All Ages':
        filtered_df_gender_melted = filtered_df_gender_melted[filtered_df_gender_melted['age_band'] == global_selected_age_band]
    else:
        age_title_part = "(All Ages)" # Already set, but explicit

    grouped_df_gender = filtered_df_gender_melted.groupby(
        ['name', 'sex']
    )['Population'].sum().reset_index()

    if grouped_df_gender.empty:
        st.plotly_chart(create_empty_figure(f"No data for selection in {year_for_gender_comp} {age_title_part}", template=plotly_template), use_container_width=True)
    else:
        grouped_df_gender = grouped_df_gender.sort_values(by=['name', 'sex'])
        fig_gender_comp = go.Figure()

        if global_selected_gender_display == "Female" or global_selected_gender_display == "Both":
            df_female = grouped_df_gender[grouped_df_gender['sex'] == 'F']
            if not df_female.empty:
                fig_gender_comp.add_trace(go.Bar(
                    x=df_female['name'],
                    y=df_female['Population'],
                    name='Female',
                    # marker_color removed
                    hovertemplate=(
                        "<b>%{x}</b><br>" +
                        "Females: %{y:,}<br>" +
                        f"Year: {year_for_gender_comp}<br>" +
                        f"Age Band: {global_selected_age_band}<extra></extra>"
                    )
                ))

        if global_selected_gender_display == "Male" or global_selected_gender_display == "Both":
            df_male = grouped_df_gender[grouped_df_gender['sex'] == 'M']
            if not df_male.empty:
                fig_gender_comp.add_trace(go.Bar(
                    x=df_male['name'],
                    y=df_male['Population'],
                    name='Male',
                    # marker_color removed
                    hovertemplate=(
                        "<b>%{x}</b><br>" +
                        "Males: %{y:,}<br>" +
                        f"Year: {year_for_gender_comp}<br>" +
                        f"Age Band: {global_selected_age_band}<extra></extra>"
                    )
                ))

        if not fig_gender_comp.data: # Check if any traces were added
            st.plotly_chart(create_empty_figure(f"No data for the selected gender(s) in {year_for_gender_comp} {age_title_part}", template=plotly_template), use_container_width=True)
        else:
            fig_gender_comp.update_layout(
                title=f'Population by Gender in {year_for_gender_comp} {age_title_part}',
                xaxis_title='Location',
                yaxis_title='Population',
                barmode='group',
                hovermode='x unified',
                legend_title_text='Gender',
                xaxis={'categoryorder': 'array',
                       'categoryarray': sorted(global_selected_locations)},
                template=plotly_template, # Use dynamic template
                margin=dict(l=40, r=20, t=60, b=40)
                # Removed paper_bgcolor, plot_bgcolor, font
            )
            st.plotly_chart(fig_gender_comp, use_container_width=True)
            
