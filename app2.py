import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configs
FONT_FAMILY = '"Helvetica Neue", Helvetica, Arial, sans-serif'
COLOR_BG = '#FFFF'
COLOR_TEXT = '#0000'
COLOR_PRIMARY = '#56B4E9'  # sky blue
COLOR_SECONDARY = '#999999'  # grey (corrected from #9999)
COLOR_MALE = '#0072B2'  # blue
COLOR_FEMALE = '#E69F00'  # orange
PLOTLY_TEMPLATE = 'plotly_white'

# Corrected background and text colors if they were shorthand
if COLOR_BG == '#FFFF':
    COLOR_BG = '#FFFFFF'
if COLOR_TEXT == '#0000':
    COLOR_TEXT = '#000000'

# load data - use st.cache_data for more responsive app
@st.cache_data
def load_data():
    df_density_raw = pd.read_csv("./data/MYE5_Table8.csv")
    df_age_gender_raw = pd.read_csv("./data/MYEB1_Table9.csv")
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
    # Ensure 'age' column is string before replace, then convert to int
    df_age_gender_detail['age_numeric'] = df_age_gender_detail['age'].astype(str).replace('90+', '90').astype(int)

    bins = [-1, 17, 24, 39, 59, 74, np.inf]
    labels = ['0-17', '18-24', '25-39', '40-59', '60-74', '75+']
    df_age_gender_detail['age_band'] = pd.cut(
        df_age_gender_detail['age_numeric'],
        bins=bins, labels=labels, right=True
    )

    # Melt data for easier year-based filtering and plotting
    df_age_gender_melted = df_age_gender_detail.melt(
        id_vars=['name', 'code', 'sex', 'age', 'age_numeric', 'age_band'], # Added 'code'
        value_vars=['population_2011', 'population_2022'],
        var_name='Year_Col',
        value_name='Population'
    )
    df_age_gender_melted['Year'] = df_age_gender_melted['Year_Col'].str.extract(r'(\d+)').astype(int)
    df_age_gender_melted = df_age_gender_melted.drop(columns=['Year_Col'])
    df_age_gender_melted = df_age_gender_melted.sort_values(by=['name', 'Year', 'sex', 'age_numeric'])
    
    # Create a version of df_age_gender_detail that also has 'Year' column for easier filtering
    # This will be used for the age distribution plot if we need data for a specific year from the original wide format
    df_age_gender_detail_2011 = df_age_gender_detail[['name', 'code', 'sex', 'age', 'age_numeric', 'age_band', 'population_2011']].rename(columns={'population_2011': 'Population'})
    df_age_gender_detail_2011['Year'] = 2011
    df_age_gender_detail_2022 = df_age_gender_detail[['name', 'code', 'sex', 'age', 'age_numeric', 'age_band', 'population_2022']].rename(columns={'population_2022': 'Population'})
    df_age_gender_detail_2022['Year'] = 2022
    df_age_gender_detail_with_year = pd.concat([df_age_gender_detail_2011, df_age_gender_detail_2022])
    
    return df_age_gender_detail_with_year, df_age_gender_melted


# prep base dfs
df_density_raw, df_age_gender_raw = load_data()
df_density = preprocess_density_data(df_density_raw)
df_age_gender_detail_with_year, df_age_gender_melted = preprocess_age_gender_data(df_age_gender_raw)


# get values for dropdowns
# Use locations present in age_gender data as it's used by more plots
all_locations = sorted(list(df_age_gender_melted['name'].unique()))
genders_map = {'M': 'Male', 'F': 'Female'}
genders_options_display = list(genders_map.values()) # For display
genders_map_reverse = {v: k for k, v in genders_map.items()} # For mapping display back to code

age_bands_raw = ['0-17', '18-24', '25-39', '40-59', '60-74', '75+']
age_bands_options = ['All Ages'] + age_bands_raw


# helper functions for empty figs / no data
def create_empty_figure(title_text):
    fig = go.Figure()
    fig.update_layout(
        title=title_text,
        xaxis={'visible': False},
        yaxis={'visible': False},
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font={'family': FONT_FAMILY, 'color': COLOR_TEXT}
    )
    return fig

# App Title and Link
url = "https://github.com/ingridientzzz"
st.markdown(f"[profile: ingridientzzz]({url})")
st.title("UK Population Dashboard")
st.markdown("---")

# --- Consolidated Controls at the Top ---
st.sidebar.header("Global Filters")

selected_locations = st.sidebar.multiselect(
    "Select Geographic Location(s):",
    options=all_locations,
    default=['ENGLAND', 'SCOTLAND', 'WALES', 'NORTHERN IRELAND']
)

selected_year = st.sidebar.selectbox(
    "Select Year:",
    options=[2011, 2022],
    index=1  # Default to 2022
)

selected_gender_display = st.sidebar.selectbox(
    "Select Gender (for Age Distribution):",
    options=genders_options_display,
    index=0  # Default to 'Male'
)
selected_gender_code = genders_map_reverse[selected_gender_display]

selected_age_band = st.sidebar.selectbox(
    "Select Age Band (for Gender Comparison):",
    options=age_bands_options,
    index=0  # Default to 'All Ages'
)
st.sidebar.markdown("---")
st.sidebar.info("Selections made here will update all charts below.")

# --- Main Page Layout ---
st.header(f"Displaying Data for Year: {selected_year}")

# 1: Population Density Comparison
st.subheader("Population Density")
if not selected_locations:
    st.plotly_chart(create_empty_figure("Please select at least one location."), use_container_width=True)
else:
    filtered_df_density = df_density[
        df_density['name'].isin(selected_locations)
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    density_col = f'density_{selected_year}'
    population_col = f'population_{selected_year}'

    if density_col not in filtered_df_density.columns or population_col not in filtered_df_density.columns:
        st.plotly_chart(create_empty_figure(f"Density data not available for {selected_year}."), use_container_width=True)
    elif filtered_df_density.empty:
        st.plotly_chart(create_empty_figure(f"No density data for selected locations."), use_container_width=True)
    else:
        filtered_df_density = filtered_df_density.sort_values('name')
        fig_density = go.Figure()
        fig_density.add_trace(go.Bar(
            x=filtered_df_density['name'],
            y=filtered_df_density[density_col],
            name=f'Density {selected_year}',
            marker_color=COLOR_PRIMARY,
            hovertemplate=(f"<b>%{{x}}</b><br>" +
                           f"{selected_year} Density: %{{y:.1f}} per sq km<br>" +
                           f"{selected_year} Population: %{{customdata[0]:,}}<extra></extra>"),
            customdata=filtered_df_density[[population_col]]
        ))
        fig_density.update_layout(
            title=f'Population Density in {selected_year}',
            xaxis_title='Location',
            yaxis_title='People per Square Kilometer',
            hovermode='x unified',
            legend_title_text='Metric',
            xaxis={'categoryorder': 'array',
                   'categoryarray': sorted(selected_locations)},
            paper_bgcolor=COLOR_BG,
            plot_bgcolor=COLOR_BG,
            font={'family': FONT_FAMILY, 'color': COLOR_TEXT},
            margin=dict(l=40, r=20, t=60, b=40)
        )
        st.plotly_chart(fig_density, use_container_width=True)

st.markdown("---")

# 2: Population Age Distribution
st.subheader(f"Population Age Distribution for {selected_gender_display}s")
if not selected_locations:
    st.plotly_chart(create_empty_figure("Please select at least one location."), use_container_width=True)
else:
    # Use df_age_gender_detail_with_year which has individual age populations and a 'Year' column
    filtered_df_age_detail = df_age_gender_detail_with_year[
        (df_age_gender_detail_with_year['name'].isin(selected_locations)) &
        (df_age_gender_detail_with_year['sex'] == selected_gender_code) &
        (df_age_gender_detail_with_year['Year'] == selected_year)
    ].copy()

    if filtered_df_age_detail.empty:
        st.plotly_chart(create_empty_figure(f"No age distribution data for {', '.join(selected_locations)} / {selected_gender_display} / {selected_year}."), use_container_width=True)
    else:
        # Sum population across selected locations for each age
        age_dist_data = filtered_df_age_detail.groupby('age')['Population'].sum().reset_index()
        # Ensure 'age' is treated as categorical and sorted correctly (especially for '90+')
        # The 'age' column in df_age_gender_detail_with_year is already string and includes '90+'
        # We need to sort it numerically then convert to string for plotting if '90+' is present
        
        # Create a numeric sort key for age
        age_dist_data['age_numeric_sort'] = age_dist_data['age'].astype(str).replace('90+', '90').astype(int)
        age_dist_data = age_dist_data.sort_values('age_numeric_sort').drop(columns=['age_numeric_sort'])


        fig_age_gender = go.Figure()
        fig_age_gender.add_trace(go.Scatter(
            x=age_dist_data['age'],
            y=age_dist_data['Population'],
            name=f'Population {selected_year}',
            mode='lines+markers',
            marker_color=COLOR_PRIMARY,
            line=dict(width=2),
            hovertemplate=(f"<b>Age: %{{x}}</b><br>" +
                           f"{selected_year} Population: %{{y:,}}<extra></extra>")
        ))
        fig_age_gender.update_layout(
            title=f'{selected_gender_display} Population Age Distribution in {selected_year} <br>(Aggregated for: {", ".join(selected_locations)})',
            xaxis_title='Age',
            yaxis_title='Estimated Population',
            xaxis={'type': 'category'}, # Keep as category to handle '90+'
            hovermode='x unified',
            legend_title_text='Year',
            paper_bgcolor=COLOR_BG,
            plot_bgcolor=COLOR_BG,
            font={'family': FONT_FAMILY, 'color': COLOR_TEXT},
            margin=dict(l=40, r=20, t=80, b=40) # Increased top margin for longer title
        )
        st.plotly_chart(fig_age_gender, use_container_width=True)

st.markdown("---")

# 3: Gender Population Comparison
st.subheader(f"Gender Population Comparison for Age Band: {selected_age_band}")
if not selected_locations:
    st.plotly_chart(create_empty_figure("Please select at least one location."), use_container_width=True)
else:
    filtered_df_gender_melted = df_age_gender_melted[
        (df_age_gender_melted['Year'] == selected_year) &
        (df_age_gender_melted['name'].isin(selected_locations))
    ].copy()

    age_title_part = f"({selected_age_band})"
    if selected_age_band != 'All Ages':
        filtered_df_gender_melted = filtered_df_gender_melted[filtered_df_gender_melted['age_band'] == selected_age_band]
    
    if filtered_df_gender_melted.empty:
        st.plotly_chart(create_empty_figure(f"No data for gender comparison for this selection in {selected_year} {age_title_part}."), use_container_width=True)
    else:
        grouped_df_gender = filtered_df_gender_melted.groupby(
            ['name', 'sex']
        )['Population'].sum().reset_index()
        grouped_df_gender = grouped_df_gender.sort_values(by=['name', 'sex'])

        if grouped_df_gender.empty: # Double check after grouping
            st.plotly_chart(create_empty_figure(f"No aggregated gender data for selection in {selected_year} {age_title_part}."), use_container_width=True)
        else:
            fig_gender_comp = go.Figure()

            df_female = grouped_df_gender[grouped_df_gender['sex'] == 'F']
            if not df_female.empty:
                fig_gender_comp.add_trace(go.Bar(
                    x=df_female['name'],
                    y=df_female['Population'],
                    name='Female',
                    marker_color=COLOR_FEMALE,
                    hovertemplate=(
                        "<b>%{x}</b><br>" +
                        "Females: %{y:,}<br>" +
                        f"Year: {selected_year}<br>" +
                        f"Age Band: {selected_age_band}<extra></extra>"
                    )
                ))
            
            df_male = grouped_df_gender[grouped_df_gender['sex'] == 'M']
            if not df_male.empty:
                fig_gender_comp.add_trace(go.Bar(
                    x=df_male['name'],
                    y=df_male['Population'],
                    name='Male',
                    marker_color=COLOR_MALE,
                    hovertemplate=(
                        "<b>%{x}</b><br>" +
                        "Males: %{y:,}<br>" +
                        f"Year: {selected_year}<br>" +
                        f"Age Band: {selected_age_band}<extra></extra>"
                    )
                ))
            
            fig_gender_comp.update_layout(
                title=f'Population by Gender in {selected_year} {age_title_part}',
                xaxis_
