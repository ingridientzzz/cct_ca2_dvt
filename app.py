import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# Configs
FONT_FAMILY = '"Helvetica Neue", Helvetica, Arial, sans-serif'
# Okabe & Ito color-blind safe palette
COLOR_BG = '#FFFFFF'
COLOR_TEXT = '#000000'
COLOR_PRIMARY = '#56B4E9'  # sky blue
COLOR_SECONDARY = '#999999'  # grey
COLOR_MALE = '#0072B2'  # blue
COLOR_FEMALE = '#E69F00'  # orange
PLOTLY_TEMPLATE = 'plotly_white'

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
density_locations = sorted(list(df_density['name'].unique()))
age_gender_locations = sorted(list(df_age_gender_melted['name'].unique()))
genders_map = {'M': 'Male', 'F': 'Female'}
genders_options = {v: k for k, v in genders_map.items()} # built selectbox display to value

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

url = "https://github.com/ingridientzzz"
# st.write(f"check out this [link]({url})")
st.markdown(f"[profile: ingridientzzz]({url})")

st.title("UK Population Dashboard: 2011 vs 2022")
st.markdown("---")  # add line to separate sections

# 1: Population Density Comparison
st.header("Population Density Comparison")
selected_density_locations = st.multiselect(
    "Select Geographic Location(s) for Density:",
    options=density_locations,
    default=['ENGLAND', 'SCOTLAND', 'WALES', 'NORTHERN IRELAND']
)

if not selected_density_locations:
    st.plotly_chart(create_empty_figure("Please select at least one location for density comparison."), use_container_width=True)
else:
    filtered_df_density = df_density[
        df_density['name'].isin(selected_density_locations)
    ].sort_values('name')

    fig_density = go.Figure()
    fig_density.add_trace(go.Bar(
        x=filtered_df_density['name'],
        y=filtered_df_density['density_2011'],
        name='Density 2011',
        marker_color=COLOR_SECONDARY,
        hovertemplate=("<b>%{x}</b><br>" +
                       "2011 Density: %{y:.1f} per sq km<extra></extra>")
    ))
    fig_density.add_trace(go.Bar(
        x=filtered_df_density['name'],
        y=filtered_df_density['density_2022'],
        name='Density 2022',
        marker_color=COLOR_PRIMARY,
        hovertemplate=("<b>%{x}</b><br>" +
                       "2022 Density: %{y:.1f} per sq km<extra></extra>")
    ))
    fig_density.update_layout(
        title='Population Density: 2011 vs 2022',
        xaxis_title='Location',
        yaxis_title='People per Square Kilometer',
        barmode='group',
        hovermode='x unified',
        legend_title_text='Year',
        xaxis={'categoryorder': 'array',
               'categoryarray': sorted(selected_density_locations)},
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font={'family': FONT_FAMILY, 'color': COLOR_TEXT},
        margin=dict(l=40, r=20, t=60, b=40)
    )
    st.plotly_chart(fig_density, use_container_width=True)

st.markdown("---")

# 2: Population Age Distribution
st.header("Population Age Distribution")

col1_age_dist, col2_age_dist = st.columns(2)

with col1_age_dist:
    selected_age_gender_location = st.selectbox(
        "Select Geographic Location for Age Distribution:",
        options=age_gender_locations,
        index=age_gender_locations.index('ENGLAND') if 'ENGLAND' in age_gender_locations else 0
    )
with col2_age_dist:
    selected_gender_display = st.selectbox(
        "Select Gender for Age Distribution:",
        options=list(genders_map.values()), # 'Male', 'Female'
        index=0 # Default to 'Male' for now
    )
    selected_gender = genders_options[selected_gender_display] # revert to 'M' or 'F'


if not selected_age_gender_location or not selected_gender:
    st.plotly_chart(create_empty_figure("Please select location and gender for age distribution."), use_container_width=True)
else:
    filtered_df_age_detail = df_age_gender_detail[
        (df_age_gender_detail['name'] == selected_age_gender_location) &
        (df_age_gender_detail['sex'] == selected_gender)
    ].sort_values('age_numeric')

    if filtered_df_age_detail.empty:
        st.plotly_chart(create_empty_figure(f"No data for {selected_age_gender_location} / {genders_map[selected_gender]}"), use_container_width=True)
    else:
        fig_age_gender = go.Figure()
        fig_age_gender.add_trace(go.Scatter(
            x=filtered_df_age_detail['age'],
            y=filtered_df_age_detail['population_2011'],
            name='Population 2011',
            mode='lines+markers',
            marker_color=COLOR_SECONDARY,
            line=dict(width=2),
            hovertemplate=("<b>Age: %{x}</b><br>" +
                           "2011 Population: %{y:,}<extra></extra>")
        ))
        fig_age_gender.add_trace(go.Scatter(
            x=filtered_df_age_detail['age'],
            y=filtered_df_age_detail['population_2022'],
            name='Population 2022',
            mode='lines+markers',
            marker_color=COLOR_PRIMARY,
            line=dict(width=2),
            hovertemplate=("<b>Age: %{x}</b><br>" +
                           "2022 Population: %{y:,}<extra></extra>")
        ))
        fig_age_gender.update_layout(
            title=f'{genders_map[selected_gender]} Population Age Distribution in {selected_age_gender_location}',
            xaxis_title='Age',
            yaxis_title='Estimated Population',
            xaxis={'type': 'category'},
            hovermode='x unified',
            legend_title_text='Year',
            paper_bgcolor=COLOR_BG,
            plot_bgcolor=COLOR_BG,
            font={'family': FONT_FAMILY, 'color': COLOR_TEXT},
            margin=dict(l=40, r=20, t=60, b=40)
        )
        st.plotly_chart(fig_age_gender, use_container_width=True)

st.markdown("---")

# 3: Gender Population Comparison
st.header("Gender Population Comparison")

col1_gender_comp, col2_gender_comp, col3_gender_comp = st.columns(3)

with col1_gender_comp:
    selected_gender_comp_year = st.selectbox(
        "Select Year for Gender Comparison:",
        options=[2011, 2022],
        index=1 # Default to 2022
    )
with col2_gender_comp:
    selected_gender_comp_locs = st.multiselect(
        "Select Location(s) for Gender Comparison:",
        options=age_gender_locations,
        default=['ENGLAND', 'SCOTLAND', 'WALES', 'NORTHERN IRELAND']
    )
with col3_gender_comp:
    selected_gender_comp_age_band = st.selectbox(
        "Select Age Band for Gender Comparison:",
        options=age_bands_options,
        index=0 # Ddefault --> 'All Ages'
    )

if not selected_gender_comp_locs or not selected_gender_comp_year or not selected_gender_comp_age_band:
    st.plotly_chart(create_empty_figure("Select year, location(s), and age band for gender comparison."), use_container_width=True)
else:
    filtered_df_gender_melted = df_age_gender_melted[
        (df_age_gender_melted['Year'] == selected_gender_comp_year) &
        (df_age_gender_melted['name'].isin(selected_gender_comp_locs))
    ].copy()

    age_title_part = f"({selected_gender_comp_age_band})"
    if selected_gender_comp_age_band != 'All Ages':
        filtered_df_gender_melted = filtered_df_gender_melted[filtered_df_gender_melted['age_band'] == selected_gender_comp_age_band]
    else:
        age_title_part = "(All Ages)"


    grouped_df_gender = filtered_df_gender_melted.groupby(
        ['name', 'sex']
    )['Population'].sum().reset_index()

    if grouped_df_gender.empty:
        st.plotly_chart(create_empty_figure(f"No data for selection in {selected_gender_comp_year} {age_title_part}"), use_container_width=True)
    else:
        grouped_df_gender = grouped_df_gender.sort_values(by=['name', 'sex'])
        fig_gender_comp = go.Figure()

        df_female = grouped_df_gender[grouped_df_gender['sex'] == 'F']
        fig_gender_comp.add_trace(go.Bar(
            x=df_female['name'],
            y=df_female['Population'],
            name='Female',
            marker_color=COLOR_FEMALE,
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Females: %{y:,}<br>" +
                f"Year: {selected_gender_comp_year}<br>" +
                f"Age Band: {selected_gender_comp_age_band}<extra></extra>"
            )
        ))
        df_male = grouped_df_gender[grouped_df_gender['sex'] == 'M']
        fig_gender_comp.add_trace(go.Bar(
            x=df_male['name'],
            y=df_male['Population'],
            name='Male',
            marker_color=COLOR_MALE,
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Males: %{y:,}<br>" +
                f"Year: {selected_gender_comp_year}<br>" +
                f"Age Band: {selected_gender_comp_age_band}<extra></extra>"
            )
        ))
        fig_gender_comp.update_layout(
            title=f'Population by Gender in {selected_gender_comp_year} {age_title_part}',
            xaxis_title='Location',
            yaxis_title='Population',
            barmode='group',
            hovermode='x unified',
            legend_title_text='Gender',
            xaxis={'categoryorder': 'array',
                   'categoryarray': sorted(selected_gender_comp_locs)},
            paper_bgcolor=COLOR_BG,
            plot_bgcolor=COLOR_BG,
            font={'family': FONT_FAMILY, 'color': COLOR_TEXT},
            margin=dict(l=40, r=20, t=60, b=40)
        )
        st.plotly_chart(fig_gender_comp, use_container_width=True)