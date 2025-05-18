import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Set Streamlit page layout to wide
st.set_page_config(layout='wide')

# Define color and font constants for consistent styling
FONT_FAMILY = '"Helvetica Neue", Helvetica, Arial, sans-serif'
COLOR_BG = '#FFFFFF'
COLOR_TEXT = '#000000'
COLOR_PRIMARY = '#6699CC'        # Muted Blue
COLOR_SECONDARY = '#44AA99'      # Teal
COLOR_MALE = '#88CCEE'           # Light Cyan / Sky Blue
COLOR_FEMALE = '#806699'         # Muted Purple
COLOR_FEMALE_2011_DIST = '#A993BD' # Lighter Muted Purple

@st.cache_data
def load_data(age_gender_csv_path, density_csv_path):
    # Load age/gender and density CSVs, validate columns, handle errors
    df_age_gender_raw = pd.DataFrame()
    df_density_raw = pd.DataFrame()
    try:
        df_age_gender_raw = pd.read_csv(age_gender_csv_path)
        expected_cols_age = [
            'code', 'name', 'geography', 'sex', 'age', 'population_2011', 'population_2022'
        ]
        if not all(col in df_age_gender_raw.columns for col in expected_cols_age):
            st.error(f"Age/Gender CSV missing columns. Found: {df_age_gender_raw.columns.tolist()}.")
            df_age_gender_raw = pd.DataFrame()
    except FileNotFoundError:
        st.error(f"Age/Gender CSV not found at {age_gender_csv_path}")
    except Exception as e:
        st.error(f"Error loading Age/Gender CSV: {e}")
    try:
        df_density_raw = pd.read_csv(density_csv_path)
        expected_cols_density = [
            'Code', 'Name', 'Geography', 'Area (sq km)',
            'Estimated Population mid-2022', '2022 people per sq. km',
            'Estimated Population mid-2011', '2011 people per sq. km'
        ]
        if not all(col in df_density_raw.columns for col in expected_cols_density):
            st.error(f"Density CSV missing columns. Found: {df_density_raw.columns.tolist()}.")
            df_density_raw = pd.DataFrame()
    except FileNotFoundError:
        st.error(f"Density CSV not found at {density_csv_path}")
    except Exception as e:
        st.error(f"Error loading Density CSV: {e}")
    return df_age_gender_raw, df_density_raw

@st.cache_data
def preprocess_density_data_wide(df_density_raw):
    # Clean and rename density columns, ensure numeric types, select relevant columns
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
    # Convert wide density data to long format for plotting by year
    needed = ['code', 'name', 'geography', 'density_2011', 'density_2022']
    if df_density_wide.empty or not all(col in df_density_wide.columns for col in needed):
        st.warning("Cannot melt density data: columns missing.")
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
    # Clean age/gender data, create age bands, melt to long format for plotting
    if df_age_gender_raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    df_age_gender_detail = df_age_gender_raw.copy()
    df_age_gender_detail['age_numeric'] = (
        df_age_gender_detail['age'].astype(str).replace('90+', '90').astype(int)
    )
    df_age_gender_detail['age_band'] = pd.cut(
        df_age_gender_detail['age_numeric'],
        bins=bins, labels=labels, right=True, include_lowest=True
    )
    df_age_gender_melted = df_age_gender_detail.melt(
        id_vars=[
            'code', 'name', 'geography', 'sex', 'age', 'age_numeric', 'age_band'
        ],
        value_vars=['population_2011', 'population_2022'],
        var_name='Year_Col', value_name='Population'
    )
    df_age_gender_melted['Year'] = (
        df_age_gender_melted['Year_Col'].str.extract(r'(\d+)').astype(int)
    )
    df_age_gender_melted = df_age_gender_melted.drop(columns=['Year_Col'])
    df_age_gender_melted['Population'] = pd.to_numeric(
        df_age_gender_melted['Population'], errors='coerce'
    ).fillna(0)
    df_age_gender_melted = df_age_gender_melted.sort_values(
        by=['name', 'Year', 'sex', 'age_numeric']
    )
    df_age_gender_detail['population_2011'] = pd.to_numeric(
        df_age_gender_detail['population_2011'], errors='coerce'
    ).fillna(0)
    df_age_gender_detail['population_2022'] = pd.to_numeric(
        df_age_gender_detail['population_2022'], errors='coerce'
    ).fillna(0)
    return df_age_gender_detail, df_age_gender_melted

@st.cache_data
def create_merged_df(df_age_gender_melted, df_density_melted):
    # Merge age/gender and density data on location, geography, and year
    if df_age_gender_melted.empty or df_density_melted.empty:
        st.warning("Cannot create merged data: input dataframes empty.")
        return pd.DataFrame()
    df_age_gender_melted['Year'] = df_age_gender_melted['Year'].astype(int)
    df_density_melted['Year'] = df_density_melted['Year'].astype(int)
    merged_df = pd.merge(
        df_age_gender_melted,
        df_density_melted,
        on=['code', 'name', 'geography', 'Year'],
        how='left'
    )
    return merged_df

# Define mappings and options for filters and age bands
genders_map = {'M': 'Male', 'F': 'Female'}
bins = [-1, 17, 24, 39, 59, 74, np.inf]
labels = ['0-17', '18-24', '25-39', '40-59', '60-74', '75+']
age_bands_options = ['All Ages'] + labels

# File paths for data
age_gender_csv_file_path = 'MYEB1_Table9.csv'
density_csv_file_path = 'MYE5_Table8.csv'

# Load raw data from CSVs
df_age_gender_raw, df_density_raw = load_data(
    age_gender_csv_file_path, density_csv_file_path
)
# Preprocess density data to wide and melted (long) formats
df_density_wide = preprocess_density_data_wide(df_density_raw)
df_density_melted = melt_density_data(df_density_wide)

# Prepare lists of all locations and geographies for filter options
all_locations_master = []
all_geographies_master = []

df_age_gender_detail = pd.DataFrame()
df_age_gender_melted = pd.DataFrame()
merged_df = pd.DataFrame()

# Preprocess age/gender data and collect unique locations/geographies
if not df_age_gender_raw.empty:
    df_age_gender_detail, df_age_gender_melted = preprocess_age_gender_data(
        df_age_gender_raw, bins, labels
    )
    if 'name' in df_age_gender_melted.columns:
        all_locations_master.extend(df_age_gender_melted['name'].unique())
    if 'geography' in df_age_gender_melted.columns:
        all_geographies_master.extend(df_age_gender_melted['geography'].unique())

# Collect unique locations/geographies from density data
if not df_density_wide.empty:
    if 'name' in df_density_wide.columns:
        all_locations_master.extend(df_density_wide['name'].unique())
    if 'geography' in df_density_wide.columns:
        all_geographies_master.extend(df_density_wide['geography'].unique())

# Merge age/gender and density data for advanced analysis
if not df_age_gender_melted.empty and not df_density_melted.empty:
    merged_df = create_merged_df(df_age_gender_melted, df_density_melted)

# Deduplicate and sort filter options
all_locations = sorted(list(set(all_locations_master)))
all_geographies = sorted(list(set(all_geographies_master)))

# Warn if any preprocessing step failed
if df_density_wide.empty and not df_density_raw.empty:
    st.warning("Failed to process density data (wide).")
if df_density_melted.empty and not df_density_wide.empty:
    st.warning("Failed to process density data (melted).")
if merged_df.empty and not (df_age_gender_melted.empty or df_density_melted.empty):
    st.warning("Failed to create merged dataset.")

def create_empty_figure(title_text):
    # Utility: create a blank Plotly figure with a title
    fig = go.Figure()
    fig.update_layout(
        title=title_text, xaxis={'visible': False}, yaxis={'visible': False},
        font={'family': FONT_FAMILY, 'color': COLOR_TEXT}
    )
    return fig

def get_vrect_coords_from_age_band(age_band_str):
    # Utility: get x0, x1 for highlighting age band in chart
    if age_band_str == 'All Ages':
        return None, None
    if age_band_str == '75+':
        return '75', '90+'
    parts = age_band_str.split('-')
    if len(parts) == 2:
        return str(parts[0]), str(parts[1])
    return None, None

# --- SIDEBAR FILTERS ---

st.sidebar.header("Global Filters")
# Geography filter
selected_geographies = st.sidebar.multiselect(
    "Select Geography Type(s):", options=all_geographies, default=all_geographies
)

# Filter locations based on selected geographies
locations_for_filter_options = all_locations
if selected_geographies:
    temp_locations = []
    if not df_age_gender_melted.empty:
        temp_locations.extend(
            df_age_gender_melted[
                df_age_gender_melted['geography'].isin(selected_geographies)
            ]['name'].unique()
        )
    if not df_density_wide.empty:
        temp_locations.extend(
            df_density_wide[
                df_density_wide['geography'].isin(selected_geographies)
            ]['name'].unique()
        )
    locations_for_filter_options = sorted(list(set(temp_locations)))

# Default locations for filter (UK countries if present)
default_locations = [
    loc for loc in ['ENGLAND', 'SCOTLAND', 'WALES', 'NORTHERN IRELAND']
    if loc in locations_for_filter_options
]
if not default_locations and locations_for_filter_options:
    default_locations = locations_for_filter_options[:min(4, len(locations_for_filter_options))]
# Location filter
global_selected_locations = st.sidebar.multiselect(
    "Select Location(s):", options=locations_for_filter_options, default=default_locations
)

# --- FILTER DATAFRAMES BASED ON FILTERS ---

df_display_melted = pd.DataFrame()
df_display_detail = pd.DataFrame()
df_display_density = pd.DataFrame()
df_display_merged = pd.DataFrame()

# Filter age/gender melted data
if not df_age_gender_melted.empty:
    master_filter_age_gender = (
        df_age_gender_melted['name'].isin(global_selected_locations) &
        df_age_gender_melted['geography'].isin(selected_geographies)
    )
    df_display_melted = df_age_gender_melted[master_filter_age_gender]
# Filter age/gender detail data
if not df_age_gender_detail.empty:
    master_filter_detail = (
        df_age_gender_detail['name'].isin(global_selected_locations) &
        df_age_gender_detail['geography'].isin(selected_geographies)
    )
    df_display_detail = df_age_gender_detail[master_filter_detail]
# Filter density data
if not df_density_wide.empty:
    master_filter_density = (
        df_density_wide['name'].isin(global_selected_locations) &
        df_density_wide['geography'].isin(selected_geographies)
    )
    df_display_density = df_density_wide[master_filter_density]
# Filter merged data
if not merged_df.empty:
    master_filter_merged = (
        merged_df['name'].isin(global_selected_locations) &
        merged_df['geography'].isin(selected_geographies)
    )
    df_display_merged = merged_df[master_filter_merged]

# --- YEAR, SEX, AGE BAND FILTERS ---

# Disable year filter if no data
disable_year_filter = (
    df_display_melted.empty and df_display_density.empty and df_display_merged.empty
)
global_selected_year_display = st.sidebar.selectbox(
    "Select Year(s) to Display:",
    options=["Comparison (2011 & 2022)", "2011 Only", "2022 Only"],
    index=0, disabled=disable_year_filter
)

# Sex filter (Both, Male, Female)
sex_options_map = {"Both": ["M", "F"], "Male": ["M"], "Female": ["F"]}
global_selected_gender_display_label = st.sidebar.selectbox(
    "Select Sex (for applicable charts):",
    options=["Both", "Male", "Female"], index=0, disabled=df_display_melted.empty
)
global_selected_sex_codes = sex_options_map[global_selected_gender_display_label]

# Age band filter
global_selected_age_band = st.sidebar.selectbox(
    "Select Age Band (for applicable charts):",
    options=age_bands_options, index=0, disabled=df_display_melted.empty
)

# --- MAIN PAGE CONTENT ---

st.title("UK Population Dashboard: 2011 vs 2022")
st.markdown("---")

# 1. Population Density Comparison
st.header("1. Population Density Comparison")
if df_display_density.empty:
    # Show empty chart if no data
    if df_density_wide.empty and not df_density_raw.empty:
        st.plotly_chart(create_empty_figure("Density data could not be processed."), use_container_width=True)
    elif df_density_raw.empty:
        st.plotly_chart(create_empty_figure("Density data not loaded."), use_container_width=True)
    else:
        st.plotly_chart(create_empty_figure("No density data for current filter selection."), use_container_width=True)
else:
    # Plot density bars for selected years and locations
    filtered_df_density_chart = df_display_density.sort_values('name')
    fig_density = go.Figure()
    plot_data_exists_density = False
    if global_selected_year_display in ["2011 Only", "Comparison (2011 & 2022)"]:
        if 'density_2011' in filtered_df_density_chart.columns and not filtered_df_density_chart['density_2011'].dropna().empty:
            fig_density.add_trace(go.Bar(
                x=filtered_df_density_chart['name'],
                y=filtered_df_density_chart['density_2011'],
                name='Density 2011',
                marker_color=COLOR_SECONDARY,
                hovertemplate="**%{x}**<br>2011 Density: %{y:.1f} per sq km<extra></extra>"
            ))
            plot_data_exists_density = True
    if global_selected_year_display in ["2022 Only", "Comparison (2011 & 2022)"]:
        if 'density_2022' in filtered_df_density_chart.columns and not filtered_df_density_chart['density_2022'].dropna().empty:
            fig_density.add_trace(go.Bar(
                x=filtered_df_density_chart['name'],
                y=filtered_df_density_chart['density_2022'],
                name='Density 2022',
                marker_color=COLOR_PRIMARY,
                hovertemplate="**%{x}**<br>2022 Density: %{y:.1f} per sq km<extra></extra>"
            ))
            plot_data_exists_density = True
    if not plot_data_exists_density:
        st.plotly_chart(create_empty_figure("No density data for selected year(s)/locations."), use_container_width=True)
    else:
        title_density = 'Population Density'
        if global_selected_year_display != "Comparison (2011 & 2022)":
            title_density += f' ({global_selected_year_display.replace(" Only", "")})'
        else:
            title_density += ': 2011 vs 2022'
        fig_density.update_layout(
            title=title_density, xaxis_title='Location', yaxis_title='People per Square Kilometer',
            barmode='group' if global_selected_year_display == "Comparison (2011 & 2022)" else 'overlay',
            hovermode='x unified', legend_title_text='Year',
            xaxis={'categoryorder': 'array', 'categoryarray': sorted(filtered_df_density_chart['name'].unique())},
            font={'family': FONT_FAMILY, 'color': COLOR_TEXT}, margin=dict(l=40, r=20, t=60, b=40)
        )
        st.plotly_chart(fig_density, use_container_width=True)
st.markdown("---")

# 2. Population Age Distribution
st.header("2. Population Age Distribution")
loc_for_age_dist = None
if not df_display_detail.empty:
    # Pick first selected location for age distribution chart
    if global_selected_locations and global_selected_locations[0] in df_display_detail['name'].unique():
        loc_for_age_dist = global_selected_locations[0]
        if len(global_selected_locations) > 1:
            st.info(f"Displaying age distribution for {loc_for_age_dist} (first selected).")
    elif not df_display_detail['name'].empty:
        loc_for_age_dist = df_display_detail['name'].unique()[0]
        st.info(f"Displaying age distribution for {loc_for_age_dist} (first available).")
    else:
        st.plotly_chart(create_empty_figure("No locations in filtered age data."), use_container_width=True)
elif not df_age_gender_raw.empty:
    st.plotly_chart(create_empty_figure("No age data for current filters."), use_container_width=True)
else:
    st.plotly_chart(create_empty_figure("Age data not loaded."), use_container_width=True)

if loc_for_age_dist and not df_display_detail.empty:
    # Plot age distribution lines for selected location, sex, and year(s)
    fig_age_gender = go.Figure()
    data_found_for_age_dist = False
    chart_df_age_detail = df_display_detail[
        (df_display_detail['name'] == loc_for_age_dist) &
        (df_display_detail['sex'].isin(global_selected_sex_codes))
    ].sort_values('age_numeric')
    if not chart_df_age_detail.empty:
        for gender_code_loop in global_selected_sex_codes:
            gender_specific_df = chart_df_age_detail[chart_df_age_detail['sex'] == gender_code_loop]
            if not gender_specific_df.empty:
                data_found_for_age_dist = True
                gender_label_loop = genders_map[gender_code_loop]
                color_2022 = COLOR_MALE if gender_code_loop == 'M' else COLOR_FEMALE
                color_2011 = COLOR_PRIMARY if gender_code_loop == 'M' else COLOR_FEMALE_2011_DIST
                if global_selected_year_display in ["2011 Only", "Comparison (2011 & 2022)"]:
                    if 'population_2011' in gender_specific_df.columns:
                        fig_age_gender.add_trace(go.Scatter(
                            x=gender_specific_df['age'],
                            y=gender_specific_df['population_2011'],
                            name=f'{gender_label_loop} 2011',
                            mode='lines',
                            marker_color=color_2011,
                            line=dict(width=2, dash='dash'),
                            hovertemplate=f"<b>{gender_label_loop} - Age: %{{x}}</b><br>2011 Pop: %{{y:,}}<extra></extra>"
                        ))
                if global_selected_year_display in ["2022 Only", "Comparison (2011 & 2022)"]:
                    if 'population_2022' in gender_specific_df.columns:
                        fig_age_gender.add_trace(go.Scatter(
                            x=gender_specific_df['age'],
                            y=gender_specific_df['population_2022'],
                            name=f'{gender_label_loop} 2022',
                            mode='lines',
                            marker_color=color_2022,
                            line=dict(width=2),
                            hovertemplate=f"<b>{gender_label_loop} - Age: %{{x}}</b><br>2022 Pop: %{{y:,}}<extra></extra>"
                        ))
    if not data_found_for_age_dist:
        st.plotly_chart(create_empty_figure(f"No age data for {loc_for_age_dist} with filters."), use_container_width=True)
    else:
        title_age_dist = f'Pop Age Distribution in {loc_for_age_dist}'
        if global_selected_year_display != "Comparison (2011 & 2022)":
            title_age_dist += f' ({global_selected_year_display.replace(" Only", "")})'
        fig_age_gender.update_layout(
            title=title_age_dist, xaxis_title='Age', yaxis_title='Est. Population',
            xaxis={'type': 'category'}, hovermode='x unified', legend_title_text='Sex & Year',
            font={'family': FONT_FAMILY, 'color': COLOR_TEXT}, margin=dict(l=40, r=20, t=60, b=40)
        )
        # Highlight selected age band if not 'All Ages'
        if global_selected_age_band != 'All Ages':
            x0_h, x1_h = get_vrect_coords_from_age_band(global_selected_age_band)
            if x0_h is not None and x1_h is not None:
                fig_age_gender.add_vrect(
                    x0=x0_h, x1=x1_h, fillcolor="rgba(128,128,128,0.2)",
                    layer="below", line_width=0
                )
        st.plotly_chart(fig_age_gender, use_container_width=True)
st.markdown("---")

# 3. Gender Population Comparison
st.header("3. Gender Population Comparison")
# Determine which year to use for gender comparison chart
year_for_gender_comp_actual = None
if global_selected_year_display == "2011 Only":
    year_for_gender_comp_actual = 2011
elif global_selected_year_display == "2022 Only":
    year_for_gender_comp_actual = 2022
elif global_selected_year_display == "Comparison (2011 & 2022)":
    year_for_gender_comp_actual = 2022
    st.info("Gender Pop Comparison chart defaults to 2022 data for 'Comparison'.")

if df_display_melted.empty:
    # Show empty chart if no data
    if not df_age_gender_raw.empty:
        st.plotly_chart(create_empty_figure("No gender data for current Location/Geo filters."), use_container_width=True)
elif not year_for_gender_comp_actual:
    st.plotly_chart(create_empty_figure("Select year for gender comparison."), use_container_width=True)
else:
    # Plot gender population bars for selected locations, sex, and age band
    chart_df_gender_melted = df_display_melted[df_display_melted['Year'] == year_for_gender_comp_actual].copy()
    age_title_part = f"({global_selected_age_band})"
    if global_selected_age_band != 'All Ages':
        chart_df_gender_melted = chart_df_gender_melted[chart_df_gender_melted['age_band'] == global_selected_age_band]
    chart_df_gender_melted = chart_df_gender_melted[chart_df_gender_melted['sex'].isin(global_selected_sex_codes)]
    if chart_df_gender_melted.empty:
        st.plotly_chart(create_empty_figure(
            f"No data for gender comparison in {year_for_gender_comp_actual} {age_title_part} with filters."
        ), use_container_width=True)
    else:
        grouped_df_gender = chart_df_gender_melted.groupby(['name', 'sex'])['Population'].sum().reset_index().sort_values(by=['name', 'sex'])
        fig_gender_comp = go.Figure()
        plot_data_exists_gender = False
        for sex_code_loop in global_selected_sex_codes:
            gender_label_loop = genders_map[sex_code_loop]
            df_sex_specific = grouped_df_gender[grouped_df_gender['sex'] == sex_code_loop]
            if not df_sex_specific.empty:
                plot_data_exists_gender = True
                color = COLOR_MALE if sex_code_loop == 'M' else COLOR_FEMALE
                fig_gender_comp.add_trace(go.Bar(
                    x=df_sex_specific['name'],
                    y=df_sex_specific['Population'],
                    name=gender_label_loop,
                    marker_color=color,
                    hovertemplate=f"<b>%{{x}}</b><br>{gender_label_loop}s: %{{y:,}}<br>Year: {year_for_gender_comp_actual}<br>Age Band: {global_selected_age_band}<extra></extra>"
                ))
        if not plot_data_exists_gender:
            st.plotly_chart(create_empty_figure(
                f"No data for selected sex(es) in {year_for_gender_comp_actual} {age_title_part}."
            ), use_container_width=True)
        else:
            fig_gender_comp.update_layout(
                title=f'Pop by Sex in {year_for_gender_comp_actual} {age_title_part}',
                xaxis_title='Location', yaxis_title='Population', barmode='group',
                hovermode='x unified', legend_title_text='Sex',
                xaxis={'categoryorder': 'array', 'categoryarray': sorted(grouped_df_gender['name'].unique())},
                font={'family': FONT_FAMILY, 'color': COLOR_TEXT}, margin=dict(l=40, r=20, t=60, b=40)
            )
            st.plotly_chart(fig_gender_comp, use_container_width=True)
st.markdown("---")

# 4. Sex Ratio by Population Density
st.header("4. Sex Ratio by Population Density")
st.info("Sex ratio is Males per 100 Females. This chart ignores the global 'Select Sex' filter.")

if df_display_merged.empty or 'Density' not in df_display_merged.columns:
    # Show empty chart if merged data is missing
    st.plotly_chart(create_empty_figure(
        "Merged data with density is not available for sex ratio analysis."
    ), use_container_width=True)
else:
    # Prepare data for sex ratio scatter plot by density
    sex_ratio_data = df_display_merged.copy()
    years_to_plot_sex_ratio = []
    if global_selected_year_display == "2011 Only":
        years_to_plot_sex_ratio = [2011]
    elif global_selected_year_display == "2022 Only":
        years_to_plot_sex_ratio = [2022]
    elif global_selected_year_display == "Comparison (2011 & 2022)":
        years_to_plot_sex_ratio = [2011, 2022]
    if not years_to_plot_sex_ratio:
        st.warning("Please select a year or 'Comparison' to display sex ratio by density.")
    else:
        sex_ratio_data = sex_ratio_data[sex_ratio_data['Year'].isin(years_to_plot_sex_ratio)]
        age_band_title_part_sr = f"(Age Band: {global_selected_age_band})"
        if global_selected_age_band != 'All Ages':
            sex_ratio_data = sex_ratio_data[sex_ratio_data['age_band'] == global_selected_age_band]
        else:
            age_band_title_part_sr = "(All Ages)"
        if sex_ratio_data.empty or sex_ratio_data['Density'].dropna().empty:
            st.plotly_chart(create_empty_figure(
                f"No data for sex ratio analysis with current filters {age_band_title_part_sr}."
            ), use_container_width=True)
        else:
            # Calculate sex ratio (M per 100 F) for each location/year
            pop_by_sex_for_ratio = sex_ratio_data.groupby(
                ['code', 'name', 'geography', 'Year', 'Density', 'sex']
            )['Population'].sum().unstack(fill_value=0).reset_index()
            if 'M' not in pop_by_sex_for_ratio.columns:
                pop_by_sex_for_ratio['M'] = 0
            if 'F' not in pop_by_sex_for_ratio.columns:
                pop_by_sex_for_ratio['F'] = 0
            pop_by_sex_for_ratio['sex_ratio'] = (
                (pop_by_sex_for_ratio['M'] / pop_by_sex_for_ratio['F'].replace(0, pd.NA)) * 100
            )
            final_plot_data_sr = pop_by_sex_for_ratio.dropna(subset=['Density', 'sex_ratio'])
            if final_plot_data_sr.empty:
                st.plotly_chart(create_empty_figure(
                    f"Not enough data to calculate sex ratios for the scatter plot {age_band_title_part_sr}."
                ), use_container_width=True)
            else:
                # Plot scatter of sex ratio vs density for each year
                fig_sex_ratio_density = go.Figure()
                plot_data_exists_sr = False
                for year_val in years_to_plot_sex_ratio:
                    year_data_sr = final_plot_data_sr[final_plot_data_sr['Year'] == year_val]
                    if not year_data_sr.empty:
                        plot_data_exists_sr = True
                        marker_color_sr = COLOR_PRIMARY if year_val == 2022 else COLOR_SECONDARY
                        fig_sex_ratio_density.add_trace(go.Scatter(
                            x=year_data_sr['Density'],
                            y=year_data_sr['sex_ratio'],
                            mode='markers',
                            name=f'Sex Ratio {year_val}',
                            marker=dict(color=marker_color_sr, size=8, opacity=0.7),
                            text=year_data_sr['name'],
                            hovertemplate=(
                                "<b>%{text} (%{customdata[0]})</b><br>"
                                "Density: %{x:.1f} per sq km<br>"
                                "Sex Ratio: %{y:.1f} (Males per 100 Females)<br>"
                                "Males: %{customdata[1]:,}<br>Females: %{customdata[2]:,}"
                                "<extra></extra>"
                            ),
                            customdata=year_data_sr[['Year', 'M', 'F']]
                        ))
                if not plot_data_exists_sr:
                    st.plotly_chart(create_empty_figure(
                        f"No valid sex ratio data to plot for selected year(s) {age_band_title_part_sr}."
                    ), use_container_width=True)
                else:
                    title_sr_density = f'Sex Ratio vs. Population Density {age_band_title_part_sr}'
                    if len(years_to_plot_sex_ratio) == 1:
                        title_sr_density += f' - {years_to_plot_sex_ratio[0]}'
                    fig_sex_ratio_density.update_layout(
                        title=title_sr_density,
                        xaxis_title='Population Density (people per sq km)',
                        yaxis_title='Sex Ratio (Males per 100 Females)',
                        font=dict(family=FONT_FAMILY, color=COLOR_TEXT),
                        hovermode='closest', legend_title_text='Year'
                    )
                    st.plotly_chart(fig_sex_ratio_density, use_container_width=True)

st.markdown("---")
# Attribution footer
url = "https://github.com/ingridientzzz"
st.markdown(f"[By: ingridientzzz]({url})")
