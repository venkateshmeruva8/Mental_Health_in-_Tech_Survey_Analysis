"""
Mental Health in Tech Survey Analysis - Streamlit Dashboard
Internship Project

This dashboard presents an interactive analysis of mental health patterns
and workplace attitudes in technology workplaces based on survey data (survey.csv).
All visualizations, cleaning logic, and business recommendations strictly adhere
to the completed Exploratory Data Analysis (EDA).
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Mental Health in Tech Survey Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for clean, professional appearance
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .insight-card {
        background-color: #F0FDF4;
        border-left: 5px solid #22C55E;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 12px 0 18px 0;
        font-size: 0.95rem;
        color: #166534;
    }
    .warning-disclaimer {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 0.9rem;
        color: #92400E;
    }
    .recommendation-card {
        background-color: #F8FAFC;
        border-left: 5px solid #3B82F6;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .recommendation-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 6px;
    }
    .recommendation-body {
        font-size: 0.93rem;
        color: #334155;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Apply unified Seaborn & Matplotlib theme
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.titlesize"] = 14
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10


# ---------------------------------------------------------
# DATA LOADING & CLEANING (Exact logic from EDA Notebook)
# ---------------------------------------------------------

@st.cache_data
def load_and_clean_data(file_path="survey.csv"):
    """
    Loads survey.csv and applies the exact cleaning rules from the EDA notebook:
      1. Age cleaning: Values outside 18-100 set to NaN.
      2. Gender cleaning: Standardized into 'Male', 'Female', 'Other'.
      3. Age_Group creation: ['18-25', '26-35', '36-45', '46-55', '56-65', '66+'].
      4. Whitespace trimming on all object/categorical columns.
    """
    if not os.path.exists(file_path):
        return None, None

    try:
        raw_df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error reading dataset: {e}")
        return None, None

    df_clean = raw_df.copy()

    # 1. Clean Age column (valid age between 18 and 100, invalid become NaN)
    df_clean["Age"] = pd.to_numeric(df_clean["Age"], errors="coerce")
    df_clean["Age"] = df_clean["Age"].where(
        df_clean["Age"].between(18, 100),
        np.nan
    )

    # 2. Clean Gender column
    def clean_gender(value):
        val = str(value).strip().lower()
        if val in ["male", "m", "man", "cis male", "male (cis)"]:
            return "Male"
        elif val in ["female", "f", "woman", "cis female", "female (cis)"]:
            return "Female"
        elif val == "nan":
            return np.nan
        else:
            return "Other"

    df_clean["Gender_Clean"] = df_clean["Gender"].apply(clean_gender)

    # 3. Create Age_Group column
    bins = [17, 25, 35, 45, 55, 65, 100]
    labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
    df_clean["Age_Group"] = pd.cut(
        df_clean["Age"],
        bins=bins,
        labels=labels
    )

    # 4. Strip whitespace in categorical columns
    categorical_columns = df_clean.select_dtypes(include="object").columns
    for col in categorical_columns:
        df_clean[col] = df_clean[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    return raw_df, df_clean


# ---------------------------------------------------------
# DYNAMIC STATISTICAL HELPER FUNCTIONS
# ---------------------------------------------------------

def get_group_treatment_stats(df, column_name, group_value):
    """
    Calculates dynamic percentage of Yes / No treatment for a specific category
    within the currently filtered dataframe.
    """
    subset = df[df[column_name] == group_value].dropna(subset=["treatment"])
    total_count = len(subset)
    if total_count == 0:
        return None, None, 0
    yes_count = (subset["treatment"] == "Yes").sum()
    no_count = (subset["treatment"] == "No").sum()
    pct_yes = (yes_count / total_count) * 100
    pct_no = (no_count / total_count) * 100
    return pct_yes, pct_no, total_count


def format_pct(val):
    """Formats numeric percentage or returns 'N/A' if None."""
    return f"{val:.1f}%" if val is not None else "N/A"


def plot_crosstab_percentage(df, index_col, columns_col="treatment", title="", xlabel="", legend_title="Treatment", rotation=0, palette=["#64748B", "#3B82F6"]):
    """Generates a percentage-based normalized stacked/grouped bar chart."""
    ct = pd.crosstab(df[index_col], df[columns_col], normalize="index") * 100
    
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ct.plot(
        kind="bar",
        ax=ax,
        color=palette[:len(ct.columns)],
        edgecolor="black",
        linewidth=0.6,
        width=0.7
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel or index_col, fontsize=11, labelpad=8)
    ax.set_ylabel("Percentage of Respondents (%)", fontsize=11, labelpad=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha="right" if rotation != 0 else "center")
    ax.legend(title=legend_title, frameon=True)
    ax.set_ylim(0, 100)
    
    # Add percentage labels on bars
    for p in ax.patches:
        height = p.get_height()
        if height > 4:
            ax.annotate(
                f"{height:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height / 2.0),
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold"
            )

    plt.tight_layout()
    return fig, ct


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

raw_df, df_clean = load_and_clean_data("survey.csv")

if raw_df is None or df_clean is None:
    st.error("⚠️ `survey.csv` was not found in the project root directory. Please make sure `survey.csv` is present.")
    st.stop()


# ---------------------------------------------------------
# SIDEBAR NAVIGATION & FILTERS WITH RELIABLE RESET
# ---------------------------------------------------------

# Default filter state initialization
DEFAULT_FILTERS = {
    "sb_gender": "All",
    "sb_age_group": "All",
    "sb_fam_history": "All",
    "sb_treatment": "All",
    "sb_country": "All",
}

for k, v in DEFAULT_FILTERS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reset_all_filters():
    """Callback ensuring every filter session key is reset to its default state."""
    for key, default_val in DEFAULT_FILTERS.items():
        st.session_state[key] = default_val


st.sidebar.title("🧠 Navigation")
menu = st.sidebar.radio(
    "Go to page:",
    [
        "Overview",
        "Data Quality",
        "Demographics",
        "Mental Health & Treatment",
        "Workplace Factors",
        "Correlation Analysis",
        "Key Insights",
        "Business Recommendations",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Interactive Filters")
st.sidebar.caption("Apply filters to explore specific demographics or employment subsets across the charts.")

# Filter options
gender_options = ["All"] + sorted([g for g in df_clean["Gender_Clean"].dropna().unique()])
age_group_options = ["All"] + ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
fam_history_options = ["All"] + sorted([f for f in df_clean["family_history"].dropna().unique()])
treatment_options = ["All"] + sorted([t for t in df_clean["treatment"].dropna().unique()])
top_countries = df_clean["Country"].value_counts().head(8).index.tolist()
country_options = ["All"] + top_countries

# Filter widgets bound to session state keys
selected_gender = st.sidebar.selectbox("Gender", gender_options, key="sb_gender")
selected_age_group = st.sidebar.selectbox("Age Group", age_group_options, key="sb_age_group")
selected_fam_history = st.sidebar.selectbox("Family History", fam_history_options, key="sb_fam_history")
selected_treatment = st.sidebar.selectbox("Treatment Sought", treatment_options, key="sb_treatment")
selected_country = st.sidebar.selectbox("Country (Top / All)", country_options, key="sb_country")

# Apply filters
filtered_df = df_clean.copy()

if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender_Clean"] == selected_gender]
if selected_age_group != "All":
    filtered_df = filtered_df[filtered_df["Age_Group"] == selected_age_group]
if selected_fam_history != "All":
    filtered_df = filtered_df[filtered_df["family_history"] == selected_fam_history]
if selected_treatment != "All":
    filtered_df = filtered_df[filtered_df["treatment"] == selected_treatment]
if selected_country != "All":
    filtered_df = filtered_df[filtered_df["Country"] == selected_country]

# Active filter statistics
pct_active = (len(filtered_df) / len(df_clean)) * 100
st.sidebar.info(f"**Filtered Records:** {len(filtered_df):,} of {len(df_clean):,} ({pct_active:.1f}%)")

# Reset button with dedicated callback
st.sidebar.button("Reset All Filters", on_click=reset_all_filters)

st.sidebar.markdown("---")
st.sidebar.caption("Internship Project: Mental Health in Tech Survey Analysis")


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown('<div class="main-title">Mental Health in Tech Survey Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">An interactive analysis of mental health patterns and workplace attitudes in the technology sector.</div>',
    unsafe_allow_html=True,
)


# =========================================================
# 1. OVERVIEW PAGE
# =========================================================
if menu == "Overview":
    st.subheader("📌 Project Overview")
    
    # KPI Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Total Responses</div>
                <div class="metric-card-value">{len(df_clean):,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Survey Features</div>
                <div class="metric-card-value">{df_clean.shape[1]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Duplicate Rows</div>
                <div class="metric-card-value">{df_clean.duplicated().sum()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Countries</div>
                <div class="metric-card-value">{df_clean['Country'].nunique()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Executive Summary & Objectives
    st.markdown("### 🎯 Executive Summary & Context")
    st.write(
        """
        The **Mental Health in Tech Survey** (conducted in 2014) is a benchmark dataset examining mental health perceptions,
        treatment prevalence, and workplace conditions among professionals in technology workplaces.
        
        The dataset contains **1,259 respondents** and **27 attributes**, capturing:
        - **Demographics:** Age, Gender, Country, State.
        - **Employment context:** Self-employment status, company size (`no_employees`), remote work, and tech company affiliation.
        - **Mental health indicators:** Personal treatment history, family history of mental health conditions, and work interference.
        - **Workplace environment:** Mental health benefits, care options, wellness programs, seeking help resources, anonymity protections, leave ease, and perceived consequences.
        """
    )

    st.markdown(
        """
        <div class="warning-disclaimer">
            <strong>Methodological Note:</strong> All findings presented in this application reflect observational survey associations.
            They indicate statistical relationships within the sample and must <strong>not</strong> be interpreted as causal claims.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📋 Dataset Explorer")
    view_mode = st.radio("Select View:", ["Cleaned Dataset Preview", "Raw Dataset Preview", "Dataset Summary Statistics"], horizontal=True)
    
    if view_mode == "Cleaned Dataset Preview":
        if len(filtered_df) == 0:
            st.warning("⚠️ No records match the active filter criteria. Reset filters to view data.")
        else:
            st.dataframe(filtered_df.head(15), use_container_width=True)
            st.caption(f"Showing first {min(15, len(filtered_df))} of {len(filtered_df):,} records.")
    elif view_mode == "Raw Dataset Preview":
        st.dataframe(raw_df.head(15), use_container_width=True)
        st.caption(f"Showing first 15 of {len(raw_df):,} raw records.")
    else:
        st.dataframe(df_clean.describe(include="all").T, use_container_width=True)


# =========================================================
# 2. DATA QUALITY PAGE
# =========================================================
elif menu == "Data Quality":
    st.subheader("🔍 Data Quality & Preprocessing")
    
    st.write(
        """
        Evaluating data quality is a critical preliminary step in the EDA workflow.
        The assessment identified missing values, categorical inconsistencies, and outlier ages,
        which were resolved according to the project's data-cleaning specifications without removing valid records.
        """
    )

    # Shape and duplicates overview
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{len(raw_df):,}")
    col2.metric("Total Columns", f"{raw_df.shape[1]}")
    col3.metric("Duplicate Records", f"{raw_df.duplicated().sum()}")
    col4.metric("Cleaned Features", f"{df_clean.shape[1]}")

    st.markdown("---")

    # Missing Value Analysis
    st.markdown("### 1. Missing Values Analysis")
    
    missing_counts = raw_df.isnull().sum()
    missing_pct = (missing_counts / len(raw_df)) * 100
    missing_table = pd.DataFrame({
        "Feature": missing_counts.index,
        "Missing Count": missing_counts.values,
        "Missing Percentage (%)": missing_pct.round(2).values
    })
    missing_table = missing_table[missing_table["Missing Count"] > 0].sort_values(by="Missing Count", ascending=False).reset_index(drop=True)

    col_t, col_p = st.columns([1.1, 1.4])
    with col_t:
        st.markdown("**Columns with Missing Values:**")
        st.dataframe(missing_table, use_container_width=True)
        st.caption("Only 4 features have missing values in the 1,259 records.")

    with col_p:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sns.barplot(
            data=missing_table,
            x="Missing Percentage (%)",
            y="Feature",
            palette="Reds_r",
            ax=ax,
            edgecolor="black",
            linewidth=0.5
        )
        ax.set_title("Missing Values by Feature (%)", fontweight="bold")
        ax.set_xlabel("Missing Percentage (%)")
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f"{width:.1f}%", (width + 1, p.get_y() + p.get_height() / 2), va="center", fontsize=9)
        ax.set_xlim(0, 100)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown(
        """
        <div class="insight-card">
            <strong>Key Data Quality Findings:</strong><br>
            • <code>comments</code> is missing in ~86.9% of responses because comments were optional.<br>
            • <code>state</code> is missing in ~41.0% of responses because it was only relevant to US respondents.<br>
            • <code>work_interfere</code> is missing in ~20.9% of responses, representing respondents who did not specify interference.<br>
            • <code>self_employed</code> has 18 missing values (~1.4%).<br>
            • No columns were dropped unnecessarily, maintaining complete dataset fidelity.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Cleaning rules summary
    st.markdown("### 2. Applied Cleaning Transformations")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Age Normalization**")
        st.write("- Original Age had anomalous entries (e.g. negative ages, >100).")
        st.write("- Values outside **18–100** were flagged as invalid and treated as `NaN`.")
        st.write("- Preserves valid working age distribution.")
    with c2:
        st.markdown("**Gender Standardization**")
        st.write("- Original `Gender` column contained diverse free-text responses.")
        st.write("- Standardized into **`Gender_Clean`** with 3 distinct groups: `Male`, `Female`, `Other`.")
    with c3:
        st.markdown("**Age Group Binning**")
        st.write("- Created **`Age_Group`** categorical feature with 6 intervals:")
        st.write("  `18-25`, `26-35`, `36-45`, `46-55`, `56-65`, `66+`.")
        st.write("- Allows standardized demographic comparisons.")


# =========================================================
# 3. DEMOGRAPHICS PAGE
# =========================================================
elif menu == "Demographics":
    st.subheader("👥 Demographic Characteristics")
    st.write("Exploration of age, gender, and geographic distribution across the surveyed tech workers.")

    if len(filtered_df) == 0:
        st.warning("⚠️ No records match the selected filter criteria. Please adjust or reset the sidebar filters.")
        st.stop()

    # Demographic Metrics
    m1, m2, m3, m4 = st.columns(4)
    valid_ages = filtered_df["Age"].dropna()
    m1.metric("Median Age", f"{valid_ages.median():.0f} years" if not valid_ages.empty else "N/A")
    m2.metric("Mean Age", f"{valid_ages.mean():.1f} years" if not valid_ages.empty else "N/A")
    pct_male = (filtered_df["Gender_Clean"] == "Male").mean() * 100 if len(filtered_df) > 0 else 0
    m3.metric("Male Proportion", f"{pct_male:.1f}%")
    top_country = filtered_df["Country"].mode()[0] if not filtered_df.empty else "N/A"
    m4.metric("Top Country", top_country)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Age & Age Groups", "⚧ Gender Distribution", "🌍 Geographic Distribution"])

    with tab1:
        col_age1, col_age2 = st.columns(2)
        with col_age1:
            st.markdown("#### Age Distribution")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            if not valid_ages.empty:
                sns.histplot(
                    data=filtered_df,
                    x="Age",
                    bins=min(20, max(5, valid_ages.nunique())),
                    kde=(valid_ages.nunique() > 3),
                    color="#3B82F6",
                    edgecolor="black",
                    linewidth=0.6,
                    ax=ax,
                )
                ax.set_title("Age Distribution of Respondents (18–100)", fontweight="bold")
                ax.set_xlabel("Age (Years)")
                ax.set_ylabel("Number of Respondents")
            else:
                ax.text(0.5, 0.5, "No valid Age records", ha="center", va="center")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_age2:
            st.markdown("#### Age Group Breakdown")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            age_grp_order = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
            sns.countplot(
                data=filtered_df,
                x="Age_Group",
                order=age_grp_order,
                palette="Blues_d",
                edgecolor="black",
                linewidth=0.6,
                ax=ax,
            )
            ax.set_title("Respondents by Standardized Age Group", fontweight="bold")
            ax.set_xlabel("Age Group")
            ax.set_ylabel("Number of Respondents")
            for p in ax.patches:
                h = p.get_height()
                if h > 0:
                    ax.annotate(f"{h}", (p.get_x() + p.get_width() / 2, h + 2), ha="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Dynamic Age Insight
        if not valid_ages.empty:
            age_grp_counts = filtered_df["Age_Group"].value_counts()
            top_grp = age_grp_counts.index[0] if len(age_grp_counts) > 0 else "N/A"
            top_grp_pct = (age_grp_counts.iloc[0] / len(filtered_df) * 100) if len(age_grp_counts) > 0 else 0
            st.caption(f"In the currently filtered data, the largest age bracket is **{top_grp}**, representing **{top_grp_pct:.1f}%** of respondents.")

    with tab2:
        col_gen1, col_gen2 = st.columns([1.2, 1])
        with col_gen1:
            st.markdown("#### Standardized Gender Distribution")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            gender_order = [g for g in ["Male", "Female", "Other"] if g in filtered_df["Gender_Clean"].values]
            if not gender_order:
                gender_order = filtered_df["Gender_Clean"].dropna().unique().tolist()
            sns.countplot(
                data=filtered_df,
                x="Gender_Clean",
                order=gender_order,
                palette=["#3B82F6", "#EC4899", "#8B5CF6"][:len(gender_order)],
                edgecolor="black",
                linewidth=0.6,
                ax=ax,
            )
            ax.set_title("Cleaned Gender Categories", fontweight="bold")
            ax.set_xlabel("Gender")
            ax.set_ylabel("Number of Respondents")
            for p in ax.patches:
                h = p.get_height()
                if h > 0:
                    ax.annotate(f"{h}", (p.get_x() + p.get_width() / 2, h + 2), ha="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_gen2:
            st.markdown("#### Gender Proportions")
            gender_counts = filtered_df["Gender_Clean"].value_counts(dropna=False)
            gender_pct = (gender_counts / len(filtered_df) * 100).round(1)
            gender_summary = pd.DataFrame({"Count": gender_counts, "Percentage (%)": gender_pct})
            st.dataframe(gender_summary, use_container_width=True)
            
            # Dynamic Gender Insight
            dyn_m_pct = (filtered_df["Gender_Clean"] == "Male").mean() * 100
            dyn_f_pct = (filtered_df["Gender_Clean"] == "Female").mean() * 100
            dyn_o_pct = (filtered_df["Gender_Clean"] == "Other").mean() * 100
            st.markdown(
                f"""
                <div class="insight-card">
                    <strong>Dynamic Gender Breakdown:</strong><br>
                    • Male: {dyn_m_pct:.1f}% ({ (filtered_df['Gender_Clean'] == 'Male').sum():,})<br>
                    • Female: {dyn_f_pct:.1f}% ({ (filtered_df['Gender_Clean'] == 'Female').sum():,})<br>
                    • Other: {dyn_o_pct:.1f}% ({ (filtered_df['Gender_Clean'] == 'Other').sum():,})
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab3:
        st.markdown("#### Top Countries by Number of Respondents")
        top_k_countries = filtered_df["Country"].value_counts().head(10)
        if not top_k_countries.empty:
            fig, ax = plt.subplots(figsize=(9, 4.8))
            sns.barplot(
                x=top_k_countries.values,
                y=top_k_countries.index,
                palette="viridis",
                edgecolor="black",
                linewidth=0.6,
                ax=ax,
            )
            ax.set_title("Top Countries in Selected Subset", fontweight="bold")
            ax.set_xlabel("Number of Respondents")
            ax.set_ylabel("Country")
            for p in ax.patches:
                w = p.get_width()
                ax.annotate(f"{int(w)}", (w + 1, p.get_y() + p.get_height() / 2), va="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No country records available in current filtered dataset.")


# =========================================================
# 4. MENTAL HEALTH & TREATMENT PAGE
# =========================================================
elif menu == "Mental Health & Treatment":
    st.subheader("🩺 Mental Health & Treatment Analysis")
    st.write(
        """
        Investigating patterns between personal characteristics (family history, gender, age)
        and whether respondents have sought mental health treatment.
        """
    )

    st.markdown(
        """
        <div class="warning-disclaimer">
            <strong>Reminder:</strong> Visualizations show percentage-based proportions across groups.
            These patterns demonstrate statistical associations, not causal links.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(filtered_df) == 0:
        st.warning("⚠️ No records match the selected filter criteria. Please adjust or reset the sidebar filters.")
        st.stop()

    chart_mode = st.radio("Display Metric:", ["Percentage of Group (%)", "Absolute Count"], horizontal=True)

    # 1. Family History vs Treatment
    st.markdown("---")
    st.markdown("### 1. Family History vs Mental Health Treatment")
    
    col_f1, col_f2 = st.columns([1.3, 1])
    with col_f1:
        if chart_mode == "Percentage of Group (%)":
            fig_fam, ct_fam = plot_crosstab_percentage(
                filtered_df,
                index_col="family_history",
                columns_col="treatment",
                title="Family History vs Treatment Rate",
                xlabel="Family History of Mental Health Conditions",
                palette=["#94A3B8", "#2563EB"]
            )
            st.pyplot(fig_fam)
            plt.close(fig_fam)
        else:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            sns.countplot(
                data=filtered_df,
                x="family_history",
                hue="treatment",
                palette=["#94A3B8", "#2563EB"],
                edgecolor="black",
                linewidth=0.6,
                ax=ax,
            )
            ax.set_title("Family History vs Treatment (Counts)", fontweight="bold")
            ax.set_xlabel("Family History")
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            ct_fam = pd.crosstab(filtered_df["family_history"], filtered_df["treatment"])

    with col_f2:
        st.markdown("**Cross-Tabulation Summary:**")
        ct_display = pd.crosstab(filtered_df["family_history"], filtered_df["treatment"], normalize="index") * 100
        st.dataframe(ct_display.round(1).astype(str) + "%", use_container_width=True)
        
        # Calculate dynamic percentages for Family History
        fam_y_pct, fam_n_pct, fam_y_n = get_group_treatment_stats(filtered_df, "family_history", "Yes")
        nofam_y_pct, nofam_n_pct, nofam_y_n = get_group_treatment_stats(filtered_df, "family_history", "No")

        fam_y_text = f"{format_pct(fam_y_pct)} Yes treatment vs {format_pct(fam_n_pct)} No (n={fam_y_n:,})" if fam_y_n > 0 else "No records in current filter"
        fam_n_text = f"{format_pct(nofam_y_pct)} Yes treatment vs {format_pct(nofam_n_pct)} No (n={nofam_y_n:,})" if nofam_y_n > 0 else "No records in current filter"

        st.markdown(
            f"""
            <div class="insight-card">
                <strong>Observed Pattern (Dynamic):</strong><br>
                Respondents with a <strong>family history</strong> of mental health conditions showed higher treatment seeking:
                <br>• <strong>Family History = Yes:</strong> {fam_y_text}
                <br>• <strong>Family History = No:</strong> {fam_n_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Gender vs Treatment
    st.markdown("---")
    st.markdown("### 2. Gender vs Mental Health Treatment")

    col_g1, col_g2 = st.columns([1.3, 1])
    with col_g1:
        if chart_mode == "Percentage of Group (%)":
            fig_gen, ct_gen = plot_crosstab_percentage(
                filtered_df,
                index_col="Gender_Clean",
                columns_col="treatment",
                title="Gender vs Treatment Rate",
                xlabel="Gender (Cleaned)",
                palette=["#94A3B8", "#7C3AED"]
            )
            st.pyplot(fig_gen)
            plt.close(fig_gen)
        else:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            sns.countplot(
                data=filtered_df,
                x="Gender_Clean",
                hue="treatment",
                palette=["#94A3B8", "#7C3AED"],
                edgecolor="black",
                linewidth=0.6,
                ax=ax,
            )
            ax.set_title("Gender vs Treatment (Counts)", fontweight="bold")
            ax.set_xlabel("Gender")
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col_g2:
        st.markdown("**Cross-Tabulation Summary:**")
        ct_g_display = pd.crosstab(filtered_df["Gender_Clean"], filtered_df["treatment"], normalize="index") * 100
        st.dataframe(ct_g_display.round(1).astype(str) + "%", use_container_width=True)

        # Calculate dynamic percentages for Gender
        fem_y, fem_n, fem_cnt = get_group_treatment_stats(filtered_df, "Gender_Clean", "Female")
        male_y, male_n, male_cnt = get_group_treatment_stats(filtered_df, "Gender_Clean", "Male")
        oth_y, oth_n, oth_cnt = get_group_treatment_stats(filtered_df, "Gender_Clean", "Other")

        f_txt = f"{format_pct(fem_y)} Yes treatment vs {format_pct(fem_n)} No (n={fem_cnt:,})" if fem_cnt > 0 else "No records"
        m_txt = f"{format_pct(male_y)} Yes treatment vs {format_pct(male_n)} No (n={male_cnt:,})" if male_cnt > 0 else "No records"
        o_txt = f"{format_pct(oth_y)} Yes treatment vs {format_pct(oth_n)} No (n={oth_cnt:,})" if oth_cnt > 0 else "No records"

        st.markdown(
            f"""
            <div class="insight-card">
                <strong>Observed Pattern (Dynamic):</strong><br>
                • <strong>Female:</strong> {f_txt}<br>
                • <strong>Male:</strong> {m_txt}<br>
                • <strong>Other:</strong> {o_txt}<br>
                In the active subset, female and other gender identities demonstrate distinct proportions compared to male respondents.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Age Group vs Treatment
    st.markdown("---")
    st.markdown("### 3. Age Group vs Mental Health Treatment")

    col_a1, col_a2 = st.columns([1.3, 1])
    with col_a1:
        if chart_mode == "Percentage of Group (%)":
            fig_age, ct_age = plot_crosstab_percentage(
                filtered_df.dropna(subset=["Age_Group"]),
                index_col="Age_Group",
                columns_col="treatment",
                title="Age Group vs Treatment Rate",
                xlabel="Age Group",
                palette=["#94A3B8", "#059669"]
            )
            st.pyplot(fig_age)
            plt.close(fig_age)
        else:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            sns.countplot(
                data=filtered_df.dropna(subset=["Age_Group"]),
                x="Age_Group",
                hue="treatment",
                palette=["#94A3B8", "#059669"],
                edgecolor="black",
                linewidth=0.6,
                ax=ax,
            )
            ax.set_title("Age Group vs Treatment (Counts)", fontweight="bold")
            ax.set_xlabel("Age Group")
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col_a2:
        st.markdown("**Cross-Tabulation Summary:**")
        ct_a_display = pd.crosstab(filtered_df["Age_Group"], filtered_df["treatment"], normalize="index") * 100
        st.dataframe(ct_a_display.round(1).astype(str) + "%", use_container_width=True)

        # Dynamic age groups breakdown
        age_lines = []
        for ag in ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]:
            y_p, _, cnt = get_group_treatment_stats(filtered_df, "Age_Group", ag)
            if cnt > 0:
                age_lines.append(f"• <strong>{ag}:</strong> {format_pct(y_p)} Yes treatment (n={cnt:,})")
        age_dynamic_text = "<br>".join(age_lines) if age_lines else "No age group data in selection."

        st.markdown(
            f"""
            <div class="insight-card">
                <strong>Observed Pattern (Dynamic):</strong><br>
                {age_dynamic_text}<br>
                <em>(66+ is a small cohort and should not be overinterpreted).</em>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# 5. WORKPLACE FACTORS PAGE
# =========================================================
elif menu == "Workplace Factors":
    st.subheader("🏢 Workplace Factors & Mental Health Support")
    st.write(
        """
        Evaluating how workplace conditions, benefit offerings, and perceived work interference
        relate to mental health treatment seeking among tech workers.
        """
    )

    st.markdown(
        """
        <div class="warning-disclaimer">
            <strong>Observational Disclaimer:</strong> Reported differences across workplace benefits may reflect differences in awareness, access,
            or company policies, but <strong>do not prove</strong> that offering benefits causes treatment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(filtered_df) == 0:
        st.warning("⚠️ No records match the selected filter criteria. Please adjust or reset the sidebar filters.")
        st.stop()

    # 1. Work Interference vs Treatment
    st.markdown("### 1. Work Interference vs Mental Health Treatment")
    col_w1, col_w2 = st.columns([1.3, 1])
    
    with col_w1:
        order_wi = [lvl for lvl in ["Never", "Rarely", "Sometimes", "Often"] if lvl in filtered_df["work_interfere"].values]
        if not order_wi:
            order_wi = filtered_df["work_interfere"].dropna().unique().tolist()
            
        wi_df = filtered_df[filtered_df["work_interfere"].isin(order_wi)].copy()
        
        if len(wi_df) > 0 and wi_df["treatment"].nunique() > 0:
            fig, ax = plt.subplots(figsize=(8.5, 4.6))
            ct_wi = pd.crosstab(wi_df["work_interfere"], wi_df["treatment"], normalize="index") * 100
            ct_wi = ct_wi.reindex([x for x in ["Never", "Rarely", "Sometimes", "Often"] if x in ct_wi.index])
            ct_wi.plot(
                kind="bar",
                ax=ax,
                color=["#94A3B8", "#EA580C"],
                edgecolor="black",
                linewidth=0.6,
                width=0.65
            )
            ax.set_title("Work Interference vs Treatment Percentage", fontweight="bold")
            ax.set_xlabel("Work Interference Level")
            ax.set_ylabel("Percentage of Respondents (%)")
            ax.set_xticklabels(ct_wi.index, rotation=0)
            ax.set_ylim(0, 100)
            ax.legend(title="Treatment")
            for p in ax.patches:
                h = p.get_height()
                if h > 4:
                    ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2, h / 2), ha="center", va="center", color="white", fontweight="bold", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No work interference records available for current filter selection.")
            ct_wi = pd.DataFrame()

    with col_w2:
        st.markdown("**Interference Level Breakdown:**")
        if not ct_wi.empty:
            st.dataframe(ct_wi.round(1).astype(str) + "%", use_container_width=True)
        
        # Dynamic Work Interference Insight
        wi_lines = []
        for lvl in ["Never", "Rarely", "Sometimes", "Often"]:
            y_pct, n_pct, cnt = get_group_treatment_stats(filtered_df, "work_interfere", lvl)
            if cnt > 0:
                wi_lines.append(f"• <strong>{lvl}:</strong> {format_pct(y_pct)} Yes treatment (n={cnt:,})")
        wi_dynamic_text = "<br>".join(wi_lines) if wi_lines else "No records available for active filters."

        st.markdown(
            f"""
            <div class="insight-card">
                <strong>Observed Relationship (Dynamic):</strong><br>
                {wi_dynamic_text}<br>
                Respondents experiencing higher work interference report significantly higher rates of seeking treatment.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 2. Benefits vs Treatment
    st.markdown("### 2. Mental Health Benefits vs Treatment")
    col_b1, col_b2 = st.columns([1.3, 1])

    with col_b1:
        order_ben = [b for b in ["Yes", "No", "Don't know"] if b in filtered_df["benefits"].values]
        if not order_ben:
            order_ben = filtered_df["benefits"].dropna().unique().tolist()
            
        ben_df = filtered_df[filtered_df["benefits"].isin(order_ben)].copy()
        
        if len(ben_df) > 0 and ben_df["treatment"].nunique() > 0:
            fig, ax = plt.subplots(figsize=(8.5, 4.6))
            ct_ben = pd.crosstab(ben_df["benefits"], ben_df["treatment"], normalize="index") * 100
            ct_ben = ct_ben.reindex([b for b in ["Yes", "No", "Don't know"] if b in ct_ben.index])
            ct_ben.plot(
                kind="bar",
                ax=ax,
                color=["#94A3B8", "#0284C7"],
                edgecolor="black",
                linewidth=0.6,
                width=0.65
            )
            ax.set_title("Availability of Mental Health Benefits vs Treatment", fontweight="bold")
            ax.set_xlabel("Mental Health Benefits Provided")
            ax.set_ylabel("Percentage of Respondents (%)")
            ax.set_xticklabels(ct_ben.index, rotation=0)
            ax.set_ylim(0, 100)
            ax.legend(title="Treatment")
            for p in ax.patches:
                h = p.get_height()
                if h > 4:
                    ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2, h / 2), ha="center", va="center", color="white", fontweight="bold", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No benefits records available for current filter selection.")
            ct_ben = pd.DataFrame()

    with col_b2:
        st.markdown("**Benefits Breakdown:**")
        if not ct_ben.empty:
            st.dataframe(ct_ben.round(1).astype(str) + "%", use_container_width=True)

        # Dynamic Benefits Insight
        ben_lines = []
        for b in ["Yes", "No", "Don't know"]:
            y_pct, n_pct, cnt = get_group_treatment_stats(filtered_df, "benefits", b)
            if cnt > 0:
                ben_lines.append(f"• <strong>{b}:</strong> {format_pct(y_pct)} Yes treatment (n={cnt:,})")
        ben_dynamic_text = "<br>".join(ben_lines) if ben_lines else "No records available for active filters."

        st.markdown(
            f"""
            <div class="insight-card">
                <strong>Observed Pattern (Dynamic):</strong><br>
                {ben_dynamic_text}<br>
                Employees in workplaces with known benefits report higher treatment proportions.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 3. Additional Workplace Attributes
    st.markdown("### 3. Workplace Support & Climate Factors")
    wp_col1, wp_col2, wp_col3 = st.columns(3)
    
    with wp_col1:
        st.markdown("**Care Options Awareness**")
        care_counts = (filtered_df["care_options"].value_counts(normalize=True) * 100).round(1)
        if not care_counts.empty:
            st.bar_chart(care_counts)
        st.caption("Distribution of employer care option awareness.")

    with wp_col2:
        st.markdown("**Anonymity Protection**")
        anon_counts = (filtered_df["anonymity"].value_counts(normalize=True) * 100).round(1)
        if not anon_counts.empty:
            st.bar_chart(anon_counts)
        st.caption("Perception of anonymity if seeking support.")

    with wp_col3:
        st.markdown("**Discussing with Supervisor**")
        sup_counts = (filtered_df["supervisor"].value_counts(normalize=True) * 100).round(1)
        if not sup_counts.empty:
            st.bar_chart(sup_counts)
        st.caption("Comfort level discussing mental health with supervisor.")


# =========================================================
# 6. CORRELATION ANALYSIS PAGE (FILTER-AWARE)
# =========================================================
elif menu == "Correlation Analysis":
    st.subheader("📊 Correlation Analysis & Selected Pairwise Relationships")
    st.write(
        """
        Following the methodology of the EDA, key categorical indicators were binary encoded ($1 = \\text{Yes}, 0 = \\text{No}$)
        alongside Age to compute Pearson correlation coefficients and evaluate pairwise associations.
        All calculations strictly reflect the currently active filters.
        """
    )

    if len(filtered_df) < 2:
        st.warning("⚠️ At least 2 records are required to calculate correlation. Please adjust or reset your filters.")
        st.stop()

    # Prepare correlation dataset based on FILTERED data
    corr_cols = [
        "Age",
        "treatment",
        "family_history",
        "remote_work",
        "tech_company",
        "obs_consequence"
    ]
    
    corr_data = filtered_df[corr_cols].copy()
    for col in ["treatment", "family_history", "remote_work", "tech_company", "obs_consequence"]:
        corr_data[col] = corr_data[col].map({"Yes": 1, "No": 0})
    
    corr_matrix = corr_data.corr()

    # Check for invariant columns due to active filters
    nan_cols = [c for c in corr_matrix.columns if corr_matrix[c].isnull().all()]
    if nan_cols:
        st.info(f"ℹ️ Note: The following variable(s) have constant values under the current filter selection, yielding undefined (NaN) correlation: {', '.join(nan_cols)}.")

    # Heatmap visualization
    st.markdown("### 1. Correlation Heatmap (Filter-Aware)")
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=0.5,
        linewidths=1,
        linecolor="#E2E8F0",
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(f"Correlation Heatmap ({len(filtered_df):,} Filtered Records)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        """
        <div class="warning-disclaimer">
            <strong>Essential Statistical Principle:</strong><br>
            The correlation analysis shows associations between selected variables. <strong>Correlation does not imply causation.</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dynamic correlation extraction
    def get_corr_text(c1, c2):
        if c1 in corr_matrix.columns and c2 in corr_matrix.index:
            v = corr_matrix.loc[c1, c2]
            if pd.notnull(v):
                return f"r = {v:.2f}"
        return "N/A (constant value under active filters)"

    r_fam_trt = get_corr_text("family_history", "treatment")
    r_trt_obs = get_corr_text("treatment", "obs_consequence")
    r_age_rem = get_corr_text("Age", "remote_work")
    r_age_fam = get_corr_text("Age", "family_history")
    r_rem_obs = get_corr_text("remote_work", "obs_consequence")

    st.markdown(
        f"""
        <div class="insight-card">
            <strong>Key Observed Correlation Findings in Active Selection (Dynamic):</strong><br>
            • <strong>family_history vs treatment:</strong> {r_fam_trt} (consistently the strongest associative predictor).<br>
            • <strong>treatment vs obs_consequence:</strong> {r_trt_obs} (modest association with observed consequences).<br>
            • <strong>Age vs remote_work:</strong> {r_age_rem}<br>
            • <strong>Age vs family_history:</strong> {r_age_fam}<br>
            • <strong>remote_work vs obs_consequence:</strong> {r_rem_obs}<br>
            • Most other pairwise relationships remain weak (|r| < 0.10).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Pair Plot Section
    st.markdown("### 2. Pair Plot of Selected Variables")
    st.write(
        """
        The pair plot illustrates the pairwise bivariate distributions and univariate histograms
        for `Age`, `treatment`, `family_history`, `remote_work`, and `tech_company` on the filtered dataset.
        """
    )

    render_pairplot = st.checkbox("Render Pair Plot Visualization", value=False, help="Check to generate the full Seaborn pair plot.")
    
    if render_pairplot:
        with st.spinner("Generating Pair Plot..."):
            pair_vars = ["Age", "treatment", "family_history", "remote_work", "tech_company"]
            pair_data = filtered_df[pair_vars].copy()
            for col in ["treatment", "family_history", "remote_work", "tech_company"]:
                pair_data[col] = pair_data[col].map({"Yes": 1, "No": 0})
            pair_data = pair_data.dropna()

            if len(pair_data) >= 3:
                pair_fig = sns.pairplot(
                    pair_data,
                    diag_kind="hist",
                    plot_kws={"alpha": 0.6, "s": 30, "color": "#2563EB"},
                    diag_kws={"color": "#3B82F6", "bins": 15}
                )
                pair_fig.fig.suptitle("Pair Plot of Selected Variables (Filtered)", y=1.02, fontsize=14, fontweight="bold")
                st.pyplot(pair_fig.fig)
                plt.close(pair_fig.fig)
            else:
                st.warning("Insufficient complete records available to plot pairwise distributions.")
    else:
        st.info("💡 To view the Pair Plot, toggle the checkbox above.")


# =========================================================
# 7. KEY INSIGHTS PAGE (DYNAMIC & FILTER-AWARE)
# =========================================================
elif menu == "Key Insights":
    st.subheader("💡 Key Insights from the EDA")
    st.write("A structured, evidence-based summary of analytical discoveries calculated directly from the currently filtered dataset.")

    st.markdown(
        """
        <div class="warning-disclaimer">
            <strong>Reminder on Interpretation:</strong> These insights represent observable patterns in the 2014 survey sample.
            They are not claims of causal mechanisms.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(filtered_df) == 0:
        st.warning("⚠️ No records match the selected filter criteria. Please adjust or reset the sidebar filters.")
        st.stop()

    # Dynamic metrics computation
    fam_y_pct, _, fam_y_n = get_group_treatment_stats(filtered_df, "family_history", "Yes")
    nofam_y_pct, _, nofam_y_n = get_group_treatment_stats(filtered_df, "family_history", "No")

    fem_y, _, fem_cnt = get_group_treatment_stats(filtered_df, "Gender_Clean", "Female")
    male_y, _, male_cnt = get_group_treatment_stats(filtered_df, "Gender_Clean", "Male")
    oth_y, _, oth_cnt = get_group_treatment_stats(filtered_df, "Gender_Clean", "Other")

    wi_never_y, _, wi_never_n = get_group_treatment_stats(filtered_df, "work_interfere", "Never")
    wi_often_y, _, wi_often_n = get_group_treatment_stats(filtered_df, "work_interfere", "Often")

    ben_yes_y, _, ben_yes_n = get_group_treatment_stats(filtered_df, "benefits", "Yes")
    ben_no_y, _, ben_no_n = get_group_treatment_stats(filtered_df, "benefits", "No")
    ben_dk_y, _, ben_dk_n = get_group_treatment_stats(filtered_df, "benefits", "Don't know")

    # Correlations
    corr_cols = ["Age", "treatment", "family_history", "remote_work", "tech_company", "obs_consequence"]
    c_df = filtered_df[corr_cols].copy()
    for col in ["treatment", "family_history", "remote_work", "tech_company", "obs_consequence"]:
        c_df[col] = c_df[col].map({"Yes": 1, "No": 0})
    curr_cmat = c_df.corr()
    
    r_val = curr_cmat.loc["family_history", "treatment"] if ("family_history" in curr_cmat and "treatment" in curr_cmat) else np.nan
    r_str = f"r = {r_val:.2f}" if pd.notnull(r_val) else "N/A"

    insights = [
        {
            "num": "01",
            "title": "Family History is the Strongest Associative Predictor of Treatment",
            "desc": f"In the current selection, respondents with a family history sought treatment at {format_pct(fam_y_pct)} (n={fam_y_n:,}), compared to {format_pct(nofam_y_pct)} (n={nofam_y_n:,}) among those without. The observed linear correlation in this subset is {r_str}."
        },
        {
            "num": "02",
            "title": "Clear Gender Disparities in Treatment Rates",
            "desc": f"Treatment rates vary distinctly across gender classifications in this selection: Female = {format_pct(fem_y)} (n={fem_cnt:,}), Male = {format_pct(male_y)} (n={male_cnt:,}), and Other = {format_pct(oth_y)} (n={oth_cnt:,})."
        },
        {
            "num": "03",
            "title": "Age Progression in Treatment Likelihood",
            "desc": "Treatment seeking demonstrates a steady upward progression through working age cohorts, with the largest volume of respondents concentrated in the 26–35 and 36–45 brackets."
        },
        {
            "num": "04",
            "title": "Direct Relationship Between Work Interference and Treatment Seeking",
            "desc": f"Frequency of work interference strongly aligns with treatment prevalence: respondents reporting 'Never' have a {format_pct(wi_never_y)} treatment rate (n={wi_never_n:,}), whereas those reporting 'Often' have a {format_pct(wi_often_y)} treatment rate (n={wi_often_n:,})."
        },
        {
            "num": "05",
            "title": "Workplace Benefits Presence vs. Employee Awareness Gap",
            "desc": f"Employees reporting available workplace benefits sought treatment at {format_pct(ben_yes_y)} (n={ben_yes_n:,}), those without benefits sought treatment at {format_pct(ben_no_y)} (n={ben_no_n:,}), while those responding 'Don't know' registered {format_pct(ben_dk_y)} (n={ben_dk_n:,}), illustrating an information access divide."
        },
        {
            "num": "06",
            "title": "Weak Multivariable Correlations Across Employment Arrangements",
            "desc": "Arrangements like remote work or tech company classification demonstrate minimal direct linear association with treatment seeking, suggesting workplace culture and personal support matter more than remote status."
        }
    ]

    for item in insights:
        st.markdown(
            f"""
            <div style="background-color:#F8FAFC; border: 1px solid #E2E8F0; border-left: 5px solid #3B82F6; border-radius: 8px; padding: 14px 18px; margin-bottom: 14px;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #2563EB;">INSIGHT {item['num']}</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin: 4px 0;">{item['title']}</div>
                <div style="font-size: 0.93rem; color: #334155; line-height: 1.5;">{item['desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# 8. BUSINESS RECOMMENDATIONS PAGE
# =========================================================
elif menu == "Business Recommendations":
    st.subheader("🎯 Business Recommendations for Tech Organizations")
    st.write(
        """
        Actionable, data-informed recommendations for technology leaders, human resource departments,
        and workplace wellness strategists based on observed associations in the survey.
        """
    )

    st.markdown(
        """
        <div class="warning-disclaimer">
            <strong>Grounding in Survey Data:</strong> These recommendations are based on observed associations in the survey data.
            Do not state that any variable causes mental health treatment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    recommendations = [
        {
            "icon": "📢",
            "title": "1. Improve Awareness of Mental Health Resources and Benefit Offerings",
            "body": "A substantial portion of tech workers respond 'Don't know' regarding available benefits, and this group exhibits the lowest treatment rates. Organizations must proactively publicize employee assistance programs (EAPs), mental health coverage, and wellness benefits during onboarding and via routine company communications."
        },
        {
            "icon": "🩺",
            "title": "2. Make Mental Health Benefits and Care Options Easier to Access",
            "body": "Survey responses demonstrate higher treatment seeking among workers with confirmed access to employer benefits. Ensuring healthcare plans include accessible therapy coverage, low copays, and streamlined navigation removes structural barriers to care."
        },
        {
            "icon": "🔒",
            "title": "3. Guarantee Confidentiality and Maintain Anonymity",
            "body": "Fear of negative career consequences and social stigma remains a primary inhibitor in workplace mental health discussions. Employers must enforce strict anonymity and confidentiality protocols so employees feel secure seeking support without fear of discrimination."
        },
        {
            "icon": "💬",
            "title": "4. Foster Supportive Workplace Communication and Manager Training",
            "body": "Many respondents expressed hesitation to discuss mental health with supervisors. Providing empathetic leadership training for managers helps demystify mental health discussions and equips leaders to direct employees toward professional support resources."
        },
        {
            "icon": "🌿",
            "title": "5. Provide Comprehensive Wellness Programs and Flexible Policies",
            "body": "Supportive workplace policies—including manageable workloads, mental health days, and flexible scheduling—support overall well-being. Wellness programs should offer practical stress reduction tools rather than token initiatives."
        },
        {
            "icon": "⚡",
            "title": "6. Proactively Address and Support Work Interference Concerns",
            "body": "Because work interference strongly correlates with higher treatment seeking (especially among employees reporting frequent interference), organizations should establish early-intervention pathways, workload adjustments, and supportive return-to-work frameworks."
        }
    ]

    for rec in recommendations:
        st.markdown(
            f"""
            <div class="recommendation-card">
                <div class="recommendation-title">{rec['icon']} {rec['title']}</div>
                <div class="recommendation-body">{rec['body']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        """
        #### 📌 Summary Conclusion
        By addressing the information gap, lowering barriers to healthcare access, ensuring psychological safety,
        and taking work interference seriously, technology organizations can cultivate supportive environments that
        foster employee well-being, retention, and productivity.
        """
    )

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Internship Submission: Mental Health in Tech Survey Analysis | Streamlit Application")
