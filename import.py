import warnings
warnings.filterwarnings('ignore')

import math
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Mind & Substance",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Playfair Display', serif !important; }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px; }
    .kicker { font-family: 'DM Mono', monospace; font-size: 0.7rem; font-weight: 500;
        letter-spacing: 0.15em; text-transform: uppercase; color: #c0392b; margin-bottom: 0.4rem; }
    .insight-box { background: #faf8f4; border-left: 3px solid #c0392b;
        padding: 0.9rem 1.1rem; border-radius: 0 4px 4px 0; margin-bottom: 0.8rem; }
    .insight-box strong { color: #c0392b; font-size: 0.75rem;
        text-transform: uppercase; letter-spacing: 0.08em; display: block; margin-bottom: 0.2rem; }
    .metric-row { display: flex; gap: 1px; background: #e8e5de; border: 1px solid #e8e5de;
        border-radius: 4px; overflow: hidden; margin-bottom: 1.5rem; }
    .metric-card { background: white; padding: 1.1rem 1rem; text-align: center; flex: 1; }
    .metric-card .num { font-family: 'Playfair Display', serif; font-size: 1.9rem;
        font-weight: 900; color: #1a1a1a; line-height: 1; }
    .metric-card .num span { color: #c0392b; }
    .metric-card .lbl { font-size: 0.72rem; color: #8a8580; text-transform: uppercase;
        letter-spacing: 0.07em; margin-top: 0.3rem; }
    .section-tag { font-family: 'DM Mono', monospace; font-size: 0.68rem; letter-spacing: 0.12em;
        text-transform: uppercase; color: #8a8580; margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS & MAPPINGS
# ─────────────────────────────────────────────────────────────

TRAIT_MAP  = {'Nscore':'Neuroticism','Escore':'Extraversion','Oscore':'Openness',
              'Ascore':'Agreeableness','cscore':'Conscientiousness','impulsive':'Impulsivity','SS':'Sensation Seeking'}
TRAIT_LIST = list(TRAIT_MAP.values())

AGE_MAP    = {-0.95197:'18-24',-0.07854:'25-34',0.49788:'35-44',
               1.09449:'45-54',1.82213:'55-64',2.59171:'65+'}
AGE_ORDER  = ['18-24','25-34','35-44','45-54','55-64','65+']

GENDER_MAP = {-0.48246:'Female', 0.48246:'Male'}

ETH_MAP = {
    -0.50212: 'Asian',
    -0.31685: 'Black',
    -0.21128: 'Mixed-Asian/Black',
    -0.14854: 'Mixed-White/Asian',
    0.11440: 'Mixed-White/Black',
    0.12608: 'Other',
    0.48246: 'White'
}

EDU_MAP    = {-2.43591:'Left school <16',-1.73790:'Left school @16',-1.43719:'Left school @17',
              -1.22751:'Left school @18',-0.61113:'Some college',-0.05921:'Professional cert.',
               0.45468:'University degree',1.16365:'Masters degree',1.98437:'Doctorate'}
EDU_ORDER  = ['Left school <16','Left school @16','Left school @17','Left school @18',
              'Some college','Professional cert.','University degree','Masters degree','Doctorate']

COUNTRY_MAP= {-0.57009:'USA',-0.46841:'New Zealand',-0.28519:'Other',-0.09765:'Australia',
               0.21128:'Ireland', 0.24923:'Canada', 0.96082:'UK'}

USAGE_ORDER  = ['CL0','CL1','CL2','CL3','CL4','CL5','CL6']
USAGE_LABELS = {'CL0':'Never','CL1':'>10 yrs ago','CL2':'Last decade',
                'CL3':'Last year','CL4':'Last month','CL5':'Last week','CL6':'Last day'}
USAGE_DISPLAY= [USAGE_LABELS[u] for u in USAGE_ORDER]
USAGE_NUM    = {v:i for i,v in enumerate(USAGE_ORDER)}

ALL_DRUGS   = ['Alcohol','Amphet','Amyl','Benzos','Caff','Cannabis','Choc','Coke',
               'Crack','Ecstasy','Heroin','Ketamine','Legalh','LSD','Meth','Mushrooms','Nicotine','VSA']
FOCUS_DRUGS = ['Cannabis','Ecstasy','LSD','Coke','Heroin','Ketamine','Mushrooms',
               'Amphet','Benzos','Nicotine','Alcohol','Legalh','Amyl']
ILLICIT     = ['Cannabis','Ecstasy','LSD','Coke','Heroin','Amphet','Ketamine','Mushrooms']

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=373)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    # Personality renaming
    trait_rename = {'nscore':'Neuroticism','escore':'Extraversion','oscore':'Openness',
                    'ascore':'Agreeableness','cscore':'Conscientiousness','impuslive':'Impulsivity','ss':'Sensation Seeking'}
    df = df.rename(columns=trait_rename)

    # Drug renaming
    drug_cols = ['alcohol','amphet','amyl','benzos','caff','cannabis','choc','coke','crack','ecstasy','heroin','ketamine','legalh','lsd','meth','mushrooms','nicotine','semer','vsa']
    rename_drugs = {c: c.capitalize() for c in drug_cols if c in df.columns}
    rename_drugs.update({'lsd':'LSD', 'vsa':'VSA'})
    df = df.rename(columns=rename_drugs)

    # Labels
    df['Age_Label']       = df['age'].map(AGE_MAP)
    df['Gender_Label']    = df['gender'].map(GENDER_MAP)
    df['Education_Label'] = df['education'].map(EDU_MAP)
    df['Country_Label']   = df['country'].map(COUNTRY_MAP)
    df['Ethnicity_Label'] = df['ethnicity'].map(ETH_MAP)

    for d in ALL_DRUGS + ['Semer']:
        if d in df.columns:
            df[f'{d}_score']   = df[d].map(USAGE_NUM)
            df[f'{d}_current'] = df[d].isin(['CL4','CL5','CL6'])

    df['n_illicit_current'] = sum(df[f'{d}_current'].astype(int) for d in ILLICIT if f'{d}_current' in df.columns)
    df['Semer_claimer'] = df['Semer'].isin(['CL3','CL4','CL5','CL6'])
    return df

df_raw = load_data()
alt.data_transformers.disable_max_rows()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Mind & Substance")
    st.divider()
    section = st.radio("ANALYSIS MODULES", [
        "Executive Summary",
        "Personality × Usage",
        "Drug Co-use & Poly-use",
        "Predictive Modeling",
        "Psychological Radar",
        "Prevalence & Demographics (Appendix)"
    ])
    st.divider()
    gender_filter = st.multiselect("Gender", ['Female','Male'], default=['Female','Male'])
    age_filter    = st.multiselect("Age group", AGE_ORDER, default=AGE_ORDER)
    excl_semer    = st.checkbox("Exclude unreliable respondents", value=True)

df = df_raw.copy()
if gender_filter: df = df[df['Gender_Label'].isin(gender_filter)]
if age_filter: df = df[df['Age_Label'].isin(age_filter)]
if excl_semer: df = df[~df['Semer_claimer']]
N = len(df)

# ─────────────────────────────────────────────────────────────
# 1 — EXECUTIVE SUMMARY
# ─────────────────────────────────────────────────────────────
if section == "Executive Summary":
    st.markdown('<div class="kicker">Core Findings</div>', unsafe_allow_html=True)
    st.header("Personality is the Primary Driver")
    
    st.markdown("""
    <div class="insight-box">
    <strong>The Central Insight</strong>
    Statistical analysis of this 1,885-person dataset suggests that <strong>Openness to Experience</strong> and 
    <strong>Sensation Seeking</strong> are the most robust predictors of drug exploration, while <strong>Conscientiousness</strong> 
    serves as the primary psychological "buffer" against frequent usage.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        high_ss = df[df['Sensation Seeking'] > 1]['n_illicit_current'].mean()
        st.metric("Avg Illicit Drugs (High SS)", f"{high_ss:.2f}", help="Mean current drugs for users 1SD+ in Sensation Seeking")
    with col2:
        high_c = df[df['Conscientiousness'] > 1]['n_illicit_current'].mean()
        st.metric("Avg Illicit Drugs (High C)", f"{high_c:.2f}", help="Mean current drugs for users 1SD+ in Conscientiousness")
    with col3:
        st.metric("Total Sample size", f"{N}")

    st.divider()
    
    # Summary Relationships Chart
    st.markdown('<div class="section-tag">Key Correlations: Personality vs. Poly-drug Use</div>', unsafe_allow_html=True)
    poly_trait = df.groupby('n_illicit_current')[TRAIT_LIST].mean().reset_index().melt(id_vars='n_illicit_current')
    
    summary_chart = alt.Chart(poly_trait).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('n_illicit_current:O', title='Number of Current Illicit Drugs Used'),
        y=alt.Y('value:Q', title='Mean Z-Score'),
        color=alt.Color('variable:N', title='Personality Trait'),
        tooltip=['variable', 'n_illicit_current', 'value']
    ).properties(height=350)
    st.altair_chart(summary_chart, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 2 — PERSONALITY × USAGE
# ─────────────────────────────────────────────────────────────
elif section == "Personality × Usage":
    st.header("The Personality Gradient")
    st.markdown("Unlike a timeline, these usage levels represent **intensity clusters**. This heatmap shows how personality profiles shift as usage frequency increases.")

    selected_drug = st.selectbox("Select Drug to Analyze:", FOCUS_DRUGS)
    
    trait_usage = []
    for trait in TRAIT_LIST:
        for cl in USAGE_ORDER:
            mean_val = df[df[selected_drug] == cl][trait].mean()
            trait_usage.append({'Trait': trait, 'Usage': USAGE_LABELS[cl], 'UsageCode': USAGE_NUM[cl], 'Score': mean_val})
    
    df_tu = pd.DataFrame(trait_usage)

    st.markdown(f'<div class="section-tag">Shift in {selected_drug} User Personality Profile</div>', unsafe_allow_html=True)
    
    # Use a Heatmap instead of a Line Chart to avoid timeline confusion
    heatmap = alt.Chart(df_tu).mark_rect().encode(
        x=alt.X('Usage:O', sort=USAGE_DISPLAY, title='Reported Usage Level'),
        y=alt.Y('Trait:N', title=None),
        color=alt.Color('Score:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1], reverse=True), 
                        title='Mean Z-Score (Darker = Higher)'),
        tooltip=['Trait', 'Usage', alt.Tooltip('Score:Q', format='.2f')]
    ).properties(height=400)

    text = heatmap.mark_text(fontSize=11).encode(
        text=alt.Text('Score:Q', format='.2f'),
        color=alt.condition(alt.abs(alt.datum.Score) > 0.5, alt.value('white'), alt.value('black'))
    )

    st.altair_chart(heatmap + text, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 3 — DRUG CO-USE & POLY-USE
# ─────────────────────────────────────────────────────────────
elif section == "Drug Co-use & Poly-use":
    st.header("Co-occurrence Patterns")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-tag">Correlation Matrix (Darker = Stronger Correlation)</div>', unsafe_allow_html=True)
        corr = df[[f'{d}_score' for d in FOCUS_DRUGS if f'{d}_score' in df.columns]].corr()
        corr.index = [c.replace('_score','') for c in corr.index]
        corr.columns = corr.index
        corr_long = corr.reset_index().melt(id_vars='index')
        
        c_matrix = alt.Chart(corr_long).mark_rect().encode(
            x=alt.X('index:N', title=None),
            y=alt.Y('variable:N', title=None),
            color=alt.Color('value:Q', scale=alt.Scale(scheme='viridis', domain=[0, 1]), title='Pearson r'),
            tooltip=['index', 'variable', 'value']
        ).properties(height=450)
        st.altair_chart(c_matrix, use_container_width=True)
        
    with col2:
        st.markdown('<div class="section-tag">Poly-drug Intensity</div>', unsafe_allow_html=True)
        poly_dist = df['n_illicit_current'].value_counts(normalize=True).reset_index()
        poly_dist.columns = ['N_Drugs', 'Pct']
        
        poly_bar = alt.Chart(poly_dist).mark_bar(color='#c0392b').encode(
            x=alt.X('N_Drugs:O', title='N Current Illicit Drugs'),
            y=alt.Y('Pct:Q', axis=alt.Axis(format='%'), title='% of Respondents'),
            tooltip=['N_Drugs', alt.Tooltip('Pct:Q', format='.1%')]
        ).properties(height=450)
        st.altair_chart(poly_bar, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 4 — PREDICTIVE MODELING
# ─────────────────────────────────────────────────────────────
elif section == "Predictive Modeling":
    st.header("Prediction: Who is most likely to use?")
    
    model_drug = st.selectbox("Predict Current Use For:", FOCUS_DRUGS)
    
    y = df[f'{model_drug}_current'].astype(int)
    X = df[TRAIT_LIST]
    
    lr = LogisticRegression()
    lr.fit(X, y)
    
    coef_df = pd.DataFrame({'Trait': TRAIT_LIST, 'Weight': lr.coef_[0]}).sort_values('Weight')
    
    st.markdown(f'<div class="section-tag">Logistic Regression Coefficients: {model_drug}</div>', unsafe_allow_html=True)
    
    lollipop = alt.layer(
        alt.Chart(coef_df).mark_rule(color='#ccc').encode(
            y=alt.Y('Trait:N', sort='x'),
            x='Weight:Q',
            x2=alt.X2(value=0)),
        alt.Chart(coef_df).mark_point(size=150, filled=True).encode(
            y=alt.Y('Trait:N', sort='x'),
            x='Weight:Q',
            color=alt.condition(alt.datum.Weight > 0, alt.value('#c0392b'), alt.value('#3498db'))
        )
    ).properties(height=400)
    
    st.altair_chart(lollipop, use_container_width=True)
    st.caption("Positive weight (Red) indicates the trait increases the likelihood of current use.")

# ─────────────────────────────────────────────────────────────
# 5 — RADAR
# ─────────────────────────────────────────────────────────────
elif section == "Psychological Radar":
    st.header("Comparative Fingerprints")
    
    d1 = st.selectbox("Drug 1:", FOCUS_DRUGS, index=0)
    d2 = st.selectbox("Drug 2:", FOCUS_DRUGS, index=1)
    
    # Simplified Radar Logic for contrast
    r_data = []
    for d in [d1, d2]:
        means = df[df[f'{d}_current']][TRAIT_LIST].mean()
        for trait in TRAIT_LIST:
            r_data.append({'Drug': d, 'Trait': trait, 'Score': means[trait]})
    
    radar_df = pd.DataFrame(r_data)
    
    st.markdown('<div class="section-tag">Comparison of "Current User" Profiles (Mean Z-Scores)</div>', unsafe_allow_html=True)
    
    radar_comp = alt.Chart(radar_df).mark_bar().encode(
        x=alt.X('Score:Q', title='Z-Score deviation from Mean'),
        y=alt.Y('Trait:N', title=None),
        color=alt.Color('Drug:N', scale=alt.Scale(range=['#c0392b', '#1a1a1a'])),
        row=alt.Row('Drug:N', title=None)
    ).properties(height=150)
    
    st.altair_chart(radar_comp, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 6 — APPENDIX: DEMOGRAPHICS
# ─────────────────────────────────────────────────────────────
else:
    st.header("Demographics Appendix")
    st.markdown("Contextual data regarding the respondent pool.")
    
    tab1, tab2, tab3 = st.tabs(["Gender Usage Gap", "Age & Education", "Ethnicity Stats"])
    
    with tab1:
        st.markdown('<div class="section-tag">Usage Gap (Higher Female Use ← | → Higher Male Use)</div>', unsafe_allow_html=True)
        gender_data = []
        for d in FOCUS_DRUGS:
            f_mean = df[df['Gender_Label']=='Female'][f'{d}_score'].mean()
            m_mean = df[df['Gender_Label']=='Male'][f'{d}_score'].mean()
            gender_data.append({'Drug': d, 'Gap': f_mean - m_mean})
        
        gap_df = pd.DataFrame(gender_data).sort_values('Gap')
        gap_chart = alt.Chart(gap_df).mark_bar().encode(
            x=alt.X('Gap:Q', title='Difference in Mean Usage Score (Scale 0-6)'),
            y=alt.Y('Drug:N', sort='x'),
            color=alt.condition(alt.datum.Gap > 0, alt.value('#e74c3c'), alt.value('#3498db'))
        ).properties(height=400)
        st.altair_chart(gap_chart, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-tag">Sample Distribution</div>', unsafe_allow_html=True)
        age_dist = df['Age_Label'].value_counts().reset_index()
        age_chart = alt.Chart(age_dist).mark_bar(color='#8a8580').encode(
            x=alt.X('Age_Label:N', sort=AGE_ORDER),
            y='count:Q'
        ).properties(height=300)
        st.altair_chart(age_chart, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-tag">Respondents by Ethnicity</div>', unsafe_allow_html=True)
        eth_counts = df['Ethnicity_Label'].value_counts().reset_index()
        eth_counts.columns = ['Ethnicity', 'Count']
        st.table(eth_counts)