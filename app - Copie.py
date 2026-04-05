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
    div[data-testid="stSidebarContent"] { background: #faf9f6; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
COLS = ['ID','Age','Gender','Education','Country','Ethnicity',
        'Nscore','Escore','Oscore','Ascore','Cscore','Impulsive','SS',
        'Alcohol','Amphet','Amyl','Benzos','Caff','Cannabis',
        'Choc','Coke','Crack','Ecstasy','Heroin','Ketamine',
        'Legalh','LSD','Meth','Mushrooms','Nicotine','Semer','VSA']

TRAIT_MAP  = {'Nscore':'Neuroticism','Escore':'Extraversion','Oscore':'Openness',
              'Ascore':'Agreeableness','Cscore':'Conscientiousness','Impulsive':'Impulsivity','SS':'Sensation Seeking'}
TRAIT_LIST = list(TRAIT_MAP.values())

AGE_MAP    = {-0.95197:'18-24',-0.07854:'25-34',0.49788:'35-44',
               1.09449:'45-54',1.82213:'55-64',2.59171:'65+'}
AGE_ORDER  = ['18-24','25-34','35-44','45-54','55-64','65+']
GENDER_MAP = {-0.48246:'Female',0.48246:'Male'}
EDU_MAP    = {-2.43591:'Left school <16',-1.73790:'Left school @16',-1.43719:'Left school @17',
              -1.22751:'Left school @18',-0.61113:'Some college',-0.05921:'Professional cert.',
               0.45468:'University degree',1.16365:'Masters degree',1.98437:'Doctorate'}
EDU_ORDER  = ['Left school <16','Left school @16','Left school @17','Left school @18',
              'Some college','Professional cert.','University degree','Masters degree','Doctorate']
COUNTRY_MAP= {-0.57009:'USA',-0.46841:'New Zealand',-0.28519:'Other',-0.09765:'Australia',
               0.21128:'Ireland',0.24923:'Canada',0.96082:'UK'}

USAGE_ORDER  = ['CL0','CL1','CL2','CL3','CL4','CL5','CL6']
USAGE_LABELS = {'CL0':'Never','CL1':'>10 yrs ago','CL2':'Last decade',
                'CL3':'Last year','CL4':'Last month','CL5':'Last week','CL6':'Last day'}
USAGE_DISPLAY= [USAGE_LABELS[u] for u in USAGE_ORDER]
USAGE_NUM    = {v:i for i,v in enumerate(USAGE_ORDER)}

ALL_DRUGS  = ['Alcohol','Amphet','Amyl','Benzos','Caff','Cannabis','Choc','Coke',
              'Crack','Ecstasy','Heroin','Ketamine','Legalh','LSD','Meth','Mushrooms','Nicotine','VSA']
FOCUS_DRUGS= ['Cannabis','Ecstasy','LSD','Coke','Heroin','Ketamine','Mushrooms',
              'Amphet','Benzos','Nicotine','Alcohol','Legalh','Amyl']
ILLICIT    = ['Cannabis','Ecstasy','LSD','Coke','Heroin','Amphet','Ketamine','Mushrooms']

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset from UCI...")
def load_data():
    from ucimlrepo import fetch_ucirepo
    alt.data_transformers.disable_max_rows()

    dataset = fetch_ucirepo(id=373)
    X = dataset.data.features   # lowercase cols: age, gender, education, country, ethnicity, nscore, escore...
    y = dataset.data.targets    # lowercase cols: alcohol, amphet, amyl, benzos, caff, cannabis...
    df = pd.concat([X, y], axis=1)

    # --- Rename personality traits (lowercase -> readable) ---
    trait_map_lower = {
        'nscore':    'Neuroticism',
        'escore':    'Extraversion',
        'oscore':    'Openness',
        'ascore':    'Agreeableness',
        'cscore':    'Conscientiousness',
        'impuslive': 'Impulsivity',
        'ss':        'Sensation Seeking'
    }
    df = df.rename(columns=trait_map_lower)

    # --- Capitalize drug columns (alcohol -> Alcohol, cannabis -> Cannabis...) ---
    drug_cols_lower = ['alcohol','amphet','amyl','benzos','caff','cannabis',
                       'choc','coke','crack','ecstasy','heroin','ketamine',
                       'legalh','lsd','meth','mushrooms','nicotine','semer','vsa']
    rename_drugs = {c: c.capitalize() for c in drug_cols_lower if c in df.columns}
    rename_drugs['lsd'] = 'LSD'   # special case
    rename_drugs['vsa'] = 'VSA'   # special case
    df = df.rename(columns=rename_drugs)

    # --- Demographic labels ---
    df['Age_Label']       = df['age'].map(AGE_MAP)
    df['Gender_Label']    = df['gender'].map(GENDER_MAP)
    df['Education_Label'] = df['education'].map(EDU_MAP)
    df['Country_Label']   = df['country'].map(COUNTRY_MAP)

    # --- Drug numeric scores & current-user flags ---
    for d in ALL_DRUGS + ['Semer']:
        if d in df.columns:
            df[f'{d}_score']   = df[d].map(USAGE_NUM)
            df[f'{d}_current'] = df[d].isin(['CL4','CL5','CL6'])

    df['n_illicit_current'] = sum(
        df[f'{d}_current'].astype(int) for d in ILLICIT if f'{d}_current' in df.columns)
    df['Semer_claimer'] = df['Semer'].isin(['CL3','CL4','CL5','CL6'])
    return df

df_raw = load_data()
alt.data_transformers.disable_max_rows()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##Mind & Substance")
    st.markdown("*Personality, Demographics & Drugs*")
    st.divider()
    st.markdown("**Navigate**")
    section = st.radio("", [
        "Overview",
        "Drug Prevalence",
        "Demographics",
        "Traits × Usage",
        "Drug Co-use",
        "Poly-drug Profiles",
        "Prediction",
        "Radar"
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("**Global filters**")
    gender_filter = st.multiselect("Gender", ['Female','Male'], default=['Female','Male'])
    age_filter    = st.multiselect("Age group", AGE_ORDER, default=AGE_ORDER)
    excl_semer    = st.checkbox("Exclude Semer claimers", value=False,
        help="Remove the 3 respondents who claimed to use the fictitious drug Semer (reliability check)")
    st.divider()
    st.caption("Dataset: UCI Drug Consumption")
    st.caption(f"N = {len(df_raw):,} respondents")

# Apply filters
df = df_raw.copy()
if gender_filter:
    df = df[df['Gender_Label'].isin(gender_filter)]
if age_filter:
    df = df[df['Age_Label'].isin(age_filter)]
if excl_semer:
    df = df[~df['Semer_claimer']]
N = len(df)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="kicker">Data Analysis · Psychology · Substance Use</div>', unsafe_allow_html=True)
st.title("Mind & Substance")
st.markdown(f"Exploring personality, demographics and drug consumption across **{N:,} respondents** · UCI Drug Consumption dataset")
st.divider()

# ═══════════════════════════════════════════════════════════════
# 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════
if section == "Overview":
    st.header("Dataset Overview")
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card"><div class="num">{N:,}</div><div class="lbl">Respondents</div></div>
      <div class="metric-card"><div class="num">18</div><div class="lbl">Substances</div></div>
      <div class="metric-card"><div class="num">7</div><div class="lbl">Personality traits</div></div>
      <div class="metric-card"><div class="num">4</div><div class="lbl">Demographic variables</div></div>
      <div class="metric-card"><div class="num">7</div><div class="lbl">Usage levels (CL0–CL6)</div></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-tag">Chart 1 — Sample by Country</div>', unsafe_allow_html=True)
        cc = df['Country_Label'].value_counts().reset_index()
        cc.columns = ['Country','Count']
        c = alt.Chart(cc).mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(
            x=alt.X('Country:N',sort='-y',title=None,axis=alt.Axis(labelAngle=-25)),
            y=alt.Y('Count:Q',title='Respondents',axis=alt.Axis(gridColor='#f0ede6')),
            color=alt.Color('Country:N',legend=None,scale=alt.Scale(scheme='tableau10')),
            tooltip=['Country','Count']
        ).properties(height=240)
        st.altair_chart(c, use_container_width=True)

    with col2:
        st.markdown('<div class="section-tag">Chart 2 — Age & Gender breakdown</div>', unsafe_allow_html=True)
        ag = df.groupby(['Age_Label','Gender_Label']).size().reset_index(name='Count')
        c = alt.Chart(ag).mark_bar().encode(
            x=alt.X('Age_Label:N',sort=AGE_ORDER,title='Age group',axis=alt.Axis(labelAngle=-20)),
            y=alt.Y('Count:Q',title='Count',axis=alt.Axis(gridColor='#f0ede6')),
            color=alt.Color('Gender_Label:N',title='Gender',
                scale=alt.Scale(domain=['Female','Male'],range=['#e74c3c','#3498db'])),
            tooltip=['Age_Label','Gender_Label','Count']
        ).properties(height=240)
        st.altair_chart(c, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-tag">Chart 3 — Education distribution</div>', unsafe_allow_html=True)
        ed = df['Education_Label'].value_counts().reset_index()
        ed.columns = ['Education','Count']
        c = alt.Chart(ed).mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(
            x=alt.X('Count:Q',title='Count',axis=alt.Axis(gridColor='#f0ede6')),
            y=alt.Y('Education:N',sort='-x',title=None),
            color=alt.Color('Count:Q',scale=alt.Scale(scheme='blues'),legend=None),
            tooltip=['Education','Count']
        ).properties(height=280)
        st.altair_chart(c, use_container_width=True)

    with col4:
        st.markdown('<div class="section-tag">Chart 4 — Personality trait distributions</div>', unsafe_allow_html=True)
        df_traits = df[TRAIT_LIST].melt(var_name='Trait',value_name='Score')
        c = alt.Chart(df_traits).transform_density(
            density='Score',groupby=['Trait'],as_=['Score','Density']
        ).mark_area(opacity=0.55,interpolate='monotone').encode(
            x=alt.X('Score:Q',title='Z-score'),
            y=alt.Y('Density:Q',title='',axis=alt.Axis(labels=False,ticks=False,domain=False)),
            color=alt.Color('Trait:N',scale=alt.Scale(scheme='category10'),legend=None),
            facet=alt.Facet('Trait:N',columns=2,title='',
                header=alt.Header(labelFontSize=11,labelFontWeight='bold'))
        ).properties(width=130,height=75).resolve_scale(y='independent')
        st.altair_chart(c, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# 2 — DRUG PREVALENCE
# ═══════════════════════════════════════════════════════════════
elif section == "Drug Prevalence":
    st.header("Drug Usage Prevalence")
    focus = [d for d in FOCUS_DRUGS if d in df.columns]

    records = []
    for drug in focus:
        for cl in USAGE_ORDER:
            pct = (df[drug] == cl).sum() / N * 100
            records.append({'Drug':drug,'Usage':USAGE_LABELS[cl],'UsageOrder':USAGE_ORDER.index(cl),'Pct':round(pct,1)})
    df_hm = pd.DataFrame(records)

    st.markdown('<div class="section-tag">Chart 5 — Usage heatmap (%)</div>', unsafe_allow_html=True)
    hm = alt.Chart(df_hm).mark_rect().encode(
        x=alt.X('Usage:O',sort=USAGE_DISPLAY,title='Usage frequency',axis=alt.Axis(labelAngle=-20)),
        y=alt.Y('Drug:N',sort=alt.EncodingSortField(field='Pct',op='sum',order='descending'),title=None),
        color=alt.Color('Pct:Q',scale=alt.Scale(scheme='viridis'),title='% respondents'),
        tooltip=['Drug','Usage',alt.Tooltip('Pct:Q',format='.1f',title='%')]
    ).properties(height=380)
    st.altair_chart(hm, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-tag">Chart 6 — % current users per drug</div>', unsafe_allow_html=True)
        curr = [{'Drug':d,'Current_pct':round(df[f'{d}_current'].mean()*100,1)}
                for d in focus if f'{d}_current' in df.columns]
        df_curr = pd.DataFrame(curr).sort_values('Current_pct',ascending=False)
        bar = alt.Chart(df_curr).mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(
            x=alt.X('Drug:N',sort='-y',title=None,axis=alt.Axis(labelAngle=-30)),
            y=alt.Y('Current_pct:Q',title='% current users (last month+)',axis=alt.Axis(gridColor='#f0ede6')),
            color=alt.Color('Current_pct:Q',scale=alt.Scale(scheme='reds'),legend=None),
            tooltip=['Drug',alt.Tooltip('Current_pct:Q',format='.1f',title='% current users')]
        ).properties(height=280)
        st.altair_chart(bar, use_container_width=True)

    with col2:
        st.markdown('<div class="section-tag">Chart 7 — Mean usage intensity per drug</div>', unsafe_allow_html=True)
        means = [{'Drug':d,'MeanScore':round(df[f'{d}_score'].mean(),2)}
                 for d in focus if f'{d}_score' in df.columns]
        df_means = pd.DataFrame(means).sort_values('MeanScore',ascending=False)
        bar2 = alt.Chart(df_means).mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(
            x=alt.X('Drug:N',sort='-y',title=None,axis=alt.Axis(labelAngle=-30)),
            y=alt.Y('MeanScore:Q',title='Mean score (0=Never, 6=Daily)',axis=alt.Axis(gridColor='#f0ede6')),
            color=alt.Color('MeanScore:Q',scale=alt.Scale(scheme='oranges'),legend=None),
            tooltip=['Drug',alt.Tooltip('MeanScore:Q',format='.2f')]
        ).properties(height=280)
        st.altair_chart(bar2, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-tag">Chart 8 — Stacked usage level distribution per drug</div>', unsafe_allow_html=True)
    drug_stack = st.multiselect("Select drugs:", focus, default=focus[:8])
    if drug_stack:
        df_stack = df_hm[df_hm['Drug'].isin(drug_stack)]
        stacked = alt.Chart(df_stack).mark_bar().encode(
            x=alt.X('Pct:Q',stack='normalize',title='% respondents (normalized)',
                axis=alt.Axis(format='%',gridColor='#f0ede6')),
            y=alt.Y('Drug:N',sort=alt.EncodingSortField(field='UsageOrder',op='max',order='descending'),title=None),
            color=alt.Color('Usage:O',sort=USAGE_DISPLAY,scale=alt.Scale(scheme='viridis'),title='Usage level'),
            tooltip=['Drug','Usage',alt.Tooltip('Pct:Q',format='.1f',title='%')]
        ).properties(height=max(200,len(drug_stack)*40))
        st.altair_chart(stacked, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# 3 — DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════════
elif section == "Demographics":
    st.header("Demographics × Drug Consumption")
    st.markdown("How do age, gender, and education shape drug use patterns?")

    focus = [d for d in FOCUS_DRUGS if d in df.columns]
    selected_drugs_demo = st.multiselect("Drugs to display:", focus,
        default=['Cannabis','Ecstasy','LSD','Coke','Nicotine','Alcohol'])

    if not selected_drugs_demo:
        st.warning("Select at least one drug.")
    else:
        # AGE
        st.divider()
        st.subheader("Age × Drug use")
        age_records = []
        for d in selected_drugs_demo:
            for age in AGE_ORDER:
                sub = df[df['Age_Label']==age]
                if len(sub)==0: continue
                pct  = sub[f'{d}_current'].mean()*100 if f'{d}_current' in sub.columns else 0
                mean = sub[f'{d}_score'].mean() if f'{d}_score' in sub.columns else 0
                age_records.append({'Drug':d,'Age':age,'Current_pct':round(pct,1),'MeanScore':round(mean,2)})
        df_age = pd.DataFrame(age_records)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-tag">Chart 9 — % current users by age group</div>', unsafe_allow_html=True)
            c = alt.Chart(df_age).mark_line(point=True,strokeWidth=2).encode(
                x=alt.X('Age:O',sort=AGE_ORDER,title='Age group'),
                y=alt.Y('Current_pct:Q',title='% current users',axis=alt.Axis(gridColor='#f0ede6')),
                color=alt.Color('Drug:N',scale=alt.Scale(scheme='tableau10')),
                tooltip=['Drug','Age',alt.Tooltip('Current_pct:Q',format='.1f',title='%')]
            ).properties(height=280)
            st.altair_chart(c, use_container_width=True)

        with col2:
            st.markdown('<div class="section-tag">Chart 10 — Heatmap: age × drug (mean score)</div>', unsafe_allow_html=True)
            hm = alt.Chart(df_age).mark_rect().encode(
                x=alt.X('Age:O',sort=AGE_ORDER,title='Age group'),
                y=alt.Y('Drug:N',title=None),
                color=alt.Color('MeanScore:Q',scale=alt.Scale(scheme='blues'),title='Mean score'),
                tooltip=['Drug','Age',alt.Tooltip('MeanScore:Q',format='.2f')]
            ).properties(height=280)
            txt = hm.mark_text(fontSize=10).encode(
                text=alt.Text('MeanScore:Q',format='.1f'),
                color=alt.condition(alt.datum.MeanScore>3,alt.value('white'),alt.value('black')))
            st.altair_chart(hm+txt, use_container_width=True)

        # GENDER
        st.divider()
        st.subheader("Gender × Drug use")
        gender_records = []
        for d in selected_drugs_demo:
            for g in ['Female','Male']:
                sub = df[df['Gender_Label']==g]
                if len(sub)==0: continue
                pct  = sub[f'{d}_current'].mean()*100 if f'{d}_current' in sub.columns else 0
                mean = sub[f'{d}_score'].mean() if f'{d}_score' in sub.columns else 0
                gender_records.append({'Drug':d,'Gender':g,'Current_pct':round(pct,1),'MeanScore':round(mean,2)})
        df_gender = pd.DataFrame(gender_records)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-tag">Chart 11 — % current users by gender</div>', unsafe_allow_html=True)
            bars = alt.Chart(df_gender).mark_bar(size=20).encode(
                x=alt.X('Current_pct:Q',title='% current users',axis=alt.Axis(gridColor='#f0ede6')),
                y=alt.Y('Drug:N',sort='-x',title=None),
                color=alt.Color('Gender:N',scale=alt.Scale(domain=['Female','Male'],range=['#e74c3c','#3498db'])),
                xOffset='Gender:N',
                tooltip=['Drug','Gender',alt.Tooltip('Current_pct:Q',format='.1f',title='%')]
            ).properties(height=max(240,len(selected_drugs_demo)*35))
            st.altair_chart(bars, use_container_width=True)

        with col2:
            st.markdown('<div class="section-tag">Chart 12 — Gender gap (Female − Male mean score)</div>', unsafe_allow_html=True)
            df_pivot = df_gender.pivot(index='Drug',columns='Gender',values='MeanScore').reset_index()
            if 'Female' in df_pivot.columns and 'Male' in df_pivot.columns:
                df_pivot['Gap'] = (df_pivot['Female']-df_pivot['Male']).round(2)
                df_pivot = df_pivot.sort_values('Gap')
                gap_chart = alt.Chart(df_pivot).mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(
                    x=alt.X('Gap:Q',title='Delta Score (Female - Male)',axis=alt.Axis(gridColor='#f0ede6')),
                    y=alt.Y('Drug:N',sort=alt.SortField('Gap'),title=None),
                    color=alt.condition(alt.datum.Gap>0,alt.value('#e74c3c'),alt.value('#3498db')),
                    tooltip=['Drug',alt.Tooltip('Gap:Q',format='+.2f',title='Gender gap')]
                ).properties(height=max(240,len(selected_drugs_demo)*35))
                st.altair_chart(gap_chart, use_container_width=True)

        # EDUCATION
        st.divider()
        st.subheader("Education × Drug use")
        edu_records = []
        for d in selected_drugs_demo:
            for edu in EDU_ORDER:
                sub = df[df['Education_Label']==edu]
                if len(sub)<5: continue
                pct  = sub[f'{d}_current'].mean()*100 if f'{d}_current' in sub.columns else 0
                mean = sub[f'{d}_score'].mean() if f'{d}_score' in sub.columns else 0
                edu_records.append({'Drug':d,'Education':edu,'Current_pct':round(pct,1),'MeanScore':round(mean,2),'N':len(sub)})
        df_edu = pd.DataFrame(edu_records)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-tag">Chart 13 — % current users by education level</div>', unsafe_allow_html=True)
            c = alt.Chart(df_edu).mark_line(point=True,strokeWidth=2).encode(
                x=alt.X('Education:O',sort=EDU_ORDER,title=None,
                    axis=alt.Axis(labelAngle=-35,labelLimit=120)),
                y=alt.Y('Current_pct:Q',title='% current users',axis=alt.Axis(gridColor='#f0ede6')),
                color=alt.Color('Drug:N',scale=alt.Scale(scheme='tableau10')),
                tooltip=['Drug','Education',alt.Tooltip('Current_pct:Q',format='.1f',title='%'),'N']
            ).properties(height=300)
            st.altair_chart(c, use_container_width=True)

        with col2:
            st.markdown('<div class="section-tag">Chart 14 — Education × drug heatmap (mean score)</div>', unsafe_allow_html=True)
            hm2 = alt.Chart(df_edu).mark_rect().encode(
                x=alt.X('Education:O',sort=EDU_ORDER,title=None,
                    axis=alt.Axis(labelAngle=-35,labelLimit=100)),
                y=alt.Y('Drug:N',title=None),
                color=alt.Color('MeanScore:Q',scale=alt.Scale(scheme='purples'),title='Mean score'),
                tooltip=['Drug','Education',alt.Tooltip('MeanScore:Q',format='.2f'),'N']
            ).properties(height=300)
            st.altair_chart(hm2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# 4 — TRAITS × USAGE
# ═══════════════════════════════════════════════════════════════
elif section == "Traits × Usage":
    st.header("Personality Traits × Drug Usage")
    focus = [d for d in FOCUS_DRUGS if d in df.columns]
    selected = st.multiselect("Select drugs:", focus,
        default=['Cannabis','Ecstasy','LSD','Coke','Alcohol','Nicotine','Heroin','Amphet'])

    @st.cache_data(show_spinner="Computing mean trait scores...")
    def build_df_mean(drugs_tuple, n):
        records = []
        for drug in drugs_tuple:
            for _, row in df.iterrows():
                uv = row[drug]
                for trait in TRAIT_LIST:
                    records.append({'Drug':drug,'Usage':USAGE_LABELS.get(uv,uv),
                        'UsageCode':USAGE_NUM.get(uv,-1),'Trait':trait,'Score':row[trait]})
        dl = pd.DataFrame(records)
        dm = dl.groupby(['Drug','Usage','UsageCode','Trait'])['Score'].mean().reset_index()
        dm.columns = ['Drug','Usage','UsageCode','Trait','MeanScore']
        return dm

    if selected:

        df_mean = build_df_mean(tuple(selected), N)
        st.markdown('<div class="section-tag">Chart 15 — Mean trait score by usage frequency (click legend to highlight)</div>', unsafe_allow_html=True)
        sel_param = alt.selection_point(fields=['Drug'],bind='legend')
        lines = alt.Chart(df_mean).mark_line(point=True,strokeWidth=2).encode(
            x=alt.X('UsageCode:O',title='Usage frequency',
                axis=alt.Axis(labelExpr="{'0':'Never','1':'>10yr','2':'Decade','3':'Year','4':'Month','5':'Week','6':'Day'}[datum.value]",
                    labelAngle=-30)),
            y=alt.Y('MeanScore:Q',title='Mean z-score',axis=alt.Axis(gridColor='#f0ede6')),
            color=alt.Color('Drug:N',scale=alt.Scale(scheme='tableau10')),
            opacity=alt.condition(sel_param,alt.value(1),alt.value(0.08)),
            facet=alt.Facet('Trait:N',columns=4,
                header=alt.Header(labelFontSize=11,labelFontWeight='bold')),
            tooltip=['Drug','Usage','Trait',alt.Tooltip('MeanScore:Q',format='.3f')]
        ).add_params(sel_param).properties(width=155,height=120).resolve_scale(y='independent')
        st.altair_chart(lines, use_container_width=True)

        
        st.divider()
        st.subheader("Sensation Seeking x Impulsivity deep-dive")
        col1, col2 = st.columns([3,1])
        with col2:
            scatter_drug = st.selectbox("Drug for scatter:", selected, index=0)
        drug_l = scatter_drug
        df_sc = df[['Sensation Seeking','Impulsivity',drug_l]].copy()
        df_sc['Score'] = df_sc[drug_l].map(USAGE_NUM)
        df_sc['Group'] = df_sc['Score'].apply(lambda c: 'Non/Rare' if c<=1 else ('Past' if c<=3 else 'Current'))
        df_sc['UsageLabel'] = df_sc[drug_l].map(USAGE_LABELS)
        scatter = alt.Chart(df_sc.sample(min(800,len(df_sc)),random_state=42)).mark_circle(size=55,opacity=0.6).encode(
            x=alt.X('Sensation Seeking:Q'),
            y=alt.Y('Impulsivity:Q'),
            color=alt.Color('Group:N',title=f'{scatter_drug} use',
                scale=alt.Scale(domain=['Non/Rare','Past','Current'],range=['#2ecc71','#f39c12','#e74c3c'])),
            tooltip=['Sensation Seeking','Impulsivity','UsageLabel']
        ).properties(height=340)
        reg = scatter.transform_regression('Sensation Seeking','Impulsivity',groupby=['Group']).mark_line(size=2)
        with col1:
            st.altair_chart((scatter+reg).resolve_scale(color='shared'), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# 5 — DRUG CO-USE
# ═══════════════════════════════════════════════════════════════
elif section == "Drug Co-use":
    st.header("Drug Co-use Patterns")
    st.markdown("Which drugs tend to be used together?")

    focus = [d for d in FOCUS_DRUGS if f'{d}_score' in df.columns]
    score_cols = [f'{d}_score' for d in focus]
    corr = df[score_cols].corr().round(2)
    corr.index   = [c.replace('_score','') for c in corr.index]
    corr.columns = [c.replace('_score','') for c in corr.columns]
    corr_long = corr.reset_index().melt(id_vars='index')
    corr_long.columns = ['Drug_A','Drug_B','Correlation']

    st.markdown('<div class="section-tag">Chart 17 — Drug co-use correlation matrix</div>', unsafe_allow_html=True)
    hm = alt.Chart(corr_long).mark_rect().encode(
        x=alt.X('Drug_A:N',title=None,axis=alt.Axis(labelAngle=-35)),
        y=alt.Y('Drug_B:N',title=None),
        color=alt.Color('Correlation:Q',scale=alt.Scale(scheme='redblue',domain=[-0.2,1],reverse=True),title='Pearson r'),
        tooltip=['Drug_A','Drug_B',alt.Tooltip('Correlation:Q',format='.2f')]
    ).properties(height=400)
    txt = hm.mark_text(fontSize=9).encode(
        text=alt.Text('Correlation:Q',format='.2f'),
        color=alt.condition(alt.datum.Correlation>0.6,alt.value('white'),alt.value('black')))
    st.altair_chart(hm+txt, use_container_width=True)

    st.divider()
    st.subheader("Joint user analysis")
    col1, col2 = st.columns(2)
    with col1:
        drug_a = st.selectbox("Drug A:", focus, index=focus.index('Cannabis') if 'Cannabis' in focus else 0)
    with col2:
        drug_b = st.selectbox("Drug B:", focus, index=focus.index('Ecstasy') if 'Ecstasy' in focus else 1)

    df_joint = df[[drug_a,drug_b]].copy()
    df_joint['A_current'] = df[drug_a].isin(['CL4','CL5','CL6'])
    df_joint['B_current'] = df[drug_b].isin(['CL4','CL5','CL6'])
    df_joint['Group'] = df_joint.apply(
        lambda r: f'Both' if r['A_current'] and r['B_current']
             else f'{drug_a} only' if r['A_current']
             else f'{drug_b} only' if r['B_current']
             else 'Neither', axis=1)

    vc = df_joint['Group'].value_counts().reset_index()
    vc.columns = ['Group','Count']
    vc['Pct'] = (vc['Count']/N*100).round(1)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-tag">Chart 18 — Co-use breakdown</div>', unsafe_allow_html=True)
        donut = alt.Chart(vc).mark_arc(innerRadius=60).encode(
            theta=alt.Theta('Count:Q'),
            color=alt.Color('Group:N',scale=alt.Scale(scheme='tableau10')),
            tooltip=['Group','Count',alt.Tooltip('Pct:Q',format='.1f',title='%')]
        ).properties(height=240)
        st.altair_chart(donut, use_container_width=True)

    with col4:
        st.markdown('<div class="section-tag">Chart 19 — Trait profiles by co-use group</div>', unsafe_allow_html=True)
        df_jt = df_joint[['Group']].join(df[TRAIT_LIST])
        jt_mean = df_jt.melt(id_vars='Group',var_name='Trait',value_name='Score')\
                       .groupby(['Group','Trait'])['Score'].mean().reset_index()
        jt_chart = alt.Chart(jt_mean).mark_bar(size=14).encode(
            x=alt.X('Score:Q',title='Mean trait score (z)',axis=alt.Axis(gridColor='#f0ede6')),
            y=alt.Y('Trait:N',title=None),
            color=alt.Color('Group:N',scale=alt.Scale(scheme='tableau10')),
            xOffset='Group:N',
            tooltip=['Group','Trait',alt.Tooltip('Score:Q',format='.3f')]
        ).properties(height=280)
        st.altair_chart(jt_chart, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-tag">Chart 20 — Co-use scatter: usage scores of two drugs</div>', unsafe_allow_html=True)
    sc2 = alt.Chart(df[[f'{drug_a}_score',f'{drug_b}_score']].rename(
        columns={f'{drug_a}_score':drug_a,f'{drug_b}_score':drug_b})
    ).mark_circle(size=40,opacity=0.35,color='#1a1a1a').encode(
        x=alt.X(f'{drug_a}:Q',title=f'{drug_a} usage score (0=Never, 6=Daily)'),
        y=alt.Y(f'{drug_b}:Q',title=f'{drug_b} usage score'),
        tooltip=[drug_a,drug_b]
    ).properties(height=320)
    reg2 = sc2.transform_regression(drug_a,drug_b).mark_line(color='#c0392b',size=2)
    st.altair_chart(sc2+reg2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# 6 — POLY-DRUG PROFILES
# ═══════════════════════════════════════════════════════════════
elif section == "Poly-drug Profiles":
    st.header("Poly-drug Use Profiles")
    st.markdown("Who uses multiple substances simultaneously? How do personality traits scale with poly-drug intensity?")

    st.markdown('<div class="section-tag">Chart 21 — Distribution of simultaneous current drug use</div>', unsafe_allow_html=True)
    vc = df['n_illicit_current'].value_counts().sort_index().reset_index()
    vc.columns = ['n_drugs','Count']
    vc['Pct'] = (vc['Count']/N*100).round(1)
    hist = alt.Chart(vc).mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(
        x=alt.X('n_drugs:O',title='Number of illicit drugs used currently',axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Count:Q',title='Number of respondents',axis=alt.Axis(gridColor='#f0ede6')),
        color=alt.Color('n_drugs:O',scale=alt.Scale(scheme='reds'),legend=None),
        tooltip=['n_drugs','Count',alt.Tooltip('Pct:Q',format='.1f',title='%')]
    ).properties(height=260)
    txt_h = hist.mark_text(dy=-8,fontSize=11).encode(text=alt.Text('Pct:Q',format='.1f'))
    st.altair_chart(hist+txt_h, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-tag">Chart 22 — Personality traits by poly-drug intensity</div>', unsafe_allow_html=True)
    poly_trait = df.groupby('n_illicit_current')[TRAIT_LIST].mean().reset_index()
    poly_melt  = poly_trait.melt(id_vars='n_illicit_current',var_name='Trait',value_name='MeanScore')
    poly_lines = alt.Chart(poly_melt).mark_line(point=True,strokeWidth=2.5).encode(
        x=alt.X('n_illicit_current:O',title='Number of current illicit drugs'),
        y=alt.Y('MeanScore:Q',title='Mean trait score (z)',axis=alt.Axis(gridColor='#f0ede6')),
        color=alt.Color('Trait:N',scale=alt.Scale(scheme='category10')),
        tooltip=['Trait','n_illicit_current',alt.Tooltip('MeanScore:Q',format='.3f')]
    ).properties(height=320)
    st.altair_chart(poly_lines, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-tag">Chart 23 — Age × poly-drug use (%)</div>', unsafe_allow_html=True)
        ap = df.groupby('Age_Label')['n_illicit_current'].value_counts(normalize=True)\
               .mul(100).round(1).reset_index()
        ap.columns = ['Age','n_drugs','Pct']
        age_poly = alt.Chart(ap).mark_rect().encode(
            x=alt.X('n_drugs:O',title='N current drugs'),
            y=alt.Y('Age:O',sort=AGE_ORDER,title=None),
            color=alt.Color('Pct:Q',scale=alt.Scale(scheme='oranges'),title='% of age group'),
            tooltip=['Age','n_drugs',alt.Tooltip('Pct:Q',format='.1f',title='%')]
        ).properties(height=240)
        txt_ap = age_poly.mark_text(fontSize=9).encode(
            text=alt.Text('Pct:Q',format='.0f'),
            color=alt.condition(alt.datum.Pct>15,alt.value('white'),alt.value('black')))
        st.altair_chart(age_poly+txt_ap, use_container_width=True)

    with col2:
        st.markdown('<div class="section-tag">Chart 24 — Education × mean poly-drug use</div>', unsafe_allow_html=True)
        ep = df.groupby('Education_Label')['n_illicit_current'].mean().reset_index()
        ep.columns = ['Education','Mean_n_drugs']
        ep = ep.sort_values('Mean_n_drugs',ascending=False)
        edu_poly = alt.Chart(ep).mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(
            x=alt.X('Mean_n_drugs:Q',title='Mean number of current illicit drugs',axis=alt.Axis(gridColor='#f0ede6')),
            y=alt.Y('Education:N',sort='-x',title=None),
            color=alt.Color('Mean_n_drugs:Q',scale=alt.Scale(scheme='purples'),legend=None),
            tooltip=['Education',alt.Tooltip('Mean_n_drugs:Q',format='.2f')]
        ).properties(height=240)
        st.altair_chart(edu_poly, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-tag">Chart 25 — Which drugs are used by heavy poly-users?</div>', unsafe_allow_html=True)
    poly_drug_records = []
    for n in range(0,7):
        sub = df[df['n_illicit_current']==n]
        if len(sub)<5: continue
        for d in [x for x in ILLICIT if x in df.columns]:
            pct = sub[f'{d}_current'].mean()*100 if f'{d}_current' in sub.columns else 0
            poly_drug_records.append({'N_drugs':n,'Drug':d,'Pct':round(pct,1)})
    df_pd = pd.DataFrame(poly_drug_records)
    pd_hm = alt.Chart(df_pd).mark_rect().encode(
        x=alt.X('N_drugs:O',title='Number of current illicit drugs'),
        y=alt.Y('Drug:N',title=None),
        color=alt.Color('Pct:Q',scale=alt.Scale(scheme='reds'),title='% current users'),
        tooltip=['Drug','N_drugs',alt.Tooltip('Pct:Q',format='.1f',title='% current')]
    ).properties(height=300)
    pd_txt = pd_hm.mark_text(fontSize=10).encode(
        text=alt.Text('Pct:Q',format='.0f'),
        color=alt.condition(alt.datum.Pct>50,alt.value('white'),alt.value('black')))
    st.altair_chart(pd_hm+pd_txt, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# 7 — PREDICTION
# ═══════════════════════════════════════════════════════════════
elif section == "Prediction":
    st.header("Predictive Modeling")
    st.markdown("Logistic regression: which traits & demographics predict current drug use?")

    focus = [d for d in FOCUS_DRUGS if d in df.columns]
    model_drugs = st.multiselect("Drugs to model:", focus,
        default=['Cannabis','Ecstasy','LSD','Coke','Nicotine','Heroin','Amphet','Benzos'])

    col1, col2 = st.columns(2)
    with col1:
        C_val = st.slider("Regularization C:", 0.01, 5.0, 1.0, 0.1)
    with col2:
        cv_k = st.slider("CV folds:", 3, 10, 5)

    features = st.multiselect("Features:", TRAIT_LIST+['Age_Label','Education_Label','Gender_Label'], default=TRAIT_LIST)

    @st.cache_data(show_spinner="Fitting models...")
    def fit_models(drugs_t, C, cv, feats_t, n):
        feat_cols = [f for f in feats_t if f in TRAIT_LIST+['Age_Label','Education_Label','Gender_Label']]
        X_all = df[feat_cols].dropna()
        records = []
        for drug in drugs_t:
            tmp = df.loc[X_all.index,drug]
            y_bin = tmp.map(lambda x: 1 if x in ['CL4','CL5','CL6'] else 0)
            valid = ~y_bin.isna()
            Xf,yf = X_all[valid],y_bin[valid]
            if yf.sum()<10: continue
            lr = LogisticRegression(max_iter=1000,C=C)
            lr.fit(Xf,yf)
            auc = cross_val_score(lr,Xf,yf,cv=cv,scoring='roc_auc').mean()
            for feat,coef in zip(feat_cols,lr.coef_[0]):
                records.append({'Drug':drug,'Feature':feat,'Coefficient':round(coef,3),
                    'AUC':round(auc,3),'Prevalence':f"{yf.mean()*100:.0f}%"})
        return pd.DataFrame(records)

    if model_drugs and features:
        df_coef = fit_models(tuple(model_drugs),C_val,cv_k,tuple(features),N)
        if not df_coef.empty:
            auc_df = df_coef.groupby('Drug')['AUC'].first().reset_index().sort_values('AUC',ascending=False)

            st.markdown('<div class="section-tag">Chart 26 — Model AUC per drug (higher = better prediction)</div>', unsafe_allow_html=True)
            auc_bar = alt.Chart(auc_df).mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(
                x=alt.X('Drug:N',sort='-y',title=None,axis=alt.Axis(labelAngle=-25)),
                y=alt.Y('AUC:Q',scale=alt.Scale(domain=[0.5,1.0]),title='CV AUC',axis=alt.Axis(gridColor='#f0ede6')),
                color=alt.Color('AUC:Q',scale=alt.Scale(scheme='blues',domain=[0.5,0.9]),legend=None),
                tooltip=['Drug',alt.Tooltip('AUC:Q',format='.3f')]
            ).properties(height=220)
            rule = alt.Chart(pd.DataFrame({'y':[0.5]})).mark_rule(color='red',strokeDash=[4,3]).encode(y='y:Q')
            st.altair_chart(auc_bar+rule, use_container_width=True)

            st.divider()
            st.markdown('<div class="section-tag">Chart 27 — Coefficient heatmap (log-odds)</div>', unsafe_allow_html=True)
            coef_hm = alt.Chart(df_coef).mark_rect().encode(
                x=alt.X('Drug:N',title=None,axis=alt.Axis(labelAngle=-30)),
                y=alt.Y('Feature:N',title=None),
                color=alt.Color('Coefficient:Q',scale=alt.Scale(scheme='redblue',domain=[-1.5,1.5],reverse=True),title='Log-odds'),
                tooltip=['Drug','Feature',alt.Tooltip('Coefficient:Q',format='.3f'),
                    alt.Tooltip('AUC:Q',title='CV AUC'),'Prevalence']
            ).properties(height=max(200,len(features)*40))
            coef_txt = coef_hm.mark_text(fontSize=10).encode(
                text=alt.Text('Coefficient:Q',format='.2f'),
                color=alt.condition(alt.datum.Coefficient>0.4,alt.value('white'),alt.value('black')))
            st.altair_chart(coef_hm+coef_txt, use_container_width=True)

            st.divider()
            st.markdown('<div class="section-tag">Chart 28 — Top predictors per drug (lollipop)</div>', unsafe_allow_html=True)
            drug_pred = st.selectbox("Select drug:", model_drugs, key='lollipop')
            sub_coef  = df_coef[df_coef['Drug']==drug_pred].sort_values('Coefficient')
            lollipop = alt.layer(
                alt.Chart(sub_coef).mark_rule(color='#ccc').encode(
                    y=alt.Y('Feature:N',sort=alt.SortField('Coefficient'),title=None),
                    x=alt.X('Coefficient:Q',title='Log-odds coefficient',axis=alt.Axis(gridColor='#f0ede6')),
                    x2=alt.X2(value=0)),
                alt.Chart(sub_coef).mark_point(size=120,filled=True).encode(
                    y=alt.Y('Feature:N',sort=alt.SortField('Coefficient')),
                    x=alt.X('Coefficient:Q'),
                    color=alt.condition(alt.datum.Coefficient>0,alt.value('#e74c3c'),alt.value('#3498db')),
                    tooltip=['Feature',alt.Tooltip('Coefficient:Q',format='.3f')])
            ).properties(height=280)
            st.altair_chart(lollipop, use_container_width=True)
            auc_val = auc_df[auc_df['Drug']==drug_pred]['AUC'].values
            if len(auc_val):
                st.caption(f"Model AUC for {drug_pred}: **{auc_val[0]:.3f}** (random baseline = 0.5)")


# ═══════════════════════════════════════════════════════════════
# 8 — RADAR
# ═══════════════════════════════════════════════════════════════
elif section == "Radar":
    st.header("Psychological Profile Radar")
    st.markdown("Compare personality fingerprints across drugs and usage levels.")

    focus_radar = [d for d in FOCUS_DRUGS if d in df.columns]
    usage_level_labels = {'CL0':'Never','CL1':'>10 years ago','CL2':'Last decade',
                          'CL3':'Last year','CL4':'Last month','CL5':'Last week','CL6':'Last day'}

    @st.cache_data(show_spinner="Building radar data...")
    def build_radar(drugs_t, n):
        nt = len(TRAIT_LIST)
        angles = [i*2*math.pi/nt for i in range(nt)]
        records = []
        for drug in drugs_t:
            tmp = df[TRAIT_LIST+[drug]].copy()
            tmp.columns = TRAIT_LIST+['CL']
            grouped = tmp.groupby('CL')[TRAIT_LIST].mean().reset_index()
            grouped['N'] = tmp.groupby('CL').size().values
            for _,row in grouped.iterrows():
                cl = row['CL']
                if cl not in usage_level_labels or row['N']<8: continue
                for i,trait in enumerate(TRAIT_LIST):
                    records.append({'Drug':drug,'UsageLevel':usage_level_labels[cl],'CL':cl,
                        'Trait':trait,'TraitIndex':i,'MeanScore':round(row[trait],3),'N':int(row['N'])})
        df_r = pd.DataFrame(records)
        if df_r.empty: return df_r,angles,nt
        df_r['NormScore'] = df_r.groupby('Trait')['MeanScore'].transform(
            lambda x: (x-x.min())/(x.max()-x.min()+1e-9))
        df_r['angle'] = df_r['TraitIndex'].apply(lambda i: angles[i]-math.pi/2)
        df_r['x'] = df_r['NormScore']*np.cos(df_r['angle'])
        df_r['y'] = df_r['NormScore']*np.sin(df_r['angle'])
        closure = df_r[df_r['TraitIndex']==0].copy()
        closure['TraitIndex'] = nt
        df_plot = pd.concat([df_r,closure],ignore_index=True).drop(columns=['angle'])
        return df_plot,angles,nt

    df_plot,angles,n_traits = build_radar(tuple(focus_radar),N)

    col1, col2 = st.columns(2)
    with col1:
        drug_options = sorted(df_plot['Drug'].unique().tolist()) if not df_plot.empty else []
        sel_drugs = st.multiselect("Drug(s):", drug_options, default=drug_options[:2])
    with col2:
        usage_opts = [u for u in ['Never','>10 years ago','Last decade','Last year',
                                   'Last month','Last week','Last day']
                      if not df_plot.empty and u in df_plot['UsageLevel'].unique()]
        sel_usage = st.multiselect("Usage level(s):", usage_opts, default=['Never','Last day'])

    if sel_drugs and sel_usage and not df_plot.empty:
        stroke_map = {'Never':[1,0],'>10 years ago':[4,2],'Last decade':[6,2],'Last year':[8,3],
                      'Last month':[2,2],'Last week':[6,2,2,2],'Last day':[1,0]}
        df_f = df_plot[df_plot['Drug'].isin(sel_drugs)&df_plot['UsageLevel'].isin(sel_usage)].copy()
        if df_f.empty:
            st.warning("No data for this selection.")
        else:
            sd = [u for u in usage_opts if u in sel_usage]
            sr = [stroke_map[u] for u in sd]
            data = alt.Data(values=df_f.to_dict(orient='records'))

            circles_data = pd.DataFrame([
                {'r':r,'xc':r*math.cos(t-math.pi/2),'yc':r*math.sin(t-math.pi/2)}
                for r in [0.25,0.5,0.75,1.0]
                for t in [i*2*math.pi/60 for i in range(61)]])
            grid = alt.Chart(circles_data).mark_line(color='#ccc',strokeWidth=0.8,opacity=0.6).encode(
                x=alt.X('xc:Q',axis=None,scale=alt.Scale(domain=[-1.5,1.5])),
                y=alt.Y('yc:Q',axis=None,scale=alt.Scale(domain=[-1.5,1.5])),detail='r:O')

            ax_data = []
            for i,tr in enumerate(TRAIT_LIST):
                a = angles[i]-math.pi/2
                ax_data+=[{'seg':i,'x':0.0,'y':0.0,'Trait':tr},
                           {'seg':i,'x':math.cos(a)*1.05,'y':math.sin(a)*1.05,'Trait':tr}]
            axes = alt.Chart(pd.DataFrame(ax_data)).mark_line(color='#bbb',strokeWidth=1).encode(
                x=alt.X('x:Q',axis=None,scale=alt.Scale(domain=[-1.5,1.5])),
                y=alt.Y('y:Q',axis=None,scale=alt.Scale(domain=[-1.5,1.5])),detail='seg:O')

            lab_data = pd.DataFrame([{'Trait':TRAIT_LIST[i],
                'x':1.3*math.cos(angles[i]-math.pi/2),'y':1.3*math.sin(angles[i]-math.pi/2)}
                for i in range(n_traits)])
            labs = alt.Chart(lab_data).mark_text(fontSize=12,fontWeight='bold',color='#333').encode(
                x=alt.X('x:Q',axis=None,scale=alt.Scale(domain=[-1.5,1.5])),
                y=alt.Y('y:Q',axis=None,scale=alt.Scale(domain=[-1.5,1.5])),text='Trait:N')

            lines = alt.Chart(data).mark_line(strokeWidth=2.8,opacity=0.85).encode(
                x=alt.X('x:Q',axis=None,scale=alt.Scale(domain=[-1.5,1.5])),
                y=alt.Y('y:Q',axis=None,scale=alt.Scale(domain=[-1.5,1.5])),
                color=alt.Color('Drug:N',scale=alt.Scale(scheme='tableau10'),title='Drug'),
                strokeDash=alt.StrokeDash('UsageLevel:N',scale=alt.Scale(domain=sd,range=sr),title='Usage'),
                order=alt.Order('TraitIndex:O'),detail=alt.Detail(['Drug:N','UsageLevel:N']),
                tooltip=['Drug:N','UsageLevel:N','Trait:N',
                    alt.Tooltip('MeanScore:Q',format='.3f'),alt.Tooltip('N:Q',title='n')])

            pts = alt.Chart(data).mark_point(size=65,filled=True,opacity=0.9).encode(
                x=alt.X('x:Q',axis=None,scale=alt.Scale(domain=[-1.5,1.5])),
                y=alt.Y('y:Q',axis=None,scale=alt.Scale(domain=[-1.5,1.5])),
                color=alt.Color('Drug:N',scale=alt.Scale(scheme='tableau10')),
                tooltip=['Drug:N','UsageLevel:N','Trait:N',alt.Tooltip('MeanScore:Q',format='.3f')])

            radar = (grid+axes+labs+lines+pts).properties(
                title=alt.TitleParams('Psychological Profile Radar',fontSize=15),
                width=520,height=520)
            col_r,_ = st.columns([2,1])
            with col_r:
                st.altair_chart(radar)

            with st.expander("Raw mean scores"):
                pivot = df_f.groupby(['Drug','UsageLevel','Trait'])['MeanScore'].mean().reset_index()
                st.dataframe(
                    pivot.pivot_table(index=['Drug','UsageLevel'],columns='Trait',values='MeanScore').round(3),
                    use_container_width=True)
    else:
        st.info("Select at least one drug and one usage level.")
