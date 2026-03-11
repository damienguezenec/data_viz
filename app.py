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
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    h1 { font-family: 'Playfair Display', serif !important; }
    h2 { font-family: 'Playfair Display', serif !important; }
    h3 { font-family: 'Playfair Display', serif !important; }

    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    .kicker {
        font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.15em; text-transform: uppercase;
        color: #c0392b; margin-bottom: 0.5rem;
    }
    .insight-box {
        background: #f8f5f0;
        border-left: 3px solid #c0392b;
        padding: 1rem 1.2rem;
        border-radius: 0 4px 4px 0;
        margin-bottom: 0.8rem;
    }
    .insight-box strong { color: #c0392b; font-size: 0.78rem;
        text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-card {
        background: white; border: 1px solid #e8e5de;
        padding: 1rem; text-align: center; border-radius: 4px;
    }
    .metric-card .num { font-family: 'Playfair Display', serif;
        font-size: 2rem; font-weight: 700; color: #1a1a1a; }
    .metric-card .lbl { font-size: 0.75rem; color: #8a8580;
        text-transform: uppercase; letter-spacing: 0.06em; }
    .stTabs [data-baseweb="tab"] { font-family: 'DM Sans', sans-serif; }
    div[data-testid="stSidebarContent"] { background: #faf9f6; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset from UCI…")
def load_data():
    from ucimlrepo import fetch_ucirepo
    alt.data_transformers.disable_max_rows()

    drug_consumption_quantified = fetch_ucirepo(id=373)
    X = drug_consumption_quantified.data.features
    y = drug_consumption_quantified.data.targets
    df = pd.concat([X, y], axis=1)

    personality_cols = {
        'nscore':    'Neuroticism',
        'escore':    'Extraversion',
        'oscore':    'Openness',
        'ascore':    'Agreeableness',
        'cscore':    'Conscientiousness',
        'impuslive': 'Impulsivity',
        'ss':        'Sensation Seeking'
    }

    drug_cols = [
        'alcohol', 'amphet', 'amyl', 'benzos', 'caff', 'cannabis',
        'choc', 'coke', 'crack', 'ecstasy', 'heroin', 'ketamine',
        'legalh', 'lsd', 'meth', 'mushrooms', 'nicotine', 'semer', 'vsa'
    ]
    drug_cols = [d for d in drug_cols if d in df.columns]

    usage_order  = ['CL0', 'CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6']
    usage_labels = {
        'CL0': 'Never', 'CL1': '>10 years ago', 'CL2': 'Last decade',
        'CL3': 'Last year', 'CL4': 'Last month', 'CL5': 'Last week', 'CL6': 'Last day'
    }
    usage_numeric = {v: i for i, v in enumerate(usage_order)}

    country_map = {
        -0.57009: 'USA',   -0.28519: 'Other',  -0.09765: 'Australia',
         0.21128: 'Republic of Ireland', 0.24923: 'Canada',
        -0.46841: 'New Zealand', 0.96082: 'UK'
    }

    df_as = df.copy()
    df_as['Country_Label'] = df_as['country'].map(country_map)
    df_as = df_as.rename(columns=personality_cols)

    for drug in drug_cols:
        if drug in df_as.columns:
            df_as[f'{drug}_score'] = df_as[drug].map(usage_numeric)

    df_as.columns = [c.lower() if c in drug_cols else c for c in df_as.columns]
    drug_cols = [d.lower() for d in drug_cols]

    return df_as, drug_cols, usage_order, usage_labels, usage_numeric, list(personality_cols.values())


df_as, drug_cols, usage_order, usage_labels, usage_numeric, trait_list = load_data()
alt.data_transformers.disable_max_rows()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Mind & Substance")
    st.markdown("*Personality Traits & Drug Consumption*")
    st.divider()

    st.markdown("**Navigation**")
    section = st.radio(
        "", 
        ["📊 Overview", "🌡️ Prevalence", "📈 Traits vs Usage",
         "🔬 Correlations", "🤖 Prediction", "👥 User Profiles", "🕸️ Radar"],
        label_visibility="collapsed"
    )
    st.divider()

    st.markdown("**Dataset info**")
    st.caption(f"N = {len(df_as):,} respondents")
    st.caption("UCI Drug Consumption (Quantified)")
    st.caption("Fehrman et al.")
    st.divider()
    st.markdown("*Built with Streamlit & Altair*")

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="kicker">Data Analysis · Psychology · Substance Use</div>', unsafe_allow_html=True)
st.title("Who uses drugs? It starts in the mind.")
st.markdown(
    "An analysis of how **Big Five personality traits**, impulsivity, and sensation seeking "
    "predict drug consumption across **{:,} Anglo-Saxon respondents**.".format(len(df_as))
)
st.divider()

# ─────────────────────────────────────────────────────────────
# SECTION: OVERVIEW
# ─────────────────────────────────────────────────────────────
if section == "📊 Overview":
    st.header("Dataset Overview")

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="num">{:,}</div><div class="lbl">Respondents</div></div>'.format(len(df_as)), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="num">19</div><div class="lbl">Substances tracked</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="num">7</div><div class="lbl">Personality traits</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="num">7</div><div class="lbl">Usage levels (CL0–CL6)</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Respondents by Country")
        country_counts = df_as['Country_Label'].value_counts().reset_index()
        country_counts.columns = ['Country', 'Count']
        chart_country = alt.Chart(country_counts).mark_bar(
            cornerRadiusTopLeft=4, cornerRadiusTopRight=4
        ).encode(
            x=alt.X('Country:N', sort='-y', title='Country'),
            y=alt.Y('Count:Q', title='Number of Respondents'),
            color=alt.Color('Country:N', legend=None, scale=alt.Scale(scheme='tableau10')),
            tooltip=['Country', 'Count']
        ).properties(width=350, height=280)
        st.altair_chart(chart_country, use_container_width=True)

    with col2:
        st.subheader("Personality Trait Distributions")
        df_traits = df_as[trait_list].melt(var_name='Trait', value_name='Score')
        chart_traits = alt.Chart(df_traits).transform_density(
            density='Score', groupby=['Trait'], as_=['Score', 'Density']
        ).mark_area(opacity=0.6, interpolate='monotone').encode(
            x=alt.X('Score:Q', title='Z-Score'),
            y=alt.Y('Density:Q', title=''),
            color=alt.Color('Trait:N', scale=alt.Scale(scheme='category10'), legend=None),
            facet=alt.Facet('Trait:N', columns=2, title='')
        ).properties(width=140, height=90).resolve_scale(y='independent')
        st.altair_chart(chart_traits, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# SECTION: PREVALENCE
# ─────────────────────────────────────────────────────────────
elif section == "🌡️ Prevalence":
    st.header("Drug Usage Prevalence")
    st.markdown("The heatmap shows the distribution of usage levels across substances. Darker cells = more recent/frequent use.")

    st.markdown('<div class="insight-box"><strong>Key observation</strong><br>Most of the participants do not use illegal drugs. However, alcohol, cannabis and nicotine are way more tried than others.</div>', unsafe_allow_html=True)

    base_drugs = ['cannabis', 'alcohol', 'nicotine', 'ecstasy', 'coke',
                  'lsd', 'ketamine', 'mushrooms', 'amphet', 'benzos', 'heroin', 'meth']
    focus_drugs = [d for d in base_drugs if d in df_as.columns]

    records = []
    for drug in focus_drugs:
        for label in usage_order:
            pct = (df_as[drug] == label).sum() / len(df_as) * 100
            records.append({
                'Drug': drug.capitalize(),
                'Usage': usage_labels[label],
                'UsageOrder': usage_order.index(label),
                'Pct': round(pct, 1)
            })
    df_heatmap = pd.DataFrame(records)
    usage_display_order = [usage_labels[u] for u in usage_order]

    chart_heatmap = alt.Chart(df_heatmap).mark_rect().encode(
        x=alt.X('Usage:O', sort=usage_display_order, title='Usage Frequency'),
        y=alt.Y('Drug:N', sort=alt.EncodingSortField(field='Pct', op='sum', order='descending'), title='Drug'),
        color=alt.Color('Pct:Q', scale=alt.Scale(scheme='viridis'), title='% Respondents'),
        tooltip=['Drug:N', 'Usage:O', alt.Tooltip('Pct:Q', title='% Respondents', format='.1f')]
    ).properties(title='Drug Usage Distribution (%)', height=380)

    st.altair_chart(chart_heatmap, use_container_width=True)

    with st.expander("Show raw data"):
        st.dataframe(df_heatmap.pivot(index='Drug', columns='Usage', values='Pct')[usage_display_order], use_container_width=True)

# ─────────────────────────────────────────────────────────────
# SECTION: TRAITS VS USAGE
# ─────────────────────────────────────────────────────────────
elif section == "📈 Traits vs Usage":
    st.header("Personality Traits vs. Drug Usage")

    # --- Build df_mean ---
    selected_drugs_lower = ['cannabis', 'ecstasy', 'coke', 'lsd', 'alcohol', 'nicotine', 'heroin', 'amphet']
    selected_drugs = [d for d in selected_drugs_lower if d in df_as.columns]

    @st.cache_data(show_spinner="Computing mean trait scores…")
    def build_df_mean(drugs):
        records2 = []
        for drug in drugs:
            for _, row in df_as.iterrows():
                usage_val = row[drug]
                for trait in trait_list:
                    records2.append({
                        'Drug': drug.capitalize(),
                        'Usage': usage_labels.get(usage_val, usage_val),
                        'UsageCode': usage_numeric.get(usage_val, -1),
                        'Trait': trait,
                        'Score': row[trait]
                    })
        df_long = pd.DataFrame(records2)
        df_mean = df_long.groupby(['Drug', 'Usage', 'UsageCode', 'Trait'])['Score'].mean().reset_index()
        df_mean.columns = ['Drug', 'Usage', 'UsageCode', 'Trait', 'MeanScore']
        return df_mean

    df_mean = build_df_mean(tuple(selected_drugs))

    st.subheader("Mean Trait Score by Usage Frequency")
    st.caption("Click the legend to highlight a drug.")

    drug_selector = alt.selection_point(fields=['Drug'], bind='legend')
    chart_lines = alt.Chart(df_mean).mark_line(point=True).encode(
        x=alt.X('UsageCode:O', title='Usage Frequency',
                axis=alt.Axis(labelExpr="{'0':'Never','1':'>10yr','2':'Decade','3':'Year','4':'Month','5':'Week','6':'Day'}[datum.value]")),
        y=alt.Y('MeanScore:Q', title='Mean Trait Score (z)'),
        color=alt.Color('Drug:N', scale=alt.Scale(scheme='tableau10')),
        opacity=alt.condition(drug_selector, alt.value(1), alt.value(0.1)),
        facet=alt.Facet('Trait:N', columns=4),
        tooltip=['Drug', 'Usage', 'Trait', alt.Tooltip('MeanScore:Q', format='.3f')]
    ).add_params(drug_selector).properties(width=160, height=130).resolve_scale(y='independent')

    st.altair_chart(chart_lines, use_container_width=True)

    st.divider()
    st.subheader("Sensation Seeking × Impulsivity")

    col1, col2 = st.columns([2, 1])
    with col2:
        scatter_drug = st.selectbox("Color scatter by drug use:", [d.capitalize() for d in selected_drugs], index=0)
    
    scatter_drug_lower = scatter_drug.lower()
    df_scatter = df_as[['Sensation Seeking', 'Impulsivity', scatter_drug_lower]].copy()
    df_scatter['Drug_Score'] = df_scatter[scatter_drug_lower].map(usage_numeric)
    
    def usage_group(code):
        if code <= 1: return 'Non/Rare user'
        elif code <= 3: return 'Past user'
        else: return 'Current user'
    
    df_scatter['Usage_Group'] = df_scatter['Drug_Score'].apply(usage_group)
    df_scatter['Usage_Label'] = df_scatter[scatter_drug_lower].map(usage_labels)

    chart_scatter = alt.Chart(df_scatter.sample(min(800, len(df_scatter)), random_state=42)).mark_circle(
        size=60, opacity=0.6
    ).encode(
        x=alt.X('Sensation Seeking:Q', title='Sensation Seeking (z-score)'),
        y=alt.Y('Impulsivity:Q', title='Impulsivity (z-score)'),
        color=alt.Color('Usage_Group:N', title=f'{scatter_drug} Use',
                        scale=alt.Scale(domain=['Non/Rare user', 'Past user', 'Current user'],
                                        range=['#2ecc71', '#f39c12', '#e74c3c'])),
        tooltip=['Sensation Seeking', 'Impulsivity', 'Usage_Label']
    ).properties(height=350)

    chart_reg = chart_scatter.transform_regression(
        'Sensation Seeking', 'Impulsivity', groupby=['Usage_Group']
    ).mark_line(size=2)

    with col1:
        st.altair_chart((chart_scatter + chart_reg).resolve_scale(color='shared'), use_container_width=True)

    st.divider()
    st.subheader("Box Plots: SS & Impulsivity by Drug & Usage Group")
    
    drugs_boxplot_opts = [d for d in ['cannabis', 'ecstasy', 'lsd', 'coke', 'nicotine', 'heroin'] if d in df_as.columns]
    selected_box_drugs = st.multiselect("Select drugs for box plots:", [d.capitalize() for d in drugs_boxplot_opts], default=[d.capitalize() for d in drugs_boxplot_opts[:4]])
    selected_box_drugs_lower = [d.lower() for d in selected_box_drugs]

    if selected_box_drugs_lower:
        df_box_records = []
        for drug in selected_box_drugs_lower:
            tmp = df_as[[drug, 'Sensation Seeking', 'Impulsivity']].copy()
            tmp['Drug'] = drug.capitalize()
            tmp['UsageCode'] = tmp[drug].map(usage_numeric)
            tmp['UsageGroup'] = tmp['UsageCode'].apply(usage_group)
            df_box_records.append(tmp[['Drug', 'UsageGroup', 'Sensation Seeking', 'Impulsivity']])

        df_box = pd.concat(df_box_records)
        df_box_melted = df_box.melt(id_vars=['Drug', 'UsageGroup'], var_name='Trait', value_name='Score')

        n_cols = min(len(selected_box_drugs_lower), 3)
        box_chart = alt.Chart(df_box_melted).mark_boxplot(extent='min-max', outliers=False).encode(
            x=alt.X('UsageGroup:N', sort=['Non/Rare user', 'Past user', 'Current user'], title='Usage Group'),
            y=alt.Y('Score:Q', title='Trait Score (z)'),
            color=alt.Color('UsageGroup:N',
                            scale=alt.Scale(domain=['Non/Rare user', 'Past user', 'Current user'],
                                            range=['#2ecc71', '#f39c12', '#e74c3c']),
                            legend=None),
            facet=alt.Facet('Drug:N', columns=n_cols, title='Sensation Seeking & Impulsivity by Usage Group')
        ).transform_filter(
            alt.FieldOneOfPredicate(field='Trait', oneOf=['Sensation Seeking', 'Impulsivity'])
        ).properties(width=200, height=150).resolve_scale(y='independent')

        st.altair_chart(box_chart, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# SECTION: CORRELATIONS
# ─────────────────────────────────────────────────────────────
elif section == "🔬 Correlations":
    st.header("Correlation: Traits × Drug Usage")

    base_drugs = ['cannabis', 'alcohol', 'nicotine', 'ecstasy', 'coke',
                  'lsd', 'ketamine', 'mushrooms', 'amphet', 'benzos', 'heroin', 'meth']
    focus_drugs = [d for d in base_drugs if d in df_as.columns]
    score_cols = [f'{d}_score' for d in focus_drugs if f'{d}_score' in df_as.columns]

    trait_score_df = df_as[trait_list + score_cols].dropna()
    corr_matrix = trait_score_df.corr().loc[trait_list, score_cols].reset_index().melt(id_vars='index')
    corr_matrix.columns = ['Trait', 'Drug', 'Correlation']
    corr_matrix['Drug'] = corr_matrix['Drug'].str.replace('_score', '', regex=False).str.capitalize()

    col_domain = st.slider("Color scale range (±)", 0.1, 0.8, 0.4, 0.05)

    corr_chart = alt.Chart(corr_matrix).mark_rect().encode(
        x=alt.X('Drug:N', title='Drug', sort=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y('Trait:N', title='Personality Trait', sort=None),
        color=alt.Color('Correlation:Q',
                        scale=alt.Scale(scheme='redblue', domain=[-col_domain, col_domain], reverse=True),
                        title='Pearson r'),
        tooltip=['Trait', 'Drug', alt.Tooltip('Correlation:Q', format='.3f')]
    ).properties(title='Correlation: Personality Traits × Drug Usage Frequency', height=300)

    text_layer = corr_chart.mark_text(fontSize=10).encode(
        text=alt.Text('Correlation:Q', format='.2f'),
        color=alt.condition(
            alt.datum.Correlation > 0.15,
            alt.value('white'), alt.value('black')
        )
    )
    st.altair_chart((corr_chart + text_layer), use_container_width=True)

    st.divider()
    st.subheader("Top correlations")
    top_n = st.slider("Show top N correlations:", 5, 30, 10)
    top_corr = corr_matrix.reindex(corr_matrix['Correlation'].abs().sort_values(ascending=False).index)
    
    bar = alt.Chart(top_corr.head(top_n)).mark_bar().encode(
        x=alt.X('Correlation:Q', title='Pearson r'),
        y=alt.Y('Trait:N', sort='-x'),
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=[-0.4, 0.4], reverse=True), legend=None),
        column=alt.Column('Drug:N', title=''),
        tooltip=['Trait', 'Drug', alt.Tooltip('Correlation:Q', format='.3f')]
    ).properties(width=80, height=180)
    st.altair_chart(bar)

# ─────────────────────────────────────────────────────────────
# SECTION: PREDICTION
# ─────────────────────────────────────────────────────────────
elif section == "🤖 Prediction":
    st.header("Predictive Modeling — Logistic Regression")
    st.markdown("For each drug, a logistic regression predicts *current user* (CL4-CL6) vs. not, using the 7 personality traits.")

    model_drugs = ['cannabis', 'ecstasy', 'lsd', 'coke', 'nicotine', 'heroin', 'amphet', 'benzos']
    model_drugs = [d for d in model_drugs if d in df_as.columns]

    col1, col2 = st.columns(2)
    with col1:
        reg_strength = st.slider("Regularization (C):", 0.01, 5.0, 1.0, 0.1, help="Higher C = less regularization")
    with col2:
        cv_folds = st.slider("Cross-validation folds:", 3, 10, 5)

    @st.cache_data(show_spinner="Fitting logistic regression models…")
    def fit_models(drugs, C, cv):
        X_model = df_as[trait_list].dropna()
        coef_records = []
        for drug in drugs:
            tmp = df_as.loc[X_model.index, drug]
            y_bin = tmp.map(lambda x: 1 if x in ['CL4', 'CL5', 'CL6'] else 0)
            valid = (~y_bin.isna()) & (~X_model.isna().any(axis=1))
            X_fit = X_model[valid]; y_fit = y_bin[valid]
            if y_fit.sum() < 10: continue
            lr = LogisticRegression(max_iter=1000, C=C)
            lr.fit(X_fit, y_fit)
            cv_score = cross_val_score(lr, X_fit, y_fit, cv=cv, scoring='roc_auc').mean()
            for feat, coef in zip(trait_list, lr.coef_[0]):
                coef_records.append({'Drug': drug.capitalize(), 'Trait': feat, 'Coefficient': coef,
                                     'AUC': round(cv_score, 3), 'Prevalence': f"{y_fit.mean()*100:.0f}%"})
        return pd.DataFrame(coef_records)

    df_coef = fit_models(tuple(model_drugs), reg_strength, cv_folds)

    if not df_coef.empty:
        # AUC summary
        auc_df = df_coef.groupby('Drug')['AUC'].first().reset_index().sort_values('AUC', ascending=False)
        auc_bar = alt.Chart(auc_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X('Drug:N', sort='-y'),
            y=alt.Y('AUC:Q', scale=alt.Scale(domain=[0.5, 1.0]), title='CV AUC (5-fold)'),
            color=alt.Color('AUC:Q', scale=alt.Scale(scheme='blues', domain=[0.5, 0.9]), legend=None),
            tooltip=['Drug', alt.Tooltip('AUC:Q', format='.3f')]
        ).properties(title='Model Performance (AUC per Drug)', height=220)
        st.altair_chart(auc_bar, use_container_width=True)

        st.divider()
        coef_chart = alt.Chart(df_coef).mark_rect().encode(
            x=alt.X('Drug:N', title='Drug', axis=alt.Axis(labelAngle=-30)),
            y=alt.Y('Trait:N', title='Personality Trait', sort=None),
            color=alt.Color('Coefficient:Q',
                            scale=alt.Scale(scheme='redblue', domain=[-1.5, 1.5], reverse=True),
                            title='Log-Odds'),
            tooltip=['Drug', 'Trait', alt.Tooltip('Coefficient:Q', format='.3f'),
                     alt.Tooltip('AUC:Q', title='CV AUC'), 'Prevalence']
        ).properties(title='Logistic Regression Coefficients: Personality → Current Drug Use', height=280)

        text_coef = coef_chart.mark_text(fontSize=10).encode(
            text=alt.Text('Coefficient:Q', format='.2f'),
            color=alt.condition(alt.datum.Coefficient > 0.4, alt.value('white'), alt.value('black'))
        )
        st.altair_chart((coef_chart + text_coef), use_container_width=True)

        with st.expander("Raw coefficients table"):
            st.dataframe(df_coef.pivot(index='Trait', columns='Drug', values='Coefficient').round(3), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# SECTION: USER PROFILES
# ─────────────────────────────────────────────────────────────
elif section == "👥 User Profiles":
    st.header("Current Users vs. Non/Rare Users")
    st.markdown("Compare mean trait scores between current users (last month or more) and non/rare users.")

    model_drugs = ['cannabis', 'ecstasy', 'lsd', 'coke', 'nicotine', 'heroin', 'amphet', 'benzos']
    model_drugs = [d for d in model_drugs if d in df_as.columns]

    @st.cache_data
    def build_compare_data():
        compare_records = []
        for drug in model_drugs:
            df_as['_is_user'] = df_as[drug].apply(
                lambda x: 'Current User (≤1 month)' if x in ['CL4', 'CL5', 'CL6'] else 'Non/Rare User'
            )
            for grp, grp_df in df_as.groupby('_is_user'):
                for trait in trait_list:
                    compare_records.append({
                        'Drug': drug, 'Group': grp, 'Trait': trait,
                        'Mean': grp_df[trait].mean(), 'SE': grp_df[trait].sem()
                    })
        return pd.DataFrame(compare_records)

    df_compare = build_compare_data()

    selected_drug_compare = st.selectbox("Select a drug:", [d.capitalize() for d in model_drugs], index=0)
    drug_lower = selected_drug_compare.lower()

    filtered = df_compare[df_compare['Drug'] == drug_lower]

    bars = alt.Chart(filtered).mark_bar(size=22).encode(
        x=alt.X('Mean:Q', title='Mean Trait Score (z)', scale=alt.Scale(domain=[-0.6, 0.6])),
        y=alt.Y('Trait:N', sort='-x', title=''),
        color=alt.Color('Group:N',
                        scale=alt.Scale(domain=['Current User (≤1 month)', 'Non/Rare User'],
                                        range=['#e74c3c', '#3498db']),
                        legend=alt.Legend(orient='bottom')),
        xOffset='Group:N',
        tooltip=['Group:N', 'Trait:N', alt.Tooltip('Mean:Q', format='.3f')]
    ).properties(title=f'Mean Trait Scores: {selected_drug_compare} — Current vs. Non/Rare Users', height=300)

    error = alt.Chart(filtered).mark_errorbar(extent='ci').encode(
        x=alt.X('Mean:Q'),
        y=alt.Y('Trait:N'),
        xOffset='Group:N',
        color=alt.Color('Group:N', scale=alt.Scale(domain=['Current User (≤1 month)', 'Non/Rare User'],
                                                    range=['#e74c3c', '#3498db']), legend=None)
    )
    st.altair_chart((bars + error), use_container_width=True)

    st.divider()
    st.subheader("All drugs side by side")
    
    all_drugs_chart = alt.Chart(df_compare).mark_bar(size=10).encode(
        x=alt.X('Mean:Q', title='Mean Trait Score (z)'),
        y=alt.Y('Drug:N', title=''),
        color=alt.Color('Group:N', scale=alt.Scale(domain=['Current User (≤1 month)', 'Non/Rare User'],
                                                    range=['#e74c3c', '#3498db'])),
        xOffset='Group:N',
        facet=alt.Facet('Trait:N', columns=4, title=''),
        tooltip=['Drug', 'Group', alt.Tooltip('Mean:Q', format='.3f')]
    ).properties(width=140, height=160).resolve_scale(x='shared')
    st.altair_chart(all_drugs_chart, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# SECTION: RADAR
# ─────────────────────────────────────────────────────────────
elif section == "🕸️ Radar":
    st.header("Psychological Profile Radar")
    st.markdown("Compare personality fingerprints across drugs and usage levels.")

    drugs_radar = ['cannabis', 'ecstasy', 'lsd', 'coke', 'heroin', 'nicotine', 'amphet', 'benzos']
    drugs_radar = [d for d in drugs_radar if d in df_as.columns]

    usage_level_labels = {
        'CL0': 'Never', 'CL1': '>10 years ago', 'CL2': 'Last decade',
        'CL3': 'Last year', 'CL4': 'Last month', 'CL5': 'Last week', 'CL6': 'Last day'
    }

    @st.cache_data
    def build_radar_data():
        n_traits = len(trait_list)
        angles = [i * 2 * math.pi / n_traits for i in range(n_traits)]
        records = []
        for drug in drugs_radar:
            tmp = df_as[trait_list + [drug]].copy()
            tmp.columns = trait_list + ['CL']
            grouped = tmp.groupby('CL')[trait_list].mean().reset_index()
            grouped['N'] = tmp.groupby('CL').size().values
            for _, row in grouped.iterrows():
                cl = row['CL']
                if cl not in usage_level_labels or row['N'] < 8: continue
                for i, trait in enumerate(trait_list):
                    records.append({
                        'Drug': drug.capitalize(), 'UsageLevel': usage_level_labels[cl],
                        'CL': cl, 'Trait': trait, 'TraitIndex': i,
                        'MeanScore': round(row[trait], 3), 'N': int(row['N'])
                    })
        df_radar = pd.DataFrame(records)
        df_radar['NormScore'] = df_radar.groupby('Trait')['MeanScore'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
        df_radar['angle'] = df_radar['TraitIndex'].apply(lambda i: angles[i] - math.pi / 2)
        df_radar['x'] = df_radar['NormScore'] * np.cos(df_radar['angle'])
        df_radar['y'] = df_radar['NormScore'] * np.sin(df_radar['angle'])
        closure = df_radar[df_radar['TraitIndex'] == 0].copy()
        closure['TraitIndex'] = n_traits
        df_plot = pd.concat([df_radar, closure], ignore_index=True).drop(columns=['angle'])
        return df_plot, angles

    df_plot, angles = build_radar_data()
    n_traits = len(trait_list)

    col1, col2 = st.columns(2)
    with col1:
        drug_options_list = sorted(df_plot['Drug'].unique().tolist())
        selected_drugs_radar = st.multiselect("💊 Select drug(s):", drug_options_list, default=['Cannabis'])
    with col2:
        usage_options_list = [u for u in ['Never', '>10 years ago', 'Last decade', 'Last year',
                                          'Last month', 'Last week', 'Last day']
                              if u in df_plot['UsageLevel'].unique()]
        selected_usages_radar = st.multiselect("📅 Select usage level(s):", usage_options_list, default=['Never', 'Last day'])

    if selected_drugs_radar and selected_usages_radar:
        stroke_map = {
            'Never': [1, 0], '>10 years ago': [4, 2], 'Last decade': [6, 2],
            'Last year': [8, 3], 'Last month': [2, 2], 'Last week': [6, 2, 2, 2], 'Last day': [1, 0],
        }
        df_filtered = df_plot[
            df_plot['Drug'].isin(selected_drugs_radar) &
            df_plot['UsageLevel'].isin(selected_usages_radar)
        ].copy()

        if df_filtered.empty:
            st.warning("No data for this selection.")
        else:
            stroke_domain = [u for u in usage_options_list if u in selected_usages_radar]
            stroke_range = [stroke_map[u] for u in stroke_domain]
            data = alt.Data(values=df_filtered.to_dict(orient='records'))

            # Grid
            circles_data = pd.DataFrame([
                {'r': r, 'xc': r * math.cos(t - math.pi/2), 'yc': r * math.sin(t - math.pi/2)}
                for r in [0.25, 0.5, 0.75, 1.0]
                for t in [i * 2 * math.pi / 60 for i in range(61)]
            ])
            grid_circles = alt.Chart(circles_data).mark_line(
                color='#cccccc', strokeWidth=0.8, opacity=0.6
            ).encode(
                x=alt.X('xc:Q', axis=None, scale=alt.Scale(domain=[-1.5, 1.5])),
                y=alt.Y('yc:Q', axis=None, scale=alt.Scale(domain=[-1.5, 1.5])),
                detail='r:O'
            )
            axis_lines_data = []
            for i, trait in enumerate(trait_list):
                angle = angles[i] - math.pi / 2
                axis_lines_data += [
                    {'seg': i, 'x': 0.0, 'y': 0.0, 'Trait': trait},
                    {'seg': i, 'x': math.cos(angle) * 1.05, 'y': math.sin(angle) * 1.05, 'Trait': trait},
                ]
            axis_lines = alt.Chart(pd.DataFrame(axis_lines_data)).mark_line(
                color='#bbbbbb', strokeWidth=1
            ).encode(
                x=alt.X('x:Q', axis=None, scale=alt.Scale(domain=[-1.5, 1.5])),
                y=alt.Y('y:Q', axis=None, scale=alt.Scale(domain=[-1.5, 1.5])),
                detail='seg:O'
            )
            label_data = pd.DataFrame([{
                'Trait': trait_list[i],
                'x': 1.28 * math.cos(angles[i] - math.pi / 2),
                'y': 1.28 * math.sin(angles[i] - math.pi / 2),
            } for i in range(n_traits)])
            labels = alt.Chart(label_data).mark_text(
                fontSize=12, fontWeight='bold', color='#444'
            ).encode(
                x=alt.X('x:Q', axis=None, scale=alt.Scale(domain=[-1.5, 1.5])),
                y=alt.Y('y:Q', axis=None, scale=alt.Scale(domain=[-1.5, 1.5])),
                text='Trait:N'
            )
            lines = alt.Chart(data).mark_line(strokeWidth=2.8, opacity=0.85).encode(
                x=alt.X('x:Q', axis=None, scale=alt.Scale(domain=[-1.5, 1.5])),
                y=alt.Y('y:Q', axis=None, scale=alt.Scale(domain=[-1.5, 1.5])),
                color=alt.Color('Drug:N', scale=alt.Scale(scheme='tableau10'), title='Drug'),
                strokeDash=alt.StrokeDash('UsageLevel:N',
                    scale=alt.Scale(domain=stroke_domain, range=stroke_range), title='Usage Level'),
                order=alt.Order('TraitIndex:O'),
                detail=alt.Detail(['Drug:N', 'UsageLevel:N']),
                tooltip=['Drug:N', 'UsageLevel:N', 'Trait:N',
                         alt.Tooltip('MeanScore:Q', format='.3f'),
                         alt.Tooltip('N:Q', title='n respondents')]
            )
            pts = alt.Chart(data).mark_point(size=65, filled=True, opacity=0.9).encode(
                x=alt.X('x:Q', axis=None, scale=alt.Scale(domain=[-1.5, 1.5])),
                y=alt.Y('y:Q', axis=None, scale=alt.Scale(domain=[-1.5, 1.5])),
                color=alt.Color('Drug:N', scale=alt.Scale(scheme='tableau10')),
                tooltip=['Drug:N', 'UsageLevel:N', 'Trait:N',
                         alt.Tooltip('MeanScore:Q', format='.3f'),
                         alt.Tooltip('N:Q', title='n respondents')]
            )
            n_combos = df_filtered[['Drug', 'UsageLevel']].drop_duplicates().shape[0]
            radar_chart = (grid_circles + axis_lines + labels + lines + pts).properties(
                title=alt.TitleParams(
                    '🧠 Psychological Profile — Multi Drug & Usage',
                    subtitle=f'{n_combos} profile(s) shown',
                    fontSize=15, subtitleFontSize=11
                ),
                width=520, height=520
            )
            col_r, _ = st.columns([2, 1])
            with col_r:
                st.altair_chart(radar_chart, use_container_width=False)
    else:
        st.info("Select at least one drug and one usage level.")
