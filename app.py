"""
Real Estate Buyer Segmentation — Streamlit Dashboard
Run: streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate ML · Buyer Segmentation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0f1e; }
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1e2d45; }
    h1, h2, h3 { color: #f0f4ff !important; }
    .metric-box {
        background: #111827;
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { font-size: 11px; color: #7c8fa8; text-transform: uppercase; letter-spacing: 1.5px; }
    .metric-value { font-size: 28px; font-weight: 700; margin: 4px 0; }
    .seg-pill {
        display: inline-block;
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    .stPlotlyChart, .stImage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

PALETTE    = ['#4361ee', '#f72585', '#4cc9f0', '#06d6a0']
DARK       = '#0f172a'
CARD       = '#1e293b'
TEXT       = '#f1f5f9'
MUTED      = '#94a3b8'
SEG_ICONS  = {
    'Global Investors' : '🌍',
    'First-Time Buyers': '🏠',
    'Corporate Buyers' : '🏢',
    'Luxury Investors' : '💎',
}
SEG_COLORS = {
    'Global Investors' : '#4361ee',
    'First-Time Buyers': '#f72585',
    'Corporate Buyers' : '#4cc9f0',
    'Luxury Investors' : '#06d6a0',
}


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def style_ax(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')


def parse_dob(d):
    for fmt in ('%d-%m-%Y', '%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y'):
        try:
            return datetime.strptime(str(d).strip(), fmt)
        except ValueError:
            pass
    return None


# ─────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_cluster():
    clients = pd.read_csv(os.path.join('data', 'clients.csv'))
    props   = pd.read_csv(os.path.join('data', 'properties.csv'))

    props['sale_price'] = (
        props['sale_price'].astype(str)
        .str.replace(r'[$,]', '', regex=True)
        .astype(float)
    )

    props_clean  = props.dropna(subset=['client_ref'])
    client_props = (
        props_clean.groupby('client_ref')
        .agg(
            num_properties  =('listing_id',      'count'),
            avg_sale_price  =('sale_price',      'mean'),
            total_investment=('sale_price',      'sum'),
            avg_sqft        =('floor_area_sqft', 'mean'),
        )
        .reset_index()
        .rename(columns={'client_ref': 'client_id'})
    )

    df = clients.merge(client_props, on='client_id', how='left')
    df['num_properties']   = df['num_properties'].fillna(0)
    df['avg_sale_price']   = df['avg_sale_price'].fillna(df['avg_sale_price'].median())
    df['total_investment'] = df['total_investment'].fillna(0)
    df['avg_sqft']         = df['avg_sqft'].fillna(df['avg_sqft'].median())

    df['dob_parsed'] = df['date_of_birth'].apply(parse_dob)
    df['age'] = (
        df['dob_parsed']
        .apply(lambda x: (datetime(2024, 1, 1) - x).days // 365 if x else None)
        .fillna(df['dob_parsed']
                .apply(lambda x: (datetime(2024, 1, 1) - x).days // 365 if x else None)
                .median())
        .clip(18, 90)
    )

    # Encoding
    le = LabelEncoder()
    df['purpose_enc']         = (df['acquisition_purpose'] == 'Investment').astype(int)
    df['loan_enc']            = (df['loan_applied'] == 'Yes').astype(int)
    df['corp_enc']            = (df['client_type'] == 'Company').astype(int)
    df['gender_enc']          = le.fit_transform(df['gender'])
    df['high_income_country'] = df['country'].isin(
        {'USA','UK','Germany','France','Australia','Canada','Belgium','Denmark'}
    ).astype(int)

    region_dummies  = pd.get_dummies(df['region'],           prefix='reg', drop_first=True)
    channel_dummies = pd.get_dummies(df['referral_channel'], prefix='ch',  drop_first=True)

    feature_df = pd.concat([
        df[['age','satisfaction_score','num_properties','avg_sale_price',
            'total_investment','avg_sqft',
            'purpose_enc','loan_enc','corp_enc',
            'gender_enc','high_income_country']],
        region_dummies, channel_dummies,
    ], axis=1)

    scaler = StandardScaler()
    X      = scaler.fit_transform(feature_df)

    # Elbow + Silhouette
    inertias, sil_scores, K_RANGE = [], [], range(2, 9)
    for k in K_RANGE:
        km     = KMeans(n_clusters=k, random_state=42, n_init=15)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))

    # Final model
    kmeans       = KMeans(n_clusters=4, random_state=42, n_init=15)
    df['cluster'] = kmeans.fit_predict(X)

    cluster_stats = (
        df.groupby('cluster')
        .agg(
            count           =('client_id',           'count'),
            pct_investment  =('acquisition_purpose',  lambda x: (x=='Investment').mean()*100),
            pct_loan        =('loan_applied',         lambda x: (x=='Yes').mean()*100),
            pct_company     =('client_type',          lambda x: (x=='Company').mean()*100),
            avg_age         =('age',                  'mean'),
            avg_satisfaction=('satisfaction_score',   'mean'),
            avg_price       =('avg_sale_price',       'mean'),
            avg_properties  =('num_properties',       'mean'),
            avg_total_inv   =('total_investment',     'mean'),
            top_country     =('country',              lambda x: x.value_counts().index[0]),
        )
        .round(2)
    )

    sorted_by_price = cluster_stats['avg_price'].sort_values(ascending=False).index.tolist()
    label_map = {
        sorted_by_price[0]: 'Global Investors',
        sorted_by_price[1]: 'Luxury Investors',
        sorted_by_price[2]: 'First-Time Buyers',
        sorted_by_price[3]: 'Corporate Buyers',
    }
    label_map[cluster_stats['pct_company'].idxmax()] = 'Corporate Buyers'
    label_map[cluster_stats['pct_loan'].idxmax()]    = 'First-Time Buyers'

    used, final_map = set(), {}
    all_labels = ['Global Investors','First-Time Buyers','Corporate Buyers','Luxury Investors']
    for idx in sorted_by_price:
        lbl = label_map[idx]
        if lbl in used:
            lbl = next(l for l in all_labels if l not in used)
        used.add(lbl)
        final_map[idx] = lbl

    cluster_stats['label'] = cluster_stats.index.map(final_map)
    df['cluster_label']    = df['cluster'].map(final_map)

    # PCA
    pca        = PCA(n_components=2, random_state=42)
    X_pca      = pca.fit_transform(X)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]

    # Hierarchical
    sample_idx = np.random.RandomState(42).choice(len(X), 300, replace=False)
    Z          = linkage(X[sample_idx], method='ward')

    return df, cluster_stats, X, Z, list(K_RANGE), inertias, sil_scores


# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
with st.spinner("Running ML pipeline..."):
    df, cluster_stats, X, Z, K_RANGE, inertias, sil_scores = load_and_cluster()


# ─────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/real-estate.png", width=60)
st.sidebar.title("🏠 Buyer Segmentation")
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

countries       = ['All'] + sorted(df['country'].unique().tolist())
sel_country     = st.sidebar.selectbox("Country", countries)

regions         = ['All'] + sorted(df['region'].unique().tolist())
sel_region      = st.sidebar.selectbox("Region", regions)

purposes        = ['All'] + sorted(df['acquisition_purpose'].unique().tolist())
sel_purpose     = st.sidebar.selectbox("Acquisition Purpose", purposes)

client_types    = ['All'] + sorted(df['client_type'].unique().tolist())
sel_client_type = st.sidebar.selectbox("Client Type", client_types)

segments        = ['All'] + sorted(df['cluster_label'].unique().tolist())
sel_segment     = st.sidebar.selectbox("Buyer Segment", segments)

# Apply filters
filtered = df.copy()
if sel_country     != 'All': filtered = filtered[filtered['country']              == sel_country]
if sel_region      != 'All': filtered = filtered[filtered['region']               == sel_region]
if sel_purpose     != 'All': filtered = filtered[filtered['acquisition_purpose']  == sel_purpose]
if sel_client_type != 'All': filtered = filtered[filtered['client_type']          == sel_client_type]
if sel_segment     != 'All': filtered = filtered[filtered['cluster_label']        == sel_segment]

st.sidebar.markdown("---")
st.sidebar.metric("Filtered Clients", len(filtered))
st.sidebar.markdown(f"*Total dataset: {len(df):,} clients*")


# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("# 🏢 Real Estate Buyer Segmentation")
st.markdown("**Machine Learning–Based Market Intelligence** · K-Means + Hierarchical Clustering")
st.markdown("---")


# ─────────────────────────────────────────────────────────
# TAB NAVIGATION
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔬 Cluster Analysis",
    "🌍 Geographic",
    "💡 Behavior",
    "🌲 Hierarchical",
])


# ══════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════
with tab1:
    st.subheader("Key Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.metric("Total Clients",     f"{len(df):,}")
    with c2: st.metric("Buyer Segments",    4)
    with c3: st.metric("Silhouette Score",  f"{sil_scores[2]:.4f}")
    with c4: st.metric("Features Used",     "68")
    with c5: st.metric("Properties",        "10,000")
    with c6: st.metric("Filtered Clients",  f"{len(filtered):,}")

    st.markdown("---")
    st.subheader("Buyer Segment Cards")

    cols = st.columns(4)
    for i, (_, row) in enumerate(cluster_stats.iterrows()):
        lbl   = row['label']
        icon  = SEG_ICONS.get(lbl, '📊')
        color = SEG_COLORS.get(lbl, '#888')
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:#111827;border:1px solid {color}40;
                        border-top:3px solid {color};border-radius:12px;padding:16px;margin-bottom:12px;">
                <div style="font-size:28px">{icon}</div>
                <div style="color:{color};font-weight:700;font-size:15px;margin:6px 0">{lbl}</div>
                <div style="color:#7c8fa8;font-size:12px">{int(row['count'])} clients</div>
                <hr style="border-color:#1e2d45;margin:10px 0">
                <div style="font-size:12px;color:#f0f4ff">
                  <b>Avg Price:</b> ${int(row['avg_price']):,}<br>
                  <b>Investment:</b> {row['pct_investment']:.1f}%<br>
                  <b>Loan:</b> {row['pct_loan']:.1f}%<br>
                  <b>Avg Age:</b> {row['avg_age']:.0f} yrs<br>
                  <b>Satisfaction:</b> {row['avg_satisfaction']:.2f}/5<br>
                  <b>Top Country:</b> {row['top_country']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Segment Comparison Table")
    display_df = cluster_stats[['label','count','pct_investment','pct_loan',
                                 'pct_company','avg_age','avg_satisfaction',
                                 'avg_price','avg_properties','top_country']].copy()
    display_df.columns = ['Segment','Clients','% Investment','% Loan','% Corporate',
                           'Avg Age','Satisfaction','Avg Price $','Avg Props','Top Country']
    st.dataframe(display_df.set_index('Segment'), use_container_width=True)


# ══════════════════════════════════════════
# TAB 2 — CLUSTER ANALYSIS
# ══════════════════════════════════════════
with tab2:
    st.subheader("Elbow Method & Silhouette Score")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=DARK)
    for ax in axes:
        style_ax(ax)

    axes[0].plot(K_RANGE, inertias, 'o-', color=PALETTE[0], lw=2, ms=7, markerfacecolor='white')
    axes[0].axvline(4, color=PALETTE[1], ls='--', lw=1.5, label='k=4')
    axes[0].set_title('Elbow Method'); axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia')
    axes[0].legend(labelcolor=MUTED, facecolor=CARD, edgecolor='#334155')

    axes[1].plot(K_RANGE, sil_scores, 's-', color=PALETTE[2], lw=2, ms=7, markerfacecolor='white')
    axes[1].axvline(4, color=PALETTE[1], ls='--', lw=1.5, label='k=4')
    axes[1].set_title('Silhouette Score'); axes[1].set_xlabel('k'); axes[1].set_ylabel('Score')
    axes[1].legend(labelcolor=MUTED, facecolor=CARD, edgecolor='#334155')
    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Segment Distribution")
        fig, ax = plt.subplots(figsize=(6, 5.5), facecolor=DARK)
        ax.set_facecolor(DARK)
        sizes      = cluster_stats['count'].values
        lpie       = [f"{cluster_stats.loc[i,'label']}\n({v})"
                      for i, v in zip(cluster_stats.index, sizes)]
        cpie       = [SEG_COLORS.get(cluster_stats.loc[i,'label'],'#888')
                      for i in cluster_stats.index]
        wedges, _, ats = ax.pie(
            sizes, autopct='%1.1f%%', colors=cpie,
            pctdistance=0.75, startangle=90,
            wedgeprops=dict(width=0.52, edgecolor=DARK, linewidth=2),
        )
        for at in ats:
            at.set_color('white'); at.set_fontsize(10); at.set_fontweight('bold')
        ax.legend(wedges, lpie, loc='lower center', bbox_to_anchor=(0.5, -0.1),
                  ncol=2, fontsize=8, labelcolor=TEXT, facecolor=CARD, edgecolor='#334155')
        ax.set_title('Buyer Segment Distribution', color=TEXT, fontsize=13)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.subheader("PCA 2D Projection")
        fig, ax = plt.subplots(figsize=(6, 5.5), facecolor=DARK)
        style_ax(ax)
        for lbl, grp in filtered.groupby('cluster_label'):
            ax.scatter(grp['pca1'], grp['pca2'],
                       c=SEG_COLORS.get(lbl,'#888'), s=14, alpha=0.55, label=lbl)
        ax.set_title('PCA Projection', color=TEXT, fontsize=13)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        ax.legend(labelcolor=TEXT, facecolor=CARD, edgecolor='#334155', fontsize=8, markerscale=2)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.subheader("Cluster Profile Heatmap")
    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=DARK)
    hd = cluster_stats[['pct_investment','pct_loan','avg_age',
                         'avg_satisfaction','avg_price','avg_properties']].copy()
    hd.columns = ['% Investment','% Loan','Avg Age','Satisfaction','Avg Price $','Avg Props']
    hd.index   = cluster_stats['label'].values
    norm_hd    = (hd - hd.min()) / (hd.max() - hd.min() + 1e-9)
    sns.heatmap(norm_hd, annot=hd.round(1), fmt='g', cmap='Blues', ax=ax,
                linecolor=DARK, linewidths=1, cbar_kws={'shrink': 0.8})
    ax.set_title('Cluster Profile (normalised)', color=TEXT, fontsize=13, pad=10)
    ax.tick_params(colors=MUTED)
    ax.figure.axes[-1].tick_params(colors=MUTED)
    plt.xticks(rotation=20, ha='right', color=MUTED)
    plt.yticks(rotation=0, color=MUTED)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════
# TAB 3 — GEOGRAPHIC
# ══════════════════════════════════════════
with tab3:
    st.subheader("Buyer Segments by Country (Top 10)")
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK)
    style_ax(ax)
    geo     = filtered.groupby(['country','cluster_label']).size().unstack(fill_value=0)
    geo_top = geo.loc[geo.sum(axis=1).nlargest(10).index]
    colors_bar = [SEG_COLORS.get(c,'#888') for c in geo_top.columns]
    geo_top.plot(kind='bar', ax=ax, color=colors_bar, edgecolor=DARK, width=0.7)
    ax.set_title('Buyer Segments by Country', color=TEXT, fontsize=13)
    ax.set_xlabel('Country'); ax.set_ylabel('Clients')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', color=MUTED)
    ax.legend(labelcolor=TEXT, facecolor=CARD, edgecolor='#334155', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.subheader("Client Count by Country")
    country_counts = filtered['country'].value_counts().head(15).reset_index()
    country_counts.columns = ['Country','Clients']
    st.dataframe(country_counts, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════
# TAB 4 — BEHAVIOR
# ══════════════════════════════════════════
with tab4:
    st.subheader("Investment Behavior by Segment")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK)
    for ax in axes:
        style_ax(ax)

    loan_d = filtered.groupby(['cluster_label','loan_applied']).size().unstack(fill_value=0)
    if not loan_d.empty:
        loan_d.plot(kind='bar', ax=axes[0], color=['#ef4444','#22c55e'], edgecolor=DARK, width=0.6)
    axes[0].set_title('Loan Applied per Segment', color=TEXT)
    axes[0].set_xlabel(''); axes[0].set_ylabel('Clients')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20, ha='right', color=MUTED)
    axes[0].legend(['No Loan','Loan Applied'], labelcolor=TEXT, facecolor=CARD, edgecolor='#334155')

    purp_d = filtered.groupby(['cluster_label','acquisition_purpose']).size().unstack(fill_value=0)
    if not purp_d.empty:
        purp_d.plot(kind='bar', ax=axes[1], color=['#f59e0b','#6366f1'], edgecolor=DARK, width=0.6)
    axes[1].set_title('Acquisition Purpose per Segment', color=TEXT)
    axes[1].set_xlabel(''); axes[1].set_ylabel('Clients')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=20, ha='right', color=MUTED)
    axes[1].legend(labelcolor=TEXT, facecolor=CARD, edgecolor='#334155')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.subheader("Referral Channel Distribution")
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK)
    style_ax(ax)
    ref = filtered['referral_channel'].value_counts()
    ref.plot(kind='barh', ax=ax, color=PALETTE[0], edgecolor=DARK)
    ax.set_title('Referral Channel', color=TEXT, fontsize=13)
    ax.set_xlabel('Clients'); ax.set_ylabel('')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.subheader("Age vs Avg Sale Price")
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK)
    style_ax(ax)
    for lbl, grp in filtered.groupby('cluster_label'):
        ax.scatter(grp['age'], grp['avg_sale_price']/1000,
                   c=SEG_COLORS.get(lbl,'#888'), s=16, alpha=0.5, label=lbl)
    ax.set_title('Age vs Avg Sale Price ($K)', color=TEXT, fontsize=13)
    ax.set_xlabel('Age'); ax.set_ylabel('Avg Price ($K)')
    ax.legend(labelcolor=TEXT, facecolor=CARD, edgecolor='#334155', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════
# TAB 5 — HIERARCHICAL
# ══════════════════════════════════════════
with tab5:
    st.subheader("Hierarchical Clustering Dendrogram")
    st.caption("Ward linkage on a 300-client random sample. Validates K-Means k=4 selection.")
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK)
    style_ax(ax)
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.6 * max(Z[:, 2]))
    ax.set_title('Agglomerative Hierarchical Clustering (Ward Linkage)', color=TEXT, fontsize=13)
    ax.set_ylabel('Ward Distance'); ax.set_xlabel('Client Index')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.subheader("Strategic Recommendations")
    recs = [
        ("🌍 Global Investors",  "#4361ee", "Highest avg price ($406K), multi-property buyers, older demographic",
         "Premium content, international campaigns, wealth management partnerships",
         "Portfolio analytics, multi-unit dashboards, cross-border investment tools"),
        ("🏠 First-Time Buyers", "#f72585", "100% investment-purpose encoded, moderate prices, loan-dependent",
         "Educational content, first-home incentives, mortgage partner referrals",
         "Affordability calculators, step-by-step guides, loan application integration"),
        ("🏢 Corporate Buyers",  "#4cc9f0", "Largest segment (814 clients), 0% investment purpose, bulk purchases",
         "B2B outreach, volume discounts, dedicated account management",
         "Bulk listing tools, corporate invoice system, API access for integration"),
        ("💎 Luxury Investors",  "#06d6a0", "Small niche (76), mid-price point, diverse geography (top: Mexico)",
         "Concierge service, exclusive property previews, luxury lifestyle marketing",
         "White-glove agent matching, private listings, VIP satisfaction programs"),
    ]
    for seg, color, insight, marketing, product in recs:
        with st.expander(seg):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**🔍 Key Insight**")
                st.info(insight)
            with c2:
                st.markdown(f"**📣 Marketing**")
                st.success(marketing)
            with c3:
                st.markdown(f"**🛠 Product**")
                st.warning(product)

    st.subheader("Download Results")
    csv = df[['client_id','client_type','country','region','acquisition_purpose',
              'loan_applied','age','satisfaction_score','avg_sale_price',
              'num_properties','cluster','cluster_label']].to_csv(index=False)
    st.download_button(
        label     = "⬇️ Download Clients with Cluster Labels (CSV)",
        data      = csv,
        file_name = "clients_with_clusters.csv",
        mime      = "text/csv",
    )
