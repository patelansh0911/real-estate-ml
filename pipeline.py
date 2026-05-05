"""
Real Estate Buyer Segmentation & Investment Profiling
ML Pipeline — Steps 1 to 6
Run: python pipeline.py
"""

import pandas as pd
import numpy as np
import warnings
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
DATA_DIR    = "data"
OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = ['#4361ee', '#f72585', '#4cc9f0', '#06d6a0']
DARK    = '#0f172a'
CARD    = '#1e293b'
TEXT    = '#f1f5f9'
MUTED   = '#94a3b8'

SEGMENT_LABELS = {
    'highest_price':  'Global Investors',
    'most_loan':      'First-Time Buyers',
    'most_corporate': 'Corporate Buyers',
    'remaining':      'Luxury Investors',
}


# ─────────────────────────────────────────────────────────
# STEP 1 – DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1 — Data Loading & Cleaning")
print("="*60)

clients = pd.read_csv(os.path.join(DATA_DIR, 'clients.csv'))
props   = pd.read_csv(os.path.join(DATA_DIR, 'properties.csv'))

print(f"  Clients loaded : {clients.shape}")
print(f"  Properties loaded : {props.shape}")

# Clean sale_price  ($300,385.62 → float)
props['sale_price'] = (
    props['sale_price']
    .astype(str)
    .str.replace(r'[$,]', '', regex=True)
    .astype(float)
)

# Aggregate property data per client
props_clean   = props.dropna(subset=['client_ref'])
client_props  = (
    props_clean
    .groupby('client_ref')
    .agg(
        num_properties  = ('listing_id',     'count'),
        avg_sale_price  = ('sale_price',     'mean'),
        total_investment= ('sale_price',     'sum'),
        avg_sqft        = ('floor_area_sqft','mean'),
    )
    .reset_index()
    .rename(columns={'client_ref': 'client_id'})
)

# Merge
df = clients.merge(client_props, on='client_id', how='left')
df['num_properties']   = df['num_properties'].fillna(0)
df['avg_sale_price']   = df['avg_sale_price'].fillna(df['avg_sale_price'].median())
df['total_investment'] = df['total_investment'].fillna(0)
df['avg_sqft']         = df['avg_sqft'].fillna(df['avg_sqft'].median())

# Parse age from date_of_birth
def parse_dob(d):
    for fmt in ('%d-%m-%Y', '%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y'):
        try:
            return datetime.strptime(str(d).strip(), fmt)
        except ValueError:
            pass
    return None

df['dob_parsed'] = df['date_of_birth'].apply(parse_dob)
df['age'] = df['dob_parsed'].apply(
    lambda x: (datetime(2024, 1, 1) - x).days // 365 if x else None
)
df['age'] = df['age'].fillna(df['age'].median()).clip(18, 90)

print(f"  Merged dataset  : {df.shape}")
print(f"  Null check      :\n{df[['age','avg_sale_price','num_properties']].isnull().sum()}")


# ─────────────────────────────────────────────────────────
# STEP 2 – FEATURE ENCODING
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2 — Feature Encoding")
print("="*60)

le = LabelEncoder()

df['purpose_enc']           = (df['acquisition_purpose'] == 'Investment').astype(int)
df['loan_enc']              = (df['loan_applied'] == 'Yes').astype(int)
df['corp_enc']              = (df['client_type'] == 'Company').astype(int)
df['gender_enc']            = le.fit_transform(df['gender'])
df['high_income_country']   = df['country'].isin(
    {'USA','UK','Germany','France','Australia','Canada','Belgium','Denmark'}
).astype(int)

region_dummies  = pd.get_dummies(df['region'],           prefix='reg',  drop_first=True)
channel_dummies = pd.get_dummies(df['referral_channel'], prefix='ch',   drop_first=True)

feature_df = pd.concat([
    df[['age','satisfaction_score','num_properties','avg_sale_price',
        'total_investment','avg_sqft',
        'purpose_enc','loan_enc','corp_enc',
        'gender_enc','high_income_country']],
    region_dummies,
    channel_dummies,
], axis=1)

print(f"  Total features  : {feature_df.shape[1]}")


# ─────────────────────────────────────────────────────────
# STEP 3 – FEATURE SCALING
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3 — Feature Scaling (StandardScaler)")
print("="*60)

scaler = StandardScaler()
X      = scaler.fit_transform(feature_df)

print(f"  Scaled matrix   : {X.shape}")


# ─────────────────────────────────────────────────────────
# STEP 4 & 5 – OPTIMAL CLUSTER SELECTION
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4/5 — Optimal k via Elbow + Silhouette")
print("="*60)

inertias   = []
sil_scores = []
K_RANGE    = range(2, 9)

for k in K_RANGE:
    km     = KMeans(n_clusters=k, random_state=42, n_init=15)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))
    print(f"  k={k}  inertia={km.inertia_:,.0f}  silhouette={sil_scores[-1]:.4f}")

OPTIMAL_K = 4   # Confirmed by PRD + silhouette analysis
print(f"\n  → Using k={OPTIMAL_K}  (silhouette={sil_scores[OPTIMAL_K-2]:.4f})")


# ─────────────────────────────────────────────────────────
# STEP 4 – FINAL K-MEANS MODEL
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4 — Final K-Means Clustering  (k=4)")
print("="*60)

kmeans     = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=15)
df['cluster'] = kmeans.fit_predict(X)

print(f"  Cluster distribution:\n{df['cluster'].value_counts().sort_index()}")


# ─────────────────────────────────────────────────────────
# STEP 6 – CLUSTER INTERPRETATION & LABELLING
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6 — Cluster Interpretation")
print("="*60)

cluster_stats = (
    df.groupby('cluster')
    .agg(
        count           = ('client_id',           'count'),
        pct_investment  = ('acquisition_purpose',  lambda x: (x == 'Investment').mean() * 100),
        pct_loan        = ('loan_applied',         lambda x: (x == 'Yes').mean() * 100),
        pct_company     = ('client_type',          lambda x: (x == 'Company').mean() * 100),
        avg_age         = ('age',                  'mean'),
        avg_satisfaction= ('satisfaction_score',   'mean'),
        avg_price       = ('avg_sale_price',       'mean'),
        avg_properties  = ('num_properties',       'mean'),
        avg_total_inv   = ('total_investment',     'mean'),
        top_country     = ('country',              lambda x: x.value_counts().index[0]),
    )
    .round(2)
)

# Assign labels based on dominant feature
sorted_by_price = cluster_stats['avg_price'].sort_values(ascending=False).index.tolist()
label_map = {
    sorted_by_price[0]: 'Global Investors',
    sorted_by_price[1]: 'Luxury Investors',
    sorted_by_price[2]: 'First-Time Buyers',
    sorted_by_price[3]: 'Corporate Buyers',
}
# Override with clearer signals
label_map[cluster_stats['pct_company'].idxmax()] = 'Corporate Buyers'
label_map[cluster_stats['pct_loan'].idxmax()]    = 'First-Time Buyers'

# Deduplicate labels
used, final_map = set(), {}
all_labels = ['Global Investors', 'First-Time Buyers', 'Corporate Buyers', 'Luxury Investors']
for idx in sorted_by_price:
    lbl = label_map[idx]
    if lbl in used:
        lbl = next(l for l in all_labels if l not in used)
    used.add(lbl)
    final_map[idx] = lbl

cluster_stats['label']  = cluster_stats.index.map(final_map)
df['cluster_label']     = df['cluster'].map(final_map)

print(cluster_stats[['count','label','pct_investment','pct_loan','avg_age','avg_price']].to_string())

# Hierarchical clustering (300-sample dendrogram)
sample_idx = np.random.RandomState(42).choice(len(X), 300, replace=False)
Z          = linkage(X[sample_idx], method='ward')

# PCA for 2D visualisation
pca         = PCA(n_components=2, random_state=42)
X_pca       = pca.fit_transform(X)
df['pca1']  = X_pca[:, 0]
df['pca2']  = X_pca[:, 1]

# ─────────────────────────────────────────────────────────
# HELPER – styled axes
# ─────────────────────────────────────────────────────────
def style_ax(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')


SEG_COLORS = dict(zip(
    ['Global Investors','First-Time Buyers','Corporate Buyers','Luxury Investors'],
    PALETTE
))


# ─────────────────────────────────────────────────────────
# CHART 1 – Elbow + Silhouette
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=DARK)
for ax in axes:
    style_ax(ax)

axes[0].plot(list(K_RANGE), inertias, 'o-', color=PALETTE[0], lw=2, ms=7, markerfacecolor='white')
axes[0].axvline(OPTIMAL_K, color=PALETTE[1], ls='--', lw=1.5, label=f'Chosen k={OPTIMAL_K}')
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('k')
axes[0].set_ylabel('Inertia')
axes[0].legend(labelcolor=MUTED, facecolor=CARD, edgecolor='#334155', fontsize=9)

axes[1].plot(list(K_RANGE), sil_scores, 's-', color=PALETTE[2], lw=2, ms=7, markerfacecolor='white')
axes[1].axvline(OPTIMAL_K, color=PALETTE[1], ls='--', lw=1.5, label=f'Chosen k={OPTIMAL_K}')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Score')
axes[1].legend(labelcolor=MUTED, facecolor=CARD, edgecolor='#334155', fontsize=9)

plt.tight_layout(pad=1.5)
plt.savefig(os.path.join(OUTPUT_DIR, 'chart_elbow_silhouette.png'), dpi=130, bbox_inches='tight', facecolor=DARK)
plt.close()
print("\n  ✅ Saved chart_elbow_silhouette.png")


# ─────────────────────────────────────────────────────────
# CHART 2 – Segment Donut
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6.5), facecolor=DARK)
ax.set_facecolor(DARK)

sizes       = cluster_stats['count'].values
labels_pie  = [f"{cluster_stats.loc[i,'label']}\n({v} clients)"
               for i, v in zip(cluster_stats.index, sizes)]
colors_pie  = [SEG_COLORS.get(cluster_stats.loc[i,'label'], '#888')
               for i in cluster_stats.index]

wedges, _, autotexts = ax.pie(
    sizes, autopct='%1.1f%%', colors=colors_pie,
    pctdistance=0.75, startangle=90,
    wedgeprops=dict(width=0.52, edgecolor=DARK, linewidth=2),
)
for at in autotexts:
    at.set_color('white')
    at.set_fontsize(11)
    at.set_fontweight('bold')

ax.legend(wedges, labels_pie,
          loc='lower center', bbox_to_anchor=(0.5, -0.08),
          ncol=2, fontsize=9, labelcolor=TEXT,
          facecolor=CARD, edgecolor='#334155')
ax.set_title('Buyer Segment Distribution', color=TEXT, fontsize=14, pad=16)

plt.savefig(os.path.join(OUTPUT_DIR, 'chart_donut.png'), dpi=130, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  ✅ Saved chart_donut.png")


# ─────────────────────────────────────────────────────────
# CHART 3 – PCA Scatter
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6), facecolor=DARK)
style_ax(ax)

for lbl, grp in df.groupby('cluster_label'):
    ax.scatter(grp['pca1'], grp['pca2'],
               c=SEG_COLORS.get(lbl, '#888'), s=14, alpha=0.5, label=lbl)

ax.set_title('PCA 2D Projection of Buyer Segments', color=TEXT, fontsize=13)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.legend(labelcolor=TEXT, facecolor=CARD, edgecolor='#334155', fontsize=9, markerscale=2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart_pca_scatter.png'), dpi=130, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  ✅ Saved chart_pca_scatter.png")


# ─────────────────────────────────────────────────────────
# CHART 4 – Cluster Heatmap
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=DARK)

hd = cluster_stats[['pct_investment','pct_loan','avg_age',
                     'avg_satisfaction','avg_price','avg_properties']].copy()
hd.columns = ['% Investment','% Loan','Avg Age','Satisfaction','Avg Price $','Avg Props']
hd.index   = cluster_stats['label'].values
norm_hd    = (hd - hd.min()) / (hd.max() - hd.min() + 1e-9)

sns.heatmap(norm_hd, annot=hd.round(1), fmt='g', cmap='Blues', ax=ax,
            linecolor=DARK, linewidths=1, cbar_kws={'shrink': 0.8, 'aspect': 20})
ax.set_title('Cluster Profile Heatmap (normalised)', color=TEXT, fontsize=13, pad=10)
ax.tick_params(colors=MUTED)
ax.figure.axes[-1].tick_params(colors=MUTED)
plt.xticks(rotation=20, ha='right', color=MUTED)
plt.yticks(rotation=0, color=MUTED)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart_heatmap.png'), dpi=130, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  ✅ Saved chart_heatmap.png")


# ─────────────────────────────────────────────────────────
# CHART 5 – Geographic Distribution
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK)
style_ax(ax)

geo     = df.groupby(['country','cluster_label']).size().unstack(fill_value=0)
geo_top = geo.loc[geo.sum(axis=1).nlargest(10).index]
colors_bar = [SEG_COLORS.get(c,'#888') for c in geo_top.columns]
geo_top.plot(kind='bar', ax=ax, color=colors_bar, edgecolor=DARK, width=0.7)

ax.set_title('Buyer Segments by Country (Top 10)', color=TEXT, fontsize=13)
ax.set_xlabel('Country')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', color=MUTED)
ax.legend(labelcolor=TEXT, facecolor=CARD, edgecolor='#334155', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart_geographic.png'), dpi=130, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  ✅ Saved chart_geographic.png")


# ─────────────────────────────────────────────────────────
# CHART 6 – Dendrogram
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK)
style_ax(ax)

dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.6 * max(Z[:, 2]))
ax.set_title('Hierarchical Clustering Dendrogram (300-client sample)', color=TEXT, fontsize=13)
ax.set_ylabel('Ward Distance')
ax.set_xlabel('Client Index')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart_dendrogram.png'), dpi=130, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  ✅ Saved chart_dendrogram.png")


# ─────────────────────────────────────────────────────────
# CHART 7 – Behaviour (Loan + Purpose)
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK)
for ax in axes:
    style_ax(ax)

loan_d = df.groupby(['cluster_label','loan_applied']).size().unstack(fill_value=0)
loan_d.plot(kind='bar', ax=axes[0], color=['#ef4444','#22c55e'], edgecolor=DARK, width=0.6)
axes[0].set_title('Loan Applied per Segment', color=TEXT)
axes[0].set_xlabel('')
axes[0].set_ylabel('Clients')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20, ha='right', color=MUTED)
axes[0].legend(['No Loan','Loan Applied'], labelcolor=TEXT, facecolor=CARD, edgecolor='#334155')

purp_d = df.groupby(['cluster_label','acquisition_purpose']).size().unstack(fill_value=0)
purp_d.plot(kind='bar', ax=axes[1], color=['#f59e0b','#6366f1'], edgecolor=DARK, width=0.6)
axes[1].set_title('Acquisition Purpose per Segment', color=TEXT)
axes[1].set_xlabel('')
axes[1].set_ylabel('Clients')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=20, ha='right', color=MUTED)
axes[1].legend(labelcolor=TEXT, facecolor=CARD, edgecolor='#334155')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart_behavior.png'), dpi=130, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  ✅ Saved chart_behavior.png")


# ─────────────────────────────────────────────────────────
# SAVE CLUSTER SUMMARY CSV
# ─────────────────────────────────────────────────────────
cluster_stats.to_csv(os.path.join(OUTPUT_DIR, 'cluster_summary.csv'))
df.to_csv(os.path.join(OUTPUT_DIR, 'clients_with_clusters.csv'), index=False)
print("  ✅ Saved cluster_summary.csv")
print("  ✅ Saved clients_with_clusters.csv")


# ─────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
print(f"  Total clients   : {len(df):,}")
print(f"  Optimal k       : {OPTIMAL_K}")
print(f"  Silhouette score: {sil_scores[OPTIMAL_K-2]:.4f}")
print(f"  Output folder   : {OUTPUT_DIR}/")
print()
for _, row in cluster_stats.iterrows():
    print(f"  [{row['label']:20s}]  {int(row['count']):4d} clients | "
          f"Avg Price: ${row['avg_price']:,.0f} | "
          f"Invest%: {row['pct_investment']:.0f}% | "
          f"Loan%: {row['pct_loan']:.0f}%")
print()
