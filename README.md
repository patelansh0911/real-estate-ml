# 🏠 Real Estate ML — Buyer Segmentation & Investment Profiling

**Machine Learning–Based Market Intelligence for Parcl Co.**  
K-Means + Hierarchical Clustering | Python | Streamlit

---

## 📁 Project Structure

```
real_estate_ml/
│
├── data/
│   ├── clients.csv          ← Client records (2,000 rows)
│   └── properties.csv       ← Property transactions (10,000 rows)
│
├── outputs/                 ← Auto-created when pipeline runs
│   ├── chart_elbow_silhouette.png
│   ├── chart_donut.png
│   ├── chart_pca_scatter.png
│   ├── chart_heatmap.png
│   ├── chart_geographic.png
│   ├── chart_dendrogram.png
│   ├── chart_behavior.png
│   ├── cluster_summary.csv
│   └── clients_with_clusters.csv
│
├── pipeline.py              ← Full ML pipeline (Steps 1–6), saves charts + CSVs
├── app.py                   ← Streamlit interactive dashboard
├── requirements.txt         ← Python dependencies
└── README.md
```

---

## ⚙️ Setup (VS Code)

### 1. Open the folder in VS Code
```
File → Open Folder → select real_estate_ml/
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the ML Pipeline

Runs all 6 steps and saves charts + CSV results to the `outputs/` folder.

```bash
python pipeline.py
```

**What it does:**
- Step 1 — Loads & cleans data (merges clients + properties, parses age)
- Step 2 — Encodes categorical features (Label + One-Hot encoding, 68 features)
- Step 3 — Scales features with StandardScaler
- Step 4 — Runs K-Means (k=2–8) + Hierarchical Clustering
- Step 5 — Selects optimal k via Elbow Method + Silhouette Score
- Step 6 — Interprets clusters, assigns buyer segment labels

---

## 🚀 Run the Streamlit Dashboard

Interactive web dashboard with filters, charts, and download.

```bash
streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

**Dashboard Tabs:**
| Tab | Content |
|-----|---------|
| 📊 Overview | KPI metrics, segment cards, comparison table |
| 🔬 Cluster Analysis | Elbow/Silhouette, Donut chart, PCA scatter, Heatmap |
| 🌍 Geographic | Country-wise segment breakdown |
| 💡 Behavior | Loan patterns, acquisition purpose, age vs price |
| 🌲 Hierarchical | Dendrogram, strategic recommendations, CSV download |

**Sidebar Filters:**
- Country, Region, Acquisition Purpose, Client Type, Buyer Segment

---

## 📊 Buyer Segments (Results)

| # | Segment | Clients | Avg Price | Key Trait |
|---|---------|---------|-----------|-----------|
| C1 | 🌍 Global Investors | 584 | $406,460 | Highest price, multi-property |
| C2 | 🏠 First-Time Buyers | 526 | $335,626 | 100% investment purpose |
| C3 | 🏢 Corporate Buyers | 814 | $312,034 | Largest group, 0% invest purpose |
| C4 | 💎 Luxury Investors | 76 | $345,679 | Niche, high satisfaction |

**Silhouette Score (k=4): 0.1610**

---

## 🧪 Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | Data loading, cleaning, merging |
| `scikit-learn` | KMeans, StandardScaler, LabelEncoder, PCA, Silhouette |
| `scipy` | Hierarchical clustering (Ward linkage) |
| `matplotlib` + `seaborn` | Chart generation |
| `streamlit` | Interactive web dashboard |

---

## 📌 Notes

- The `data/` folder must contain both CSV files before running
- `outputs/` is created automatically by `pipeline.py`
- All charts use a dark theme (`#0f172a` background)
- PCA is used only for 2D visualisation, not for clustering
