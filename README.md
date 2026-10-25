# Customer Segmentation Using K-Means Clustering

**Portfolio:** Muhammad Awais  
**Stack:** Python, pandas, scikit-learn, matplotlib

## Overview
This beginner-friendly machine learning project groups **240 synthetic customer records** using K-Means clustering. Features include age, engagement score, and annual spend in illustrative Pakistani rupees (PKR). The exercise demonstrates feature scaling, unsupervised learning, and cluster profiling.

## Key workflow
1. Load and validate the dataset.
2. Standardize numeric features using `StandardScaler` so annual spend does not dominate due to its scale.
3. Fit K-Means with three clusters and a fixed random seed.
4. Summarize clusters and save charts.

## Run
Python 3.10+ is recommended. From the project folder:

```bash
python -m pip install -r requirements.txt
python -m src.analysis
python -m unittest discover -s tests -v
```

Try another cluster count:

```bash
python -m src.analysis --clusters 4 --out outputs_k4
```

## Files
- `data/customers.csv`: synthetic input dataset
- `src/analysis.py`: model and visualizations
- `outputs/customers_with_clusters.csv`: assigned cluster IDs
- `outputs/cluster_profile.csv`: mean feature values per cluster
- `outputs/summary.json`: counts and K-Means inertia
- `outputs/*.png`: two charts
- `tests/`: smoke and input validation tests

![Customer segments](outputs/customer_segments.png)
![Cluster sizes](outputs/cluster_sizes.png)

## Reproduced example results
The included data contains **240 simulated customers**, grouped into **3 clusters**. Cluster sizes are 0: 79, 1: 83, 2: 78; inertia is **133.831**. These results depend on the supplied data, selected features, scaling, and fixed random seed.

## Important interpretation notes
Cluster IDs are arbitrary numeric labels, **not ground-truth customer categories**. The dataset was simulated with underlying patterns; K-Means results are not evidence about real customer behavior. Choosing `k=3` is a demonstration setting, not a validated optimal number of segments. The scatterplot displays two of the three clustering features.

## Potential improvements
Use the elbow method or silhouette score to compare cluster counts, explore PCA visualization, and test a properly sourced real-world dataset.

## About this portfolio project
This repository documents a learning-oriented clustering exercise maintained for the portfolio of **Muhammad Awais**. The dataset is simulated, and cluster labels are illustrative. See `PROJECT_GUIDE.md` for reproduction and extension ideas.
