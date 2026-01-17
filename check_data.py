
import pandas as pd
import numpy as np
import os

base_path = r'c:\Users\MarcellNemeth\Documents\BME\phd\codes\HierarchicalTS\notebooks\data\hierarchical\TourismSmall'
agg_mat = pd.read_csv(os.path.join(base_path, 'agg_mat.csv'), index_col=0)
data = pd.read_csv(os.path.join(base_path, 'data.csv'))

print("Agg Mat Shape:", agg_mat.shape)
print("Data Shape:", data.shape)
print("Agg Mat Indices:", agg_mat.index[:5])
print("Agg Mat Columns:", agg_mat.columns[:5])
print("Data Columns:", data.columns[:5])
