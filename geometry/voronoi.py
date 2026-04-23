# geometry/voronoi.py
from __future__ import annotations
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from pathlib import Path

def read_capitals() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
  # wget.download("https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip")
  _project_root = Path.cwd().resolve()
  data = _project_root.parents[0] / 'data'
  gdf = gpd.read_file(data / 'cb_2018_us_state_500k.zip')
  gdf.head()
  # Removing territories and hawaii, alaska
  gdf = gdf[~gdf.STATEFP.isin(["72", "69", "60", "66", "78", "11", "02", "15"])]


  df = pd.read_csv(data / 'state_capitals_coords.csv')
  geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
  capitals_gdf = gpd.GeoDataFrame(df, geometry=geometry)
  state_series: pd.Series = getattr(capitals_gdf, 'State')
  if state_series is None: 
    raise
  capitals_gdf = capitals_gdf[~state_series.isin(["Alaska", "Hawaii"])]
  return gdf, capitals_gdf

def main():
  gdf, capitals_gdf = read_capitals()


  fig, ax = plt.subplots(figsize=(15, 10))
  gdf.plot(ax=ax, color='lightgrey', edgecolor='black')
  capitals_gdf.plot(ax=ax, color='red', markersize=50, marker="*")

  plt.title('US State Capitals')
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.show()


if __name__ == "__main__":
  main()


