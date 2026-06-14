import re

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if line.strip() == 'df["CombinedScore"] = combined':
        new_lines.append(line.replace('combined', 'combined.fillna(0.0)'))
        continue

    if line.strip() == 'fog_df_rounded = fog_df.copy()':
        new_lines.append(line.replace('fog_df_rounded = fog_df.copy()', 'fog_grid = fog_df.groupby([fog_df["Lat"].round(4).rename("Lat_g"), fog_df["Lon"].round(4).rename("Lon_g")])["MatchRate"].mean().reset_index()'))
        continue
        
    if line.strip() == 'fog_df_rounded["Lat_g"] = fog_df_rounded["Lat"].round(4)':
        new_lines.append(line.replace('fog_df_rounded["Lat_g"] = fog_df_rounded["Lat"].round(4)', 'temp_grid = grid_df.copy()'))
        continue
        
    if line.strip() == 'fog_df_rounded["Lon_g"] = fog_df_rounded["Lon"].round(4)':
        new_lines.append(line.replace('fog_df_rounded["Lon_g"] = fog_df_rounded["Lon"].round(4)', 'grid_set = set(zip(temp_grid["Lat_g"], temp_grid["Lon_g"]))'))
        continue
        
    if line.strip() == '# Compute mean MatchRate per grid point (many points have 0, so average is lower)':
        new_lines.append(line.replace('# Compute mean MatchRate per grid point (many points have 0, so average is lower)', 'fog_mask = pd.Series(list(zip(fog_df["Lat"].round(4), fog_df["Lon"].round(4)))).isin(grid_set)'))
        continue
        
    if line.strip() == 'fog_grid = fog_df_rounded.groupby(["Lat_g", "Lon_g"])["MatchRate"].mean().reset_index()':
        new_lines.append(line.replace('fog_grid = fog_df_rounded.groupby(["Lat_g", "Lon_g"])["MatchRate"].mean().reset_index()', 'filtered_df = fog_df[fog_mask.values].copy()'))
        continue
        
    if line.strip() == '# Score only points with data':
        new_lines.append(line.replace('# Score only points with data', 'filtered_df["Lat_g"] = filtered_df["Lat"].round(4)'))
        continue
        
    if line.strip() == 'filtered_df = pd.merge(fog_df_rounded, grid_df, on=["Lat_g", "Lon_g"], how="inner")':
        new_lines.append(line.replace('filtered_df = pd.merge(fog_df_rounded, grid_df, on=["Lat_g", "Lon_g"], how="inner")', 'filtered_df["Lon_g"] = filtered_df["Lon"].round(4)'))
        continue

    new_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
