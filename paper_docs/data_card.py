import pandas as pd

metadata_b = "metadata_butterflies.csv"
metadata_p = "metadata_plants.csv"

df_b = pd.read_csv(metadata_b)
df_p = pd.read_csv(metadata_p)
# print(df_p.columns)

df_b['date'] = pd.to_datetime(df_b['date'], format="%Y/%m/%d", errors='coerce')
df_b['month'] = df_b['date'].dt.month_name()
df_p['date'] = pd.to_datetime(df_p['date'], format="%Y/%m/%d", errors='coerce')
df_p['month'] = df_p['date'].dt.month_name()
# print(df_b['month'])
# Missing Field Coverage
def get_missing_stats(df, fields):
    stats = []
    total = len(df)
    for field in fields:
        if field in df.columns:
            present = df[field].notnull().sum()
            missing = total - present
            pct_missing = (missing / total) * 100
            stats.append([field, present, missing, f"{pct_missing:.1f}%"])
    return pd.DataFrame(stats, columns=['Field', 'Present', 'Missing', '% Missing'])

bf_fields = ['common_name', 'sex', 'media_code', 'location', 'state', 'date', 'credit']
pl_fields = ['media_code', 'location', 'state', 'date', 'credit']

bf_missing_df = get_missing_stats(df_b, bf_fields)
pl_missing_df = get_missing_stats(df_p, pl_fields)

# drop_cols = ['image_id', 'filename', 'raw_filename', 'media_code', 'credit', 'source_url', 'source']

# df_b.drop(columns=drop_cols, inplace=True)
# df_p.drop(columns=drop_cols, inplace=True)

b_grouped_lf = df_b.groupby(by=['species', 'life_stage'], dropna=False).size().reset_index(name='count')
b_grouped = df_b.groupby(by='species', dropna=False).size().reset_index(name='count')
p_grouped = df_p.groupby(by='species', dropna=False).size().reset_index(name='count')

b_grouped_split = df_b.groupby(by=['split'], dropna=False).size().reset_index(name='count')
p_grouped_split = df_p.groupby(by=['split'], dropna=False).size().reset_index(name='count')

b_grouped_geotemp = df_b.groupby(by=['species', 'state', 'month'], dropna=False).size().reset_index(name='count')
p_grouped_geotemp = df_p.groupby(by=['species', 'state', 'month'], dropna=False).size().reset_index(name='count')

b_grouped_fam = df_b.groupby(by=['family'], dropna=False).size().reset_index(name='count')
p_grouped_fam = df_p.groupby(by=['family'], dropna=False).size().reset_index(name='count')

# print(b_grouped_lf.columns)
# print(b_grouped.columns)
# print(p_grouped.columns)
# print(p_grouped)
b_grouped_lf[['species', 'life_stage', 'count']].to_csv('stats/butterflylf_ct.csv', index=False, header=False)
b_grouped[['species', 'count']].to_csv('stats/butterfly_ct.csv', index=False, header=False)
p_grouped[['species', 'count']].to_csv('stats/plants_ct.csv', index=False, header=False)
# print(p_grouped_split)
p_grouped_split[['split', 'count']].to_csv('stats/plants_split_ct.csv', index=False, header=False)
b_grouped_split[['split', 'count']].to_csv('stats/butterfly_split_ct.csv', index=False, header=False)

b_grouped_geotemp[['species', 'state', 'month', 'count']].to_csv('stats/butterfly_geotemp_ct.csv', index=False, header=False)
p_grouped_geotemp[['species', 'state', 'month', 'count']].to_csv('stats/plants_geotemp_ct.csv', index=False, header=False)

bf_missing_df.to_csv('stats/butterfly_missing.csv', index=False)
pl_missing_df.to_csv('stats/plants_missing.csv', index=False)

b_grouped_fam[['family', 'count']].to_csv('stats/butterfly_fam.csv', index=False)
p_grouped_fam[['family', 'count']].to_csv('stats/plants_fam.csv', index=False)
# for group in b_grouped:
#     # print(name)
#     print(group)

# print("Butterflies: \n", df_b[:5])
# print("Plants: \n", df_p[:5])