import pandas as pd

metadata_b = "metadata_butterflies.csv"
metadata_p = "metadata_plants.csv"

df_b = pd.read_csv(metadata_b)
df_p = pd.read_csv(metadata_p)

drop_cols = ['image_id', 'filename', 'raw_filename', 'media_code', 'credit', 'source_url', 'source']

df_b.drop(columns=drop_cols, inplace=True)
df_p.drop(columns=drop_cols, inplace=True)

b_grouped_lf = df_b.groupby(by=['species', 'life_stage'], dropna=True, as_index=False).count()
b_grouped = df_b.groupby(by='species', dropna=True, as_index=False).count()
p_grouped = df_p.groupby(by='species', dropna=True, as_index=False).count()

b_grouped_split = df_b.groupby(by=['species', 'split'], dropna=True, as_index=False).count()
p_grouped_split = df_p.groupby(by=['species', 'split'], dropna=True, as_index=False).count()

# print(b_grouped_lf.columns)
# print(b_grouped.columns)
# print(p_grouped.columns)
# print(p_grouped)
# b_grouped_lf[['species', 'life_stage', 'common_name']].to_csv('stats/butterflylf_ct.csv', index=False, header=False)
# b_grouped[['species', 'common_name']].to_csv('stats/butterfly_ct.csv', index=False, header=False)
# p_grouped[['species', 'family']].to_csv('stats/plants_ct.csv', index=False, header=False)
# print(p_grouped_split)
p_grouped_split[['species', 'split', 'family']].to_csv('stats/plants_split_ct.csv', index=False, header=False)
b_grouped_split[['species', 'split', 'common_name']].to_csv('stats/butterfly_split_ct.csv', index=False, header=False)

# for group in b_grouped:
#     # print(name)
#     print(group)

# print("Butterflies: \n", df_b[:5])
# print("Plants: \n", df_p[:5])