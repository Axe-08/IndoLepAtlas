
import pandas as pd
import os

stats_dir = r"c:\Users\Kriti\OneDrive\Desktop\sem6\dlcv\dataset\IndoLepAtlas\paper_docs\stats"

# Load Butterfly Stats
butterfly_ct = pd.read_csv(os.path.join(stats_dir, "butterfly_ct.csv"), header=None, names=["Species", "Count", "Total"])
butterflylf_ct = pd.read_csv(os.path.join(stats_dir, "butterflylf_ct.csv"), header=None, names=["Species", "LifeStage", "Count", "Total"])
butterfly_split_ct = pd.read_csv(os.path.join(stats_dir, "butterfly_split_ct.csv"), header=None, names=["Species", "Split", "Count"])

# Load Plant Stats
plants_ct = pd.read_csv(os.path.join(stats_dir, "plants_ct.csv"), header=None, names=["Species", "Count", "Total"])
plants_split_ct = pd.read_csv(os.path.join(stats_dir, "plants_split_ct.csv"), header=None, names=["Species", "Split", "Count"])

# Totals
total_butterflies = butterfly_ct["Count"].sum()
total_plants = plants_ct["Count"].sum()
total_images = total_butterflies + total_plants

print(f"Total images: {total_images}")
print(f"- Butterflies: {total_butterflies}")
print(f"- Plants: {total_plants}")

# Species Top 30
def get_top_30(df):
    sorted_df = df.sort_values("Count", ascending=False)
    top_30 = sorted_df.head(30)
    others = sorted_df.iloc[30:]
    return top_30, others

top_30_bf, others_bf = get_top_30(butterfly_ct)
top_30_pl, others_pl = get_top_30(plants_ct)

# Life Stage
lstage = butterflylf_ct.groupby("LifeStage")["Count"].sum()

# Splits
bf_splits = butterfly_split_ct.groupby("Split")["Count"].sum()
pl_splits = plants_split_ct.groupby("Split")["Count"].sum()
total_splits = bf_splits.add(pl_splits, fill_value=0)

# Generate Markdown
md = []
md.append(f"**Total images:** {total_images}")
md.append(f"- Butterflies: {total_butterflies}")
md.append(f"- Plants: {total_plants}")
md.append("")
md.append("## 1. Images per Species")
md.append("")
md.append("### Butterflies")
md.append("")
md.append("| Species | Count |")
md.append("|---|---|")
for _, row in top_30_bf.iterrows():
    md.append(f"| {row['Species']} | {row['Count']} |")
md.append("")
md.append(f"**Total: {total_butterflies} images across {len(butterfly_ct)} unique values**")
md.append(f"*(showing top 30, {len(others_bf)} more with {others_bf['Count'].sum()} images)*")
md.append("")
md.append("### Plants")
md.append("")
md.append("| Species | Count |")
md.append("|---|---|")
for _, row in top_30_pl.iterrows():
    md.append(f"| {row['Species']} | {row['Count']} |")
md.append("")
md.append(f"**Total: {total_plants} images across {len(plants_ct)} unique values**")
md.append(f"*(showing top 30, {len(others_pl)} more with {others_pl['Count'].sum()} images)*")
md.append("")

# Life Stage
md.append("## 5. Life Stage Distribution (Butterflies)")
md.append("")
md.append("| Life Stage | Count |")
md.append("|---|---|")
for ls, count in lstage.items():
    md.append(f"| {ls} | {count} |")
md.append("")
md.append(f"**Total: {total_butterflies} images across {len(lstage)} unique values**")
md.append("")

# Splits
md.append("## 7. Split Distribution")
md.append("")
md.append("| Split | Count | % |")
md.append("|---|---|---|")
total_all_splits = total_splits.sum()
for split in ["train", "val", "test"]:
    count = int(total_splits.get(split, 0))
    percent = (count / total_all_splits) * 100
    md.append(f"| {split} | {count} | {percent:.1f}% |")

print("\n".join(md))
