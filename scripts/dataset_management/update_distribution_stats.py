
import pandas as pd
import os

def generate_stats(stats_dir, output_file):
    # Load Butterfly Stats
    butterfly_ct = pd.read_csv(os.path.join(stats_dir, "butterfly_ct.csv"), header=None, names=["Species", "Count", "Total"])
    butterflylf_ct = pd.read_csv(os.path.join(stats_dir, "butterflylf_ct.csv"), header=None, names=["Species", "LifeStage", "Count", "Total"])
    butterfly_split_ct = pd.read_csv(os.path.join(stats_dir, "butterfly_split_ct.csv"), header=None, names=["Split", "Count"])
    butterfly_geotemp = pd.read_csv(os.path.join(stats_dir, "butterfly_geotemp_ct.csv"), header=None, names=["Species", "State", "Month", "Count"])
    butterfly_fam = pd.read_csv(os.path.join(stats_dir, "butterfly_fam.csv"), header=None, names=["Family", "Count"])
    # Load Plant Stats
    plants_ct = pd.read_csv(os.path.join(stats_dir, "plants_ct.csv"), header=None, names=["Species", "Count", "Total"])
    plants_split_ct = pd.read_csv(os.path.join(stats_dir, "plants_split_ct.csv"), header=None, names=["Split", "Count"])
    plants_geotemp = pd.read_csv(os.path.join(stats_dir, "plants_geotemp_ct.csv"), header=None, names=["Species", "State", "Month", "Count"]) 
    plants_fam = pd.read_csv(os.path.join(stats_dir, "plants_fam.csv"), header=None, names=["Family", "Count"])
    # Totals
    total_butterflies = butterfly_ct["Count"].sum()
    total_plants = plants_ct["Count"].sum()
    total_images = total_butterflies + total_plants

    butterfly_fam_count = butterfly_fam["Count"].sum()
    plants_fam_count = plants_fam["Count"].sum()

    # Species Top 30
    def get_top_n(df, n):
        sorted_df = df.sort_values("Count", ascending=False)
        top_n = sorted_df.head(n)
        others = sorted_df.iloc[n:]
        return top_n, others

    top_30_bf, others_bf = get_top_n(butterfly_ct, 30)
    top_30_pl, others_pl = get_top_n(plants_ct, 30)

    bf_fam = butterfly_fam.sort_values("Count", ascending=False)
    pl_fam = plants_fam.sort_values("Count", ascending=False)
    top_20_plan_fam, other_20_plan_fam = get_top_n(pl_fam, 20)
    # Life Stage
    lstage = butterflylf_ct.groupby("LifeStage")["Count"].sum()

    # Splits
    bf_splits = butterfly_split_ct.groupby("Split")["Count"].sum()
    pl_splits = plants_split_ct.groupby("Split")["Count"].sum()
    total_splits = bf_splits.add(pl_splits, fill_value=0)

    # Geographic
    bf_states = butterfly_geotemp.groupby("State")["Count"].sum()
    pl_states = plants_geotemp.groupby("State")["Count"].sum()
    total_states = bf_states.add(pl_states, fill_value=0).sort_values(ascending=False)

    # Temporal
    bf_months = butterfly_geotemp.groupby("Month")["Count"].sum()
    pl_months = plants_geotemp.groupby("Month")["Count"].sum()
    total_months = bf_months.add(pl_months, fill_value=0)

    # Missing Stats
    butterfly_missing = pd.read_csv(os.path.join(stats_dir, "butterfly_missing.csv"))
    plants_missing = pd.read_csv(os.path.join(stats_dir, "plants_missing.csv"))
    
    month_order = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
    
    # Generate Markdown
    md = []
    md.append("# Dataset Distribution Statistics")
    md.append("")
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

    md.append("## 2. Images per Family")
    md.append("")
    md.append("### Butterflies")
    md.append("")
    md.append("| Family | Count |")
    md.append("|---|---|")
    for _, row in bf_fam.iterrows():
        md.append(f"| {row['Family']} | {row['Count']} |")
    md.append("")
    md.append(f"**Total: {butterfly_fam_count} images across {len(bf_fam)} unique values**")
    md.append("")
    md.append("### Plants")
    md.append("")
    md.append("| Family | Count |")
    md.append("|---|---|")
    for _, row in top_20_plan_fam.iterrows():
        md.append(f"| {row['Family']} | {row['Count']} |")
    md.append("")
    md.append(f"**Total: {plants_fam_count} images across {len(pl_fam)} unique values**")
    md.append("")


    md.append("## 3. Geographic Distribution (by State)")
    md.append("")
    md.append("| State | Count |")
    md.append("|---|---|")
    for state, count in total_states.items():
        md.append(f"| {state} | {int(count)} |")
    md.append("")
    md.append(f"**Total: {int(total_states.sum())} images across {len(total_states)} unique values**")
    md.append("")

    md.append("## 4. Temporal Distribution (by Month)")
    md.append("")
    md.append("| Month | Count |")
    md.append("|---|---|")
    for month in month_order:
        if month in total_months:
            md.append(f"| {month} | {int(total_months[month])} |")
    md.append("")

    md.append("## 5. Life Stage Distribution (Butterflies)")
    md.append("")
    md.append("| Life Stage | Count |")
    md.append("|---|---|")
    for ls, count in lstage.sort_values(ascending=False).items():
        md.append(f"| {ls} | {count} |")
    md.append("")
    md.append(f"**Total: {total_butterflies} images across {len(lstage)} unique values**")
    md.append("")

    md.append("## 6. Missing Field Coverage")
    md.append("")
    md.append("### Butterflies")
    md.append("")
    md.append("| Field | Present | Missing | % Missing |")
    md.append("|---|---|---|---|")
    for _, row in butterfly_missing.iterrows():
        md.append(f"| {row['Field']} | {row['Present']} | {row['Missing']} | {row['% Missing']} |")
    md.append("")
    md.append("### Plants")
    md.append("")
    md.append("| Field | Present | Missing | % Missing |")
    md.append("|---|---|---|---|")
    for _, row in plants_missing.iterrows():
        md.append(f"| {row['Field']} | {row['Present']} | {row['Missing']} | {row['% Missing']} |")
    md.append("")

    md.append("## 7. Split Distribution")
    md.append("")
    md.append("| Split | Count | % |")
    md.append("|---|---|---|")
    total_all_splits = total_splits.sum()
    for split in ["train", "val", "test"]:
        count = int(total_splits.get(split, 0))
        percent = (count / total_all_splits) * 100 if total_all_splits > 0 else 0
        md.append(f"| {split} | {count} | {percent:.1f}% |")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

if __name__ == "__main__":
    # Use absolute paths for reliability
    base_dir = r"c:\Users\Kriti\OneDrive\Desktop\sem6\dlcv\dataset\IndoLepAtlas\paper_docs"
    stats_dir = os.path.join(base_dir, "stats")
    output_file = os.path.join(base_dir, "distribution_stats.md")
    
    generate_stats(stats_dir, output_file)
    print(f"Updated {output_file}")
