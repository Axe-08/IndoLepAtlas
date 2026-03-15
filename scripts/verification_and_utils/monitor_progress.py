#!/usr/bin/env python3
import os
import time
import sys
from pathlib import Path

def count_files(directory, extension):
    count = 0
    if not os.path.isdir(directory):
        return 0
    for root, dirs, files in os.walk(directory):
        count += len([f for f in files if f.lower().endswith(extension)])
    return count

def get_status():
    base_dir = "."
    datasets = ["butterflies", "plants"]
    
    status = []
    
    for ds in datasets:
        raw_dir = os.path.join(base_dir, "data", ds, "raw")
        proc_dir = os.path.join(base_dir, "data", ds, "images")
        anno_dir = os.path.join(base_dir, "annotations", ds)
        
        raw_count = count_files(raw_dir, (".jpg", ".jpeg", ".png"))
        proc_count = count_files(proc_dir, (".jpg", ".jpeg", ".png"))
        anno_count = count_files(anno_dir, ".txt")
        
        status.append({
            "name": ds.capitalize(),
            "raw": raw_count,
            "processed": proc_count,
            "annotated": anno_count
        })
    
    return status

def print_dashboard(status):
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 60)
    print(" IndoLepAtlas - Pipeline Progress Monitor ".center(60, "="))
    print("=" * 60)
    print(f"{'Dataset':<15} | {'Raw':<10} | {'Processed':<10} | {'Annotated':<10}")
    print("-" * 60)
    
    for s in status:
        proc_pct = (s['processed'] / s['raw'] * 100) if s['raw'] > 0 else 0
        anno_pct = (s['annotated'] / s['processed'] * 100) if s['processed'] > 0 else 0
        
        line = f"{s['name']:<15} | {s['raw']:<10} | {s['processed']:<10} | {s['annotated']:<10}"
        print(line)
        print(f"{'':<15} | {'Trimmed:':<10} {proc_pct:>5.1f}% | {'Bbox:':<10} {anno_pct:>5.1f}%")
        print("-" * 60)

    print("\n[Press Ctrl+C to exit monitor]")
    
    # Show last few lines of log if exists
    if os.path.exists("annotate.log"):
        print("\n--- Last 5 Log Entries (annotate.log) ---")
        try:
            with open("annotate.log", "r") as f:
                lines = f.readlines()
                for l in lines[-5:]:
                    print(l.strip())
        except:
            pass

def main():
    while True:
        try:
            status = get_status()
            print_dashboard(status)
            time.sleep(10)
        except KeyboardInterrupt:
            print("\nExiting monitor.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
