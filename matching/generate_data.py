import pandas as pd
import os

INPUT_FILE = 'traces/container_usage.csv'
print("Extracting representative utilization rows for Alibaba Test...")

# Based on the official Alibaba Trace schema
col_names = ['container_id', 'machine_id', 'time_stamp', 'cpu_util_percent', 
             'mem_util_percent', 'cpi', 'mem_gps', 'mpki', 'net_in', 'net_out', 'disk_io_percent']

collected_data = []

# Process in 50k chunks to use minimal RAM
try:
    for chunk in pd.read_csv(INPUT_FILE, chunksize=50000, header=None, names=col_names):
        # Clean invalid data (keep only physically valid utilization percentages)
        valid_data = chunk[(chunk['cpu_util_percent'] >= 0) & (chunk['cpu_util_percent'] <= 100)]
        valid_data = valid_data[(valid_data['mem_util_percent'] >= 0) & (valid_data['mem_util_percent'] <= 100)]
        
        # Extract ALL valid rows — the natural Alibaba distribution
        # This gives a representative mix of idle, moderate, and stressed servers
        collected_data.append(valid_data)
        
        # Stop when we have 5000 rows (enough for a 5000 timeslot simulation)
        total_rows = sum(len(df) for df in collected_data)
        print(f"Extracted {total_rows}/5000 rows...", end='\r')
        if total_rows >= 5000:
            break

    final_df = pd.concat(collected_data).head(5000)

    # We map CPU to Energy (high CPU = high power draw) and Memory to Delay (high RAM usage = swap/paging latency)
    clean_df = final_df[['cpu_util_percent', 'mem_util_percent']]
    os.makedirs('traces', exist_ok=True)
    clean_df.to_csv('traces/alibaba_clean_trace.csv', index=False)
    
    print(f"\n[SUCCESS] Extraction complete! Saved {len(clean_df)} rows to traces/alibaba_clean_trace.csv")
    print(f"  CPU range: {clean_df['cpu_util_percent'].min():.0f}% - {clean_df['cpu_util_percent'].max():.0f}% (mean: {clean_df['cpu_util_percent'].mean():.1f}%)")
    print(f"  MEM range: {clean_df['mem_util_percent'].min():.0f}% - {clean_df['mem_util_percent'].max():.0f}% (mean: {clean_df['mem_util_percent'].mean():.1f}%)")

except FileNotFoundError:
    print("\n[ERROR] traces/container_usage.csv not found! Did you run the wget and tar commands?")