import csv
import sys
from collections import defaultdict

def parse_float(val):
    val = val.replace('"', '').replace(',', '.').strip()
    try:
        return float(val)
    except ValueError:
        return 0.0

def analyze_csv(filepath):
    stats = defaultdict(lambda: {
        'count': 0,
        'total_duration': 0.0,
        'total_compute': 0.0,
        'total_memory': 0.0,
        'registers': 0
    })

    try:
        with open(filepath, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                func_name = row.get('Function Name')
                if not func_name or func_name == 'Function Name':
                    continue
                
                dur = parse_float(row.get('Duration [us]', '0'))
                comp = parse_float(row.get('Compute Throughput [%]', '0'))
                mem = parse_float(row.get('Memory Throughput [%]', '0'))
                regs = int(parse_float(row.get('# Registers [register/thread]', '0')))

                s = stats[func_name]
                s['count'] += 1
                s['total_duration'] += dur
                s['total_compute'] += comp
                s['total_memory'] += mem
                s['registers'] = max(s['registers'], regs)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    if not stats:
        print("No valid kernel data found in the CSV.")
        return

    print(f"Analysis of {filepath}")
    print(f"{'Function Name':<25} | {'Calls':<6} | {'Avg Duration (us)':<18} | {'Avg Compute (%)':<16} | {'Avg Memory (%)':<15} | {'Registers'}")
    print("-" * 110)
    
    total_time = 0.0
    for func, s in stats.items():
        total_time += s['total_duration']

    for func, s in sorted(stats.items(), key=lambda x: x[1]['total_duration'], reverse=True):
        count = s['count']
        avg_dur = s['total_duration'] / count
        avg_comp = s['total_compute'] / count
        avg_mem = s['total_memory'] / count
        regs = s['registers']
        print(f"{func[:24].ljust(25)} | {count:<6} | {avg_dur:<18.2f} | {avg_comp:<16.2f} | {avg_mem:<15.2f} | {regs}")

    print("-" * 110)

    for func, s in sorted(stats.items(), key=lambda x: x[1]['total_duration'], reverse=True):
        avg_comp = s['total_compute'] / s['count']
        avg_mem = s['total_memory'] / s['count']
        
    print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_ncu.py <path_to_csv>")
    else:
        analyze_csv(sys.argv[1])
