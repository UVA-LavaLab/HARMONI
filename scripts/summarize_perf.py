import os
import re
import argparse
import pandas as pd

def parse_performance_summary_files(output_dir):
    pattern = re.compile(
        r"performance_summary_([A-Za-z0-9\.-]+)_([A-Za-z0-9\.-]+)_[A-Za-z0-9]+_(\d+)i_(\d+)o_B(\d+)\.txt"
    )
    records = []
    for fname in os.listdir(output_dir):
        if fname.startswith("performance_summary_") and fname.endswith(".txt"):
            m = pattern.match(fname)
            if not m:
                continue
            model, dram, input_tokens, output_tokens, batch = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
            with open(os.path.join(output_dir, fname), "r") as f:
                lines = f.readlines()
            prefill_latency = decode_latency = e2e_latency = e2e_throughput = tpot = None
            for line in lines:
                if "Prefill latency" in line:
                    prefill_latency = float(line.split(":")[1].split()[0])
                elif "Decode latency" in line:
                    decode_latency = float(line.split(":")[1].split()[0])
                elif "E2E Latency" in line:
                    e2e_latency = float(line.split(":")[1].split()[0])
                elif "E2E Throughput" in line:
                    e2e_throughput = float(line.split(":")[1].split()[0])
                elif "TPOT (ms/token)" in line:
                    tpot = float(line.split(":")[1].split()[0])
                elif "Total Energy (J)" in line:
                    total_energy = float(line.split(":")[1].split()[0])
                elif "Energy per token (mJ)" in line:
                    energy_per_token = float(line.split(":")[1].split()[0])
                elif "Tokens per J" in line:
                    tokens_per_J = float(line.split(":")[1].split()[0])
                elif "Compute (%)" in line: # % of E2E latency
                    comp_pct = float(line.split(":")[1].split()[0])
                elif "Communication (%)" in line: # % of E2E latency
                    comm_pct = float(line.split(":")[1].split()[0])
                elif "Queueing (%)" in line: # % of E2E latency
                    queue_pct = float(line.split(":")[1].split()[0])
            
            records.append({
                "model": model,
                "dram": dram,
                "input": input_tokens,
                "output": output_tokens,
                "batch": batch,
                "prefill_latency(ms)": prefill_latency,
                "decode_latency(ms)": decode_latency,
                "decode_throughput(tok/s)": round((output_tokens*batch*1000)/decode_latency,2), 
                "e2e_latency(ms)": e2e_latency,
                "e2e_throughput(tok/s)": e2e_throughput,
                "TPOT(ms/token)": tpot,
                "total_energy(J)": total_energy,
                "energy_per_token(mJ)": energy_per_token,
                "tokens_per_J": tokens_per_J,
                "comp_pct": comp_pct,
                "comm_pct": comm_pct,
                "queue_pct": queue_pct
            })
    df = pd.DataFrame(records)
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality performance summary plot from performance_summary files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory containing performance_summary files and to save plot.")
    args = parser.parse_args()
    df = parse_performance_summary_files(args.output_dir)
    if df.empty:
        print("No performance summary files found.")
        return
    
    # Rename 'dram' column to 'system'
    df.rename(columns={"dram": "system"}, inplace=True)

    # Save DataFrame to CSV named after output_dir.
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir_name = os.path.basename(os.path.normpath(args.output_dir))
    file_path = os.path.join(args.output_dir, f"{output_dir_name}.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved data to {file_path}")

if __name__ == "__main__":
    main()
