#!/usr/bin/env python3
import pandas as pd


KEY_COLS = ["model", "system", "batch", "lin", "lout"]


def format_key(key_vals) -> str:
    return (
        f"model={key_vals[0]}, system={key_vals[1]}, batch={key_vals[2]}, "
        f"lin={key_vals[3]}, lout={key_vals[4]}"
    )


def to_python_scalar(value):
    """Convert pandas/numpy scalar wrappers to plain Python values for clean logs."""
    if pd.isna(value):
        return "NaN"
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def main() -> int:
    ref_df = pd.read_csv("reference/hierarchical_data.csv")
    res_df = pd.read_csv("results/hierarchical_data.csv")

    ref_df = ref_df.set_index(KEY_COLS, drop=False)
    res_df = res_df.set_index(KEY_COLS, drop=False)

    compare_cols = [c for c in res_df.columns if c in ref_df.columns and c not in KEY_COLS]

    errors = 0
    for key, res_row in res_df.iterrows():
        if key not in ref_df.index:
            print(f"[WARN] Missing key in reference: {format_key(key)}")
            continue

        ref_row = ref_df.loc[key]
        for col in compare_cols:
            res_val = res_row[col]
            ref_val = ref_row[col]
            if pd.isna(res_val) and pd.isna(ref_val):
                continue
            if res_val != ref_val:
                errors += 1
                print(
                    "[ERROR] Mismatch "
                    f"({format_key(key)}), column={col}, "
                    f"results={to_python_scalar(res_val)}, "
                    f"reference={to_python_scalar(ref_val)}"
                )

    if errors == 0:
        print("[PASS] All relevant metrics matched the reference values.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
