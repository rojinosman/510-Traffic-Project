# generate_synthetic_syntrac_data.py
import numpy as np
import pandas as pd

def main(rows=5000, seed=0, out_path="syntrac_merged.csv"):
    rng = np.random.default_rng(seed)
    t = np.arange(rows)

    # Signal phase: 0 or 1 with slow drift
    phase = (np.sin(t / 50.0) > 0).astype(int)

    # Queue length depends on phase + noise
    queue = (
        30
        + 12 * (phase == 0).astype(float)  # longer when red
        + rng.normal(0, 8, size=rows)
    )
    queue = np.clip(queue, 0, 80)

    # Waiting time roughly correlated with queue
    wait = np.clip(queue / 8 + rng.normal(0, 1.2, size=rows), 0, 15)

    # Throughput higher when green, inversely with queue & wait
    base_thru = 6 * (phase == 1).astype(float) + 2 * (phase == 0).astype(float)
    throughput = np.clip(base_thru - 0.03 * queue - 0.1 * wait + rng.normal(0, 0.7, size=rows), 0, 10)

    df = pd.DataFrame(
        {
            "timestamp": t,
            "signal_phase": phase,
            "queue_length": queue.round(2),
            "waiting_time": wait.round(2),
            "throughput": throughput.round(2),
        }
    )
    df.to_csv(out_path, index=False)
    print(f"✅ Synthetic dataset generated: {out_path} ({len(df)} rows)")
    print(df.head())

if __name__ == "__main__":
    main()
