import matplotlib.pyplot as plt

def plot_runtime(runtimes_s, title, outfile):
    labels = list(runtimes_s.keys())
    values = [runtimes_s[k] for k in labels]

    baseline = values[0]  # Serial baseline
    speedups = [baseline / v for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(values)), values)

    # Annotate with runtime (s) + speedup
    for bar, sec, sp in zip(bars, values, speedups):
        txt = f"{sec:.2f} s\n×{sp:.1f}"
        ax.annotate(
            txt,
            xy=(bar.get_x() + bar.get_width() / 2, sec),
            xytext=(0, 5),  # offset 5 px above bar
            textcoords="offset points",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Runtime (s)")
    ax.set_title(title)

    # Add margin at top so text never overlaps with title
    ax.set_ylim(0, max(values) * 1.25)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    print(f"Saved {outfile}")


# ===== European Monte Carlo runtimes =====
eu_runtimes_s = {
    "Serial": 386.0,
    "OMP x2": 205.1,
    "OMP x4": 102.9,
    "OMP x8": 79.0,
    "CUDA":   2.26,
}
plot_runtime(eu_runtimes_s, 
             "European Monte Carlo Runtime by Threads (50M paths - 1 option chain)", 
             "runtime_european.png")

# ===== American Monte Carlo runtimes =====
am_runtimes_s = {
    "Serial":  462.0,
    "OMP x2": 224.0,
    "OMP x4": 180.0,
    "OMP x8": 156.0,
    "CUDA":     8.5,
}
plot_runtime(am_runtimes_s, 
             "American Monte Carlo Runtime by Threads (500k paths - 1 option chain)", 
             "runtime_american.png")

