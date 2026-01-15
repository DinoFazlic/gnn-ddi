# 🏆 Leaderboard

Competition: **GNN Molecular Graph Classification Challenge**

Primary Metric: **Macro F1 Score** (higher is better)

Efficiency Metric: $\text{Efficiency} = \frac{F_1^2}{\log_{10}(\text{time}_{ms}) \times \log_{10}(\text{params})}$

---

| Rank | Participant | Macro-F1 | Efficiency | Params | Time (ms) | Last Updated |
|------|-------------|----------|------------|--------|-----------|---------------|
| 🥇 1 | *Baseline-Spectral* | 0.7215 | 0.6360 | 40 | 4.4 | 2026-01-15 |
| 🥈 2 | *Baseline-DMPNN* | 0.6674 | 0.0833 | 53 | 62.4 | 2026-01-15 |
| 🥉 3 | *Baseline-GCN* | 0.6153 | - | - | - | 2026-01-07 |
| 4 | *Baseline-GIN* | 0.6103 | - | - | - | 2026-01-07 |
| 5 | *Baseline-GraphSAGE* | 0.5835 | - | - | - | 2026-01-07 |
| 6 | muuki2 | 0.5048 | - | - | - | 2026-01-07 |

---

### Legend

- **Macro-F1**: Primary ranking metric (harmonic mean of class-wise F1 scores)
- **Efficiency**: Higher is better - rewards both accuracy and computational efficiency
- **Params**: Total number of trainable parameters
- **Time (ms)**: Average inference time per batch

*Italic entries are baseline models provided by organizers.*

*Last updated: 2026-01-15 19:45 UTC*
