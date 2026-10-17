# Policy success rate (%)

Independent paper reproduction; 50 tasks × 100 episodes per model/protocol. All 50 tasks with 100 episodes.

| Model | RoboTwin-Phys randomization |
|---|---:|
| G0.5 | — (0/5000 episodes) |
| π0.5 | — (0/5000 episodes) |
| FACT | — (0/5000 episodes) |
| Motus | — (0/5000 episodes) |
| Fast-WAM | 45.18 |

Each score is the arithmetic mean of task success rates. Missing, interrupted, invalid, or debug results do not count as failures and suppress the aggregate score. Completed policy failures remain in the denominator.

The same verified manifest is shared by every policy for each task/protocol. Different protocols can accept different seeds after expert filtering; this is not a paired comparison across protocols.

Detailed task results and completion counts: `tasks.csv`, `summary.csv`, `results.json`.
