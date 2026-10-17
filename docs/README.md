# Documentation index

The notes below were written in Korean during development and are kept as the primary record. One-line English summaries:

| File | Topic |
|---|---|
| [operations.md](operations.md) | Former Korean README: running notes for the cluster, patch ZIPs, live view and each pool version, in chronological order. |
| [reproduction.md](reproduction.md) | What the paper and the official repositories publish, the pinned RoboTwin revision, Appendix A/B ranges, seed and fairness rules, and every interpretation this implementation made. |
| [validation.md](validation.md) | Infrastructure validation records (Docker, dual-GPU, live view, MP4, reproducibility of frozen episodes). |
| [docker.md](docker.md) | Building and running the simulator image; GPU and Vulkan rendering notes. |
| [policies.md](policies.md) | The five checkpoints, their adapters, inference arguments (Fast-WAM `sigma_shift=5.0`) and preparation steps. |
| [cluster.md](cluster.md) | CSI cluster: Apptainer bundle image, Open OnDemand launchers, patch ZIP workflow, result tables. |
| [parallel-evaluation.md](parallel-evaluation.md), [parallel-evaluation-5090.md](parallel-evaluation-5090.md), [evaluation-pro6000-single.md](evaluation-pro6000-single.md), [pool-parallel.md](pool-parallel.md) | GPU layouts of the scheduler: shared, split, dual; pool preparation with several workers per GPU. |
| [fresh-pool.md](fresh-pool.md), [com-scope-fresh-pool.md](com-scope-fresh-pool.md) | The v5 and v6 pools; the decision to apply CoM/mass/size/force to task objects only. |
| [oidn-blackwell.md](oidn-blackwell.md) | SAPIEN's bundled OIDN 2.0.1 denoiser fails on Blackwell; evidence, the OIDN 2.3.3 bind fix and the in-job render probe. |
| [g05-runtime-bundle.md](g05-runtime-bundle.md) | The pinned GalaxeaVLA adapter resolves the checkpoint symlink and read the original config (job 1690); the hard-linked runtime bundle fix. |
| [success-context-fix.md](success-context-fix.md) | Three tasks whose `check_success` reads expert-only attributes; the evaluator fix and the provenance continuation. |
| [put-object-cabinet-feasibility.md](put-object-cabinet-feasibility.md) | Why `put_object_cabinet` accepts about 2% of candidates and why no deviation was taken. |
| [expert-planning-review.md](expert-planning-review.md), [planner-failure-recovery.md](planner-failure-recovery.md), [grasp-contact-recovery.md](grasp-contact-recovery.md), [empty-grasp-recovery.md](empty-grasp-recovery.md) | How the official expert's failure modes (exhausted grasp search, planner exceptions, missing or empty contact annotations) are classified without hiding runtime errors, and how saved pool progress was preserved across each fix. |
| [geometry-failure-review.md](geometry-failure-review.md) | Finite-rotation repair of the shared RoboTwin transform helpers and re-verification of saved pools. |
| [curobo-mask-review.md](curobo-mask-review.md) | CuRobo v0.7.8 batch result ownership and IK mask fixes. |
