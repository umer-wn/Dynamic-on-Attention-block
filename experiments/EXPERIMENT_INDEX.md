# Experiment Index

| Experiment | Status | Plan | Report | Data root |
|---|---|---|---|---|
| Pythia operator-output gain sweep | complete | `plan/pythia_operator_gain_sweep_plan.md` | `reports/pythia_operator_gain_sweep_findings.md` | legacy: project `results/` |
| Pythia residual identity calibration | complete | `plan/pythia_residual_calibration_plan.md` | `reports/pythia_residual_calibration_findings.md` | legacy: project `results/` |
| Pythia epsilon sensitivity | complete | `plan/pythia_epsilon_sensitivity_plan.md` | `reports/pythia_epsilon_sensitivity_findings.md` | `/home/luohaoming/model_feature_experiments/pythia_epsilon_sensitivity` |
| Core-paper alignment audit | complete | method gate | `reports/core_paper_alignment_audit_20260712.md` | no experiment data |
| Paper-aligned Lyapunov remeasurement | complete | `plan/paper_aligned_lyapunov_remeasurement_plan.md` | `reports/paper_aligned_lyapunov_remeasurement_findings.md` | `/home/luohaoming/model_feature_experiments/paper_aligned_lyapunov_remeasurement` |
| Pythia checkpoint criticality | complete | `plan/pythia_checkpoint_criticality_plan.md` | `reports/pythia_checkpoint_criticality_findings.md` | `/home/luohaoming/model_feature_experiments/pythia_checkpoint_criticality` |
| Pythia checkpoint long-asymptotic follow-up | complete | `plan/pythia_checkpoint_long_asymptotic_followup_plan.md` | included in `reports/pythia_checkpoint_criticality_findings.md` | `/home/luohaoming/model_feature_experiments/pythia_checkpoint_criticality/long_asymptotic` |
| Dynamics visualization and research audit (original phases 02–06) | complete; static-spectrum phase retired 2026-07-13 | `plan/experiment_visualization_report_plan.md` | `reports/experiment_visualization_review.md` | full figures/manifest: `/home/luohaoming/model_feature_reports/experiment_visualization_review`; projection rerun: `/home/luohaoming/model_feature_experiments/pythia_checkpoint_criticality/visualization_rerun` |
| Transformer paper-validation small experiments | rolling main complete; low-priority training gated | `plan/transformer_paper_validation_small_experiments_plan.md` | `reports/rolling_next_token_criticality_report.md` | `/home/luohaoming/model_feature_experiments/rolling_next_token_criticality`; full-embedding training not started |
| Rolling next-token visualization supplement | complete | `plan/rolling_next_token_visualization_supplement_plan.md` | `reports/rolling_next_token_visualization_guide.md` | `/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/visualization_supplement` |
| Rolling token-block Jacobian follow-up | planned | `plan/rolling_token_block_jacobian_followup_plan.md` | pending | planned: `/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/token_block_jacobian_pilot` |
| Generation-aligned rolling follow-up | planned; method derivation complete | `plan/generation_aligned_rolling_followup_plan.md` | design audit: `reports/rolling_application_scenario_derivation.md`; experiment report pending | planned: `/home/luohaoming/model_feature_experiments/rolling_generation_aligned_followup` |
| Phase 6 Poincaré visualization correction | complete | `plan/phase6_poincare_visualization_correction_plan.md` | `reports/phase6_poincare_projection_audit.md` | reuses `/home/luohaoming/model_feature_experiments/pythia_checkpoint_criticality/visualization_rerun` |
| Three-group single-token/context frequency dynamics | v1.3 four-checkpoint Pilot complete; Main gated by float32 nearby-floor audit | `plan/single_token_frequency_dynamics_plan.md` | `reports/single_token_frequency_dynamics_report.md` | `/home/luohaoming/model_feature_experiments/single_token_frequency_dynamics` |
| Core-paper alignment audit | complete | n/a | `reports/core_paper_alignment_audit_1909_05176.md` | n/a |
| Pythia checkpoint training-to-edge test | planned, download gate | `plan/pythia_checkpoint_edge_test_plan.md` | pending | `/home/luohaoming/model_feature_experiments/pythia_checkpoint_edge_test` |

Policy from 2026-07-12 onward: experimental data and logs are stored under `/home/luohaoming/model_feature_experiments`; repository files contain configs, plans, reports, analysis code, and small summary tables only.
