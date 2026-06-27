# Repository inventory — script -> result mapping + results-file catalog

Read-only pre-submission inventory. Engine note: **two recovery pipelines exist; only Pipeline B
(`phase1_recovery.py`) produced results** — Pipeline A (`stepramp.py`/`phase2_validate.py`/
`phase2_recovery_sweep.py`) was never run (`phase2_validation.txt`, `phase2_recovery.csv` absent).

## A. Scripts (purpose -> results written)

| script | purpose | writes |
|---|---|---|
| `phase1_behavior.py` | Phase-1 identifiability gate (behavior only): RT distributions per contrast | phase1_summary.csv |
| `phase1b_engagement.py` | Phase-1b: slow-RT = engaged deliberation; accuracy-by-RT, window pool curation | phase1b_accuracy_by_rt.csv |
| `phase1_recovery.py` | **(a) RECOVERY SWEEP** Pipeline B engine: step-HMM vs ramp-DDM forward-LL + CV recovery map | phase1_recovery_sweep.csv, phase1_recovery_validation.json |
| `phase1_recovery_hardened.py` / `phase1_recovery_nb_grid.py` / `src/rerun_*.py` | Hardening: NB overdispersion + intermediate stepiness; realistic FR×N recovery grid + Poisson-misspecification (R=40) | phase1_recovery_stepiness.csv, phase1_recovery_nb_grid.csv (R=20, Fig S1 note), phase1_recovery_nb_grid_R40.csv, phase1_recovery_poisson_misspec_R40.csv |
| `run_recovery.py / run_hardened.py` | Launchers for phase1_recovery / _hardened (loky-safe) | (none) |
| `phase2_census.py` | Phase-2 census (trimmed spikes): region coverage + poolable-N + FR distribution | phase2_coverage.csv, phase2_session_delib.csv, phase2_census_units.csv, phase2_region_census.csv |
| `phase2_selectivity.py` | **(b) CASCADE** hi-FR -> choice-AUC(perm,FDR) -> movement -> stimulus/prior -> (require leading) = triple | phase2_sel_features{,_full}.csv, phase2_sel_cells{,_full}.csv, phase2_sel_region{,_full}.csv |
| `phase2_choice_vs_stim.py` | Choice-vs-stimulus on decorrelated (0%+error) trials; pooled error effect | phase2_choicestim_cells{,_full}.csv, phase2_choicestim_region{,_full}.csv |
| `phase2_population_decode.py` | **(d) DECODE** pseudo-population L2 decode of stim-indep choice (within-fold residual) | phase2_population_decode.csv, phase2_population_decode_curve.csv |
| `phase2_mrn_recovery_preflight.py` | Sec 3.5 gate: shared-latent MRN population step-vs-ramp recovery pre-flight | phase2_mrn_recovery_preflight.csv, phase2_mrn_recovery_headline.csv |
| `run_mrn_preflight.py` | Launcher for phase2_mrn_recovery_preflight | (none) |
| `steinmetz_coverage.py` | **(e) STEINMETZ** coverage feasibility (binned NMA release) | steinmetz_coverage.csv, steinmetz_coverage_sessions.csv |
| `steinmetz_adapter.py` | Steinmetz binned -> IBL feature schema adapter (+smoke PSTH) | steinmetz_features.csv (gitignored, ~15MB) |
| `steinmetz_replicate.py` | **(e) STEINMETZ** replicate cascade + choice-vs-stim + decode + 0.57 gate | steinmetz_selectivity_region.csv, steinmetz_choicestim_region.csv, steinmetz_decode.csv, steinmetz_gate.csv, steinmetz_vs_ibl.csv |
| `steinmetz_population_preflight.py` | Steinmetz population step-vs-ramp recovery pre-flight (reuses 3.5 engine) | steinmetz_population_preflight.csv, steinmetz_population_aggregation.csv |
| `run_steinmetz_preflight.py` | Launcher for steinmetz_population_preflight | (none) |
| `audit_corrections.py` | **(c) BOOTSTRAP EFFECTS** + FDR cascade + proper decode perm + gate debias (audit fixes) | audit_pooled_effects.csv, audit_cascade_fdr.csv, audit_gate_debias.csv, audit_decode_2000perm.csv |
| `audit_realtrial_decode.py` | **(d) DECODE/decisive** real-trial per-session CV decode + nonlinear movement control | audit_realtrial_decode.csv |
| `audit_ramp_validate.py` | Ramp-model validation: brute-force + hmmlearn + two-engine + sigma/K robustness | ramp_validation_robustness.csv, ramp_validation_reconcile.csv |
| `run_ramp_validate.py` | Launcher for audit_ramp_validate | (none) |
| `stepramp.py` | Pipeline A engine (cell-integration ramp; brute-force self-test). UNUSED for headline | (none; self-test only) |
| `phase2_validate.py` | Pipeline A validation (hmmlearn + grid-convergence + Part-A2). **NEVER RUN** | results/phase2_validation.txt (ABSENT) |
| `phase2_recovery_sweep.py` | Pipeline A recovery sweep (M=25,k-fold,3:1,sigma0.7). **NEVER RUN** | results/phase2_recovery.csv (ABSENT) |
| `ibl_one.py` | Shared Open-Alyx ONE connection helper | (none) |
| `list_sessions.py / psth_smoketest.py` | IBL public-data access smoke tests | (none tracked) |

## B. The five headline-analysis -> script mappings

| analysis | script / function | result file(s) |
|---|---|---|
| (a) recovery map / step-vs-ramp identifiability (Fig 2 realistic, R=40; Fig S1 clean-Poisson best case) | `phase1_recovery_nb_grid.py` + `src/rerun_*.py` (shared engine `ramp_hmm`/`step_hmm`/`model_loglik`; `phase1_recovery.py::run_sweep` for the Poisson best case) | `phase1_recovery_nb_grid_R40.csv`, `phase1_recovery_poisson_misspec_R40.csv` (Fig 2, R=40); `phase1_recovery_sweep.csv` (Fig S1), `phase1_recovery_validation.json` |
| (b) confound/selectivity cascade (hiFR->choice->move->stim/prior->+leading->triple) | `phase2_selectivity.py::run_analyze`/`run_report` | `phase2_sel_cells_full.csv`, `phase2_sel_region_full.csv` |
| (c) cell-clustered bootstrap effect sizes | `audit_corrections.py::pooled_effect` (orig: `phase2_choice_vs_stim.py::pooled_error_effect`) | `audit_pooled_effects.csv` |
| (d) decode incl. nonlinear movement control | `audit_realtrial_decode.py::cv_decode` (pseudo: `phase2_population_decode.py`; proper-perm: `audit_corrections.py::decode_2000`) | `audit_realtrial_decode.csv`, `audit_decode_2000perm.csv`, `phase2_population_decode.csv` |
| (e) Steinmetz cross-dataset replication | `steinmetz_replicate.py::main` (+ `steinmetz_adapter.py`, `steinmetz_coverage.py`) | `steinmetz_{selectivity_region,choicestim_region,decode,gate,vs_ibl}.csv` |

## C. results/ files (columns + first 3 rows)


**`results/audit_cascade_fdr.csv`** (465B)  
cols: `region, hiFR, choice_raw, choice_fdr, leading_raw, leading_fdr, move_surv_raw, move_surv_fdr, stim_surv_raw, stim_surv_fdr, triple_raw, triple_fdr, dataset`  
- `['GRN', '184', '22', '7', '12', '2', '12', '0', '5', '0', '2', '0', 'IBL']`
- `['MRN', '777', '143', '47', '97', '14', '64', '0', '62', '0', '29', '0', 'IBL']`
- `['SNr', '47', '19', '9', '17', '4', '6', '0', '11', '0', '3', '0', 'IBL']`

**`results/audit_decode_2000perm.csv`** (327B)  
cols: `auc, p, null_mean, null_sd, null_p95, n_cells, n_perm, R_reps, label`  
- `['0.8648194444444444', '0.0024937655860349127', '0.48284789062499994', '0.1456542534684899', '0.7340921874999999', '243', '400', '20', 'IBL MRN error-only']`
- `['0.9131944444444444', '0.007481296758104738', '0.5092054340277778', '0.17881991729484323', '0.8091369791666666', '222', '400', '20', 'Steinmetz MRN equal-only']`

**`results/audit_gate_debias.csv`** (438B)  
cols: `gscale, clean_percell_auc, debiased_percell_auc, eff_sd`  
- `['0.015', '0.5286536242016777', '0.5241903213751868', '0.09749414871898904']`
- `['0.022', '0.5403296770533548', '0.5338874813153961', '0.13600924361846609']`
- `['0.032', '0.551714628354498', '0.5344726083707025', '0.17516600946694838']`

**`results/audit_pooled_effects.csv`** (454B)  
cols: `label, eff, ci_lo, ci_hi, n_cells, n_trials, mwu_p_descriptive, sig_by_ci`  
- `['IBL MRN (error trials)', '0.0588658234539174', '0.0012809847007796639', '0.11663177056320785', '356', '6027', '0.03376792606846113', 'True']`
- `['Steinmetz MRN (equal-contrast)', '0.09455165690529829', '0.04971958820739415', '0.13902359242321152', '223', '10139', '5.591807921292445e-07', 'True']`
- `['IBL ALL-REGION (error) [context only]', '0.09485532960124732', '0.059885161891934824', '0.13061883546590708', '815', '13186', '1.8904923098043992e-07', 'True']`

**`results/audit_realtrial_decode.csv`** (2570B)  
cols: `dataset, region, mode, n_sessions, mean_cells, mean_trials, cv_auc, perm_p, null_mean, null_p95, frac_sess_auc_gt55`  
- `['IBL', 'MRN', 'none', '28', '19.75', '30.071428571428573', '0.6149432031323208', '0.0024937655860349127', '0.48726417861043336', '0.5358935754153101', '0.6428571428571429']`
- `['IBL', 'MRN', 'linear', '28', '19.75', '30.071428571428573', '0.5805305473059421', '0.0024937655860349127', '0.4909352131448137', '0.5357643087005689', '0.6785714285714286']`
- `['IBL', 'MRN', 'expanded', '28', '19.75', '30.071428571428573', '0.5279040684148412', '0.057356608478802994', '0.49862767290389165', '0.5289185134792893', '0.35714285714285715']`

**`results/number_trace.csv`** (4363B)  
cols: `claim_id, manuscript_value, source_file, file_value, match, note`  
- `['R1', 'recovery >80% needs N>=160-320 pooled at FR>=20Hz (sigma=0.4)', 'results/phase1_recovery_sweep.csv', 'FR20: N160=84%, N320=85%; FR40: N80=85%; FR10: N320=81% -> >=80% first at N>=160/FR>=20 (or N>=80/FR>=40)', 'Y', 'sweep generator uses GEN_SIGMA=0.4; mean over step+ramp']`
- `['R2', 'sigma=0.4 recovery: N160/FR20 ~83%, N320 ~89%', 'results/ramp_validation_robustness.csv', 'sigma=0.4,K=50: N160/FR20=83%, N320/FR20=89%', 'Y', 'matches the sigma-validation run; original phase1_recovery_sweep.csv gives 84%/85% (consistent within MC noise)']`
- `['R3', 'sigma=1.0 true-ramp recovery -> ~chance (50%)', 'results/ramp_validation_robustness.csv', 'true-ramp,sigma=1.0,K=50,FR20: N160=50%, N320=57%, N40=38%', 'Y', 'N160/FR20 = exactly 50% (chance)']`

**`results/phase1_recovery_stepiness.csv`** (11705B)  
cols: `stepiness, N, fr, step_won`  
- `['0.0', '160', '20.0', '0']`
- `['0.0', '160', '20.0', '0']`
- `['0.0', '160', '20.0', '0']`

**`results/phase1_recovery_sweep.csv`** (121477B)  
cols: `true, N, fr, pred, correct, d_cvll`  
- `['step', '20', '2.0', 'step', '1', '0.060861184198429896']`
- `['step', '20', '2.0', 'ramp', '0', '-0.1770550438555034']`
- `['step', '20', '2.0', 'step', '1', '7.739097765041784']`

**`results/phase1_recovery_validation.json`** (709B, json) keys: `['hmmlearn_check', 'grid_convergence', 'part_a2', 'part_a2_overall']`

**`results/phase1_summary.csv`** (2882B)  
cols: `metric, contrast_pct, n, median_ms, p10_ms, p25_ms, p75_ms, p90_ms, IQR_ms, IDR_ms, QCD, CV, frac_gt_200ms, frac_gt_500ms`  
- `['reaction_time', 'ALL', '42836', '177.85206087205907', '47.32403901039106', '79.92450221360059', '847.9823152412109', '3406.7241860994955', '768.0578130276103', '3359.4001470891044', '0.8277316197916762', '2.7072148517718873', '0.46243813614716595', '0.29802035670930993']`
- `['reaction_time', '0.0', '4592', '219.1110494951758', '47.092755849780595', '88.42019555902425', '1179.6376869645578', '3574.4015600381854', '1091.2174914055336', '3527.308804188405', '0.8605423352078253', '2.569210517639398', '0.5237369337979094', '0.3471254355400697']`
- `['reaction_time', '6.25', '5793', '186.55180163159457', '45.221186827075144', '82.62619912926539', '619.5008980037073', '3139.203008046391', '536.874698874442', '3093.981821219316', '0.7646403351568208', '3.009564038810757', '0.4736751251510444', '0.2753322976005524']`

**`results/phase1b_accuracy_by_rt.csv`** (5471B)  
cols: `metric, contrast_pct, rt_bin, n, accuracy, wilson_lo, wilson_hi`  
- `['reaction_time', '0.0', '<150', '1738', '0.6271576524741082', '0.6041662306336263', '0.6495881865303684']`
- `['reaction_time', '0.0', '150-300', '897', '0.5529542920847269', '0.5202599173228905', '0.5851970242748333']`
- `['reaction_time', '0.0', '300-500', '363', '0.6005509641873278', '0.5493656163680096', '0.6496303535547034']`

**`results/phase2_census_units.csv`** (195295B)  
cols: `region, pid, eid, cluster, fr_overall, fr_window, auc, selectivity, n_delib, n_left, n_right`  
- `['GRN', 'cc72fdb7-92e8-47e6-9cea-94f27c0da2d8', 'c958919c-2e75-435d-845d-5b62190b520e', '0', '29.36302320031947', '66.60550830865989', '0.56', '0.06000000000000005', '62', '37', '25']`
- `['GRN', 'cc72fdb7-92e8-47e6-9cea-94f27c0da2d8', 'c958919c-2e75-435d-845d-5b62190b520e', '3', '6.7722695701687785', '12.855156270343947', '0.6216216216216215', '0.12162162162162149', '62', '37', '25']`
- `['GRN', 'cc72fdb7-92e8-47e6-9cea-94f27c0da2d8', 'c958919c-2e75-435d-845d-5b62190b520e', '4', '20.73830199511263', '16.98906470534216', '0.5005405405405405', '0.000540540540540535', '62', '37', '25']`

**`results/phase2_choicestim_cells.csv`** (86427B)  
cols: `cell, region, eid, n_trials, n_err, n0, fr_window, choice_beta, p_choice, corr_diff, err_diff, q_choice, sig_choice_raw, sig_choice_fdr, follows_choice`  
- `['0143d3fe-79c2-4922-8332-62c3e4e0ba85:160', 'MRN', '22e04698-b974-4805-b241-3b547dbf37bf', '37', '5', '15', '13.679727324135914', '0.4091816100968866', '0.6363636363636364', '', '', '0.9027759126119781', 'False', 'False', 'False']`
- `['0143d3fe-79c2-4922-8332-62c3e4e0ba85:183', 'MRN', '22e04698-b974-4805-b241-3b547dbf37bf', '37', '5', '15', '14.782704559665994', '-0.22429880848013584', '0.7262737262737263', '', '', '0.9043456543456544', 'False', 'False', 'False']`
- `['0143d3fe-79c2-4922-8332-62c3e4e0ba85:188', 'MRN', '22e04698-b974-4805-b241-3b547dbf37bf', '37', '5', '15', '17.479509551541764', '0.643766588150321', '0.5364635364635365', '', '', '0.8850013622740894', 'False', 'False', 'False']`

**`results/phase2_choicestim_cells_full.csv`** (326750B)  
cols: `cell, region, eid, n_trials, n_err, n0, fr_window, choice_beta, p_choice, corr_diff, err_diff, q_choice, sig_choice_raw, sig_choice_fdr, follows_choice`  
- `['00a824c0-e060-495f-9ebc-79c82fef4c67:144', 'SCm', 'fa704052-147e-46f6-b190-a65b837e605e', '164', '28', '36', '15.48487752369754', '0.7083783320891106', '0.16183816183816183', '1.593538820724838', '-0.2833301253674101', '0.5634793472052573', 'False', 'False', 'False']`
- `['00a824c0-e060-495f-9ebc-79c82fef4c67:215', 'SCm', 'fa704052-147e-46f6-b190-a65b837e605e', '164', '28', '36', '29.54882732643965', '-0.7530058528290624', '0.18681318681318682', '1.618023292955023', '-7.007278197554474', '0.5861943881361357', 'False', 'False', 'False']`
- `['00a824c0-e060-495f-9ebc-79c82fef4c67:220', 'SCm', 'fa704052-147e-46f6-b190-a65b837e605e', '164', '28', '36', '39.45679981700536', '-1.005859440659473', '0.08591408591408592', '-2.2599351502694347', '1.4131036157264063', '0.4658965195877948', 'False', 'False', 'False']`

**`results/phase2_choicestim_region.csv`** (227B)  
cols: `region, cells, err_testable, follow_choice, follow_stim, choice_sig_raw, choice_sig_fdr, pooledN_choice_raw`  
- `['GRN', '146', '66', '40', '26', '15', '0', '728']`
- `['MRN', '142', '48', '26', '22', '12', '0', '811']`
- `['SNr', '17', '7', '3', '4', '4', '0', '604']`

**`results/phase2_choicestim_region_full.csv`** (248B)  
cols: `region, cells, err_testable, follow_choice, follow_stim, choice_sig_raw, choice_sig_fdr, pooledN_choice_raw`  
- `['GRN', '184', '86', '49', '37', '19', '0', '985']`
- `['MRN', '777', '274', '153', '121', '102', '18', '7171']`
- `['SNr', '47', '31', '17', '14', '13', '0', '1173']`

**`results/phase2_coverage.csv`** (252B)  
cols: `region, insertions, sessions, poolable_N, median_session_N`  
- `['SCm', '78', '68', '3257', '37']`
- `['MRN', '140', '120', '5509', '36']`
- `['SNr', '22', '22', '963', '27']`

**`results/phase2_mrn_recovery_headline.csv`** (461B)  
cols: `label, N_cells, N_trials, gscale, recovery, se`  
- `['easy_sanity', '80.0', '120.0', '0.2', '1.0', '0.0']`
- `['null_nocoupling', '243.0', '200.0', '0.0', '0.5', '0.04564354645876384']`
- `['mrn_Nt25', '243.0', '25.0', '0.015', '0.6666666666666666', '0.04303314829119352']`

**`results/phase2_mrn_recovery_preflight.csv`** (2399B)  
cols: `N_cells, N_trials, fano, gscale, n, recovery, se, rec_step, rec_ramp`  
- `['243', '200', '2.0', '0.0', '120', '0.5', '0.04564354645876384', '0.0', '1.0']`
- `['10', '25', '2.0', '0.015', '48', '0.6041666666666666', '0.07058524774394487', '0.5', '0.7083333333333334']`
- `['15', '25', '2.0', '0.015', '48', '0.5416666666666666', '0.07191776103197216', '0.5', '0.5833333333333334']`

**`results/phase2_population_decode.csv`** (1407B)  
cols: `region, n_cells, n_cells_err, decode_auc, ci_lo, ci_hi, null_auc, null_hi, perm_p, err_auc, err_ci_lo, err_ci_hi, err_null, err_p, percell_mean_auc, percell_p95_auc`  
- `['GRN', '130', '56', '0.1669878472222222', '0.09694878472222222', '0.24401519097222216', '0.4924618055555555', '0.679302083333333', '1.0', '0.6729982638888888', '0.5554943142361111', '0.7884469184027777', '0.47997055844907405', '0.024691358024691357', '0.5085919497169497', '0.8212499999999999']`
- `['MRN', '669', '243', '0.7071996527777777', '0.6226027343749999', '0.7761333767361112', '0.5019127604166667', '0.6611945891203702', '0.012345679012345678', '0.7944253472222222', '0.7133196180555554', '0.8684656250000001', '0.5169509548611112', '0.012345679012345678', '0.5068538032897418', '0.7714285714285714']`
- `['SNr', '41', '29', '0.5890000000000001', '0.5207533420138888', '0.6558746961805556', '0.4917800925925926', '0.6096338252314809', '0.06172839506172839', '0.33363541666666663', '0.26499266493055557', '0.40660889756944446', '0.5063865740740742', '0.9382716049382716', '0.5358390305493965', '0.725']`

**`results/phase2_population_decode_curve.csv`** (1869B)  
cols: `region, K, auc, auc_sd`  
- `['GRN', '1', '0.4619097222222222', '0.2777357310480546']`
- `['GRN', '2', '0.4602361111111111', '0.34294117273265234']`
- `['GRN', '5', '0.47592592592592603', '0.34088822385683276']`

**`results/phase2_region_census.csv`** (310B)  
cols: `region, poolable_N, sessions, median_session_N, sampled_insertions, good_units, good_hiFR20, sel_hiFR20_obs, sel_hiFR20_excess, est_sel_hiFR20_full`  
- `['GRN', '678', '17', '30', '8', '300', '94', '22', '5.9', '12']`
- `['MRN', '5509', '120', '36', '12', '336', '91', '15', '5.6', '66']`
- `['SNr', '963', '22', '27', '5', '35', '12', '4', '3.4', '15']`

**`results/phase2_sel_cells.csv`** (154103B)  
cols: `cell, region, eid, n_left, n_right, fr_window, auc_delib, p_delib, auc_early, p_early, auc_peri, p_peri, auc_move, p_move, auc_stim, p_stim, q_delib, sig_fdr, sig_raw, move_survive, stim_survive, leading, locked, triple`  
- `['0143d3fe-79c2-4922-8332-62c3e4e0ba85:160', 'MRN', '22e04698-b974-4805-b241-3b547dbf37bf', '10', '27', '13.679727324135914', '0.42962962962962964', '0.5367316341829086', '0.4925925925925926', '0.9685157421289355', '0.4203703703703704', '0.4777611194402799', '0.42962962962962964', '0.56071964017991', '0.4703703703703704', '0.7976011994002998', '0.8611738906666071', 'False', 'False', 'False', 'False', 'False', 'False', 'False']`
- `['0143d3fe-79c2-4922-8332-62c3e4e0ba85:183', 'MRN', '22e04698-b974-4805-b241-3b547dbf37bf', '10', '27', '14.782704559665994', '0.44814814814814813', '0.663168415792104', '0.5185185185185185', '0.8725637181409296', '0.30925925925925923', '0.079960019990005', '0.44074074074074077', '0.6116941529235382', '0.5592592592592592', '0.6036981509245377', '0.9126218140929536', 'False', 'False', 'False', 'False', 'False', 'False', 'False']`
- `['0143d3fe-79c2-4922-8332-62c3e4e0ba85:188', 'MRN', '22e04698-b974-4805-b241-3b547dbf37bf', '10', '27', '17.479509551541764', '0.34074074074074073', '0.15242378810594703', '0.34074074074074073', '0.15642178910544727', '0.37777777777777777', '0.2753623188405797', '0.32592592592592595', '0.11594202898550725', '0.4740740740740741', '0.824087956021989', '0.5468404009377426', 'False', 'False', 'False', 'False', 'False', 'False', 'False']`

**`results/phase2_sel_cells_full.csv`** (599011B)  
cols: `cell, region, eid, n_left, n_right, fr_window, has_dlc, auc_delib, p_delib, auc_early, p_early, auc_peri, p_peri, p_move_wheel, p_move_full, p_stim, q_delib, sig_fdr, sig_raw, p_move, move_survive, move_survive_wheel, stim_survive, leading, locked, triple`  
- `['00a824c0-e060-495f-9ebc-79c82fef4c67:144', 'SCm', 'fa704052-147e-46f6-b190-a65b837e605e', '75', '89', '15.48487752369754', 'True', '0.44719101123595506', '0.23738130934532733', '0.45123595505617975', '0.28835582208895555', '0.5061423220973783', '0.881559220389805', '0.20689655172413793', '0.5112443778110944', '0.32733633183408295', '0.5809029020233085', 'False', 'False', '0.5112443778110944', 'False', 'False', 'False', 'False', 'False', 'False']`
- `['00a824c0-e060-495f-9ebc-79c82fef4c67:215', 'SCm', 'fa704052-147e-46f6-b190-a65b837e605e', '75', '89', '29.54882732643965', 'True', '0.5069662921348315', '0.8920539730134932', '0.5402247191011236', '0.37031484257871067', '0.5113857677902621', '0.8010994502748626', '0.872063968015992', '0.4427786106946527', '0.5552223888055972', '0.9751197275856', 'False', 'False', '0.4427786106946527', 'False', 'False', 'False', 'False', 'False', 'False']`
- `['00a824c0-e060-495f-9ebc-79c82fef4c67:220', 'SCm', 'fa704052-147e-46f6-b190-a65b837e605e', '75', '89', '39.45679981700536', 'True', '0.5695880149812734', '0.12143928035982009', '0.5610486891385768', '0.19140429785107446', '0.5943071161048689', '0.043478260869565216', '0.11294352823588207', '0.3713143428285857', '0.551224387806097', '0.42390821756503727', 'False', 'False', '0.3713143428285857', 'False', 'False', 'False', 'False', 'False', 'False']`

**`results/phase2_sel_features.csv`** (4805202B)  
cols: `cell, region, eid, fr_overall, fr_window, choice, absc, signed, pL, rate_delib, rate_early, rate_peri, wheel_speed, wheel_disp, paw_speed`  
- `['cc72fdb7-92e8-47e6-9cea-94f27c0da2d8:0', 'GRN', 'c958919c-2e75-435d-845d-5b62190b520e', '29.36302320031947', '66.60550830865989', '-1', '0.0', '0.0', '0.5', '117.29011338823292', '119.83015755277303', '44.99999999998977', '0.07256574017918638', '-1.1722589192686428e-05', '']`
- `['cc72fdb7-92e8-47e6-9cea-94f27c0da2d8:0', 'GRN', 'c958919c-2e75-435d-845d-5b62190b520e', '29.36302320031947', '66.60550830865989', '-1', '0.0', '0.0', '0.8', '87.57891914498248', '97.983755985649', '9.999999999997726', '0.2817236872734769', '-0.001588141828932521', '']`
- `['cc72fdb7-92e8-47e6-9cea-94f27c0da2d8:0', 'GRN', 'c958919c-2e75-435d-845d-5b62190b520e', '29.36302320031947', '66.60550830865989', '1', '6.25', '-0.0625', '0.8', '38.37974393399782', '37.35182393894516', '0.0', '0.33186152639883704', '0.07410226102467732', '']`

**`results/phase2_sel_features_full.csv`** (26758471B)  
cols: `cell, region, eid, fr_overall, fr_window, choice, absc, signed, pL, rate_delib, rate_early, rate_peri, wheel_speed, wheel_disp, paw_speed, nose_speed`  
- `['cc72fdb7-92e8-47e6-9cea-94f27c0da2d8:0', 'GRN', 'c958919c-2e75-435d-845d-5b62190b520e', '29.36302320031947', '66.60550830865989', '-1', '0.0', '0.0', '0.5', '117.29011338823292', '119.83015755277303', '44.99999999998977', '0.07256574017918638', '-1.1722589192686428e-05', '8.14447185319241', '9.92249199862312']`
- `['cc72fdb7-92e8-47e6-9cea-94f27c0da2d8:0', 'GRN', 'c958919c-2e75-435d-845d-5b62190b520e', '29.36302320031947', '66.60550830865989', '-1', '0.0', '0.0', '0.8', '87.57891914498248', '97.983755985649', '9.999999999997726', '0.2817236872734769', '-0.001588141828932521', '18.32892256291373', '18.422877469621707']`
- `['cc72fdb7-92e8-47e6-9cea-94f27c0da2d8:0', 'GRN', 'c958919c-2e75-435d-845d-5b62190b520e', '29.36302320031947', '66.60550830865989', '1', '6.25', '-0.0625', '0.8', '38.37974393399782', '37.35182393894516', '0.0', '0.33186152639883704', '0.07410226102467732', '113.95326446420103', '28.721471025045123']`

**`results/phase2_sel_region.csv`** (287B)  
cols: `region, raw_hiFR, choice_raw, choice_fdr, leading, locked, move_surv, stim_surv, triple, sessions, pooledN_leading, pooledN_triple`  
- `['GRN', '146', '21', '4', '12', '5', '14', '4', '2', '7', '624', '82']`
- `['MRN', '142', '27', '0', '15', '9', '13', '8', '3', '11', '1134', '162']`
- `['SNr', '17', '6', '1', '6', '0', '4', '2', '2', '4', '809', '234']`

**`results/phase2_sel_region_full.csv`** (379B)  
cols: `region, raw_hiFR, choice_raw, choice_fdr, leading, locked, move_surv_wheel, move_surv, stim_surv, triple, dlc_cells, sessions, pooledN_leading, pooledN_triple`  
- `['GRN', '184', '22', '7', '12', '5', '15', '12', '5', '2', '173', '12', '624', '82']`
- `['MRN', '777', '143', '47', '97', '27', '77', '64', '62', '29', '737', '61', '7914', '2147']`
- `['SNr', '47', '19', '9', '17', '1', '8', '6', '11', '3', '45', '11', '1554', '309']`

**`results/phase2_session_delib.csv`** (8582B)  
cols: `eid, n_delib, n_left, n_right`  
- `['037d75ca-c90a-43f2-aca6-e86611916779', '49', '22', '27']`
- `['03cf52f6-fba6-4743-a42e-dd1ac3072343', '24', '17', '7']`
- `['03d9a098-07bf-4765-88b7-85f8d8f620cc', '9', '2', '7']`

**`results/ramp_validation_reconcile.csv`** (385B)  
cols: `K, transmat_maxdiff, ll_B, ll_A, ll_reldiff`  
- `['10', '0.050830007462897286', '-1010.6418644634723', '-1010.9906719679789', '0.00034513462856769874']`
- `['25', '0.22053749365847536', '-1049.7316997323019', '-1050.1135168957778', '0.0003637283351291698']`
- `['50', '0.37965047280722786', '-990.1597798828626', '-989.9003026307977', '0.0002620559401994922']`

**`results/ramp_validation_robustness.csv`** (39867B)  
cols: `true, N, fr, sigma, K, correct`  
- `['step', '40', '20', '0.2', '50', '1']`
- `['step', '40', '20', '0.2', '50', '1']`
- `['step', '40', '20', '0.2', '50', '0']`

**`results/steinmetz_choicestim_region.csv`** (419B)  
cols: `region, equal_testable, transfer_choice, no_transfer, choice_sig_raw, choice_sig_fdr, pooled_eff_sd, eff_lo, eff_hi, pooled_p`  
- `['MRN', '223', '142', '81', '17', '0', '0.09455165690529829', '0.05065554938621854', '0.13829593743584756', '5.591807921292445e-07']`
- `['SC', '168', '104', '64', '4', '0', '0.12330586175084002', '0.0651420340776344', '0.1834259328039204', '3.0108736571230814e-05']`
- `['SNr', '79', '46', '33', '5', '0', '0.07406879635373459', '-0.013142850016520461', '0.1622701337825663', '0.0227032003785972']`

**`results/steinmetz_coverage.csv`** (562B)  
cols: `region, target, total_cells, hiFR_cells, n_sessions, max_simult, median_simult, max_simult_hiFR, median_simult_hiFR`  
- `['MRN', 'True', '1016', '239', '11', '217', '72.0', '45', '20.0']`
- `['SCm', 'True', '356', '55', '7', '87', '48.0', '14', '7.0']`
- `['SNr', 'True', '306', '75', '4', '129', '81.0', '40', '13.5']`

**`results/steinmetz_coverage_sessions.csv`** (3304B)  
cols: `sess, mouse, date, n_neurons, n_regions, MRN_cells, SCm_cells, SNr_cells, GRN_cells, IRN_cells, MRN_hiFR, SCm_hiFR, SNr_hiFR, GRN_hiFR, IRN_hiFR, n_trials, n_go, n_nogo, n_err, n_equal, n_zero, n_decorr, has_wheel, has_face, has_pupil`  
- `['0', 'Cori', '2016-12-14', '734', '8', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '214', '140', '74', '33', '22', '12', '37', '1', '1', '1']`
- `['1', 'Cori', '2016-12-17', '1070', '5', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '251', '157', '94', '45', '39', '19', '54', '1', '1', '1']`
- `['2', 'Cori', '2016-12-18', '619', '11', '41', '0', '0', '0', '0', '20', '0', '0', '0', '0', '228', '151', '77', '47', '50', '29', '59', '1', '1', '1']`

**`results/steinmetz_decode.csv`** (411B)  
cols: `region, n_cells, n_cells_equal, decode_auc, perm_p, equal_auc, equal_p, percell_mean_auc`  
- `['MRN', '223', '222', '0.8041493055555555', '0.012345679012345678', '0.9221406249999999', '0.012345679012345678', '0.45779867396534063']`
- `['SC', '153', '126', '0.870921875', '0.012345679012345678', '0.8761006944444445', '0.012345679012345678', '0.5042302823572665']`
- `['SNr', '79', '79', '0.6691736111111112', '0.012345679012345678', '0.7350850694444444', '0.012345679012345678', '0.49428481012658215']`

**`results/steinmetz_features.csv`** (14895422B)  
cols: `cell, region, eid, fr_overall, fr_window, choice, absc, signed, pL, feedback, rate_delib, rate_early, rate_peri, wheel_speed, wheel_disp, paw_speed, nose_speed`  
- `['Cori_2016-12-18:349', 'MRN', 'Cori_2016-12-18', '8.880701754385965', '12.97547408661884', '1', '0.5', '-0.5', '0.5', '1', '13.043478260869565', '7.692307692307692', '20.0', '0.5217391304347826', '0.12', '1.1817052366705234', '0.17236056271364525']`
- `['Cori_2016-12-18:349', 'MRN', 'Cori_2016-12-18', '8.880701754385965', '12.97547408661884', '1', '1.0', '-1.0', '0.5', '1', '20.0', '20.0', '60.0', '1.1333333333333333', '0.17', '2.1857424187982724', '0.15828985470311166']`
- `['Cori_2016-12-18:349', 'MRN', 'Cori_2016-12-18', '8.880701754385965', '12.97547408661884', '-1', '1.0', '1.0', '0.5', '1', '60.0', '', '70.0', '0.8', '0.08', '1.453100931706723', '0.12989484622804234']`

**`results/steinmetz_gate.csv`** (221B)  
cols: `metric, steinmetz, bar, ibl`  
- `['per-cell AUC (raw, equal-contrast)', '0.6025365198679556', '0.57', '0.53']`
- `['per-cell AUC (finite-sample debiased)', '0.5348604713896887', '0.57', '0.53']`
- `['pooled decision effect (SD)', '0.09455165690529829', '0.24', '0.085']`

**`results/steinmetz_population_aggregation.csv`** (1058B)  
cols: `gscale, percell_auc, agg_majority, agg_pooled, agg_maj_step, agg_maj_ramp, best_single, k1, k3, k5, k7, k11`  
- `['0.015', '0.5286536242016777', '0.51', '0.5433333333333333', '0.2966666666666667', '0.7233333333333334', '0.6', '0.495', '0.4866666666666667', '0.47', '0.5466666666666666', '0.5416666666666666']`
- `['0.022', '0.5403296770533548', '0.6283333333333333', '0.6666666666666666', '0.4033333333333333', '0.8533333333333334', '0.66', '0.5016666666666667', '0.5983333333333334', '0.6483333333333333', '0.58', '0.6083333333333333']`
- `['0.032', '0.551714628354498', '0.78', '0.8683333333333333', '0.7233333333333334', '0.8366666666666667', '0.69', '0.6766666666666666', '0.7416666666666667', '0.7483333333333333', '0.7666666666666667', '0.7933333333333333']`

**`results/steinmetz_population_preflight.csv`** (6891B)  
cols: `label, N_cells, N_trials, gscale, percell_auc, recovery, se, n, rec_step, rec_ramp`  
- `['Cori_2016-12-18', '20', '59', '0.015', '0.5286536242016777', '0.57', '0.04950757517794625', '100', '0.42', '0.72']`
- `['Cori_2016-12-18', '20', '59', '0.022', '0.5403296770533548', '0.55', '0.049749371855330994', '100', '0.44', '0.66']`
- `['Cori_2016-12-18', '20', '59', '0.032', '0.551714628354498', '0.63', '0.048280430818293245', '100', '0.6', '0.66']`

**`results/steinmetz_selectivity_region.csv`** (178B)  
cols: `region, raw_hiFR, choice_raw, choice_fdr, leading, locked, move_surv, stim_surv, triple, sessions`  
- `['MRN', '223', '86', '60', '66', '13', '51', '7', '4', '11']`
- `['SC', '168', '64', '45', '52', '6', '35', '4', '4', '11']`
- `['SNr', '79', '21', '13', '16', '3', '14', '1', '0', '4']`

**`results/steinmetz_vs_ibl.csv`** (443B)  
cols: `region, ibl_triple, stein_triple, ibl_choice_fdr, stein_choice_fdr, ibl_decode_strict, stein_decode_equal, stein_decode_decorr, stein_pooled_sd, stein_pooled_p`  
- `['MRN', '29', '4', '47', '60', '0.79', '0.9221406249999999', '0.8041493055555555', '0.09455165690529829', '5.591807921292445e-07']`
- `['SC', '17', '4', '27', '45', '0.37', '0.8761006944444445', '0.870921875', '0.12330586175084002', '3.0108736571230814e-05']`
- `['SNr', '3', '0', '9', '13', '0.33', '0.7350850694444444', '0.6691736111111112', '0.07406879635373459', '0.0227032003785972']`
