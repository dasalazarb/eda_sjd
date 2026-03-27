# CTDB 11D/15D Reproducible EDA + Linkage Pipeline (Python)

This repository implements a **patient-centric backbone pipeline** for:

- `CTDB Data Download 11D.xlsx`
- `CTDB Data Download 15D.xlsx`

Core rule: **merge by `SUBJECT_NUMBER` at patient level**; do **not** perform a raw row-to-row merge between protocols.

## Design goals

1. Exhaustive EDA per source file.
2. Transparent cross-protocol linkage.
3. Deterministic deduplication and conflict resolution.
4. Auditable analytical backbone (`patient_master`, `visits_long`).
5. Objective-specific cohort datasets.
6. Max traceability: persistent logs, audit tables, and report outputs.

## Project structure

```
project/
  data_raw/
    CTDB Data Download 11D.xlsx
    CTDB Data Download 15D.xlsx
  data_intermediate/
  data_analytic/
  reports/
  logs/
  src/
    common.py
    01_ingest.py
    02_profile_individual.py
    03_linkage.py
    04_dedup_rules.py
    05_build_backbone.py
    06_postmerge_eda.py
    07_build_cohorts.py
    08_visit_patterns.py
```

## Run order

```bash
python src/01_ingest.py
python src/02_profile_individual.py
python src/03_linkage.py
python src/04_dedup_rules.py
python src/05_build_backbone.py
python src/06_postmerge_eda.py
python src/07_build_cohorts.py
python src/08_visit_patterns.py
python src/09_interval_collapse_audit.py --collapse no
```

Every script:
- writes to `logs/pipeline.log`
- emits console prints for quick traceability
- persists machine-readable outputs (`.parquet`, `.csv`)


## Header and privacy rules during ingest

- Column names are built as `{category}__{variable}` from the first two header rows in the raw workbooks.
  - Example: `ids__subject_number`.
  - If a category cell is empty, the fallback category is `uncategorized`, keeping the same format.
- During ingest, the pipeline removes any `FIRST_NAME` and `LAST_NAME` fields (after standardization: `first_name`, `last_name`, `*__first_name`, `*__last_name`) from raw-enriched outputs to avoid propagating direct identifiers.

## Script-by-script execution narrative

Each script prints an English, step-by-step console narrative with numbered steps for traceability.

1. `01_ingest.py`
   - Reads each Excel raw file.
   - Builds grouped headers via `build_group_prefixed_columns(...)`.
   - Standardizes names with `standardize_columns(...)`, replaces missing tokens with `replace_empty_with_nan(...)`, parses temporal fields with `parse_datetime_columns(...)`, and removes sensitive name variables with `drop_sensitive_name_columns(...)`.
   - Adds lineage columns (`source_protocol`, `source_file`, `row_id_raw`) and saves raw-enriched outputs.
2. `02_profile_individual.py`
   - Loads both enriched raw datasets.
   - Profiles each dataset using `profile_dataframe(...)`.
   - Saves independent EDA reports and prints shape/key-ID summaries.
3. `03_linkage.py`
   - Builds patient overlap map with `overlap_table(...)`.
   - Builds exact episode candidates using `build_episode_candidates(...)` after canonical column resolution.
   - Saves linkage tables and prints overlap/candidate counts.
4. `04_dedup_rules.py`
   - Applies `deduplicate_within_protocol(...)` on each source.
   - Consolidates kept rows and conflict/audit decisions.
   - Saves deduplicated visits + conflict log and prints dedup summary.
5. `05_build_backbone.py`
   - Builds patient-level aggregates using `build_patient_master(...)`.
   - Saves `patient_master` and `visits_long`.
   - Prints backbone counts.
6. `06_postmerge_eda.py`
   - Computes post-merge quality/readiness metrics from `patient_master` and `visits_long`.
   - Saves `eda_postmerge.csv` and prints metric summary.
7. `07_build_cohorts.py`
   - Derives baseline, longitudinal, and time-to-event cohorts.
   - Saves cohorts and `analysis_readiness.csv`.
   - Prints cohort-level summary counts.
8. `08_visit_patterns.py`
   - Maps visit distribution by `INTERVAL_NAME` and patient-level timeline order.
   - Builds complementary visit-pattern visualizations (swimmer, violin by transition, KDE+hist in log scale, and patient×time heatmap).
   - Saves both plot-ready data tables and rendered figures under `reports/visit_patterns/`.
9. `09_interval_collapse_audit.py`
   - Audits repeated measurements inside `SUBJECT_NUMBER × INTERVAL_NAME` groups.
   - Quantifies whether variables are mostly complementary (e.g., `yes` + `NaN` -> `yes`) or conflicting (e.g., `yes` + `no` -> `yes | no`).
   - Exports temporal-window distribution reports and optionally saves a collapsed one-row-per-group table via `--collapse yes|no`.

## Operational definitions encoded in the pipeline

- Patient unit: `SUBJECT_NUMBER`
- Episode unit: `PATIENT_RECORD_NUMBER`
- Source marker: `source_protocol` (`11D` / `15D`)
- Operational time: `visit_datetime = VISIT_DATE + TIME_24_HOUR`

## Conflict policy (default)

When same episode candidate appears in both protocols:
1. keep non-missing values first
2. if equal, keep one and flag agreement
3. if conflicting, prioritize 11D for longitudinal follow-up (but always log conflict)
4. if anchor-specific analysis is used later, choose nearest record to anchor in cohort scripts

## Outputs

Intermediate examples:
- `data_intermediate/11d_raw_enriched.parquet`
- `data_intermediate/15d_raw_enriched.parquet`
- `data_intermediate/overlap_subjects.parquet`
- `data_intermediate/episode_candidates.parquet`
- `data_intermediate/conflict_log.parquet`

Analytical examples:
- `data_analytic/patient_master.parquet`
- `data_analytic/visits_long.parquet`
- `data_analytic/cohort_baseline.parquet`
- `data_analytic/cohort_longitudinal.parquet`
- `data_analytic/cohort_time_to_event.parquet`

Reports:
- `reports/eda_11d.csv`
- `reports/eda_15d.csv`
- `reports/eda_postmerge.csv`
- `reports/analysis_readiness.csv`
- `reports/eda_unificado.xlsx` (workbook consolidado de EDA por dataset y etapa)
- `reports/visit_patterns/interval_name_map.csv`
- `reports/visit_patterns/swimmer_plot.png`
- `reports/visit_patterns/violin_transition_plot.png`
- `reports/visit_patterns/kde_hist_gapdays_plot.png`
- `reports/visit_patterns/heatmap_patient_time.png`
- `reports/interval_collapse_repeated_groups.csv`
- `reports/interval_collapse_variable_audit.csv`
- `reports/interval_collapse_window_stats.csv`
- `reports/interval_collapse_window_summary.csv`
- `reports/interval_collapse_conflict_examples.csv`
- `data_analytic/visits_long_collapsed_by_interval.parquet` (optional, with `--collapse yes`)

## EDA unificada (`reports/eda_unificado.xlsx`)

La salida unificada de EDA vive en:

- `reports/eda_unificado.xlsx`

### Convención uniforme de nombres de hoja

Cada hoja sigue el patrón:

- `<dataset_base>_<sufijo_eda>`

Donde `dataset_base` representa etapa+dataset con convención homogénea. Ejemplos:

- `01_11D_input`
- `01_11D_output`
- `01_15D_input`
- `01_15D_output`
- `03_overlap_subjects`
- `05_patient_master`
- `07_cohort_baseline`

### Hojas derivadas por dataset

Para **cada** `dataset_base`, el pipeline escribe múltiples hojas con los siguientes sufijos:

- `_summary`
- `_missing`
- `_cat_dist`
- `_date_stats`
- `_visit_dist`

Ejemplo completo para `05_patient_master`:

- `05_patient_master_summary`
- `05_patient_master_missing`
- `05_patient_master_cat_dist`
- `05_patient_master_date_stats`
- `05_patient_master_visit_dist`
