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
    CTDB_Data_Download_11D.xlsx
    CTDB_Data_Download_15D.xlsx
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
```

Every script:
- writes to `logs/pipeline.log`
- emits console prints for quick traceability
- persists machine-readable outputs (`.parquet`, `.csv`)

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
