# Technical Specification for Python Code Generation
## Exhaustive EDA, Linkage, Deduplication, and Analytical Dataset Construction

**Sources:** `CTDB Data Download 11D.xlsx` and `CTDB Data Download 15D.xlsx`.

**Purpose:** Functional specification for a reproducible Python pipeline with maximum logging and auditability.

### Guiding principle
Primary merge must be by `SUBJECT_NUMBER` at patient level. Avoid raw row-to-row protocol merge.

### Operational definitions
- Patient level: `SUBJECT_NUMBER`
- Episode level: `PATIENT_RECORD_NUMBER`
- Source marker: `source_protocol` (`11D`, `15D`)
- Time marker: `visit_datetime = VISIT_DATE + TIME_24_HOUR`

### Recommended phases
1. Ingestion and minimal standardization
2. Per-file EDA (11D and 15D separately)
3. Cross-protocol linkage
4. Deduplication and conflict handling
5. Backbone build (`patient_master`, `visits_long`)
6. Post-merge EDA
7. Objective-specific cohort generation

### Required auditability controls
- Keep `row_id_raw` and `source_file` in all transformations.
- Never drop conflicts silently; flag and quantify them.
- Save intermediate outputs for every phase.
- Use unified logs with timestamps (`logs/pipeline.log`).
- Keep human-readable print summaries and machine-readable reports.

### Conflict handling defaults
- Prefer non-missing over missing.
- If equal values: keep one, log agreement.
- If conflicting values: prioritize 11D for longitudinal follow-up.
- Always store conflict flags and chosen source.

### Outputs
- Intermediate: overlap, episode candidates, dedup outputs, conflict log.
- Analytic: patient master + visit-long + baseline/longitudinal/time-to-event cohorts.
- Reports: individual EDA, post-merge EDA, analysis readiness.
