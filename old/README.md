# Archivo histórico

Cada traslado conserva la ruta relativa y el contenido del archivo de origen. El orquestador y pytest usan las rutas activas. Las notas históricas pueden mencionar rutas anteriores.

| Ruta original | Ruta nueva | Motivo | Evidencia |
| --- | --- | --- | --- |
| `code_del_check_pcr_info.py` | `old/code_del_check_pcr_info.py` | Análisis de disponibilidad PCR | Imports de seaborn/matplotlib y escritura fuera del contrato |
| `docs/technical_report_en.md` | `old/docs/technical_report_en.md` | Descripción del pipeline histórico | Describe el backbone histórico y sus reportes; se conserva como referencia |
| `src/02_profile_individual.py` | `old/src/02_profile_individual.py` | Perfil exploratorio | Lecturas L43–44; solo reportes/workbook |
| `src/03_linkage.py` | `old/src/03_linkage.py` | Auditoría histórica de enlaces | Salidas L124–125; 04 L136–137 lee ingestión directa |
| `src/06_postmerge_eda.py` | `old/src/06_postmerge_eda.py` | EDA del backbone | main L34–35 y exportación de reportes |
| `src/07_build_cohorts.py` | `old/src/07_build_cohorts.py` | Cohortes de EDA | Salidas L104–125; obj1 Step00 construye su propia cohorte |
| `src/08_visit_patterns.py` | `old/src/08_visit_patterns.py` | Patrones y visualizaciones | main L1146; salidas bajo visit_patterns |
| `src/08b_visit_episode_temporal_audit.py` | `old/src/08b_visit_episode_temporal_audit.py` | Revisión histórica de proximidad | L17–20,866–889; reglas de 08c están implementadas localmente |
| `src/10_interval_collapse_plots.py` | `old/src/10_interval_collapse_plots.py` | Visualizaciones de colapso | L17–26; único llamador anterior era el orquestador |
| `src/12_interval_temporal_order_audit.py` | `old/src/12_interval_temporal_order_audit.py` | Auditoría exploratoria de orden | L206; usa visitas y publica diagnósticos |
| `src/13_longitudinal_plausibility_audit.py` | `old/src/13_longitudinal_plausibility_audit.py` | Plausibilidad longitudinal | Carga L138–142; sus informes alimentan únicamente diagnósticos/recodificación exploratoria |
| `src/14_longitudinal_plausibility_plots.py` | `old/src/14_longitudinal_plausibility_plots.py` | Figuras de plausibilidad | Carga L89 y parámetros de entrada apuntan a reportes de 13 |
| `src/15_codebook_cleanliness_check.py` | `old/src/15_codebook_cleanliness_check.py` | Informe de limpieza | L301–308 exporta diagnósticos sin corregir datos |
| `src/16_missingness_patterns.py` | `old/src/16_missingness_patterns.py` | Informe de faltantes | L250; reporte de faltantes sin escritura sobre la entrada |
| `src/18_recode_longitudinal_values_by_type.py` | `old/src/18_recode_longitudinal_values_by_type.py` | Recodificación exploratoria | L344 y L59–63; 24 excluye explícitamente el producto |
| `src/19_collapse_patients_one_row.py` | `old/src/19_collapse_patients_one_row.py` | Colapso descriptivo a paciente | L96 y escritura en ruta de salida separada |
| `src/20_merge_optional_nh_same_year.py` | `old/src/20_merge_optional_nh_same_year.py` | Condensación alternativa para análisis | L36–59; no sobrescribe la ruta consumida |
| `src/21_compare_essdai_prefixes.py` | `old/src/21_compare_essdai_prefixes.py` | Comparación de versiones | L99 lee input y solo compara prefijos |
| `src/22_merge_optional_nh_same_year.py` | `old/src/22_merge_optional_nh_same_year.py` | Segunda condensación alternativa | L36–59; ruta distinta del contrato |
| `src/23_eval_cohorts_sample.py` | `old/src/23_eval_cohorts_sample.py` | Viabilidad exploratoria de cohortes | L520–546,578–605; obj1 Step00 lee el padrón externo list_ids_longitudinal.csv, con otro nombre y esquema |
| `src/24_protocol_flow_table.py` | `old/src/24_protocol_flow_table.py` | Tabla descriptiva de flujo | L483 y lógica de fallback descrita en README previo |
| `src/tier1_longitudinal.py` | `old/src/tier1_longitudinal.py` | Modelado longitudinal y supervivencia | L623 y CLI local; no se consume desde obj1 |
| `tests/test_08b_visit_episode_temporal_audit.py` | `old/tests/test_08b_visit_episode_temporal_audit.py` | Prueba sintética de 08b_visit_episode_temporal_audit | carga del módulo bajo prueba en el encabezado |
| `tests/test_23_eval_cohorts_sample.py` | `old/tests/test_23_eval_cohorts_sample.py` | Prueba sintética de 23_eval_cohorts_sample | carga del módulo bajo prueba en el encabezado |
| `.gitkeep` | `old/.gitkeep` | Marcador vacío de la raíz | Archivo de cero bytes. La raíz contiene código y documentación versionados. El plan, los productores y pytest carecen de referencias al marcador. |
