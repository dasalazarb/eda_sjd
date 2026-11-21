# Validación de la limpieza

La comparación parte de `eda_sjd/main@ffe59ecc5711b9e2e0bf4e611e32e8feffc5e2b4` y consulta `obj1_sjd/main@2f4f2eb7c8c1d9019964d81a5b9df69fe0d9a857`. Se usó Python 3.12.14 en un entorno recién instalado desde `requirements.txt`. `pip check` pasó. Se usaron únicamente fixtures sintéticos y copias temporales.

## Repetir la comparación

Ejecuta el bloque desde la raíz de la propuesta, con el entorno activado. La primera copia ejecuta los productores originales directamente porque el orquestador histórico no publicaba la espina. La segunda usa el orquestador propuesto.

```bash
set -euo pipefail
root=$PWD
sandbox=$(mktemp -d)
mkdir "$sandbox/base" "$sandbox/candidate"
git archive ffe59ecc5711b9e2e0bf4e611e32e8feffc5e2b4 | tar -x -C "$sandbox/base"
cp -R src "$sandbox/candidate/"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$sandbox/matplotlib"
python tests/cleanup_fixture.py fixture "$sandbox/base"
python tests/cleanup_fixture.py fixture "$sandbox/candidate"
(
  cd "$sandbox/base"
  for step in 01_ingest 04_dedup_rules 05_build_backbone \
    09_interval_collapse_audit 09b_merge_essdai_versions \
    08c_build_clinical_episode_map 09c_clinical_episode_collapse \
    09d_finalize_clinical_episode_dataset 10_clinical_baseline_qc \
    11_build_clinical_episode_spine 19_filter_btris_patients \
    20c_btris_lab_rule_harmonization 20_btris_visit_date_match_report \
    20b_btris_baseline_labs; do
    python "src/$step.py"
  done
)
(cd "$sandbox/candidate" && python src/00_super_main.py)
python "$root/tests/cleanup_fixture.py" compare "$sandbox/base" "$sandbox/candidate" \
  > "$sandbox/comparison.json"
```

El comparador exige igualdad exacta de columnas, tipos inferidos, orden de filas, valores y nulos. Comprueba claves no nulas y únicas donde corresponden y registra el esquema y los nulos por columna. Los SHA-256 se calculan antes de normalizar la ruta absoluta de `source_file` en Step 20. Las dos copias contienen el mismo nombre relativo de origen. Los hashes de ese Parquet varían entre ubicaciones.

Una primera comparación sin semilla fija reveló diferencias en el orden de campos de 09c. La implementación original construye `AUTHORITATIVE_COLUMNS` a partir del conjunto `MANIFEST_REQUIRED` (líneas 49 y 89). Con `PYTHONHASHSEED=0` ambas copias usan el mismo orden. Se conservó el código original de los productores. Otro ensayo con IDs sintéticos exclusivamente numéricos falló en el merge de 11 porque `read_csv` infirió `int64` y el Parquet de episodios conservó texto. Ocurrió también con el productor original. La comparación satisfactoria emplea IDs sintéticos alfanuméricos.

## Pruebas de la segunda pasada

Con Python 3.12.14, el mismo entorno instalado desde `requirements.txt` y `PYTHONHASHSEED=0`, la referencia en copia aislada dio **214 aprobadas y 15 fallidas**. La referencia solo incorpora la corrección de import dinámico de 08c descrita en la primera pasada, necesaria para recolectar esos tests. El árbol activo antes de esta pasada dio **202 aprobadas y las mismas 15 fallidas**. Al aplicar los parches se obtuvieron **218 aprobadas y cero fallos** con semillas 0 y 31. La diferencia de un caso corresponde a la regresión CLI de IDs.

### Diagnóstico individual de la referencia

Los 15 casos de la tabla reproducían el fallo en la referencia. Todos pasan en el árbol actual sin exclusiones.

| Test | Responsable y aserción original | Clasificación | Causa, impacto y acción |
| --- | --- | --- | --- |
| `test_sparse_candidates_filter_distant_optional_pairs` | 08c, pares evaluados < 20 000 | TEST_DESACTUALIZADO | La ventana vigente y la excepción de mes/día generan 25 378 pares. Se verifica el conteo exacto con un cálculo exhaustivo independiente sobre las fechas del fixture y se exige que sea menor al espacio de pares. El mapa conserva sus reglas. |
| `test_nearest_date_distance_uses_all_source_dates` | 08c, distancia mínima de 2 días | TEST_DESACTUALIZADO | El test invocaba un helper eliminado en la reconstrucción `418bd8e`. Ahora ejercita el algoritmo vigente con las mismas fechas, conservando la aserción de dos días. |
| `test_conflicts_preserve_falsey_and_negative_values` | 08c, valores seleccionados 0 y falso | TEST_DESACTUALIZADO | El informe de 08c conserva desacuerdos del mismo intervalo en vez de elegir un valor. Se exige conservar los pares completos `0 / -1` y `False / 0` en el informe. La resolución analítica posterior es responsabilidad de 09d. |
| `test_merge_same_date_intervals_prefers_non_natural_history_values` | 09, acceso por etiqueta 0 | TEST_DESACTUALIZADO | La fila conservada retiene su índice original. El test verifica los mismos valores y acción por posición. El código publica con `index=False`. |
| `test_preexisting_source_pipe_is_missing_and_unresolved` | 09d, etiquetas `preexisting_source_pipe` | TEST_DESACTUALIZADO | `ddec924` eliminó la rama especial y dirige los tokens al resolvedor general. Se verifica la etiqueta vigente de conflicto del mismo día, ambos valores de origen, el estado no resuelto y el valor analítico nulo. |
| `test_build_comparison_uses_first_clinical_anchor` | 10, fecha y clasificación de baseline | FIXTURE_INSUFICIENTE | Faltaban sequedad, fatiga y clasificación SjD exigidas por el productor. Se completó el fixture sin cambiar aserciones de fecha, shift o clasificabilidad. |
| `test_missing_clinical_baseline_is_flagged_for_review` | 10, bandera clínica sin baseline | FIXTURE_INSUFICIENTE | El mismo esquema incompleto bloqueaba la evaluación. Se conservó el paciente sin baseline y la aserción de revisión. |
| `test_duplicate_episode_identifier_is_rejected` | 10, fallo por clave duplicada | FIXTURE_INSUFICIENTE | El esquema incompleto causaba un error anterior al de unicidad. Con esquema completo se comprueba el mismo rechazo. |
| `test_summary_reports_requested_counts_and_reasons` | 10, conteos de baseline y motivos | FIXTURE_INSUFICIENTE | Se completaron los campos que 10 lee. Se conservan las aserciones de cobertura y motivos. |
| `test_build_spines_preserves_every_episode_for_sjd_patients` | 11, episodios y valores sin pérdidas | DEFECTO_PRODUCTOR | La comparación de arrays object con `pd.NA` abortaba antes del filtro de ausentes. Ahora solo se comparan IDs basales presentes. Sin cambio de elegibilidad o cardinalidad. |
| `test_missing_baseline_produces_no_baseline_episode` | 11, bandera falsa y QC | DEFECTO_PRODUCTOR | El mismo error impedía publicar pacientes sin baseline. Se conserva el nulo en la clave y se produce bandera falsa con dtype booleano. |
| `test_write_outputs_creates_csv_and_parquet_pairs` | 11, cuatro productos y QC | DEFECTO_PRODUCTOR | El mismo error impedía alcanzar la escritura. El test vuelve a ejercitar las dos espinas y sus formatos. |
| `test_semantic_intent_distinguishes_reviewed_unused_from_unexpected` | 20, flags ordenados como el input | TEST_DESACTUALIZADO | El informe agrupa y ordena por clave canónica. Se exige el mismo flag para cada orden, sin imponer orden de entrada. |
| `test_identical_results_without_strong_identity_are_not_specimen_deduplicated` | 20b, leucopenia verdadera y sin deduplicación | FIXTURE_INSUFICIENTE | Una referencia baja sin límite alto ni estructura parseada no demuestra el estado requerido. Se completó un rango bilateral y se conservaron la aserción clínica y la de no deduplicación. |
| `test_core_input_evidence_qc_counts_only_aggregate_availability` | 20b, conteos agregados y ausencia de IDs | FIXTURE_INSUFICIENTE | Faltaban columnas exigidas del paso 20. El caso C4 que pretendía carecer de referencia traía un límite alto. El fixture corregido representa ambos escenarios previstos y mantiene aserciones. |

El otro defecto reproducido de 11 estaba en la lectura del CSV. La inferencia convertía IDs numéricos a enteros e impedía el join con el Parquet textual. El test nuevo usa `0001` y `1` como claves distintas y verifica la publicación y el baseline de cada una.

### Comparación reproducible y consumo aguas abajo

- Los 14 productores completaron tres ejecuciones aisladas. Referencia y propuesta con semilla 0 dieron **21 artefactos equivalentes por contenido** y **20 hashes idénticos**. En Step 20 la única diferencia fue el prefijo de la ruta absoluta de procedencia.
- Las propuestas con semillas 0 y 31 discrepan en orden de columnas en siete artefactos. Son el colapso 09c en CSV y Parquet, el finalizado 09d en Parquet y las espinas `all` y `sjd` en ambos formatos. Un diagnóstico adicional alineó las columnas por nombre después de verificar su igualdad como conjunto. Valores, tipos, filas y nulos fueron idénticos en los 21 artefactos. La comparación estricta original sigue fallando entre semillas. Sin un snapshot autorizado que establezca el orden, 09c queda intacto.
- En `obj1_sjd` estaban vigentes los mismos consumidores. Steps 00 y 01 ingirieron las salidas nuevas en una copia sintética. Step 08 reconstruyó cuatro documentos en modo de simulación. Las pruebas dirigidas de Step 00/01, importador, Step 10, Pharma y Graph N3 pasaron. Los diez fallos de publicación observados en la copia exportada se debían a que el productor requiere el SHA de Git. La suite específica de medicación pasó sus 12 tests en el checkout consultado, para un total dirigido de 142 casos aprobados.

Para repetir el diagnóstico entre semillas, crea una tercera copia con el comando anterior, provisiona el mismo fixture y ejecuta el orquestador con `PYTHONHASHSEED=31`. Ejecuta el comparador estricto entre ambas propuestas. Para aislar la discrepancia, verifica nombres y orden de columnas por archivo y vuelve a comparar el contenido después de alinear por nombre, manteniendo el orden de filas.

## Puertas de la segunda pasada

| Puerta | Estado | Evidencia |
| --- | --- | --- |
| G1 Inventario | PASS | Cubiertos los archivos rastreados, las adiciones de la primera pasada y las fuentes externas conocidas. |
| G2 Nombres | PASS | 18 rutas bajo `src/`, sin renombres. |
| G3 Cierre mínimo | BLOCKED | 17 scripts del cierre y un escritor alternativo retenido en REVIEW. |
| G4 Codebook | BLOCKED | El importador acredita el nombre público. Falta el snapshot utilizado o una procedencia verificable que identifique su escritor. |
| G5 Archivo y colisiones | PASS | Ningún import ni comando activo lee `old/`. 09b es el único escritor automático por intervalo. |
| G6 Plan | PASS | 14 etapas, en el orden establecido, con `--list-steps` y fallo inmediato por subprocess. |
| G7 Suite activa | PASS | 218 aprobadas, incluidos los 15 casos reparados y la regresión nueva de IDs. |
| G8 Equivalencia y determinismo | BLOCKED | Equivalencia estricta con semilla común. Pendiente el orden canónico de siete artefactos que cambian de orden entre semillas. |
| G9 Snapshots | BLOCKED | Las fuentes y snapshots autorizados no están en el workspace. La ejecución fue sintética y aislada. |
| G10 Contrato | BLOCKED | Contrato, lectura y contenido sintético verificados. Pendiente confirmar la equivalencia operativa. |
| G11 Archivo íntegro | PASS | 25 traslados acumulados, verificados byte a byte. Solo `.gitkeep` en la segunda pasada. |
| G12 Custodia y alcance | PASS | `obj1_sjd` conserva su checkout sin cambios. Los cambios preparados son locales y las escrituras de prueba quedaron en copias temporales. |

### Procedimiento pendiente por el custodio

1. Identificar el snapshot por intervalo que usa el importador, con hash local, revisión, productor, parámetros y fuentes. Buscar esa cadena en el manifiesto, los logs de publicación o un registro autorizado. Reunir el snapshot de episodios con su orden de campos, referencias BTRIS y parquets históricos, y el padrón de obj1.
2. Preparar dos copias aisladas con entradas congeladas en las [rutas acordadas](pipeline_minimal.md) y salidas nuevas. Usar la misma versión de Python, dependencias y semilla para reconstruir la referencia y la propuesta, siempre con autorización. Si hay evidencia de uso de 11 Codebook, ejecutar su CLI con su hoja y tabla de origen verificadas, apuntando `--corrected-output-path` y `--output-path` a una salida aislada distinta.
3. Comparar localmente SHA-256, esquemas y orden, claves no nulas y únicas, cobertura del mapa, filas, nulos por columna, fechas, tipos de visita, elegibilidad, baseline, conflictos y valores clínicos. Tratar procedencia absoluta como una diferencia del entorno y conservar el orden del snapshot cuando sea contractual. Reportar solo agregados y discrepancias desidentificadas.
4. Ejecutar Steps 00, 01, 08 y 10 y los perfiles Pharma y Graph N3 en una copia independiente de obj1 con rutas CLI hacia esas salidas, padrón y credenciales institucionales. Revisar el resultado antes de sincronizar cualquier publicación operativa.

## Huellas de las salidas sintéticas

La comparación con semilla 0 calcula los hashes antes de normalizar el prefijo de procedencia de Step 20. Un solo hash indica igualdad binaria.

| Artefacto | Filas | Pacientes | Columnas | Nulos | SHA-256 (referencia primero) |
| --- | ---: | ---: | ---: | ---: | --- |
| `data_analytic/visits_long.parquet` | 8 | 3 | 22 | 0 | `04276853113e4795230fc53b9d09ca24c9e390597953e8ec713bc28853551d1e` |
| `data_intermediate/clinical_episode_row_map.parquet` | 8 | 3 | 49 | 0 | `cce88b682a9d698d197e5275f02ef128996e99592a81c32ed152bc1287e66e2a` |
| `data_analytic/clinical_episode_manifest.parquet` | 7 | 3 | 25 | 0 | `83b4ce5514fb19a80fc46cb5b5771a9686a1f2982bc5bdc27a14926332fc1403` |
| `data_analytic/visits_long_collapsed_by_clinical_episode_codebook_not_clean.parquet` | 7 | 3 | 45 | 22 | `6521ce83d8803031043740fb6f97d8e7fa75bcdc8bdcd1c5697d38b2102a2f8e` |
| `data_analytic/visits_long_collapsed_by_clinical_episode_codebook_not_clean.csv` | 7 | 3 | 45 | 32 | `aca67b740be93eaf45db1fd09b241f52a792b8e4a4f7906e5f168a47235ac8b3` |
| `reports/clinical_episode_collapse/09c_episode_variable_conflicts.csv` | 0 | 0 | 14 | 0 | `cdfafeafbfe5c7ad01eba74b34b96650eccf50c3c00a88b1aaac7ab604871b41` |
| `data_analytic/visits_long_collapsed_by_clinical_episode_codebook_corrected.parquet` | 7 | 3 | 47 | 33 | `890008a18412d11cd85c1c640e57c0928b49e6a2678e322f3fca8995a965f8d9` |
| `reports/clinical_baseline/10_clinical_baseline_comparison.csv` | 3 | 3 | 16 | 0 | `a518fabad90cb204b44cf4256494296ca91b1e2f35282a2befab84b1d8cc5a3f` |
| `data_analytic/clinical_episode_spine_all.parquet` | 7 | 3 | 52 | 33 | `b08a4ead3cb36e26e6734cbfc1a7f84a17041fc4ff404b9f066718eabd98010c` |
| `data_analytic/clinical_episode_spine_all.csv` | 7 | 3 | 52 | 45 | `a8d59714bf9b9c33b4d662e372ef3446d922c413cca37843b76ab21c893f0eb9` |
| `data_analytic/clinical_episode_spine_sjd.parquet` | 6 | 2 | 52 | 30 | `b6cbf1a488f29426d6887d7425d5b58d7fd53d191004dd3a9a12c33520e33f89` |
| `data_analytic/clinical_episode_spine_sjd.csv` | 6 | 2 | 52 | 40 | `5035ab0c242d13d3b34805318500bd600398679f6d0b32c3224ead84f1b0e160` |
| `data_analytic/visits_long_collapsed_by_interval_codebook_corrected.parquet` | 7 | 3 | 23 | 15 | `0d4cee57db21d938ff193c48d210ca695a1071f88d9295148361ed7432be7185` |
| `data_analytic/visits_long_collapsed_by_interval_codebook_corrected.csv` | 7 | 3 | 23 | 15 | `c4e63036a5378af283d5050f6e86f7347f9635038cd1be79c41117f1d18ca156` |
| `data_intermediate/BTRIS/11D/Lab Synthetic.csv` | 2 | 1 | 13 | 2 | `818e486593815344b43a653e9032d918dd4e2b7c52af22cd917d01384e815116` |
| `data_intermediate/BTRIS/15D/Lab Synthetic.csv` | 2 | 1 | 13 | 2 | `818d7a82026077c74cb834e5982a9b5e91695185c49f240509c8e267a28786f8` |
| `data_intermediate/BTRIS/11D/Medication288444-1.csv` | 1 | 1 | 6 | 0 | `efb7694d1b33e8519ac2113848cbbeead98b072487cf4aaaadcc76fe88ae89d2` |
| `data_intermediate/BTRIS/15D/Medication288476-1.csv` | 1 | 1 | 6 | 0 | `22438b6a032dfdba6c344fb1e8523f88edce51d753e2a234e021b2fb79aeab0e` |
| `data_analytic/BTRIS/20_btris_lab_records_long.parquet` | 4 | 2 | 43 | 36 | `3677c412a087e41e098654d8220a03a71a1d02ef62c2300ee265f6312c16a6df`<br>`7780191d17bdc3eaeba68763367fce605b22f87e8a98eb3f66107c1f51a2d8a4` |
| `data_analytic/BTRIS/20b_baseline_lab_selected_long.parquet` | 14 | 2 | 57 | 414 | `4153d7c893227ef71b7a97a3404f85ffc2ca83559cb9cbbbcf68bfbf56e0e4c7` |
| `data_analytic/BTRIS/20b_baseline_labs_patient_level.parquet` | 2 | 2 | 88 | 50 | `1f744f34ad102349ded8e020ec8ceb03d3c7e3c64ab845a4f1fc5d67b70494e4` |

## Esquemas y nulos por columna

Tipos leídos por pandas, en el orden de cada artefacto. Los ceros se omiten. Esquemas y métricas coinciden entre ambas ejecuciones con la semilla común.

<details>
<summary>visits_long.parquet</summary>

- `data_analytic/visits_long.parquet` · Clave row_id_raw.

```text
ids__subject_number: object
ids__patient_record_number: object
ids__visit_date: object
ids__time_24_hour: object
ids__interval_name: object
visit_summary_form__sjogrens_class: object
essdai__essdai_total_score: object
esspri_questionnaire__dryness: object
esspri_questionnaire__fatigue: object
esspri_questionnaire__pain: object
medications__rx_1_name: object
visit_date: datetime64[ns]
time_24_hour: object
visit_datetime: datetime64[ns]
source_protocol: object
source_file: object
row_id_raw: object
dup_rank: int64
duplicate_group_id: int64
visit_datetime_adjustment_seconds: int64
comparison_type: object
comparison_detail: object
```

</details>

<details>
<summary>clinical_episode_row_map.parquet</summary>

- `data_intermediate/clinical_episode_row_map.parquet` · Clave row_id_raw.

```text
ids__subject_number: object
ids__patient_record_number: object
ids__visit_date: object
ids__time_24_hour: object
ids__interval_name: object
visit_summary_form__sjogrens_class: object
essdai__essdai_total_score: object
esspri_questionnaire__dryness: object
esspri_questionnaire__fatigue: object
esspri_questionnaire__pain: object
medications__rx_1_name: object
visit_date: datetime64[ns]
time_24_hour: object
visit_datetime: datetime64[ns]
source_protocol: object
source_file: object
row_id_raw: object
dup_rank: int64
duplicate_group_id: int64
visit_datetime_adjustment_seconds: int64
comparison_type: object
comparison_detail: object
_source_order: int64
patient_id: object
interval_name: object
interval_normalized: object
collection_date: datetime64[ns]
collection_year: Int64
has_essdai: bool
has_esspri: bool
has_systems_review: bool
has_visit_summary: bool
has_eye_examination: bool
has_salivary_flow: bool
has_oral_examination: bool
has_physical_examination: bool
clinical_episode_id: object
assignment_rule: object
merge_stage: object
merge_rule: object
manual_review_required: bool
manual_review_reason: object
representative_interval: object
representative_date: datetime64[ns]
episode_precedence: int64
optional_cluster_id: object
optional_adjudication: object
visit_type: object
clinical_visit: bool
```

</details>

<details>
<summary>clinical_episode_manifest.parquet</summary>

- `data_analytic/clinical_episode_manifest.parquet` · Clave patient_id, clinical_episode_id.

```text
patient_id: object
clinical_episode_id: object
intervals_involved: object
representative_interval: object
episode_start_date: datetime64[ns]
clinical_anchor_date: datetime64[ns]
episode_end_date: datetime64[ns]
episode_span_days: int64
n_source_dates: int64
source_dates: object
n_raw_rows: int64
visit_type: object
clinical_visit: bool
manual_review_required: bool
manual_review_reason: object
assignment_rule: object
merge_stage: object
cross_interval_merge: bool
cross_year_merge: bool
long_interval_span_warning: bool
possible_date_entry_error: bool
optional_adjudication: object
optional_cluster_ids: object
interval_order_anomaly: bool
exceptional_merge: bool
```

</details>

<details>
<summary>visits_long_collapsed_by_clinical_episode_codebook_not_clean.parquet</summary>

- `data_analytic/visits_long_collapsed_by_clinical_episode_codebook_not_clean.parquet` · Clave patient_id, clinical_episode_id.

```text
patient_id: object
clinical_episode_id: object
ids__visit_date: datetime64[ns]
visit_date: datetime64[ns]
visit_datetime: datetime64[ns]
ids__subject_number: object
ids__patient_record_number: object
ids__time_24_hour: object
ids__interval_name: object
visit_summary_form__sjogrens_class: object (nulos: 2)
essdai__essdai_total_score: object (nulos: 2)
esspri_questionnaire__dryness: object (nulos: 2)
esspri_questionnaire__fatigue: object (nulos: 2)
esspri_questionnaire__pain: object (nulos: 2)
medications__rx_1_name: object (nulos: 5)
time_24_hour: object
source_protocol: object
source_file: object
row_id_raw: object
dup_rank: int64
duplicate_group_id: object
visit_datetime_adjustment_seconds: int64
comparison_type: object
comparison_detail: object (nulos: 7)
assignment_rule: object
n_raw_rows_in_episode: int64
n_collection_dates_in_episode: int64
manual_review_required: bool
manual_review_reason: object
episode_start_date: datetime64[ns]
clinical_anchor_date: datetime64[ns]
clinical_visit: bool
episode_end_date: datetime64[ns]
episode_span_days: int64
visit_type: object
intervals_involved: object
cross_interval_merge: bool
cross_year_merge: bool
long_interval_span_warning: bool
possible_date_entry_error: bool
optional_adjudication: object
optional_cluster_ids: object
n_source_dates: int64
source_dates: object
n_raw_rows: int64
```

</details>

<details>
<summary>visits_long_collapsed_by_clinical_episode_codebook_not_clean.csv</summary>

- `data_analytic/visits_long_collapsed_by_clinical_episode_codebook_not_clean.csv` · Clave patient_id, clinical_episode_id.

```text
patient_id: object
clinical_episode_id: object
ids__visit_date: object
visit_date: object
visit_datetime: object
ids__subject_number: object
ids__patient_record_number: object
ids__time_24_hour: object
ids__interval_name: object
visit_summary_form__sjogrens_class: float64 (nulos: 2)
essdai__essdai_total_score: float64 (nulos: 2)
esspri_questionnaire__dryness: float64 (nulos: 2)
esspri_questionnaire__fatigue: float64 (nulos: 2)
esspri_questionnaire__pain: float64 (nulos: 2)
medications__rx_1_name: object (nulos: 5)
time_24_hour: object
source_protocol: object
source_file: object
row_id_raw: object
dup_rank: int64
duplicate_group_id: object
visit_datetime_adjustment_seconds: int64
comparison_type: object
comparison_detail: float64 (nulos: 7)
assignment_rule: object
n_raw_rows_in_episode: int64
n_collection_dates_in_episode: int64
manual_review_required: bool
manual_review_reason: object (nulos: 5)
episode_start_date: object
clinical_anchor_date: object
clinical_visit: bool
episode_end_date: object
episode_span_days: int64
visit_type: object
intervals_involved: object
cross_interval_merge: bool
cross_year_merge: bool
long_interval_span_warning: bool
possible_date_entry_error: bool
optional_adjudication: object
optional_cluster_ids: object (nulos: 5)
n_source_dates: int64
source_dates: object
n_raw_rows: int64
```

</details>

<details>
<summary>09c_episode_variable_conflicts.csv</summary>

- `reports/clinical_episode_collapse/09c_episode_variable_conflicts.csv` · Clave patient_id, clinical_episode_id, variable.

```text
patient_id: object
clinical_episode_id: object
clinical_anchor_date: object
intervals_involved: object
variable: object
n_nonmissing_values: object
n_unique_values: object
observed_values: object
collection_dates: object
row_ids_involved: object
visit_type: object
clinical_visit: object
manual_review_required: object
manual_review_reason: object
```

</details>

<details>
<summary>visits_long_collapsed_by_clinical_episode_codebook_corrected.parquet</summary>

- `data_analytic/visits_long_collapsed_by_clinical_episode_codebook_corrected.parquet` · Clave patient_id, clinical_episode_id.

```text
patient_id: object
clinical_episode_id: object
ids__visit_date: datetime64[ns]
visit_date: datetime64[ns]
visit_datetime: datetime64[ns]
ids__subject_number: object
ids__patient_record_number: object
ids__time_24_hour: object
ids__interval_name: object
visit_summary_form__sjogrens_class: object (nulos: 2)
essdai__essdai_total_score: float64 (nulos: 2)
esspri_questionnaire__dryness: object (nulos: 2)
esspri_questionnaire__fatigue: object (nulos: 2)
esspri_questionnaire__pain: object (nulos: 2)
medications__rx_1_name: object (nulos: 5)
source_protocol: object
assignment_rule: object
n_raw_rows_in_episode: int64
n_collection_dates_in_episode: int64
manual_review_required: bool
manual_review_reason: object
episode_start_date: datetime64[ns]
clinical_anchor_date: datetime64[ns]
clinical_visit: bool
episode_end_date: datetime64[ns]
episode_span_days: int64
visit_type: object
intervals_involved: object
cross_interval_merge: bool
cross_year_merge: bool
long_interval_span_warning: bool
possible_date_entry_error: bool
optional_adjudication: object
optional_cluster_ids: object
n_source_dates: int64
source_dates: object
n_raw_rows: int64
_essdai_resolution_method: object
_essdai_version: object (nulos: 2)
_essdai_total_version: object (nulos: 2)
preferred_essdai_r_raw_row: object (nulos: 7)
episode_has_unresolved_conflict: bool
essdai_has_unresolved_conflict: bool
analytic_resolution_quality: object
essdai_total_derived_from_domains: object (nulos: 7)
essdai_total_source: object
essdai_total_consistency: object
```

</details>

<details>
<summary>10_clinical_baseline_comparison.csv</summary>

- `reports/clinical_baseline/10_clinical_baseline_comparison.csv` · Clave patient_id.

```text
patient_id: object
first_recorded_episode_id: object
first_recorded_date: object
first_recorded_visit_type: object
clinical_baseline_episode_id: object
clinical_baseline_date: object
baseline_visit_type: object
days_shifted: int64
baseline_shifted: bool
reason_for_shift: object
baseline_has_essdai: bool
baseline_has_esspri: bool
baseline_pop_classifiable: bool
baseline_manual_review_required: bool
sjd_ever_1_2_4: bool
sjogrens_class_patient_values: int64
```

</details>

<details>
<summary>clinical_episode_spine_all.parquet, clinical_episode_spine_sjd.parquet</summary>

- `data_analytic/clinical_episode_spine_all.parquet` · Clave patient_id, clinical_episode_id.
- `data_analytic/clinical_episode_spine_sjd.parquet` · Clave patient_id, clinical_episode_id.

```text
patient_id: object
clinical_episode_id: object
ids__visit_date: datetime64[ns]
visit_date: datetime64[ns]
visit_datetime: datetime64[ns]
ids__subject_number: object
ids__patient_record_number: object
ids__time_24_hour: object
ids__interval_name: object
visit_summary_form__sjogrens_class: object (nulos: 2, 2)
essdai__essdai_total_score: float64 (nulos: 2, 2)
esspri_questionnaire__dryness: object (nulos: 2, 2)
esspri_questionnaire__fatigue: object (nulos: 2, 2)
esspri_questionnaire__pain: object (nulos: 2, 2)
medications__rx_1_name: object (nulos: 5, 4)
source_protocol: object
assignment_rule: object
n_raw_rows_in_episode: int64
n_collection_dates_in_episode: int64
manual_review_required: bool
manual_review_reason: object
episode_start_date: datetime64[ns]
clinical_anchor_date: datetime64[ns]
clinical_visit: bool
episode_end_date: datetime64[ns]
episode_span_days: int64
visit_type: object
intervals_involved: object
cross_interval_merge: bool
cross_year_merge: bool
long_interval_span_warning: bool
possible_date_entry_error: bool
optional_adjudication: object
optional_cluster_ids: object
n_source_dates: int64
source_dates: object
n_raw_rows: int64
_essdai_resolution_method: object
_essdai_version: object (nulos: 2, 2)
_essdai_total_version: object (nulos: 2, 2)
preferred_essdai_r_raw_row: object (nulos: 7, 6)
episode_has_unresolved_conflict: bool
essdai_has_unresolved_conflict: bool
analytic_resolution_quality: object
essdai_total_derived_from_domains: object (nulos: 7, 6)
essdai_total_source: object
essdai_total_consistency: object
sjd_ever_1_2_4: bool
sjogrens_class_patient_values: int64
clinical_baseline_episode_id: object
clinical_baseline_date: object
is_clinical_baseline: bool
```

</details>

<details>
<summary>clinical_episode_spine_all.csv, clinical_episode_spine_sjd.csv</summary>

- `data_analytic/clinical_episode_spine_all.csv` · Clave patient_id, clinical_episode_id.
- `data_analytic/clinical_episode_spine_sjd.csv` · Clave patient_id, clinical_episode_id.

```text
patient_id: object
clinical_episode_id: object
ids__visit_date: object
visit_date: object
visit_datetime: object
ids__subject_number: object
ids__patient_record_number: object
ids__time_24_hour: object
ids__interval_name: object
visit_summary_form__sjogrens_class: float64 (nulos: 2, 2)
essdai__essdai_total_score: float64 (nulos: 2, 2)
esspri_questionnaire__dryness: float64 (nulos: 2, 2)
esspri_questionnaire__fatigue: float64 (nulos: 2, 2)
esspri_questionnaire__pain: float64 (nulos: 2, 2)
medications__rx_1_name: object (nulos: 5, 4)
source_protocol: object
assignment_rule: object
n_raw_rows_in_episode: int64
n_collection_dates_in_episode: int64
manual_review_required: bool
manual_review_reason: object (nulos: 5, 4)
episode_start_date: object
clinical_anchor_date: object
clinical_visit: bool
episode_end_date: object
episode_span_days: int64
visit_type: object
intervals_involved: object
cross_interval_merge: bool
cross_year_merge: bool
long_interval_span_warning: bool
possible_date_entry_error: bool
optional_adjudication: object
optional_cluster_ids: object (nulos: 5, 4)
n_source_dates: int64
source_dates: object
n_raw_rows: int64
_essdai_resolution_method: object (nulos: 2, 2)
_essdai_version: object (nulos: 2, 2)
_essdai_total_version: object (nulos: 2, 2)
preferred_essdai_r_raw_row: float64 (nulos: 7, 6)
episode_has_unresolved_conflict: bool
essdai_has_unresolved_conflict: bool
analytic_resolution_quality: object
essdai_total_derived_from_domains: float64 (nulos: 7, 6)
essdai_total_source: object
essdai_total_consistency: object
sjd_ever_1_2_4: bool
sjogrens_class_patient_values: int64
clinical_baseline_episode_id: object
clinical_baseline_date: object
is_clinical_baseline: bool
```

</details>

<details>
<summary>visits_long_collapsed_by_interval_codebook_corrected.parquet</summary>

- `data_analytic/visits_long_collapsed_by_interval_codebook_corrected.parquet` · Clave sin restricción de unicidad en el comparador.

```text
ids__patient_record_number: object
ids__interval_name: object
ids__subject_number: object
ids__visit_date: object
ids__time_24_hour: object
visit_summary_form__sjogrens_class: object (nulos: 2)
essdai__essdai_total_score: object (nulos: 2)
esspri_questionnaire__dryness: object (nulos: 2)
esspri_questionnaire__fatigue: object (nulos: 2)
esspri_questionnaire__pain: object (nulos: 2)
medications__rx_1_name: object (nulos: 5)
visit_date: object
time_24_hour: object
visit_datetime: object
source_protocol: object
source_file: object
row_id_raw: object
dup_rank: object
duplicate_group_id: object
visit_datetime_adjustment_seconds: object
comparison_type: object
sjogrens_class_patient_values: object
sjogrens_class_patient_cohort: object
```

</details>

<details>
<summary>visits_long_collapsed_by_interval_codebook_corrected.csv</summary>

- `data_analytic/visits_long_collapsed_by_interval_codebook_corrected.csv` · Clave sin restricción de unicidad en el comparador.

```text
ids__patient_record_number: object
ids__interval_name: object
ids__subject_number: object
ids__visit_date: object
ids__time_24_hour: object
visit_summary_form__sjogrens_class: float64 (nulos: 2)
essdai__essdai_total_score: float64 (nulos: 2)
esspri_questionnaire__dryness: float64 (nulos: 2)
esspri_questionnaire__fatigue: float64 (nulos: 2)
esspri_questionnaire__pain: float64 (nulos: 2)
medications__rx_1_name: object (nulos: 5)
visit_date: object
time_24_hour: object
visit_datetime: object
source_protocol: object
source_file: object
row_id_raw: object
dup_rank: int64
duplicate_group_id: object
visit_datetime_adjustment_seconds: int64
comparison_type: object
sjogrens_class_patient_values: int64
sjogrens_class_patient_cohort: object
```

</details>

<details>
<summary>Lab Synthetic.csv, Lab Synthetic.csv</summary>

- `data_intermediate/BTRIS/11D/Lab Synthetic.csv` · Clave sin restricción de unicidad en el comparador.
- `data_intermediate/BTRIS/15D/Lab Synthetic.csv` · Clave sin restricción de unicidad en el comparador.

```text
MRN: object
Order Name: object
Cluster Name: object
Collected Date Time: object
Observation Value: int64
Reference Range: object
Units: object
Interpretation: object
Observation Value Original: int64
rule_family: object
result_category: float64 (nulos: 2, 2)
harmonization_status: object
harmonization_source: object
```

</details>

<details>
<summary>Medication288444-1.csv, Medication288476-1.csv</summary>

- `data_intermediate/BTRIS/11D/Medication288444-1.csv` · Clave sin restricción de unicidad en el comparador.
- `data_intermediate/BTRIS/15D/Medication288476-1.csv` · Clave sin restricción de unicidad en el comparador.

```text
MRN: object
Medication Name: object
Drug Ingredient: object
Order Date/Start Date: object
Order End Date: object
Initial Consent Date: object
```

</details>

<details>
<summary>20_btris_lab_records_long.parquet</summary>

- `data_analytic/BTRIS/20_btris_lab_records_long.parquet` · Clave sin restricción de unicidad en el comparador.

```text
patient_id: string
has_clinical_baseline: bool
clinical_baseline_episode_id: object
clinical_baseline_date: datetime64[ns]
days_from_clinical_baseline: int64
order_name_original: object
order_name_canonical: object
cluster_name_original: object
cluster_name_canonical: object
mapping_status: object
canonical_analyte: object
lab_family: object
analytic_role: object
semantic_mapping_status: object
lab_date: datetime64[ns]
result_raw: string
result_numeric: float64
result_numeric_exact: float64
result_operator: string (nulos: 4)
result_numeric_bound: float64 (nulos: 4)
result_text: string (nulos: 4)
unit: object
reference_range_raw: object
reference_low: float64
reference_high: float64
reference_operator: string (nulos: 4)
reference_bound: float64 (nulos: 4)
reference_range_parse_status: string
observation_comment: object (nulos: 4)
observation_note: object (nulos: 4)
reported_interpretation: object
result_status: object (nulos: 4)
result_valid_for_analysis: bool
invalid_result_reason: string (nulos: 4)
matched_clinical_episode_id: object
matched_clinical_anchor_date: datetime64[ns]
days_from_clinical_anchor: int64
episode_match_method: object
episode_match_ambiguous: bool
unexpected_cluster_name: bool
source_protocol: object
source_file: string
specimen_datetime: object
```

</details>

<details>
<summary>20b_baseline_lab_selected_long.parquet</summary>

- `data_analytic/BTRIS/20b_baseline_lab_selected_long.parquet` · Clave sin restricción de unicidad en el comparador.

```text
patient_id: object
clinical_baseline_episode_id: object
clinical_baseline_date: datetime64[ns]
lab_baseline_eligible: bool
lab_baseline_exclusion_reason: object (nulos: 14)
has_any_btris_lab: bool
baseline_feature: object
source_canonical_analyte: object (nulos: 12)
source_order_name: object (nulos: 12)
source_cluster_name: object (nulos: 12)
primary_baseline_status: boolean (nulos: 12)
selected_result_raw: object (nulos: 12)
selected_result_numeric: float64 (nulos: 12)
selected_result_text: object (nulos: 14)
selected_unit: object (nulos: 12)
selected_lab_date: datetime64[ns] (nulos: 12)
days_from_clinical_baseline: float64 (nulos: 12)
temporal_window_category: object
selection_rule: object
interpretation_source: object
result_interpretation: object (nulos: 12)
interpretation_qc: object
reference_low: float64 (nulos: 12)
reference_high: float64 (nulos: 12)
selected_reference_range_raw: object (nulos: 12)
selected_reference_low: float64 (nulos: 12)
selected_reference_high: float64 (nulos: 12)
selected_reference_operator: object (nulos: 14)
selected_reference_bound: object (nulos: 14)
selected_reference_range_parse_status: object (nulos: 12)
selected_result_operator: object (nulos: 14)
selected_result_numeric_bound: object (nulos: 14)
n_prebaseline_measurements: int64
n_eligible_prebaseline_measurements: int64
n_interpretable_prebaseline_measurements: int64
n_postbaseline_measurements: int64
n_total_measurements: int64
n_positive_prebaseline: int64
n_negative_prebaseline: int64
ever_positive_prebaseline: boolean (nulos: 12)
ever_abnormal_prebaseline: boolean (nulos: 14)
same_day_conflict: bool
longitudinal_discordance: bool
postbaseline_rescue: bool
deduplicate_same_specimen: bool
selected_from_final_verified: bool
source_row_ids: object
closest_prebaseline_status: boolean (nulos: 12)
closest_prebaseline_value: float64 (nulos: 12)
closest_prebaseline_date: datetime64[ns] (nulos: 12)
postbaseline_90d_sensitivity_status: boolean (nulos: 14)
postbaseline_90d_sensitivity_date: datetime64[ns] (nulos: 14)
qc_status: object
ana_source_priority_used: object (nulos: 12)
cryoglobulin_tested_prebaseline: object (nulos: 12)
cryoglobulin_tested_in_primary_window: object (nulos: 12)
cryoglobulins_ife_available: object (nulos: 12)
```

</details>

<details>
<summary>20b_baseline_labs_patient_level.parquet</summary>

- `data_analytic/BTRIS/20b_baseline_labs_patient_level.parquet` · Clave patient_id.

```text
patient_id: object
clinical_baseline_episode_id: object
clinical_baseline_date: datetime64[ns]
lab_baseline_eligible: bool
lab_baseline_exclusion_reason: object (nulos: 2)
has_any_btris_lab: bool
baseline_anti_ro_ssa: boolean (nulos: 2)
ssa_measurement_date: datetime64[ns] (nulos: 2)
ssa_days_from_clinical_baseline: object (nulos: 2)
ssa_temporal_window_category: object
ssa_selection_rule: object
ssa_interpretation_source: object
ssa_n_prebaseline_measurements: int64
ssa_n_total_measurements: int64
ssa_conflict_flag: bool
ssa_postbaseline_rescue: bool
ssa_ever_positive_prebaseline: boolean (nulos: 2)
ssa_closest_prebaseline_status: boolean (nulos: 2)
ssa_longitudinal_discordance: bool
baseline_anti_la_ssb: boolean (nulos: 2)
ssb_measurement_date: datetime64[ns] (nulos: 2)
ssb_days_from_clinical_baseline: object (nulos: 2)
ssb_temporal_window_category: object
ssb_selection_rule: object
ssb_interpretation_source: object
ssb_n_prebaseline_measurements: int64
ssb_n_total_measurements: int64
ssb_conflict_flag: bool
ssb_postbaseline_rescue: bool
ssb_ever_positive_prebaseline: boolean (nulos: 2)
ssb_closest_prebaseline_status: boolean (nulos: 2)
ssb_longitudinal_discordance: bool
baseline_ana: boolean (nulos: 2)
ana_measurement_date: datetime64[ns] (nulos: 2)
ana_days_from_clinical_baseline: object (nulos: 2)
ana_temporal_window_category: object
ana_selection_rule: object
ana_interpretation_source: object
ana_n_prebaseline_measurements: int64
ana_n_total_measurements: int64
ana_conflict_flag: bool
ana_postbaseline_rescue: bool
ana_ever_positive_prebaseline: boolean (nulos: 2)
ana_closest_prebaseline_status: boolean (nulos: 2)
ana_longitudinal_discordance: bool
baseline_rf: boolean
rf_measurement_date: datetime64[ns]
rf_days_from_clinical_baseline: int64
rf_temporal_window_category: object
rf_selection_rule: object
rf_interpretation_source: object
rf_n_prebaseline_measurements: int64
rf_n_total_measurements: int64
rf_conflict_flag: bool
rf_postbaseline_rescue: bool
rf_ever_positive_prebaseline: boolean
rf_closest_prebaseline_status: boolean
rf_longitudinal_discordance: bool
baseline_cryoglobulinemia: boolean (nulos: 2)
cryoglobulinemia_measurement_date: datetime64[ns] (nulos: 2)
cryoglobulinemia_days_from_clinical_baseline: object (nulos: 2)
cryoglobulinemia_temporal_window_category: object
cryoglobulinemia_selection_rule: object
cryoglobulinemia_interpretation_source: object
cryoglobulinemia_n_prebaseline_measurements: int64
cryoglobulinemia_n_total_measurements: int64
cryoglobulinemia_conflict_flag: bool
cryoglobulinemia_postbaseline_rescue: bool
baseline_low_c4: boolean (nulos: 2)
low_c4_measurement_date: datetime64[ns] (nulos: 2)
low_c4_days_from_clinical_baseline: object (nulos: 2)
low_c4_temporal_window_category: object
low_c4_selection_rule: object
low_c4_interpretation_source: object
low_c4_n_prebaseline_measurements: int64
low_c4_n_total_measurements: int64
low_c4_conflict_flag: bool
low_c4_postbaseline_rescue: bool
baseline_leukopenia: boolean (nulos: 2)
leukopenia_measurement_date: datetime64[ns] (nulos: 2)
leukopenia_days_from_clinical_baseline: object (nulos: 2)
leukopenia_temporal_window_category: object
leukopenia_selection_rule: object
leukopenia_interpretation_source: object
leukopenia_n_prebaseline_measurements: int64
leukopenia_n_total_measurements: int64
leukopenia_conflict_flag: bool
leukopenia_postbaseline_rescue: bool
```

</details>
