# Cierre de dependencias de `eda_sjd`

Revisiones y rutas exactas en [el contrato](obj1_input_contract.md). El [inventario](cleanup_inventory.csv) registra las decisiones por archivo. Las flechas representan lecturas de datos, salvo la relación de import explícita.

```text
CTDB 11D/15D
  → 01 → raw_enriched → 04 → deduped_visits → 05 → visits_long
     visits_long ──────────────────────────────────────────→ obj1 Step 08
     visits_long → 08c → row_map ──────────────────────────→ obj1 Step 08
                      └→ manifest
     visits_long + row_map + manifest → 09c → episodios sin finalizar + conflictos
     visitas + row_map + manifest + salida 09c → 09d → episodios finalizados
     episodios finalizados → 10 → comparación baseline
     episodios finalizados + comparación baseline → 11 → spine SjD → obj1 Step 00
     visits_long → 09 → colapso por intervalo → 09b → corregido por intervalo
                                                         → importador y auditorías obj1
BTRIS raw 11D/15D + spine SjD + unique_OrderSets → 19 → CSV BTRIS filtrados
     CSV Medication* ──────────────────────────────────────→ obj1 Step 08
     CSV Lab* + reglas → 20c → CSV Lab* armonizados
     CSV Lab* armonizados + spine SjD + referencia 256 pares → 20 → labs long
     labs long + spine SjD → 20b → baseline labs
     labs long + baseline labs + obj1 Step 00 → obj1 Step 01
     BTRIS raw, intermedios y labs long  ───────────────────→ obj1 Graph 03 N3
obj1 Step 00 + Step 01 + Step 08 + otros derivados internos → obj1 Step 10
obj1 Step 10 → Pharma, Graph, Longitudinal Graph y análisis internos
```

## Evidencia de arcos críticos

| Arco | Referencias en las revisiones auditadas |
| --- | --- |
| Exportaciones → 01 → 04 → 05 | `src/01_ingest.py:125–152`, `src/04_dedup_rules.py:136–148`, `src/05_build_backbone.py:166–173`. 04 no lee los informes de enlace de 03. |
| 05 → 08c → 09c | `src/08c_build_clinical_episode_map.py:33–35,1245`, `src/09c_clinical_episode_collapse.py:832–851`. |
| 09c, 08c, 05 → 09d → 10 → 11 | `src/09d_finalize_clinical_episode_dataset.py:1187–1231`, `src/10_clinical_baseline_qc.py:18–24,320`, `src/11_build_clinical_episode_spine.py:28–39,257–259`. |
| 11 → obj1 Step 00 → Step 10 | Obj1 `common.py:16–24,44–50`, `src/00_build_visit_spine.py:331–368`, y validaciones y defaults de `src/block_A/10_build_integrated_longitudinal_dataset.py`. |
| 05 → 09 → 09b → copy legado | EDA `src/09_interval_collapse_audit.py:470,545–549`, `src/09b_merge_essdai_versions.py:68–76,610–611`, obj1 `src/00_input data.py:20–24,142–154`. |
| 11, raw y referencias → 19 → 20c → 20 → 20b | EDA `src/19_filter_btris_patients.py:55–77,368–377`, `src/20c_btris_lab_rule_harmonization.py:18,30–35,288–367`, `src/20_btris_visit_date_match_report.py:811–846,1851–1911`, `src/20b_btris_baseline_labs.py:24–27,1704–1714`. |
| 05, 08c, 19 → obj1 Step 08 | Obj1 `src/block_A/08_build_medications.py:43–48`; `src/block_A/_internal/medications/source_inputs.py:9–10,30–47`. |
| 20 y 20b → obj1 Step 01, 19/20 → Graph N3 | Obj1 `src/block_A/01_serological_profile.py:48–53,271–277`; `src/studies/graph/configs/n3_auto.yaml:4–14` y adaptador calendar-day. |

## Dependencias condicionales y antiguas

- El colapso por intervalo es necesario para el consumidor legado. Su entrada a 09c es condicional y únicamente alimenta un informe comparativo (`09c:859–863`). No reemplaza el mapa, las fechas ni la espina por episodio.
- El productor alternativo `11_codebook_value_audit.py` acepta su propia tabla de entrada o una CLI y publica el mismo nombre legado que 09b. Está en REVIEW y fuera del plan automático hasta identificar el productor del snapshot. Ejecutarlos en secuencia sobrescribiría el mismo nombre.
- Las reglas 20c (`btris_lab_rule_families.xlsx`) y la referencia Step 20 de 256 pares son dependencias de entrada. El código versionado no las genera. Los parquets BTRIS históricos de medicamentos tienen consumidores, aunque tampoco se encontró un productor en EDA.
- Los archivos en [old](../old/README.md) producen perfiles, figuras, auditorías exploratorias, ramas alternativas o diagnósticos sin un arco a estas entradas. Las pruebas asociadas viajan con su código. Sigue habiendo reportes incidentales de productores retenidos.
- `common.py` y `btris_lab_harmonization.py` siguen activos por imports directos. Ningún productor activo importa desde el archivo histórico.

## Identidad, fecha y alcance científico

08c conserva IDs de filas, abre la asignación con fecha de colección y clasifica el episodio según precedencia e información del componente. El manifiesto recoge inicio, ancla, fin, tipo y bandera clínica. 09c verifica cobertura y unicidad de la unión, preserva conflictos y separa la procedencia. 09d consulta los valores fuente y la fecha de colección para aplicar reglas existentes, incluida preferencia por el ancla cuando corresponde. Una selección por proximidad puede considerar otras fechas y no equivale a observación disponible al ancla. 10 deriva el primer baseline clínico y elegibilidad. 11 une sus atributos sin eliminar episodios ambiguos o de investigación de pacientes SjD. Step 00 de obj1 construye aparte el subconjunto clínico.

Las fechas de espécimen, orden, documentación y consentimiento tienen semánticas distintas a la fecha de visita. BTRIS conserva la fecha de observación y marca su relación con el ancla. 20b tiene rescates posteriores documentados y auditados. La evidencia presentada cubre preservación de las reglas existentes. El uso temporal de cada variable derivada en modelos requiere el contrato de observabilidad de obj1 y validación con fuentes autorizadas.

## Cierre de la segunda pasada

La auditoría del árbol actual y las pruebas de Steps 00, 01 y 08 mantienen el cierre anterior. Graph N3 sigue leyendo BTRIS raw y filtrado directamente, además del Parquet largo. 09c consulta el colapso por intervalo para una comparación auxiliar. El plan de 14 pasos conserva un escritor automático para el Parquet legado, 09b. La ausencia de procedencia del snapshot impide retirar `11_codebook_value_audit.py`. El paso 11 ahora interpreta como texto los IDs del CSV de 10 y compara IDs basales presentes. El parche preserva las rutas y las reglas de selección.
