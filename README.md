# Entradas clínicas y BTRIS para `obj1_sjd`

Este repositorio prepara las visitas CTDB, la espina de episodios clínicos, las mediciones BTRIS y el archivo por intervalo que todavía usan las auditorías históricas de `obj1_sjd`.

Desde la raíz de una copia de trabajo autorizada, con Python 3.12 y las fuentes congeladas, instala las dependencias con `python -m pip install -r requirements.txt`. Consulta el orden sin escribir archivos con `python src/00_super_main.py --list-steps`.

El comando canónico en el entorno autorizado es:

```bash
python src/00_super_main.py
```

La ejecución escribe las rutas operativas relativas a la raíz. Usa una copia con salidas aisladas para reproducir resultados y valida el contenido antes de publicar. Necesita los workbooks CTDB 11D/15D, los extractos BTRIS de ambos protocolos, la lista de órdenes, el mapa de reglas y la referencia de 256 pares. Los nombres, prerrequisitos y controles están en [Pipeline reproducible](docs/pipeline_minimal.md).

| Producto | Granularidad |
| --- | --- |
| `data_analytic/clinical_episode_spine_sjd.parquet` | Paciente SjD × episodio. Step 00 de `obj1_sjd` sincroniza este archivo y registra su hash. |
| `data_analytic/clinical_episode_spine_all.parquet` y CSV de ambas espinas | Paciente × episodio, incluida la cohorte completa en `all`. |
| `data_analytic/visits_long.parquet` y `data_intermediate/clinical_episode_row_map.parquet` | Visita y fila fuente. Step 08 de `obj1_sjd` los lee directamente para reconstruir documentación de medicamentos. |
| `data_intermediate/BTRIS/{11D,15D}/*.csv` | Registros fuente filtrados por MRN, incluidos los CSV de medicamentos. |
| `data_analytic/BTRIS/20_btris_lab_records_long.parquet` | Medición BTRIS por fecha con contexto de episodio. |
| `data_analytic/BTRIS/20b_baseline_labs_patient_level.parquet` | Paciente con variables basales de laboratorio. |
| `data_analytic/visits_long_collapsed_by_interval_codebook_corrected.parquet` | Paciente × intervalo. Lo lee el importador histórico de `obj1_sjd`. |

Los productores verifican sus entradas, la cobertura de asignación, la unicidad de episodios y los controles de laboratorio. El orquestador se detiene al primer proceso fallido. Revisa [el contrato de entradas](docs/obj1_input_contract.md) y [la validación de esta limpieza](docs/cleanup_validation.md) antes de usar las salidas como una publicación clínica.

`old/` conserva código y documentación históricos fuera de la ejecución y del descubrimiento de tests. El inventario y la evidencia de traslados están en [old/README.md](old/README.md).

La validación de la segunda pasada reprodujo los 15 fallos de la referencia y los resolvió en la suite activa. El paso 11 conserva los IDs textuales al leer el CSV de baseline, incluidos los ceros a la izquierda. Para comparar reproducciones fija `PYTHONHASHSEED=0` en ambas ejecuciones. El paso 09c todavía deriva parte del orden de columnas de un conjunto y falta el snapshot operativo que permitiría fijar el orden contractual.

La equivalencia operativa sigue pendiente de datos congelados y procedencia. El paso 09b es el único escritor automático del Parquet por intervalo. `11_codebook_value_audit.py` conserva su ruta en revisión hasta que el custodio identifique el productor del snapshot legado.
