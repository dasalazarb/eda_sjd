# Pipeline reproducible

Usa una copia de trabajo autorizada con Python 3.12 y las versiones de `requirements.txt`. Instala las dependencias en un entorno virtual desde la raíz. En Biowulf usa el Python disponible que pueda reproducir las versiones verificadas, y registra cualquier diferencia antes de publicar. Para una ejecución sin pantalla se puede exportar `MPLBACKEND=Agg`.

## Fuentes congeladas

| Ruta relativa a la copia | Contenido |
| --- | --- |
| `data_raw/CTDB Data Download 11D.xlsx`, `data_raw/CTDB Data Download 15D.xlsx` | Exportaciones CTDB con dos filas de encabezado. 01 admite las variantes históricas de nombre registradas en su código. |
| `data_raw/BTRIS/11D/`, `data_raw/BTRIS/15D/` | Extractos CSV autorizados, incluidos `Lab*.csv` y los originales de `Medication288444-1.csv` y `Medication288476-1.csv`. |
| `data_raw/unique_OrderSets.csv` | Encabezado `Order Name`. El paso 19 admite alternativa XLSX o CLI. |
| `btris_lab_rule_families.xlsx` | Hoja `Lab rule map` con `Order Name`, `Cluster Name` y `Suggested Rule Family`. |
| `data_intermediate/BTRIS/20_lab_order_cluster_reference.csv` | Referencia autorizada de 256 pares exactos, con anotaciones de analito, familia y rol cuando proceda. |

Conserva una copia inmutable y hashes de estas fuentes. Trabaja con directorios de salida nuevos o aislados. El paso 20c escribe sobre **los intermedios filtrados** y no debe apuntar al directorio raw. Usa la misma estructura de rutas en la copia para que las CLI de los productores resuelvan los mismos defaults.

## Orden canónico

`python src/00_super_main.py --list-steps` imprime el plan. `python src/00_super_main.py` lo ejecuta con el mismo intérprete y se detiene al primer error.

| Paso | Razón y salida usada |
| --- | --- |
| 01 | Normaliza CTDB, aplica el etiquetado de intervalos vigente y asigna linaje. |
| 04 | Deduplica dentro de protocolo. Lee los dos productos de 01. |
| 05 | Publica `visits_long` desde `deduped_visits`. |
| 09 → 09b | Genera el colapso por intervalo y armoniza ESSDAI por la ruta anterior de 00. Preserva el archivo requerido por el importador y auditorías históricas de obj1. |
| 08c | Asigna filas, construye el mapa y el manifiesto y conserva procedencia. |
| 09c | Colapsa por episodio con el mapa y manifiesto. Escribe la tabla y los conflictos que requiere 09d. El colapso por intervalo solo alimenta un resumen comparativo. |
| 09d | Finaliza los episodios con las reglas vigentes de precedencia, conflictos, ESSDAI y serología. |
| 10 | Deriva baseline y elegibilidad a partir del resultado de 09d. El CSV de comparación alimenta al paso 11. |
| 11 | Valida el merge, las claves, la cardinalidad y el baseline. Publica las espinas. |
| 19 → 20c | Filtra BTRIS por la cohorte del spine y armoniza archivos de laboratorio filtrados según sus reglas. Los CSV de medicamentos se conservan como fuente documental para obj1. |
| 20 → 20b | Asocia mediciones con contexto clínico y deriva perfiles de laboratorio basales. Step 20 exige 256 pares únicos. Step 20b genera QC y devuelve un fallo si hay violaciones bloqueantes. |

01, 04 y 05 todavía generan reportes dentro de los propios productores. 05 conserva también `patient_master` y su boxplot. Se preservaron estos efectos junto con las implementaciones de los productores. 11 Codebook conserva su CLI mientras se comprueba el linaje del snapshot histórico. El orquestador usa únicamente la rama histórica 09 → 09b para ese nombre público.

## Entrega y control

Verifica todo el pipeline y sus informes antes de ofrecer las salidas a obj1. Algunos productores escriben resultados antes de devolver un error de QC. Un archivo existente puede pertenecer a una ejecución parcial. Compara esquemas, hashes, cardinalidad, claves, fechas, nulos y valores en la copia aislada antes de sincronizar. El Step 00 de obj1 copia la espina y registra su procedencia. Sus pasos 01 y 08 leen los productos complementarios directamente. Para el legado, revisa el contrato y el origen del snapshot antes de sincronizar o reconstruirlo.

Ejecuta los tests sintéticos con `python -m pytest`. Consulta los resultados y prerrequisitos externos en [cleanup_validation.md](cleanup_validation.md).

### Reproducir sin escribir sobre una publicación

Desde una copia aislada que contenga las fuentes autorizadas en las rutas anteriores y un entorno virtual activado, usa `PYTHONHASHSEED=0 MPLBACKEND=Agg python src/00_super_main.py`. Aplica la misma semilla a la referencia. Este ajuste permite una comparación controlada del orden de columnas de 09c. Antes de una publicación, confirma el orden contractual contra el snapshot autorizado y revisa el resultado de la comparación por contenido. El paso 11 lee los IDs de comparación basal como texto y asigna `False` a la bandera de baseline si el episodio basal está ausente.
