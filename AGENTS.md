# Instrucciones para trabajar en `eda_sjd`

El pipeline publica entradas clínicas y BTRIS para `obj1_sjd`. Lee [el orden de ejecución](docs/pipeline_minimal.md), [el contrato](docs/obj1_input_contract.md) y [el cierre de dependencias](docs/cleanup_dependency_graph.md) antes de cambiar productores o rutas.

## Protección de datos y resultados

- Trata `data_raw/` como solo lectura. Conserva los extractos, reglas, identificadores, rutas y archivos previos. No copies datos clínicos a Git, al archivo histórico, a entornos sin autorización ni a fixtures.
- Pide autorización antes de ejecutar el pipeline productivo o cualquier paso que escriba resultados clínicos. Usa datos sintéticos y copias aisladas para pruebas. Cualquier reemplazo de una publicación requiere respaldo, comparación de contenido y revisión humana.
- No hagas commit, push, merge ni operaciones destructivas sin autorización explícita.
- Los cambios de cohortes, criterios de enlace, claves públicas, fechas, baseline, precedencia, serología, valores o formatos públicos requieren revisión humana. No cambies la lógica para satisfacer un test.

## Código y validación

- Mantén los nombres y rutas públicas de los productores. Los ejecutables en `src/` conservan su prefijo numérico. Usa `src/common.py` para rutas, logging e I/O compartidos.
- Usa `pathlib.Path`, funciones pequeñas, tipos en las firmas y docstrings NumPy para funciones públicas nuevas. Conserva las convenciones de trazabilidad de salidas y registra las dependencias nuevas.
- El orquestador es `src/00_super_main.py`. Usa `--list-steps` para verificar su orden sin generar resultados. Antes de un cambio estructural o de secuencia, presenta el impacto y espera autorización si el alcance aún no está aprobado.
- Ejecuta las pruebas en `tests/` con `python -m pytest`. Usa fixtures sintéticos o anonimizados y añade pruebas de regresión cuando cambie un contrato.
- `old/` conserva el historial de archivos retirados. No lo añadas al import path ni ejecutes su código como parte del pipeline, pruebas activas, linters o trabajos SLURM. Si se prepara una entrega para Biowulf, ejecuta el comando canónico desde una copia de trabajo autorizada.
