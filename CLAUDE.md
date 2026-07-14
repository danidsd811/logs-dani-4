# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Run the full stack
```bash
cd log_analyzer_v2
docker-compose up --build      # primera vez o tras cambios en Dockerfile
docker-compose up              # arranque normal (datos persistentes en volumen postgres_data)
docker-compose down            # parar sin borrar datos
docker-compose down -v         # parar Y borrar la BD (reset completo)
```

- Backend FastAPI: http://localhost:8000
- Frontend React: http://localhost:3000
- PostgreSQL: localhost:5432 (`loguser` / `logpass` / `logs_analyzer`)

### Desarrollo sin Docker
```bash
# Backend
cd log_analyzer_v2/backend
pip install -r requirements.txt
DATABASE_URL=postgresql://loguser:logpass@localhost:5432/logs_analyzer uvicorn main:app --reload --port 8000

# Frontend
cd log_analyzer_v2/frontend
npm install
npm start   # dev server con hot reload en :3000
```

## Arquitectura

```
log_analyzer_v2/
├── backend/
│   ├── main.py               # Toda la lógica: FastAPI, parsers, analytics
│   └── configs/
│       ├── clients_registry.json   # Lista de todos los clientes (mínimo: id + name)
│       └── <cliente_id>.json       # Config completa con analytics (opcional)
└── frontend/
    └── src/
        ├── App.jsx                          # Orquestador: estado global, fetching, routing entre pestañas
        ├── constants.js                     # API_BASE, SCALE_COLORS, ODS_*, BLOCKED_* (compartidos)
        └── components/
            ├── CustomerCombobox.jsx         # Dropdown con búsqueda para selección de cliente
            ├── UploadTab.jsx                # Formulario de subida de logs
            ├── ViewLogsTab.jsx              # Tabla paginada con búsqueda
            ├── HostpicsTable.jsx            # Tabla de HOSTPICs con filtros y paginación propia
            └── ChartsTab/
                ├── index.jsx                # Selector de BD + loading state + monta los charts
                ├── SortQualityChart.jsx     # Calidad de Clasificación por ODS (stacked bar)
                ├── InductionQualityChart.jsx # Calidad de Inducción por Infeed + detalle HOSTPICs
                ├── ScaleQualityChart.jsx    # Calidad de Básculas por Infeed
                └── BlockedStatusChart.jsx   # Motivos de Mal Inducido (Parcel Blocked Status)
```

### Backend (`main.py`)

Un solo archivo. Clases y funciones principales por orden de aparición:

| Elemento | Propósito |
|---|---|
| `load_customer_configs()` | Carga todos los JSON de `configs/` al arrancar. Construye `_zone_reverse` como índice invertido entry_point→zone. |
| `UltraSchema` | Parser para logs **eDS** (`.log` + `.json`). Lee `ChannelsConfiguration[].MessageConfiguration[]`, usa MessageID 20 como orden de columnas, fallback a MessageID 20 → resto de IDs. |
| `SCNETSchemaOptimized` | Parser para logs **SCNET** (`.fsc/.log` + `.xml`). Detecta automáticamente el formato del XML. |
| `process_ultra_fast()` | Pipeline eDS: construye UltraSchema → crea tabla → lee log en batches de 50k → inserta. |
| `process_scnet_ultra_fast()` | Pipeline SCNET: igual que el anterior pero con SCNETSchemaOptimized. Devuelve `debug_info` para diagnóstico cuando `records_processed == 0`. |
| `cleanup_old_tables()` | Rotación automática: mantiene máximo `MAX_TABLES = 10` tablas, borra las más antiguas. |
| Endpoints analytics | `/databases/{id}/induction_quality`, `sort_quality`, `scale_quality`, `good_hostpics`, `bad_hostpics` — todos leen la tabla de logs con SQL y aplican la lógica del config del cliente. |

La tabla de metadatos `databases` (no las tablas de logs) guarda: `id`, `table_name`, `record_count`, `file_size_mb`, `columns_info` (JSONB), `customer_id`.

Las tablas de logs se nombran `logs_<número>` (eDS) o `scnet_<número>` (SCNET), donde el número se extrae del nombre del archivo.

### Dos formatos SCNET

`SCNETSchemaOptimized._build_schema_from_xml()` detecta el formato por el elemento raíz:

| Formato | XML | Log FSC |
|---|---|---|
| **Moderno** (`ParcelDataReportConfig.xml`) | `<ParcelDataReport id="20">` con `<property>` hijos | `timestamp\|MSG_ID\|campo1\|campo2\|...` |
| **Legado** (`NetworkConfig.xml`) | `<parceldata reportid="...">` con `<text>N</text>` + `<property>` hijos | `DD/MM/YYYY HH:MM:SS:mmm,\x02MSG_ID\|campo1\|campo2\|...\|[checksum]@` |

La flag `_is_network_config = True` activa:
- `field_offset = 1` (campos empiezan en `parts[1]`; en moderno es 2)
- El STX (`\x02`) se elimina con `_CTRL_CHARS_RE` antes del split
- El `MSG_ID` va embebido en `parts[0]` tras la última coma: `ts_str, mid_str = parts[0].rsplit(',', maxsplit desde rfind)`
- Orden de columnas por mensaje 56 (DivertSuccessSortReport), no por mensaje 20
- Mensaje 51 hardcodeado: `{pic: parts[1], hostpic: parts[2]}` (nunca aparece en el XML)

`col_offset` (0 o 1) es independiente: compensa la columna FSC extra en Crossorter XXL.

### Configs de cliente

Cada archivo `configs/<id>.json` tiene esta estructura:

```json
{
  "id": "cliente_ciudad_pais",
  "name": "Nombre legible",
  "charts": ["induction_quality", "sort_quality", "scale_quality"],
  "hostpic_column": "hostpic",
  "entry_point": {
    "column": "parcelentrancepoint",
    "column_aliases": ["entrancepoint", "entrance_point"],
    "pattern": "0*50\\.([0-9]{2})\\.",
    "group": 1
  },
  "zones": [
    {"id": 1, "name": "INFEED 1", "codes": ["04","05","06"]},
    {"id": 99, "name": "Loop", "codes": []}
  ],
  "default_zone_id": 99,
  "sort_report": {"message_id": 20},
  "good_package": {
    "message_id": 20,
    "conditions": [
      {"column": "lastdestination", "op": "!=", "value": "998"},
      {"column": "originaldestinationstate", "op": "=", "value": "1", "optional": true}
    ]
  },
  "bad_package": {
    "message_id": 21,
    "conditions": [{"column": "lastdestination", "op": "=", "value": "998"}]
  },
  "scale_quality": {
    "message_id": 20,
    "field": "ScannerData1",
    "ok_prefix": "6",
    "ok_contains": "g",
    "no_scan_value": "1  0",
    "error_codes": {"1": "No Data received", ...},
    "exclude_from_pct": ["8", "9"],
    "pre_scale_entry_points": ["50.04.01", ...]
  }
}
```

- `charts` controla qué gráficas se muestran en Analytics. Valores posibles: `induction_quality`, `sort_quality`, `scale_quality`, `blocked_status`.
- `column_aliases` en `entry_point` permite encontrar la columna aunque se llame distinto en cada instalación.
- Las condiciones de `good_package`/`bad_package` usan `find_column()` que normaliza nombre (minúsculas, sin guiones bajos) para ser agnóstico al nombre exacto de la columna.
- Los clientes sin analytics completo van en `clients_registry.json` (solo `id` + `name`).

### Frontend

El estado global (databases, selectedDatabase, analyticsLoading, todos los datos de analytics) vive en `LogAnalyzerApp` (`App.jsx`). Tres pestañas principales:

- **Upload & Process** (`UploadTab.jsx`): formulario de subida; cliente obligatorio (valida en `handleUpload` y deshabilita el botón). Usa `CustomerCombobox` para el selector.
- **View Logs** (`ViewLogsTab.jsx`): tabla paginada con búsqueda en todas las columnas de texto (`ILIKE`). La búsqueda resalta coincidencias con `<span>`.
- **Analytics & Charts** (`ChartsTab/index.jsx`): selector siempre visible (nunca oculto por early return); estado `analyticsLoading` propio independiente del `loading` de View Logs.

Cada chart (`SortQualityChart`, `InductionQualityChart`, `ScaleQualityChart`, `BlockedStatusChart`) gestiona su propio estado de minimizado internamente. Todos los charts nuevos deben ser minimizables. Reciben `selectedDatabaseId` para resetear filtros cuando cambia la BD. `fetchDatabases()` auto-selecciona `data[0]` cuando `selectedDatabase` es null.

El endpoint `/databases/{id}/blocked_status` usa bitmask: `parcel_blocked_status` es suma de potencias de 2 (FrontFault=1, RearFault=2, MultipleCarriers=4, etc.). El chart descompone cada valor en flags individuales; un HOSTPIC puede contabilizarse en varios flags simultáneamente.
