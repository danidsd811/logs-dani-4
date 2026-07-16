from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import tempfile
import os
import json
import re
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel
from datetime import datetime
import asyncpg
import time
import logging
import xml.etree.ElementTree as ET

# Configuración
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# *** NUEVO: Constante para rotación de tablas ***
MAX_TABLES = 10

# ─── Customer configs ──────────────────────────────────────────────────────────
CUSTOMER_CONFIGS: Dict[str, Dict] = {}

def load_customer_configs():
    """Carga todos los JSON de clientes desde configs/"""
    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    if not os.path.exists(configs_dir):
        logger.warning(f"configs/ directory not found at {configs_dir}")
        return

    # Cargar configs completos (cualquier JSON que tenga campo "id" como objeto)
    for fname in sorted(os.listdir(configs_dir)):
        if not fname.endswith('.json') or fname == 'clients_registry.json':
            continue
        path = os.path.join(configs_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            if not isinstance(cfg, dict) or 'id' not in cfg:
                continue
            cfg['_zone_reverse'] = {
                code: zone
                for zone in cfg.get('zones', [])
                for code in zone.get('codes', [])
            }
            CUSTOMER_CONFIGS[cfg['id']] = cfg
            logger.info(f"Loaded customer config: {cfg['name']} ({fname})")
        except Exception as e:
            logger.error(f"Failed to load config {fname}: {e}")

    # Cargar registro de clientes (entradas mínimas sin analytics)
    registry_path = os.path.join(configs_dir, 'clients_registry.json')
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            added = 0
            for entry in registry:
                cid = entry.get('id')
                if cid and cid not in CUSTOMER_CONFIGS:
                    CUSTOMER_CONFIGS[cid] = entry
                    added += 1
            logger.info(f"Loaded {added} clients from registry ({len(registry)} total entries)")
        except Exception as e:
            logger.error(f"Failed to load clients registry: {e}")

def get_customer_config(customer_id: Optional[str]) -> Optional[Dict]:
    """Devuelve el config del cliente. Si no hay customer_id, usa el primero con analytics completo."""
    if customer_id:
        return CUSTOMER_CONFIGS.get(customer_id)
    # Backward compat: bases de datos antiguas sin customer_id → primer config completo
    return next((cfg for cfg in CUSTOMER_CONFIGS.values() if cfg.get('good_package')), None)

def find_column(columns: List[str], *patterns: str) -> Optional[str]:
    """Busca en columns cualquier nombre que coincida con los patrones (ignora mayúsculas y guiones bajos)"""
    needles = {p.lower().replace('_', '') for p in patterns}
    return next((c for c in columns if c.lower().replace('_', '') in needles), None)

def get_zone_for_entry_point(entry_point: str, config: Dict) -> Dict:
    """Devuelve el dict de zona para el entry point dado, según el config del cliente"""
    default_id = config.get('default_zone_id', 99)
    zones = config.get('zones', [])
    default_zone = next((z for z in zones if z['id'] == default_id), {'id': 99, 'name': 'Unknown'})

    if not entry_point or entry_point.strip() in ('-', ''):
        return default_zone

    ep_cfg = config.get('entry_point', {})
    pattern = ep_cfg.get('pattern')
    group = ep_cfg.get('group', 1)
    if not pattern:
        return default_zone

    match = re.search(pattern, entry_point.strip())
    if match:
        zone = config.get('_zone_reverse', {}).get(match.group(group))
        if zone:
            return zone

    return default_zone

def build_condition_sql(conditions: List[Dict], columns: List[str]) -> str:
    """Genera cláusula SQL AND a partir de las condiciones del config de cliente"""
    parts = []
    for cond in conditions:
        col = find_column(columns, cond['column'])
        if col is None:
            if cond.get('optional', False):
                continue
            # Condición requerida ausente → query no debe devolver filas falsas
            logger.warning(f"Required condition column '{cond['column']}' not found — query will return no rows")
            return '1=0'
        op = cond['op']
        val = cond['value'].replace("'", "''")
        parts.append(f"{col}::text {op} '{val}'")
    return ' AND '.join(parts) if parts else '1=1'

def _safe_sql_name(name: str) -> str:
    """Sanitiza un identificador SQL eliminando caracteres no permitidos."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

# ───────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Log Analyzer - Ultra Optimized", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://loguser:logpass@postgres:5432/logs_analyzer")
db_pool = None

# Regex compilados para timestamp parsing ultra-rápido
TIMESTAMP_PATTERNS = [
    re.compile(r'^(\d{2})/(\d{2})/(\d{4}) (\d{2}):(\d{2}):(\d{2})\.(\d{3})$'),
    re.compile(r'^(\d{2})/(\d{2})/(\d{4}) (\d{2}):(\d{2}):(\d{2})$')
]
# Regex compilada para limpiar caracteres de control (usada en hot path SCNET)
_CTRL_CHARS_RE = re.compile(r'[\x00-\x1F]')
_NON_DIGITS_RE = re.compile(r'[^0-9]')

# *** NUEVA FUNCIÓN: Sistema de rotación automática ***
async def cleanup_old_tables():
    """Mantener máximo 10 tablas, eliminar las más antiguas"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # VERIFICAR que la tabla databases existe antes de hacer queries
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'databases'
            )
        """)
        
        if not table_exists:
            logger.info("Table 'databases' does not exist yet - skipping cleanup")
            return
        
        # Obtener todas las tablas ordenadas por fecha de creación
        tables = await conn.fetch("""
            SELECT id, table_name, created_at, record_count 
            FROM databases 
            ORDER BY created_at ASC
        """)
        
        if len(tables) > MAX_TABLES:
            tables_to_delete = tables[:-MAX_TABLES]  # Mantener solo las últimas 10
            
            for table in tables_to_delete:
                table_name = table['table_name']
                try:
                    # Eliminar la tabla física
                    await conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
                    
                    # Eliminar registro de metadatos
                    await conn.execute("DELETE FROM databases WHERE id = $1", table['id'])
                    
                    logger.info(f"Deleted old table: {table_name} ({table['record_count']:,} records)")
                    
                except Exception as e:
                    logger.error(f"Error deleting table {table_name}: {e}")
            
            logger.info(f"Cleanup complete: Removed {len(tables_to_delete)} old tables")
        else:
            logger.info(f"Current tables: {len(tables)}/{MAX_TABLES} - no cleanup needed")

# Models
class ProcessResponse(BaseModel):
    success: bool
    message: str
    processing_time: float
    records_processed: int
    database_id: str
    table_name: str
    debug_info: Optional[str] = None

class LogQuery(BaseModel):
    page: int = 1
    limit: int = 100
    search: Optional[str] = None
    message_id: Optional[int] = None

class LogResponse(BaseModel):
    data: List[Dict[str, Any]]
    total_records: int
    page: int
    total_pages: int
    processing_time: float
    columns: List[str]

class DatabaseInfo(BaseModel):
    id: str
    name: str
    table_name: str
    created_at: datetime
    record_count: int
    file_size_mb: float
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None

class UltraSchema:
    """Esquema ultra-optimizado con mapeo por MessageField (CORREGIDO)"""
    
    def __init__(self, config: Dict):
        self.columns = []  # [(column_name, sql_type)]
        self.message_field_to_table_index = {}  # MessageField -> índice en tabla
        self.message_id_to_field_positions = {}  # {msg_id: {field_name: log_position}}
        self.column_count = 0
        self._build_schema_fixed(config)
    
    def _build_schema_fixed(self, config: Dict):
        """Construir esquema usando MessageField en lugar de Order"""
        
        # Extraer configuración del MessageID 20
        msg_20_fields = self._get_message_20_fields(config)
        if not msg_20_fields:
            # Si no hay MessageID 20, usar el primer MessageID disponible
            msg_20_fields = self._get_first_available_message_fields(config)
            
        logger.info(f"Using reference fields from MessageID for column ordering: {len(msg_20_fields)} fields")
        
        # Construir esquema de tabla basado en DefaultOrder
        self.columns = [("timestamp", "TIMESTAMP"), ("message_id", "INTEGER")]
        
        # Ordenar por DefaultOrder y crear columnas
        for default_order, field_name in sorted(msg_20_fields):
            safe_name = _safe_sql_name(field_name.lower())
            self.columns.append((safe_name, "TEXT"))
            # Mapear MessageField -> índice en tabla
            self.message_field_to_table_index[field_name] = len(self.columns) - 1
        
        self.column_count = len(self.columns)
        self.columns.append(("_search_text", "TEXT"))
        self.column_count = len(self.columns)

        # Pre-computar mapeos para cada MessageID
        self._build_message_mappings(config)
        
        logger.info(f"Schema built: {self.column_count} columns, {len(self.message_id_to_field_positions)} MessageIDs mapped")
    
    def _get_message_20_fields(self, config: Dict):
        """Extraer campos del MessageID 20 para definir orden de columnas"""
        for channel in config.get("ChannelsConfiguration", []):
            for msg in channel.get("MessageConfiguration", []):
                if msg.get("MessageId") == 20:
                    fields = []
                    for field in msg.get("EnabledFields", []):
                        field_name = field.get("MessageField")
                        default_order = field.get("DefaultOrder", field.get("Order", 999))
                        if field_name:
                            fields.append((default_order, field_name))
                    return fields
        return []
    
    def _get_first_available_message_fields(self, config: Dict):
        """Fallback: usar primer MessageID disponible si no hay MessageID 20"""
        for channel in config.get("ChannelsConfiguration", []):
            for msg in channel.get("MessageConfiguration", []):
                msg_id = msg.get("MessageId")
                if 1 <= msg_id <= 100 and msg_id not in [30, 99]:
                    fields = []
                    for field in msg.get("EnabledFields", []):
                        field_name = field.get("MessageField")
                        default_order = field.get("DefaultOrder", field.get("Order", 999))
                        if field_name:
                            fields.append((default_order, field_name))
                    if fields:
                        logger.warning(f"MessageID 20 not found, using MessageID {msg_id} for column order")
                        return fields
        return []
    
    def _build_message_mappings(self, config: Dict):
        """Pre-computar mapeo de posiciones para cada MessageID con fallback a MessageID 20"""
    
        # Primero mapear todos los MessageIDs explícitamente definidos
        explicitly_defined = set()
    
        for channel in config.get("ChannelsConfiguration", []):
            for msg in channel.get("MessageConfiguration", []):
                msg_id = msg.get("MessageId")
                if 1 <= msg_id <= 100 and msg_id not in [30, 99]:
                
                    # Mapear MessageField -> posición en log
                    field_positions = {}
                    for log_position, field in enumerate(msg.get("EnabledFields", [])):
                        field_name = field.get("MessageField")
                        if field_name:
                            # Posición en log = posición en EnabledFields + 2 (por timestamp y message_id)
                            field_positions[field_name] = log_position + 2
                
                    if field_positions:
                        self.message_id_to_field_positions[msg_id] = field_positions
                        explicitly_defined.add(msg_id)
                        logger.info(f"MessageID {msg_id}: mapped {len(field_positions)} fields (explicit)")
    
        # Obtener la estructura del MessageID 20 para usar como fallback
        fallback_structure = self.message_id_to_field_positions.get(20)
    
        if fallback_structure:
            # Aplicar fallback para MessageIDs válidos no definidos
            valid_message_ids = set(range(1, 101)) - {30, 99}  # 1-100 excepto 30 y 99
            undefined_message_ids = valid_message_ids - explicitly_defined
        
            for msg_id in undefined_message_ids:
                # Usar la misma estructura del MessageID 20
                self.message_id_to_field_positions[msg_id] = fallback_structure.copy()
                logger.info(f"MessageID {msg_id}: using fallback structure from MessageID 20 ({len(fallback_structure)} fields)")
        
            logger.info(f"Applied MessageID 20 fallback to {len(undefined_message_ids)} undefined MessageIDs: {sorted(undefined_message_ids)}")
        else:
            logger.warning("MessageID 20 not found - no fallback structure available for undefined MessageIDs")
    
        total_mapped = len(self.message_id_to_field_positions)
        logger.info(f"Total MessageIDs mapped: {total_mapped} (explicit: {len(explicitly_defined)}, fallback: {total_mapped - len(explicitly_defined)})")
    
    def parse_timestamp_fast(self, ts_str: str) -> Optional[datetime]:
        """Timestamp parsing ultra-rápido con regex compilados"""
        for pattern in TIMESTAMP_PATTERNS:
            match = pattern.match(ts_str)
            if match:
                try:
                    groups = [int(g) for g in match.groups()]
                    if len(groups) == 7:  # Con milisegundos
                        d, m, y, h, min, s, ms = groups
                        return datetime(y, m, d, h, min, s, ms * 1000)
                    else:  # Sin milisegundos
                        d, m, y, h, min, s = groups
                        return datetime(y, m, d, h, min, s)
                except ValueError:
                    continue
        return None
    
    def parse_line_ultra_fast(self, line: str) -> Optional[List]:
        """Parse ultra-optimizado usando mapeo por MessageField (CORREGIDO)"""
        if not line.strip():
            return None
        
        parts = line.split('|', self.column_count + 10)  # Margen extra para seguridad
        if len(parts) < 2:
            return None
        
        try:
            # Parse timestamp y message_id
            timestamp = self.parse_timestamp_fast(parts[0].strip())
            message_id = int(parts[1].strip())
            
            if message_id not in self.message_id_to_field_positions:
                return None
            
            # Inicializar record con "-" por defecto
            record = ["-"] * self.column_count
            record[0] = timestamp
            record[1] = message_id
            
            # MAPEO CORREGIDO: Usar MessageField como clave
            field_positions = self.message_id_to_field_positions[message_id]
            
            for field_name, log_position in field_positions.items():
                # Verificar que el campo existe en el esquema de tabla
                if field_name in self.message_field_to_table_index:
                    # Verificar que hay datos en esa posición del log
                    if log_position < len(parts):
                        value = parts[log_position].strip()
                        if value:  # Solo sobrescribir "-" si hay valor real
                            table_index = self.message_field_to_table_index[field_name]
                            record[table_index] = value
            
            record[-1] = ' '.join(' '.join(v.split()) for v in record[2:-1] if v and v != '-')
            return record

        except (ValueError, IndexError) as e:
            logger.debug(f"Parse error for line: {line[:50]}... Error: {e}")
            return None

    def get_create_table_sql(self, table_name: str) -> str:
        """SQL para crear tabla (sin índices — se crean después del COPY para mayor velocidad)"""
        safe_name = _safe_sql_name(table_name)
        column_defs = ["id BIGSERIAL PRIMARY KEY"]
        for col_name, col_type in self.columns:
            column_defs.append(f"{col_name} {col_type}")
        return f"CREATE TABLE {safe_name} ({', '.join(column_defs)})"


class SCNETSchemaOptimized:
    """Esquema SCNET corregido con orden del MessageID 20 y parsing funcional"""
    
    def __init__(self, xml_path: str):
        self.columns = []  # [(column_name, sql_type)]
        self.message_id_to_field_positions = {}  # {msg_id: {field_name: log_position}}
        self.property_to_table_index = {}  # property_name -> índice en tabla
        self.column_count = 0
        self.col_offset = 0  # 1 para Crossorter XXL (columna FSC extra tras timestamp)
        self._is_network_config = False  # True si el XML es NetworkConfig.xml (SCNET legado)
        self._build_schema_from_xml(xml_path)
    
    def _build_schema_from_xml(self, xml_path: str):
        """Construir esquema soportando ParcelDataReportConfig.xml y NetworkConfig.xml (SCNET legado)"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Detectar formato:
            # - ParcelDataReportConfig.xml → <ParcelDataReport id="20">
            # - NetworkConfig.xml (legado) → <parceldata reportid="DivertSuccessSortReport"> <text>56</text>
            report_elements = root.findall('.//ParcelDataReport')
            if not report_elements:
                report_elements = root.findall('.//parceldata')
                if report_elements:
                    self._is_network_config = True
                    logger.info("Detected NetworkConfig.xml format (legacy SCNET)")

            # PASO 1: Extraer TODOS los MessageIDs y sus propiedades
            parcel_reports = {}
            all_properties = set()
            reference_properties_order = []  # Para MessageID 20 (o el más rico disponible)

            for report in report_elements:
                # ParcelDataReportConfig usa 'id'; NetworkConfig usa 'reportid'
                report_id = report.get('id') or report.get('reportid') or ''

                # Determinar MessageID real
                message_number = None
                text_element = report.find('text')

                if text_element is not None and text_element.text:
                    try:
                        message_number = int(text_element.text.strip())
                    except ValueError:
                        continue
                else:
                    try:
                        message_number = int(report_id)
                    except ValueError:
                        continue

                # Extraer propiedades EN ORDEN (búsqueda recursiva para capturar sub-elementos)
                properties = []
                for prop in report.findall('.//property'):
                    prop_text = prop.text
                    if prop_text:
                        prop_name = prop_text.strip()
                        properties.append(prop_name)
                        all_properties.add(prop_name)

                if properties and message_number is not None:
                    # Si el mensaje aparece en varias secciones, conservar la primera definición
                    if message_number not in parcel_reports:
                        parcel_reports[message_number] = properties

                    # Capturar MessageID 20 como orden de referencia (formato moderno)
                    if message_number == 20:
                        reference_properties_order = properties.copy()

                    logger.info(f"SCNET MessageID {message_number}: {len(properties)} properties")

            if not parcel_reports:
                raise HTTPException(400, "No valid message definitions found in XML (checked ParcelDataReport and parceldata elements)")

            # Si no hay MessageID 20 (e.g. NetworkConfig.xml), usar el sort report principal como referencia.
            # Orden de preferencia: 56 (DivertSuccessSortReport), 58 (SortReport), 81 (AlibiLog),
            # 57 (DivertSortReport); fallback al mensaje con más propiedades.
            if not reference_properties_order and parcel_reports:
                for preferred_id in [56, 58, 81, 57]:
                    if preferred_id in parcel_reports:
                        reference_properties_order = parcel_reports[preferred_id].copy()
                        logger.info(f"No MessageID 20 — using MessageID {preferred_id} (sort report) as column order reference ({len(reference_properties_order)} properties)")
                        break
                if not reference_properties_order:
                    ref_id, ref_props = max(parcel_reports.items(), key=lambda x: len(x[1]))
                    reference_properties_order = ref_props.copy()
                    logger.info(f"No MessageID 20 — using MessageID {ref_id} (richest) as column order reference ({len(ref_props)} properties)")

            # PASO 2: Crear esquema de tabla usando ORDEN del mensaje de referencia
            self._create_table_schema_with_reference_order(all_properties, reference_properties_order)

            # PASO 3: Crear mapeos para cada mensaje
            self._create_message_mappings(parcel_reports)

            logger.info(f"SCNET Schema built: {self.column_count} columns, {len(self.message_id_to_field_positions)} MessageIDs")
            
        except ET.ParseError as e:
            logger.error(f"Error parsing XML: {e}")
            raise HTTPException(400, f"Invalid XML file: {e}")
        except Exception as e:
            logger.error(f"Error building SCNET schema: {e}")
            raise HTTPException(500, f"Error processing XML: {e}")
    
    def _create_table_schema_with_reference_order(self, all_properties: set, reference_order: list):
        """Crear esquema de tabla: columnas en el orden del mensaje de referencia, resto al final."""
        self.columns = [("timestamp", "TIMESTAMP"), ("message_id", "INTEGER")]

        if reference_order:
            ordered_properties = reference_order.copy()
            remaining = sorted(all_properties - set(reference_order))
            ordered_properties.extend(remaining)
            logger.info(f"Column order: {len(reference_order)} reference properties + {len(remaining)} additional")
        else:
            ordered_properties = sorted(all_properties)
            logger.warning("No reference message found - using alphabetical column order")
        
        # Crear columnas y mapeo
        for i, prop_name in enumerate(ordered_properties):
            safe_name = _safe_sql_name(prop_name.lower())
            self.columns.append((safe_name, "TEXT"))
            # Mapear property_name -> índice en tabla
            self.property_to_table_index[prop_name] = i + 2  # +2 por timestamp, message_id
        
        self.column_count = len(self.columns)
        self.columns.append(("_search_text", "TEXT"))
        self.column_count = len(self.columns)
        logger.info(f"Table schema created: {self.column_count} columns (incl. _search_text)")
    
    def _create_message_mappings(self, parcel_reports: dict):
        """Crear mapeos de posición para cada MessageID - LÓGICA SIMPLIFICADA"""

        # En NetworkConfig.xml (legado) el message_id va en parts[0] tras la coma:
        #   "timestamp,MSG_ID|campo1|campo2|..." → campos empiezan en parts[1] (offset=1)
        # En ParcelDataReportConfig.xml (moderno):
        #   "timestamp|MSG_ID|campo1|..." → campos empiezan en parts[2] (offset=2)
        # Mensajes especiales 11, 13, 31 del formato moderno tienen PIC fuera de los campos XML.
        no_text_message_ids = set() if self._is_network_config else {11, 13, 31}
        field_offset = 1 if self._is_network_config else 2

        for message_number, properties in parcel_reports.items():
            field_positions = {}

            if message_number in no_text_message_ids:
                # Mensajes sin <text> (solo formato moderno): PIC en pos 2, propiedades desde pos 3
                field_positions['pic'] = 2
                for i, prop_name in enumerate(properties):
                    field_positions[prop_name] = i + 3
                logger.info(f"MessageID {message_number} (sin <text>): PIC directo pos 2 + {len(properties)} properties del XML")

            else:
                # Regla normal: propiedades en orden desde field_offset
                for log_position, prop_name in enumerate(properties):
                    field_positions[prop_name] = log_position + field_offset
                logger.info(f"MessageID {message_number} (con <text>): {len(properties)} properties @ offset {field_offset}")
            
            if field_positions:
                self.message_id_to_field_positions[message_number] = field_positions
        
        logger.info(f"Message mappings created for {len(self.message_id_to_field_positions)} MessageIDs")

        # En NetworkConfig.xml el mensaje 51 (HostpicResponse) raramente se define.
        # En los logs FSC siempre tiene: parts[1]=PIC, parts[2]=HOSTPIC.
        # Forzamos este mapeo independientemente de lo que diga (o no diga) el XML.
        if self._is_network_config:
            self.message_id_to_field_positions[51] = {'pic': 1, 'hostpic': 2}
            logger.info("NetworkConfig: mensaje 51 hardcoded → pic@parts[1], hostpic@parts[2]")
    
    def parse_timestamp_fast(self, ts_str: str) -> Optional[datetime]:
        """Parse timestamp SCNET en cualquiera de sus variantes:
          DD/MM/YYYY HH:MM:SS:mmm        (dos puntos para ms — SCNET moderno)
          DD/MM/YYYY HH:MM:SS:mmm,µµµ   (ms + microsegundos con coma — SCNET legado)
          DD/MM/YYYY HH:MM:SS.mmm        (punto para ms)
          DD/MM/YYYY HH:MM:SS            (sin milisegundos)
        """
        try:
            space_idx = ts_str.find(' ')
            if space_idx < 0:
                return None
            date_part = ts_str[:space_idx]
            time_part = ts_str[space_idx + 1:]

            day, month, year = date_part.split('/')

            if '.' in time_part:
                # HH:MM:SS.mmm
                t, ms_str = time_part.rsplit('.', 1)
                hour, minute, second = t.split(':')
                ms = int(ms_str.split(',')[0])  # ignorar microsegundos tras coma
            else:
                # HH:MM:SS:mmm[,µµµ]  o  HH:MM:SS
                tp = time_part.split(':')
                hour, minute, second = tp[0], tp[1], tp[2]
                ms_raw = tp[3] if len(tp) >= 4 else '0'
                ms = int(ms_raw.split(',')[0])  # ignorar microsegundos tras coma

            return datetime(
                int(year), int(month), int(day),
                int(hour), int(minute), int(second),
                ms * 1000
            )
        except (ValueError, IndexError):
            pass
        return None
    
    def parse_line_ultra_fast(self, line: str) -> Optional[List]:
        """Parse línea FSC — soporta formato moderno y NetworkConfig (SCNET legado).

        Formato moderno (ParcelDataReportConfig):
            timestamp|MSG_ID|campo1|campo2|...
        Formato legado (NetworkConfig):
            timestamp,STX+MSG_ID|campo1|campo2|...|[checksum]
            Tras strip de STX: timestamp,MSG_ID|campo1|campo2|...
        """
        if not line.strip():
            return None

        line = _CTRL_CHARS_RE.sub('', line.rstrip('@').strip())
        if not line:
            return None

        parts = line.split('|')

        try:
            if self._is_network_config:
                # En el formato legado parts[0] = "timestamp,MSG_ID"
                if len(parts) < 2:
                    return None
                comma_idx = parts[0].rfind(',')
                if comma_idx < 0:
                    return None
                ts_str = parts[0][:comma_idx].strip()
                mid_str = parts[0][comma_idx + 1:].strip()
            else:
                # Formato moderno: parts[0]=timestamp, parts[1+offset]=MSG_ID
                if len(parts) < 3 + self.col_offset:
                    return None
                ts_str = parts[0].strip()
                mid_str = parts[1 + self.col_offset]

            message_id_clean = _NON_DIGITS_RE.sub('', mid_str)
            if not message_id_clean:
                return None

            timestamp = self.parse_timestamp_fast(ts_str)
            if not timestamp:
                return None

            message_id = int(message_id_clean)
            if message_id not in self.message_id_to_field_positions:
                return None

            record = ["-"] * self.column_count
            record[0] = timestamp
            record[1] = message_id

            field_positions = self.message_id_to_field_positions[message_id]
            mapped_count = 0
            for prop_name, log_position in field_positions.items():
                # NetworkConfig: log_position ya es el índice directo en parts (sin col_offset)
                # Moderno:       log_position + col_offset (para Crossorter XXL)
                actual_pos = log_position if self._is_network_config else log_position + self.col_offset
                if prop_name in self.property_to_table_index and actual_pos < len(parts):
                    value = parts[actual_pos].strip()
                    if value:
                        record[self.property_to_table_index[prop_name]] = value
                        mapped_count += 1

            if mapped_count > 0:
                record[-1] = ' '.join(' '.join(v.split()) for v in record[2:-1] if v and v != '-')
                return record
            return None

        except (ValueError, IndexError):
            return None

    def get_create_table_sql(self, table_name: str) -> str:
        """SQL para crear tabla SCNET (sin índices — se crean después del COPY para mayor velocidad)"""
        safe_name = _safe_sql_name(table_name)
        column_defs = ["id BIGSERIAL PRIMARY KEY"]
        for col_name, col_type in self.columns:
            column_defs.append(f"{col_name} {col_type}")
        return f"CREATE TABLE {safe_name} ({', '.join(column_defs)})"

# Database functions
async def get_db_pool():
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=5)
    return db_pool

async def init_database():
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS databases (
                id VARCHAR(100) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                table_name VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                record_count INTEGER DEFAULT 0,
                file_size_mb FLOAT DEFAULT 0,
                columns_info JSONB,
                customer_id VARCHAR(100)
            )
        """)
        # Migración: añadir columna si la BD ya existía
        await conn.execute("""
            ALTER TABLE databases ADD COLUMN IF NOT EXISTS customer_id VARCHAR(100)
        """)
        await conn.execute("""
            ALTER TABLE databases ADD COLUMN IF NOT EXISTS analytics_cache JSONB DEFAULT NULL
        """)
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        # Tabla para configs de log guardadas por cliente (analytics)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS customer_log_configs (
                customer_id VARCHAR(100) PRIMARY KEY,
                filename TEXT NOT NULL,
                content BYTEA NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    # Normalizar _search_text de tablas existentes en segundo plano
    asyncio.create_task(_migrate_normalize_search_text(pool))


async def _migrate_normalize_search_text(pool):
    """Normaliza espacios múltiples en _search_text y sincroniza columns_info (one-shot al arrancar)."""
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT id, table_name, columns_info FROM databases")
        for row in rows:
            tname = row['table_name']
            safe = _safe_sql_name(tname)
            try:
                async with pool.acquire() as conn:
                    has_col = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM information_schema.columns "
                        "WHERE table_name=$1 AND column_name='_search_text')", tname
                    )
                    if not has_col:
                        continue
                    # Sincronizar columns_info si _search_text no está en los metadatos
                    cols = json.loads(row['columns_info'] or '[]')
                    if '_search_text' not in cols:
                        logger.info(f"Updating columns_info for {tname} to include _search_text")
                        await conn.execute(
                            "UPDATE databases SET columns_info = columns_info::jsonb || '[\"_search_text\"]' "
                            "WHERE id = $1",
                            row['id']
                        )
                    needs_fix = await conn.fetchval(
                        f"SELECT EXISTS(SELECT 1 FROM {safe} WHERE _search_text ~ '  ' LIMIT 1)"
                    )
                    if not needs_fix:
                        continue
                    logger.info(f"Normalizing _search_text whitespace in {tname}…")
                    await conn.execute(
                        f"UPDATE {safe} SET _search_text = regexp_replace(_search_text, '\\s+', ' ', 'g')"
                    )
                    logger.info(f"_search_text normalized in {tname}, running VACUUM ANALYZE…")
                async with pool.acquire() as conn:
                    await conn.execute(f"VACUUM ANALYZE {safe}")
                    logger.info(f"VACUUM ANALYZE done for {tname}")
            except Exception as e:
                logger.warning(f"Could not migrate _search_text in {tname}: {e}")
    except Exception as e:
        logger.warning(f"_migrate_normalize_search_text failed: {e}")


async def get_unique_table_name(base_name: str) -> str:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        counter = 0
        table_name = base_name
        while True:
            exists = await conn.fetchval(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)", 
                table_name
            )
            if not exists:
                return table_name
            counter += 1
            table_name = f"{base_name}_{counter}"

def extract_table_number(filename: str) -> str:
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else str(int(time.time()))

async def process_ultra_fast(log_path: str, config: Dict, database_id: str, filename: str, customer_id: Optional[str] = None) -> Tuple[int, float, str]:
    """Procesamiento ultra-optimizado en una sola pasada"""
    start_time = time.time()
    
    # Construir esquema
    schema = UltraSchema(config)
    if not schema.message_id_to_field_positions:
        raise HTTPException(400, "No valid MessageIDs in config")
    
    # Crear tabla
    table_number = extract_table_number(filename)
    table_name = await get_unique_table_name(f"logs_{table_number}")
    
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(schema.get_create_table_sql(table_name))
    
    file_size_mb = os.path.getsize(log_path) / (1024 * 1024)
    logger.info(f"Ultra-fast processing: {filename} ({file_size_mb:.1f}MB) -> {table_name}")
    
    # Procesamiento ultra-optimizado
    batch = []
    batch_size = 50000  # Batch grande para menos overhead
    total_processed = 0
    lines_read = 0
    last_log = start_time
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore', buffering=8*1024*1024) as f:
        for line in f:
            lines_read += 1

            # Limpiar caracteres problemáticos ANTES del parsing
            line = line.replace('\x00', '').replace('\r', '').strip()
            
            record_array = schema.parse_line_ultra_fast(line)
            if record_array:
                batch.append(tuple(record_array))
                
                if len(batch) >= batch_size:
                    inserted = await insert_ultra_fast(pool, table_name, batch, schema.columns)
                    total_processed += inserted
                    batch = []
                    
                    # Log menos frecuente
                    if time.time() - last_log > 10:
                        rate = total_processed / (time.time() - start_time)
                        logger.info(f"Progress: {lines_read:,} lines, {total_processed:,} records ({rate:,.0f}/sec)")
                        last_log = time.time()
    
    # Último batch
    if batch:
        inserted = await insert_ultra_fast(pool, table_name, batch, schema.columns)
        total_processed += inserted
    
    processing_time = time.time() - start_time
    speed = total_processed / processing_time if processing_time > 0 else 0
    
    # Guardar metadatos
    column_names = [col[0] for col in schema.columns]
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO databases (id, name, table_name, record_count, file_size_mb, columns_info, customer_id)
            VALUES ($1, $1, $2, $3, $4, $5, $6)
        """, database_id, table_name, total_processed, file_size_mb, json.dumps(column_names), customer_id)

    # *** LIMPIEZA DESPUÉS de guardar - ahora que tenemos la nueva tabla ***
    await cleanup_old_tables()
    await create_analytics_indexes(pool, table_name, column_names)

    logger.info(f"ULTRA-FAST COMPLETE: {total_processed:,} records in {processing_time:.2f}s ({speed:,.0f} rec/sec)")

    return total_processed, processing_time, table_name

async def process_scnet_ultra_fast(fsc_path: str, xml_path: str, database_id: str, filename: str, customer_id: Optional[str] = None, crossorter_type: str = 'standard') -> Tuple[int, float, str]:
    """Procesamiento SCNET — soporta ParcelDataReportConfig.xml y NetworkConfig.xml (SCNET legado)"""
    start_time = time.time()

    # Construir esquema
    schema = SCNETSchemaOptimized(xml_path)
    if not schema.message_id_to_field_positions:
        raise HTTPException(400, "No valid ParcelDataReport found in XML")

    # Crossorter XXL: columna FSC extra tras timestamp → offset de 1
    # El selector de UI tiene prioridad; el config del cliente es fallback
    if crossorter_type == 'xxl':
        schema.col_offset = 1
        logger.info(f"Crossorter XXL mode (UI) for customer {customer_id}")
    elif customer_id:
        cfg = get_customer_config(customer_id)
        if cfg and cfg.get('crossorter_xxl'):
            schema.col_offset = 1
            logger.info(f"Crossorter XXL mode (config) for customer {customer_id}")
    
    # Crear tabla
    table_number = extract_table_number(filename)
    table_name = await get_unique_table_name(f"scnet_{table_number}")
    
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(schema.get_create_table_sql(table_name))
    
    file_size_mb = os.path.getsize(fsc_path) / (1024 * 1024)
    logger.info(f"SCNET processing: {filename} ({file_size_mb:.1f}MB) -> {table_name}")
    
    # Log esquema para debug
    logger.info(f"Available MessageIDs: {sorted(schema.message_id_to_field_positions.keys())}")
    logger.info(f"Table will have {schema.column_count} columns")
    
    # Procesamiento
    batch = []
    batch_size = 50000
    total_processed = 0
    lines_read = 0
    lines_skipped = 0
    last_log = start_time

    # Diagnóstico: capturar las primeras líneas rechazadas y por qué
    diag_samples = []   # [(raw_line, reason)]

    with open(fsc_path, 'r', encoding='utf-8', errors='ignore', buffering=8*1024*1024) as f:
        for line in f:
            lines_read += 1

            # Limpieza robusta de caracteres problemáticos
            line = line.replace('\x00', '').replace('\r', '').strip()

            record_array = schema.parse_line_ultra_fast(line)
            if record_array:
                batch.append(tuple(record_array))
                if len(batch) >= batch_size:
                    inserted = await insert_ultra_fast(pool, table_name, batch, schema.columns)
                    total_processed += inserted
                    batch = []
                    if time.time() - last_log > 5:
                        rate = total_processed / (time.time() - start_time)
                        skip_rate = (lines_skipped / lines_read) * 100 if lines_read > 0 else 0
                        logger.info(f"SCNET Progress: {lines_read:,} lines read, {total_processed:,} records ({rate:,.0f}/sec, {skip_rate:.1f}% skipped)")
                        last_log = time.time()
            else:
                lines_skipped += 1
                # Guardar muestra de las primeras líneas rechazadas para diagnóstico
                if line and len(diag_samples) < 5:
                    parts = line.split('|')
                    if schema._is_network_config:
                        # Formato legado: parts[0] = "timestamp,MSG_ID"
                        if len(parts) < 2:
                            reason = f"muy_pocas_columnas({len(parts)})"
                        else:
                            comma_idx = parts[0].rfind(',')
                            if comma_idx < 0:
                                reason = "sin_coma_en_campo0_formato_invalido"
                            else:
                                ts_str = parts[0][:comma_idx].strip()
                                mid_str = parts[0][comma_idx + 1:].strip()
                                ts_ok = schema.parse_timestamp_fast(ts_str) is not None
                                if not ts_ok:
                                    reason = f"timestamp_invalido('{ts_str[:40]}')"
                                else:
                                    mid_clean = _NON_DIGITS_RE.sub('', mid_str)
                                    try:
                                        mid = int(mid_clean)
                                        if mid not in schema.message_id_to_field_positions:
                                            reason = f"message_id_{mid}_no_en_XML(disponibles:{sorted(schema.message_id_to_field_positions.keys())})"
                                        else:
                                            reason = "sin_campos_mapeados"
                                    except ValueError:
                                        reason = f"message_id_no_numerico('{mid_str[:20]}')"
                    else:
                        # Formato moderno: parts[0]=timestamp, parts[1+offset]=MSG_ID
                        if len(parts) < 3 + schema.col_offset:
                            reason = f"muy_pocas_columnas({len(parts)})"
                        else:
                            ts_ok = schema.parse_timestamp_fast(parts[0].strip()) is not None
                            if not ts_ok:
                                reason = f"timestamp_invalido('{parts[0].strip()[:40]}')"
                            else:
                                mid_raw = parts[1 + schema.col_offset].strip()
                                mid_clean = _NON_DIGITS_RE.sub('', mid_raw)
                                try:
                                    mid = int(mid_clean)
                                    if mid not in schema.message_id_to_field_positions:
                                        reason = f"message_id_{mid}_no_en_XML(disponibles:{sorted(schema.message_id_to_field_positions.keys())[:8]})"
                                    else:
                                        reason = "sin_campos_mapeados"
                                except ValueError:
                                    reason = f"message_id_no_numerico('{mid_raw[:20]}')"
                    diag_samples.append((line[:120], reason))

    # Último batch
    if batch:
        inserted = await insert_ultra_fast(pool, table_name, batch, schema.columns)
        total_processed += inserted

    processing_time = time.time() - start_time
    speed = total_processed / processing_time if processing_time > 0 else 0

    # Stats finales
    skip_rate = (lines_skipped / lines_read) * 100 if lines_read > 0 else 0
    logger.info(f"SCNET Final Stats: {lines_read:,} lines read, {total_processed:,} records, {lines_skipped:,} skipped ({skip_rate:.1f}%)")

    # Construir debug_info si hay problemas
    debug_info = None
    if total_processed == 0 and diag_samples:
        lines_diag = [f"• [{r}]  {l}" for l, r in diag_samples]
        debug_info = f"0 registros procesados de {lines_read:,} líneas leídas. Primeras líneas rechazadas:\n" + "\n".join(lines_diag)
        logger.warning(f"SCNET DIAGNÓSTICO:\n{debug_info}")
    
    # Guardar metadatos
    column_names = [col[0] for col in schema.columns]
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO databases (id, name, table_name, record_count, file_size_mb, columns_info, customer_id)
            VALUES ($1, $1, $2, $3, $4, $5, $6)
        """, database_id, table_name, total_processed, file_size_mb, json.dumps(column_names), customer_id)

    # *** LIMPIEZA DESPUÉS de guardar - ahora que tenemos la nueva tabla ***
    await cleanup_old_tables()
    await create_analytics_indexes(pool, table_name, column_names)

    logger.info(f"SCNET COMPLETE: {total_processed:,} records in {processing_time:.2f}s ({speed:,.0f} rec/sec)")

    return total_processed, processing_time, table_name, debug_info

async def insert_ultra_fast(pool, table_name: str, batch: List[tuple], columns: List[tuple]) -> int:
    """Inserción ultra-optimizada"""
    if not batch:
        return 0

    safe_name = _safe_sql_name(table_name)
    column_names = [col[0] for col in columns]
    
    async with pool.acquire() as conn:
        try:
            await conn.copy_records_to_table(safe_name, records=batch, columns=column_names)
            return len(batch)
        except Exception as e:
            logger.error(f"COPY failed: {e}")
            # Fallback a executemany
            placeholders = ', '.join(f'${i+1}' for i in range(len(column_names)))
            sql = f"INSERT INTO {safe_name} ({', '.join(column_names)}) VALUES ({placeholders})"
            try:
                await conn.executemany(sql, batch)
                return len(batch)
            except Exception as e2:
                logger.error(f"Executemany failed: {e2}")
                return 0

async def _build_gin_index(pool, safe_table: str, table_name: str):
    """Crea el índice GIN de trigramas en segundo plano (no bloquea la respuesta de upload)."""
    idx_name = f"idx_{safe_table[:40]}_search"
    try:
        async with pool.acquire() as conn:
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {idx_name}
                ON {safe_table} USING GIN (_search_text gin_trgm_ops)
            """)
        logger.info(f"GIN trigram index created: {idx_name}")
    except Exception as e:
        logger.warning(f"Could not create GIN index for {table_name}: {e}")


async def create_analytics_indexes(pool, table_name: str, column_names: List[str]):
    """Crea todos los índices DESPUÉS del COPY completo (mucho más rápido que durante la ingesta).
    Orden: msgid + ts (bloqueantes, necesarios para analytics) → composite analytics → GIN (background)."""
    safe_table = _safe_sql_name(table_name)

    async with pool.acquire() as conn:
        # Índices básicos de navegación — creados aquí, no en CREATE TABLE, para no ralentizar el COPY
        try:
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{safe_table[:50]}_msgid ON {safe_table}(message_id)")
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{safe_table[:50]}_ts ON {safe_table}(timestamp)")
            logger.info(f"Base indexes created for {table_name}")
        except Exception as e:
            logger.warning(f"Could not create base indexes for {table_name}: {e}")

        # ANALYZE: da al planner estadísticas reales tras el COPY masivo
        try:
            await conn.execute(f"ANALYZE {safe_table}")
            logger.info(f"ANALYZE completed for {table_name}")
        except Exception as e:
            logger.warning(f"ANALYZE failed for {table_name}: {e}")

    # Índice compuesto para queries de analytics
    hostpic_col    = find_column(column_names, 'hostpic')
    lastdest_col   = find_column(column_names, 'lastdestination', 'last_destination')
    entrypoint_col = find_column(column_names,
                                 'parcelentrypoint', 'parcelentrancepoint',
                                 'parcel_entry_point', 'parcel_entrance_point',
                                 'entrancepoint', 'entrypoint', 'entrance_point', 'entry_point')

    if all([hostpic_col, lastdest_col, entrypoint_col]):
        idx_name = f"idx_{safe_table[:40]}_analytics"
        async with pool.acquire() as conn:
            try:
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {idx_name}
                    ON {safe_table}(message_id, {lastdest_col}, {hostpic_col}, {entrypoint_col})
                """)
                logger.info(f"Analytics index created: {idx_name}")
            except Exception as e:
                logger.warning(f"Could not create analytics index for {table_name}: {e}")
    else:
        logger.info(f"Analytics columns not detected for {table_name} — skipping composite index")

    # GIN trigram index en segundo plano (30-60s) — no bloquea la respuesta al usuario
    if '_search_text' in column_names:
        asyncio.create_task(_build_gin_index(pool, safe_table, table_name))


async def _get_db_info(conn, database_id: str):
    """Carga metadatos de la BD, parsea columnas y resuelve el config del cliente.
    Lanza HTTPException(404) si la BD no existe."""
    row = await conn.fetchrow(
        "SELECT table_name, columns_info, customer_id FROM databases WHERE id = $1",
        database_id
    )
    if not row:
        raise HTTPException(404, "Database not found")
    table_name = row['table_name']
    columns = json.loads(row['columns_info'] or '[]')
    if not columns:
        col_rows = await conn.fetch(
            "SELECT column_name FROM information_schema.columns WHERE table_name = $1 ORDER BY ordinal_position",
            table_name
        )
        columns = [r['column_name'] for r in col_rows]
    cfg = get_customer_config(row['customer_id'])
    return table_name, _safe_sql_name(table_name), columns, cfg


async def _read_cache(conn, database_id: str, key: str):
    """Devuelve el resultado cacheado o None si no existe."""
    row = await conn.fetchrow(
        "SELECT analytics_cache->>$2 AS v FROM databases WHERE id=$1",
        database_id, key
    )
    return json.loads(row['v']) if (row and row['v']) else None

async def _write_cache(conn, database_id: str, key: str, value) -> None:
    """Persiste un resultado de analytics en la caché JSONB. No-fatal si falla."""
    try:
        await conn.execute(
            "UPDATE databases SET analytics_cache = COALESCE(analytics_cache, '{}'::jsonb) || $2::jsonb WHERE id=$1",
            database_id, json.dumps({key: value})
        )
    except Exception as e:
        logger.warning(f"Cache write failed [{key}]: {e}")


# API Routes
@app.on_event("startup")
async def startup():
    load_customer_configs()
    await init_database()
    logger.info("Ultra-Optimized Log Analyzer started")

@app.get("/")
async def root():
    return {"message": "Ultra-Optimized Log Analyzer", "version": "4.0.0"}

@app.get("/customers")
async def list_customers():
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT customer_id, filename FROM customer_log_configs")
    saved = {row['customer_id']: row['filename'] for row in rows}
    return [
        {
            "id": cfg["id"],
            "name": cfg["name"],
            "charts": cfg.get("charts", []),
            "crossorter_type": "xxl" if cfg.get("crossorter_xxl") else "standard",
            "server_type": cfg.get("server_type", "eDS"),
            "log_config_filename": saved.get(cfg["id"]) if cfg.get("charts") else None,
        }
        for cfg in CUSTOMER_CONFIGS.values()
    ]

@app.post("/upload", response_model=ProcessResponse)
async def upload_files(
    log_file: UploadFile = File(...),
    config_file: Optional[UploadFile] = File(None),
    server_type: str = Query("eDS"),
    customer_id: Optional[str] = Query(None),
    crossorter_type: str = Query("standard")
):
    start_time = time.time()
    pool = await get_db_pool()

    customer_cfg = CUSTOMER_CONFIGS.get(customer_id) if customer_id else None
    has_analytics = bool(customer_cfg and customer_cfg.get('charts'))

    # Resolver config: del upload o de la caché en BD
    if config_file is not None:
        config_content = await config_file.read()
        config_filename = config_file.filename
        # Guardar en BD si el cliente tiene analytics
        if has_analytics and customer_id:
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO customer_log_configs (customer_id, filename, content, uploaded_at)
                    VALUES ($1, $2, $3, NOW())
                    ON CONFLICT (customer_id) DO UPDATE
                    SET filename = EXCLUDED.filename,
                        content  = EXCLUDED.content,
                        uploaded_at = NOW()
                """, customer_id, config_filename, config_content)
    elif has_analytics and customer_id:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT filename, content FROM customer_log_configs WHERE customer_id = $1",
                customer_id
            )
        if row is None:
            raise HTTPException(400, "No hay config guardada para este cliente. Sube el fichero de config la primera vez.")
        config_content = bytes(row['content'])
        config_filename = row['filename']
    else:
        raise HTTPException(400, "Config file is required.")

    log_extension    = log_file.filename.split('.')[-1].lower()
    config_extension = config_filename.split('.')[-1].lower()

    # Validar combinaciones de archivos
    if server_type.lower() == "eds":
        if log_extension != 'log':
            raise HTTPException(400, "eDS requires a .log file")
        if config_extension != 'json':
            raise HTTPException(400, "eDS requires a .json configuration file")
    elif server_type.lower() == "scnet":
        if log_extension not in ['fsc', 'log']:  # Permitir ambas extensiones para logs SCNET
            raise HTTPException(400, "SCNET requires a .fsc or .log file")
        if config_extension != 'xml':
            raise HTTPException(400, "SCNET requires a .xml configuration file")
    else:
        raise HTTPException(400, f"Unsupported server_type: {server_type}. Use 'eDS' or 'SCNET'")

    # Generar ID único para la base de datos
    database_id = f"{server_type.lower()}_{int(time.time())}_{log_file.filename.split('.')[0]}"

    # Guardar archivos temporales
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{log_extension}') as tmp_log, \
         tempfile.NamedTemporaryFile(delete=False, suffix=f'.{config_extension}') as tmp_config:

        tmp_log.write(await log_file.read())
        tmp_config.write(config_content)
        log_path, config_path = tmp_log.name, tmp_config.name
    
    try:
        # Procesar según el tipo de tecnología
        if server_type.lower() == "eds":
            logger.info(f"Processing eDS files: {log_file.filename} + {config_filename}")
            
            # Cargar configuración JSON
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Procesar con eDS
            records_processed, processing_time, table_name = await process_ultra_fast(
                log_path, config, database_id, log_file.filename, customer_id
            )
            debug_info = None

            message = f"eDS: {records_processed:,} records processed -> {table_name}"

        elif server_type.lower() == "scnet":
            logger.info(f"Processing SCNET files: {log_file.filename} + {config_filename}")

            # Procesar con SCNET (XML + FSC)
            records_processed, processing_time, table_name, debug_info = await process_scnet_ultra_fast(
                log_path, config_path, database_id, log_file.filename, customer_id, crossorter_type
            )

            message = f"SCNET: {records_processed:,} records processed -> {table_name}"

        else:
            raise HTTPException(400, f"Invalid server_type: {server_type}")

        logger.info(f"Upload successful: {message}")

        return ProcessResponse(
            success=True,
            message=message,
            processing_time=time.time() - start_time,
            records_processed=records_processed,
            database_id=database_id,
            table_name=table_name,
            debug_info=debug_info
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Processing error for {server_type}: {str(e)}")
        raise HTTPException(500, f"Error processing {server_type} files: {str(e)}")
    
    finally:
        # Limpiar archivos temporales
        for path in [log_path, config_path]:
            try:
                os.unlink(path)
            except:
                pass

@app.get("/databases", response_model=List[DatabaseInfo])
async def list_databases():
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, name, table_name, created_at, record_count, file_size_mb, customer_id
            FROM databases ORDER BY created_at DESC LIMIT 20
        """)
        result = []
        for row in rows:
            d = dict(row)
            cid = d.get("customer_id")
            cfg = CUSTOMER_CONFIGS.get(cid) if cid else None
            d["customer_name"] = cfg["name"] if cfg else None
            result.append(DatabaseInfo(**d))
        return result

@app.post("/logs/{database_id}", response_model=LogResponse)
async def query_logs(database_id: str, query: LogQuery):
    start_time = time.time()
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        db_info = await conn.fetchrow(
            "SELECT table_name, columns_info FROM databases WHERE id = $1", 
            database_id
        )
        if not db_info:
            raise HTTPException(404, "Database not found")
        
        table_name = db_info['table_name']
        columns = json.loads(db_info['columns_info'] or '[]')
        # columns_info puede estar desactualizado (tablas migradas antes de que se añadiera _search_text)
        # Comprobamos directamente en el esquema real de la tabla
        has_search_text = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.columns "
            "WHERE table_name=$1 AND column_name='_search_text')",
            table_name
        )
        display_columns = [c for c in columns if c != '_search_text']

        conditions, params = [], []

        if query.message_id:
            conditions.append(f"message_id = ${len(params) + 1}")
            params.append(query.message_id)

        if query.search:
            normalized_search = ' '.join(query.search.split())
            if has_search_text:
                conditions.append(f"_search_text ILIKE ${len(params) + 1}")
            else:
                text_cols = columns[2:]
                search_conds = [f"{col}::text ILIKE ${len(params) + 1}" for col in text_cols]
                if search_conds:
                    conditions.append(f"({' OR '.join(search_conds)})")
            params.append(f"%{normalized_search}%")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # COUNT aproximado (O(1)) cuando no hay filtros; exacto cuando los hay
        if not conditions:
            total_records = await conn.fetchval(
                "SELECT reltuples::bigint FROM pg_class WHERE relname = $1", table_name
            ) or 0
        else:
            total_records = await conn.fetchval(
                f"SELECT COUNT(*) FROM {table_name} {where_clause}", *params
            )

        offset = (query.page - 1) * query.limit
        safe_cols = ', '.join(display_columns) if display_columns else '*'
        rows = await conn.fetch(f"""
            SELECT {safe_cols} FROM {table_name} {where_clause}
            ORDER BY timestamp ASC LIMIT {query.limit} OFFSET {offset}
        """, *params)

        data = [{k: v.isoformat() if isinstance(v, datetime) else v
                 for k, v in row.items()} for row in rows]

        return LogResponse(
            data=data,
            total_records=total_records,
            page=query.page,
            total_pages=(total_records + query.limit - 1) // query.limit,
            processing_time=time.time() - start_time,
            columns=display_columns
        )

@app.get("/table/{table_name}/stats")
async def get_table_stats(table_name: str):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        safe_name = _safe_sql_name(table_name)
        
        exists = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)", 
            safe_name
        )
        if not exists:
            raise HTTPException(404, "Table not found")
        
        total_records = await conn.fetchval(f"SELECT COUNT(*) FROM {safe_name}")
        message_dist = await conn.fetch(f"SELECT message_id, COUNT(*) as count FROM {safe_name} GROUP BY message_id ORDER BY message_id")
        
        return {
            'total_records': total_records,
            'message_distribution': [dict(row) for row in message_dist]
        }

@app.get("/databases/{database_id}/induction_quality")
async def get_induction_quality(database_id: str):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        table_name, safe_table, columns, cfg = await _get_db_info(conn, database_id)
        if not cfg or not cfg.get('good_package') or not cfg.get('bad_package'):
            return {"data": [], "error": "Analytics not configured for this customer"}
        cached = await _read_cache(conn, database_id, 'induction_quality')
        if cached is not None:
            return cached

        hostpic_col    = find_column(columns, cfg.get('hostpic_column', 'hostpic'))
        ep_cfg         = cfg.get('entry_point', {})
        ep_aliases     = ep_cfg.get('column_aliases', [])
        entrypoint_col = find_column(columns, ep_cfg.get('column', 'parcelentrypoint'), *ep_aliases)
        bad_cfg        = cfg.get('bad_package', {})
        good_cfg       = cfg.get('good_package', {})

        if not hostpic_col or not entrypoint_col:
            return {"data": [], "columns_found": {"hostpic": hostpic_col, "entry_point": entrypoint_col}}

        bad_msg_id  = bad_cfg.get('message_id', 21)
        good_msg_id = good_cfg.get('message_id', 20)
        bad_where   = build_condition_sql(bad_cfg.get('conditions', []), columns)
        good_where  = build_condition_sql(good_cfg.get('conditions', []), columns)

        bad_rows = await conn.fetch(f"""
            SELECT DISTINCT ON ({hostpic_col}) {hostpic_col}, {entrypoint_col},
                EXTRACT(HOUR FROM timestamp)::int AS hour
            FROM {safe_table}
            WHERE message_id = {bad_msg_id}
            AND {bad_where}
            AND {hostpic_col} NOT IN ('-', '-1', '')
            AND {entrypoint_col} NOT IN ('-', '')
            ORDER BY {hostpic_col}, timestamp
        """)

        good_rows = await conn.fetch(f"""
            SELECT DISTINCT ON ({hostpic_col}) {hostpic_col}, {entrypoint_col},
                EXTRACT(HOUR FROM timestamp)::int AS hour
            FROM {safe_table}
            WHERE message_id = {good_msg_id}
            AND {good_where}
            AND {hostpic_col} NOT IN ('-', '-1', '')
            AND {entrypoint_col} NOT IN ('-', '')
            ORDER BY {hostpic_col}, timestamp
        """)

        hour_infeed_stats: Dict[tuple, Dict] = {}

        for row in good_rows:
            zone = get_zone_for_entry_point(str(row[entrypoint_col] or ''), cfg)
            key = (int(row['hour']) if row['hour'] is not None else 0, zone['id'])
            s = hour_infeed_stats.setdefault(key, {'zone': zone, 'good': 0, 'bad': 0})
            s['good'] += 1

        for row in bad_rows:
            zone = get_zone_for_entry_point(str(row[entrypoint_col] or ''), cfg)
            key = (int(row['hour']) if row['hour'] is not None else 0, zone['id'])
            s = hour_infeed_stats.setdefault(key, {'zone': zone, 'good': 0, 'bad': 0})
            s['bad'] += 1

        result = []
        for (hour, _), stats in sorted(hour_infeed_stats.items()):
            good  = stats['good']
            bad   = stats['bad']
            total = good + bad
            if total > 0:
                zone = stats['zone']
                result.append({
                    'infeed':     zone['name'],
                    'infeed_num': zone['id'],
                    'good':       good,
                    'bad':        bad,
                    'total':      total,
                    'good_pct':   round((good / total) * 100, 1),
                    'bad_pct':    round((bad  / total) * 100, 1),
                    'hour':       hour,
                })

        _out = {
            'data': result,
            'columns_used': {'hostpic': hostpic_col, 'entry_point': entrypoint_col},
            'customer': cfg['name'],
        }
        await _write_cache(conn, database_id, 'induction_quality', _out)
        return _out

BLOCKED_STATUS_FLAGS = [
    (1,   "FrontFault"),
    (2,   "RearFault"),
    (4,   "MultipleCarriers"),
    (8,   "MultipleFault"),
    (16,  "SmallItemOverlappingGap"),
    (32,  "Unsortable"),
    (64,  "ScreenFault"),
    (128, "SortRestricted"),
    (256, "MultipleDataForOneItem"),
]

@app.get("/databases/{database_id}/blocked_status")
async def get_blocked_status(database_id: str):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        table_name, safe_table, columns, cfg = await _get_db_info(conn, database_id)
        bs_cfg  = cfg.get('blocked_status') if cfg else None
        bad_cfg = cfg.get('bad_package')    if cfg else None
        if not bs_cfg or not bad_cfg:
            return {"data": [], "error": "Blocked status not configured for this customer"}
        cached = await _read_cache(conn, database_id, 'blocked_status')
        if cached is not None:
            return cached

        hostpic_col = find_column(columns, cfg.get('hostpic_column', 'hostpic'))
        status_col  = find_column(columns, bs_cfg.get('column', 'parcel_blocked_status'), 'parcelblockedstatus')
        if not hostpic_col or not status_col:
            return {"data": [], "error": "Required columns not found"}

        bad_msg_id = bad_cfg.get('message_id', 21)
        bad_where  = build_condition_sql(bad_cfg.get('conditions', []), columns)

        rows = await conn.fetch(f"""
            WITH first_occ AS (
                SELECT DISTINCT ON ({hostpic_col})
                    {status_col}::integer AS status
                FROM {safe_table}
                WHERE message_id = {bad_msg_id}
                  AND {bad_where}
                  AND {hostpic_col} NOT IN ('-', '-1', '')
                  AND {status_col} IS NOT NULL
                  AND {status_col} != ''
                ORDER BY {hostpic_col}, timestamp ASC NULLS LAST
            )
            SELECT status, COUNT(*)::integer AS cnt
            FROM first_occ
            WHERE status > 0
            GROUP BY status
        """)

        flag_counts = {}
        total_hostpics = 0

        for row in rows:
            status = int(row['status'])
            cnt    = int(row['cnt'])
            total_hostpics += cnt
            for flag_val, flag_name in BLOCKED_STATUS_FLAGS:
                if status & flag_val:
                    flag_counts[flag_name] = flag_counts.get(flag_name, 0) + cnt

        data = sorted([
            {
                "flag":  flag_name,
                "value": flag_val,
                "count": flag_counts[flag_name],
                "pct":   round(flag_counts[flag_name] / total_hostpics * 100, 1) if total_hostpics > 0 else 0,
            }
            for flag_val, flag_name in BLOCKED_STATUS_FLAGS
            if flag_name in flag_counts
        ], key=lambda x: -x["count"])

        _out = {"data": data, "total": total_hostpics}
        await _write_cache(conn, database_id, 'blocked_status', _out)
        return _out


@app.get("/databases/{database_id}/bad_hostpics")
async def get_bad_hostpics(database_id: str):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        table_name, safe_table, columns, cfg = await _get_db_info(conn, database_id)
        if not cfg or not cfg.get('good_package') or not cfg.get('bad_package'):
            return {"data": [], "total": 0}
        cached = await _read_cache(conn, database_id, 'bad_hostpics')
        if cached is not None:
            return cached

        hostpic_col    = find_column(columns, cfg.get('hostpic_column', 'hostpic'))
        ep_cfg         = cfg.get('entry_point', {})
        ep_aliases     = ep_cfg.get('column_aliases', [])
        entrypoint_col = find_column(columns, ep_cfg.get('column', 'parcelentrypoint'), *ep_aliases)
        bad_cfg        = cfg.get('bad_package', {})

        if not hostpic_col or not entrypoint_col:
            return {"data": [], "total": 0}

        bad_msg_id = bad_cfg.get('message_id', 21)
        bad_where  = build_condition_sql(bad_cfg.get('conditions', []), columns)

        bad_rows = await conn.fetch(f"""
            SELECT DISTINCT ON ({hostpic_col}) {hostpic_col}, {entrypoint_col}
            FROM {safe_table}
            WHERE message_id = {bad_msg_id}
            AND {bad_where}
            AND {hostpic_col} NOT IN ('-', '-1', '')
            AND {entrypoint_col} NOT IN ('-', '')
            ORDER BY {hostpic_col}, {entrypoint_col}
        """)

        data = []
        for row in bad_rows:
            ep   = str(row[entrypoint_col] or '')
            zone = get_zone_for_entry_point(ep, cfg)
            data.append({
                'hostpic':     str(row[hostpic_col] or ''),
                'entry_point': ep,
                'infeed':      zone['name'],
            })

        data.sort(key=lambda x: (x['infeed'], x['hostpic']))
        _out = {"data": data, "total": len(data)}
        await _write_cache(conn, database_id, 'bad_hostpics', _out)
        return _out


@app.get("/databases/{database_id}/good_hostpics")
async def get_good_hostpics(database_id: str):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        table_name, safe_table, columns, cfg = await _get_db_info(conn, database_id)
        if not cfg or not cfg.get('good_package') or not cfg.get('bad_package'):
            return {"data": [], "total": 0}
        cached = await _read_cache(conn, database_id, 'good_hostpics')
        if cached is not None:
            return cached

        hostpic_col    = find_column(columns, cfg.get('hostpic_column', 'hostpic'))
        ep_cfg         = cfg.get('entry_point', {})
        ep_aliases     = ep_cfg.get('column_aliases', [])
        entrypoint_col = find_column(columns, ep_cfg.get('column', 'parcelentrypoint'), *ep_aliases)
        good_cfg       = cfg.get('good_package', {})

        if not hostpic_col or not entrypoint_col:
            return {"data": [], "total": 0}

        good_msg_id = good_cfg.get('message_id', 20)
        good_where  = build_condition_sql(good_cfg.get('conditions', []), columns)

        good_rows = await conn.fetch(f"""
            SELECT DISTINCT ON ({hostpic_col}) {hostpic_col}, {entrypoint_col}
            FROM {safe_table}
            WHERE message_id = {good_msg_id}
            AND {good_where}
            AND {hostpic_col} NOT IN ('-', '-1', '')
            AND {entrypoint_col} NOT IN ('-', '')
            ORDER BY {hostpic_col}, {entrypoint_col}
        """)

        data = []
        for row in good_rows:
            ep   = str(row[entrypoint_col] or '')
            zone = get_zone_for_entry_point(ep, cfg)
            data.append({
                'hostpic':     str(row[hostpic_col] or ''),
                'entry_point': ep,
                'infeed':      zone['name'],
            })

        data.sort(key=lambda x: (x['infeed'], x['hostpic']))
        _out = {"data": data, "total": len(data)}
        await _write_cache(conn, database_id, 'good_hostpics', _out)
        return _out


@app.get("/databases/{database_id}/sort_quality")
async def get_sort_quality(database_id: str):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        table_name, safe_table, columns, cfg = await _get_db_info(conn, database_id)
        if not cfg or 'sort_report' not in cfg or 'sort_quality' not in cfg.get('charts', []):
            return {"data": [], "error": "Sort quality not configured for this customer"}
        cached = await _read_cache(conn, database_id, 'sort_quality')
        if cached is not None:
            return cached

        sort_cfg       = cfg['sort_report']
        msg_id         = sort_cfg['message_id']
        state_col_name = sort_cfg.get('state_column', 'originaldestinationstate')

        ep_cfg         = cfg.get('entry_point', {})
        hostpic_col    = find_column(columns, cfg.get('hostpic_column', 'hostpic'))
        entrypoint_col = find_column(columns, ep_cfg.get('column', 'parcelentrancepoint'), *ep_cfg.get('column_aliases', []))
        state_col      = find_column(columns, state_col_name, 'original_destination_state', 'originaldestinationstate')

        columns_found = {
            'hostpic':     bool(hostpic_col),
            'entry_point': bool(entrypoint_col),
            'state':       bool(state_col),
        }
        if not all(columns_found.values()):
            return {"data": [], "error": "Required columns not found", "columns_found": columns_found}

        # DISTINCT (hostpic, entry_point, state): cada par único (hostpic, ODS) cuenta una vez
        rows = await conn.fetch(f"""
            SELECT DISTINCT
                {hostpic_col}    AS hp,
                {entrypoint_col} AS ep,
                {state_col}      AS ods
            FROM {safe_table}
            WHERE message_id = {msg_id}
              AND {hostpic_col}    NOT IN ('-', '-1', '')
              AND {entrypoint_col} NOT IN ('-', '')
              AND {state_col}      NOT IN ('-', '')
        """)

        counts: Dict[tuple, int] = {}
        for row in rows:
            zone = get_zone_for_entry_point(str(row['ep']), cfg)
            key  = (zone['id'], zone['name'], str(row['ods']))
            counts[key] = counts.get(key, 0) + 1

        zone_order = {z['id']: i for i, z in enumerate(cfg.get('zones', []))}
        result = [
            {'zone_id': zid, 'zone_name': zname, 'state': ods, 'count': cnt}
            for (zid, zname, ods), cnt in counts.items()
        ]
        result.sort(key=lambda r: (zone_order.get(r['zone_id'], 99), r['state']))

        _out = {"data": result, "customer": cfg['name'], "columns_found": columns_found}
        await _write_cache(conn, database_id, 'sort_quality', _out)
        return _out


@app.get("/databases/{database_id}/scale_quality")
async def get_scale_quality(database_id: str):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        table_name, safe_table, columns, cfg = await _get_db_info(conn, database_id)
        sq_cfg = (cfg or {}).get('scale_quality')
        if not sq_cfg:
            return {"data": [], "error": "Scale quality not configured for this customer"}
        cached = await _read_cache(conn, database_id, 'scale_quality')
        if cached is not None:
            return cached

        ep_cfg         = cfg.get('entry_point', {})
        entrypoint_col = find_column(columns, ep_cfg.get('column', 'parcelentrypoint'), *ep_cfg.get('column_aliases', []))
        field_name     = sq_cfg.get('field', 'scanner_data_3')
        scale_col      = find_column(columns, field_name)
        msg_id         = sq_cfg.get('message_id', 20)
        ok_prefix      = sq_cfg.get('ok_prefix', '6')
        ok_contains    = sq_cfg.get('ok_contains', '')
        no_scan_value  = sq_cfg.get('no_scan_value', '')
        error_codes      = sq_cfg.get('error_codes', {})
        exclude_from_pct = set(sq_cfg.get('exclude_from_pct', []))
        pre_scale_eps    = sq_cfg.get('pre_scale_entry_points', [])

        def _is_pre_scale(ep_str: str) -> bool:
            if not pre_scale_eps:
                return True
            normalized = ep_str.strip().lstrip('0')
            return any(normalized == pep or normalized.startswith(pep + '.') for pep in pre_scale_eps)

        if not entrypoint_col or not scale_col:
            return {"data": [], "error": f"Columns not found (entry_point: {entrypoint_col}, scale: {scale_col})"}

        # Condición OK: empieza por ok_prefix Y contiene ok_contains (ej. 'g' de gramos)
        ok_condition     = f"LEFT({scale_col}, 1) = '{ok_prefix}'"
        ok_full          = ok_condition + (f" AND {scale_col} LIKE '%{ok_contains}%'" if ok_contains else "")
        # Unknown: empieza por ok_prefix pero NO contiene ok_contains (6 sin 'g')
        unknown_condition = (ok_condition + f" AND {scale_col} NOT LIKE '%{ok_contains}%'") if ok_contains else "FALSE"

        # Normalizar espacios internos para comparación robusta (ej. "1  0", "1   0" → "1 0")
        # El parser ya hace strip() exterior, pero pueden quedar dobles espacios internos en este campo
        no_scan_normalized = re.sub(r'\s+', ' ', no_scan_value.strip()) if no_scan_value else ''
        no_scan_clause = (
            f"WHEN regexp_replace({scale_col}, '\\s+', ' ', 'g') = '{no_scan_normalized}' THEN 'noscan'"
            if no_scan_normalized else ""
        )

        rows = await conn.fetch(f"""
            SELECT ep, scale_code, COUNT(*) AS cnt
            FROM (
                SELECT {entrypoint_col} AS ep,
                       CASE
                           {no_scan_clause}
                           WHEN {ok_full} THEN 'ok'
                           WHEN {unknown_condition} THEN 'unknown'
                           ELSE LEFT({scale_col}, 1)
                       END AS scale_code
                FROM {safe_table}
                WHERE message_id = {msg_id}
                  AND {scale_col} NOT IN ('-', '')
                  AND {scale_col} IS NOT NULL
            ) sub
            GROUP BY ep, scale_code
        """)

        zone_stats:      Dict[int, Dict] = {}
        otros_by_zone:   Dict[int, Dict] = {}
        for row in rows:
            ep_str = str(row['ep'] or '')
            code   = row['scale_code']
            count  = int(row['cnt'])
            target = zone_stats if _is_pre_scale(ep_str) else otros_by_zone
            zone   = get_zone_for_entry_point(ep_str, cfg)
            zid    = zone['id']
            if zid not in target:
                target[zid] = {'zone': zone, 'ok': 0, 'noscan': 0, 'unknown': 0, 'errors': {}}
            if code == 'ok':
                target[zid]['ok'] += count
            elif code == 'noscan':
                target[zid]['noscan'] += count
            elif code == 'unknown':
                target[zid]['unknown'] += count
            elif code in error_codes:
                target[zid]['errors'][code] = target[zid]['errors'].get(code, 0) + count

        result = []
        zone_order = {z['id']: i for i, z in enumerate(cfg.get('zones', []))}

        def _build_rows(stats_dict: Dict, is_other: bool) -> list:
            rows_out = []
            for zid, stats in sorted(stats_dict.items(), key=lambda x: zone_order.get(x[0], 99)):
                ok           = stats['ok']
                noscan       = stats['noscan']
                unknown      = stats['unknown']
                total_errors = sum(stats['errors'].values())
                total        = ok + noscan + unknown + total_errors
                if total == 0:
                    continue
                if is_other:
                    ok_pct = None
                else:
                    excluded = sum(stats['errors'].get(c, 0) for c in exclude_from_pct)
                    pct_base = total - excluded
                    ok_pct   = round((ok / pct_base) * 100, 1) if pct_base > 0 else 0.0
                row_data = {
                    'infeed':     stats['zone']['name'],
                    'infeed_num': zid,
                    'ok':         ok,
                    'noscan':     noscan,
                    'unknown':    unknown,
                    'total':      total,
                    'ok_pct':     ok_pct,
                    'is_other':   is_other,
                }
                for code in error_codes:
                    row_data[f'err_{code}'] = stats['errors'].get(code, 0)
                rows_out.append(row_data)
            return rows_out

        otros_rows = _build_rows(otros_by_zone, is_other=True)
        if otros_rows:
            agg = {'ok': 0, 'noscan': 0, 'unknown': 0, 'total': 0}
            err_agg: Dict[str, int] = {}
            for r in otros_rows:
                agg['ok']      += r['ok']
                agg['noscan']  += r['noscan']
                agg['unknown'] += r['unknown']
                agg['total']   += r['total']
                for code in error_codes:
                    err_agg[code] = err_agg.get(code, 0) + r.get(f'err_{code}', 0)
            otros_header = {
                'infeed':          'Otros (paquetes registrados después de WSO)',
                'infeed_num':      -1,
                'ok_pct':          None,
                'is_other_header': True,
                **agg,
                **{f'err_{c}': v for c, v in err_agg.items()},
            }
            result = _build_rows(zone_stats, is_other=False) + [otros_header] + otros_rows
        else:
            result = _build_rows(zone_stats, is_other=False)

        _out = {
            'data':             result,
            'error_codes':      error_codes,
            'exclude_from_pct': list(exclude_from_pct),
            'customer':         cfg['name'],
        }
        await _write_cache(conn, database_id, 'scale_quality', _out)
        return _out


@app.get("/databases/{database_id}/tracking_losses")
async def get_tracking_losses(database_id: str):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        table_name, safe_table, columns, cfg = await _get_db_info(conn, database_id)
        if not cfg or not cfg.get('entry_point'):
            return {"data": [], "error": "Analytics not configured for this customer"}
        cached = await _read_cache(conn, database_id, 'tracking_losses')
        if cached is not None:
            return cached

        sort_msg_id   = cfg.get('sort_report', {}).get('message_id', 20)
        exitpoint_col = find_column(columns, 'parcelexitpoint', 'exitpoint', 'exit_point',
                                    'parcel_exit_point', 'parceldestinationpoint')
        exitstate_col = find_column(columns, 'parcelexitstate', 'exitstate', 'exit_state',
                                    'parcel_exit_state', 'parcelstate')

        if not exitpoint_col or not exitstate_col:
            return {"data": [], "error": f"Required columns not found (exitpoint={exitpoint_col}, exitstate={exitstate_col})"}

        rows = await conn.fetch(f"""
            SELECT {exitpoint_col}, COUNT(*) AS cnt
            FROM {safe_table}
            WHERE message_id = {sort_msg_id}
              AND {exitstate_col}::text = '2'
              AND {exitpoint_col} IS NOT NULL
              AND {exitpoint_col}::text NOT IN ('-', '')
            GROUP BY {exitpoint_col}
            ORDER BY cnt DESC
        """)

        zone_counts: Dict[int, Dict] = {}
        total = 0
        for row in rows:
            ep  = str(row[exitpoint_col] or '')
            cnt = int(row['cnt'])
            zone = get_zone_for_entry_point(ep, cfg)
            zid  = zone['id']
            if zid not in zone_counts:
                zone_counts[zid] = {'zone': zone, 'count': 0, 'points': []}
            zone_counts[zid]['count'] += cnt
            zone_counts[zid]['points'].append({'exit_point': ep, 'count': cnt})
            total += cnt

        zone_order = {z['id']: i for i, z in enumerate(cfg.get('zones', []))}
        result = sorted([
            {
                'zone_id':   zid,
                'zone_name': d['zone']['name'],
                'count':     d['count'],
                'pct':       round(d['count'] / total * 100, 1) if total > 0 else 0,
                'points':    sorted(
                    [{'exit_point': p['exit_point'], 'count': p['count'],
                      'pct_of_zone': round(p['count'] / d['count'] * 100, 1) if d['count'] > 0 else 0}
                     for p in d['points']],
                    key=lambda x: -x['count']
                ),
            }
            for zid, d in zone_counts.items()
        ], key=lambda r: zone_order.get(r['zone_id'], 99))

        _out = {'data': result, 'total': total, 'customer': cfg.get('name')}
        await _write_cache(conn, database_id, 'tracking_losses', _out)
        return _out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)