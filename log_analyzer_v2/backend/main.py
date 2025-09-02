from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import json
import re
import asyncio
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

app = FastAPI(title="Log Analyzer - Ultra Optimized", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://loguser:logpass@postgres:5432/logs_analyzer")
db_pool = None

# Regex compilados para timestamp parsing ultra-rápido
TIMESTAMP_PATTERNS = [
    re.compile(r'^(\d{2})/(\d{2})/(\d{4}) (\d{2}):(\d{2}):(\d{2})\.(\d{3})$'),
    re.compile(r'^(\d{2})/(\d{2})/(\d{4}) (\d{2}):(\d{2}):(\d{2})$')
]

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
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', field_name.lower())
            self.columns.append((safe_name, "TEXT"))
            # Mapear MessageField -> índice en tabla
            self.message_field_to_table_index[field_name] = len(self.columns) - 1
        
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
            
            return record
            
        except (ValueError, IndexError) as e:
            logger.debug(f"Parse error for line: {line[:50]}... Error: {e}")
            return None
    
    def get_create_table_sql(self, table_name: str) -> str:
        """SQL para crear tabla"""
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
        
        column_defs = ["id BIGSERIAL PRIMARY KEY"]
        for col_name, col_type in self.columns:
            column_defs.append(f"{col_name} {col_type}")
        
        return f"""
        CREATE TABLE {safe_name} ({', '.join(column_defs)});
        CREATE INDEX idx_{safe_name}_msgid ON {safe_name}(message_id);
        CREATE INDEX idx_{safe_name}_ts ON {safe_name}(timestamp);
        """


class SCNETSchemaOptimized:
    """Esquema SCNET corregido con orden del MessageID 20 y parsing funcional"""
    
    def __init__(self, xml_path: str):
        self.columns = []  # [(column_name, sql_type)]
        self.message_id_to_field_positions = {}  # {msg_id: {field_name: log_position}}
        self.property_to_table_index = {}  # property_name -> índice en tabla
        self.column_count = 0
        self.reference_order = []  # Orden de MessageID 20
        self._build_schema_from_xml(xml_path)
    
    def _build_schema_from_xml(self, xml_path: str):
        """Construir esquema con orden del MessageID 20"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # PASO 1: Extraer TODOS los MessageIDs y sus propiedades
            parcel_reports = {}
            all_properties = set()
            reference_properties_order = []  # Para MessageID 20
            
            for report in root.findall('.//ParcelDataReport'):
                report_id = report.get('id', '')
                
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
                
                # Extraer propiedades EN ORDEN
                properties = []
                for prop in report.findall('property'):
                    prop_text = prop.text
                    if prop_text:
                        prop_name = prop_text.strip()
                        properties.append(prop_name)
                        all_properties.add(prop_name)
                
                if properties and message_number is not None:
                    parcel_reports[message_number] = properties
                    
                    # CAPTURAR ORDEN DEL MessageID 20 como referencia
                    if message_number == 20:
                        reference_properties_order = properties.copy()
                        logger.info(f"MessageID 20 found - using as column order reference: {len(properties)} properties")
                
                logger.info(f"SCNET MessageID {message_number}: {len(properties)} properties")
            
            if not parcel_reports:
                raise HTTPException(400, "No valid ParcelDataReport found in XML")
            
            # PASO 2: Crear esquema de tabla usando ORDEN del MessageID 20
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
        """Crear esquema usando el orden del MessageID 20"""
        # Columnas base
        self.columns = [("timestamp", "TIMESTAMP"), ("message_id", "INTEGER")]
        
        # Usar orden del MessageID 20 si existe
        if reference_order:
            ordered_properties = reference_order.copy()
            # Agregar propiedades que no estén en MessageID 20 al final
            remaining = sorted(all_properties - set(reference_order))
            ordered_properties.extend(remaining)
            logger.info(f"Using MessageID 20 order: {len(reference_order)} properties + {len(remaining)} additional")
        else:
            # Fallback: orden alfabético
            ordered_properties = sorted(all_properties)
            logger.warning("MessageID 20 not found - using alphabetical order")
        
        # Crear columnas y mapeo
        for i, prop_name in enumerate(ordered_properties):
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', prop_name.lower())
            self.columns.append((safe_name, "TEXT"))
            # Mapear property_name -> índice en tabla
            self.property_to_table_index[prop_name] = i + 2  # +2 por timestamp, message_id
        
        self.column_count = len(self.columns)
        logger.info(f"Table schema created: {self.column_count} columns")
    
    def _create_message_mappings(self, parcel_reports: dict):
        """Crear mapeos de posición para cada MessageID - LÓGICA SIMPLIFICADA"""
        
        # MessageIDs que no tienen <text> en el XML - leen PIC directamente de posición 2
        no_text_message_ids = {11, 13, 31}
        
        for message_number, properties in parcel_reports.items():
            field_positions = {}
            
            if message_number in no_text_message_ids:
                # LÓGICA SIMPLE: MessageIDs sin <text>
                # Posición 2: PIC (leído directamente del archivo)
                field_positions['pic'] = 2
                
                # Posición 3+: Las propiedades definidas en XML
                for i, prop_name in enumerate(properties):
                    field_positions[prop_name] = i + 3
                
                logger.info(f"MessageID {message_number} (sin <text>): PIC directo pos 2 + {len(properties)} properties del XML")
                
            else:
                # REGLA NORMAL: MessageIDs con <text>
                # Posición 2+: Las propiedades definidas en XML
                for log_position, prop_name in enumerate(properties):
                    field_positions[prop_name] = log_position + 2
                
                logger.info(f"MessageID {message_number} (con <text>): {len(properties)} properties normales")
            
            if field_positions:
                self.message_id_to_field_positions[message_number] = field_positions
        
        logger.info(f"Message mappings created for {len(self.message_id_to_field_positions)} MessageIDs")
    
    def parse_timestamp_fast(self, ts_str: str) -> Optional[datetime]:
        """Parse timestamp formato SCNET: DD/MM/YYYY HH:MM:SS:mmm"""
        try:
            # Formato: 07/05/2025 00:00:00:659
            if ':' in ts_str and len(ts_str.split(':')) >= 4:
                date_part, time_part = ts_str.split(' ', 1)
                day, month, year = date_part.split('/')
                time_parts = time_part.split(':')
                if len(time_parts) >= 4:
                    hour, minute, second, millisecond = time_parts[0], time_parts[1], time_parts[2], time_parts[3]
                    return datetime(
                        int(year), int(month), int(day),
                        int(hour), int(minute), int(second),
                        int(millisecond) * 1000
                    )
        except (ValueError, IndexError):
            pass
        return None
    
    def parse_line_ultra_fast(self, line: str) -> Optional[List]:
        """Parse línea FSC con limpieza robusta de caracteres de control"""
        if not line.strip():
            return None
    
        # PASO 1: Limpiar caracteres de control y @ al final
        line = line.rstrip('@').strip()
        
        # Remover caracteres de control comunes (0x00 a 0x1F)
        import re
        line = re.sub(r'[\x00-\x1F]', '', line)
        
        if not line:
            return None
    
        parts = line.split('|')
        if len(parts) < 3:
            return None
    
        try:
            # Parse básico
            timestamp_str = parts[0].strip()
            message_id_str = parts[1].strip()
            
            # PASO 2: Limpiar message_id de caracteres no numéricos residuales
            message_id_clean = re.sub(r'[^0-9]', '', message_id_str)
            
            if not message_id_clean:
                return None
        
            timestamp = self.parse_timestamp_fast(timestamp_str)
            if not timestamp:
                return None
            
            message_id = int(message_id_clean)
        
            # Verificar si existe mapeo para este MessageID
            if message_id not in self.message_id_to_field_positions:
                # DEBUG TEMPORAL: comentar después de validar
                # logger.debug(f"MessageID {message_id} not in schema. Available: {list(self.message_id_to_field_positions.keys())[:5]}...")
                return None
        
            # Inicializar record
            record = ["-"] * self.column_count
            record[0] = timestamp
            record[1] = message_id
        
            # Mapear campos
            field_positions = self.message_id_to_field_positions[message_id]
            mapped_count = 0
            
            # DEBUG TEMPORAL: para las primeras 10 líneas de cada MessageID
            is_debug = hasattr(self, '_debug_count')
            if not is_debug:
                self._debug_count = {}
            
            if message_id not in self._debug_count:
                self._debug_count[message_id] = 0
            
            debug_this_line = self._debug_count[message_id] < 3
            if debug_this_line:
                self._debug_count[message_id] += 1
                logger.info(f"DEBUG MessageID {message_id}: Line has {len(parts)} parts, expects {len(field_positions)} fields")
            
            for prop_name, log_position in field_positions.items():
                if prop_name in self.property_to_table_index:
                    if log_position < len(parts):
                        value = parts[log_position].strip()
                        if value:  # Solo mapear valores no vacíos
                            table_index = self.property_to_table_index[prop_name]
                            record[table_index] = value
                            mapped_count += 1
                            
                            if debug_this_line:
                                logger.info(f"  Mapped {prop_name} (pos {log_position}): '{value[:20]}...' -> col {table_index}")
                    elif debug_this_line:
                        logger.warning(f"  Missing {prop_name} at position {log_position} (line has {len(parts)} parts)")
        
            if debug_this_line:
                logger.info(f"  Result: {mapped_count}/{len(field_positions)} fields mapped for MessageID {message_id}")
            
            # Solo retornar si se mapeó al menos algo
            return record if mapped_count > 0 else None
        
        except (ValueError, IndexError) as e:
            # DEBUG TEMPORAL
            logger.debug(f"Parse error: {str(e)} - Line: {line[:100]}...")
            return None
    
    def get_create_table_sql(self, table_name: str) -> str:
        """SQL para crear tabla SCNET"""
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
        
        column_defs = ["id BIGSERIAL PRIMARY KEY"]
        for col_name, col_type in self.columns:
            column_defs.append(f"{col_name} {col_type}")
        
        return f"""
        CREATE TABLE {safe_name} ({', '.join(column_defs)});
        CREATE INDEX idx_{safe_name}_msgid ON {safe_name}(message_id);
        CREATE INDEX idx_{safe_name}_ts ON {safe_name}(timestamp);
        """

# Database functions
async def get_db_pool():
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=5, max_size=20)
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
                columns_info JSONB
            )
        """)

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

async def process_ultra_fast(log_path: str, config: Dict, database_id: str, filename: str) -> Tuple[int, float, str]:
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
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
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
            INSERT INTO databases (id, name, table_name, record_count, file_size_mb, columns_info)
            VALUES ($1, $1, $2, $3, $4, $5)
        """, database_id, table_name, total_processed, file_size_mb, json.dumps(column_names))
    
    # *** LIMPIEZA DESPUÉS de guardar - ahora que tenemos la nueva tabla ***
    await cleanup_old_tables()
    
    logger.info(f"ULTRA-FAST COMPLETE: {total_processed:,} records in {processing_time:.2f}s ({speed:,.0f} rec/sec)")
    
    return total_processed, processing_time, table_name

async def process_scnet_ultra_fast(fsc_path: str, xml_path: str, database_id: str, filename: str) -> Tuple[int, float, str]:
    """Procesamiento SCNET con debug temporal"""
    start_time = time.time()
    
    # Construir esquema
    schema = SCNETSchemaOptimized(xml_path)
    if not schema.message_id_to_field_positions:
        raise HTTPException(400, "No valid ParcelDataReport found in XML")
    
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
    
    # Procesamiento con contadores de debug
    batch = []
    batch_size = 50000
    total_processed = 0
    lines_read = 0
    lines_skipped = 0
    last_log = start_time
    
    with open(fsc_path, 'r', encoding='utf-8', errors='ignore') as f:
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
                    
                    # Log progreso
                    if time.time() - last_log > 5:  # Más frecuente para debug
                        rate = total_processed / (time.time() - start_time)
                        skip_rate = (lines_skipped / lines_read) * 100 if lines_read > 0 else 0
                        logger.info(f"SCNET Progress: {lines_read:,} lines read, {total_processed:,} records processed ({rate:,.0f}/sec, {skip_rate:.1f}% skipped)")
                        last_log = time.time()
            else:
                lines_skipped += 1
    
    # Último batch
    if batch:
        inserted = await insert_ultra_fast(pool, table_name, batch, schema.columns)
        total_processed += inserted
    
    processing_time = time.time() - start_time
    speed = total_processed / processing_time if processing_time > 0 else 0
    
    # Stats finales
    skip_rate = (lines_skipped / lines_read) * 100 if lines_read > 0 else 0
    logger.info(f"SCNET Final Stats: {lines_read:,} lines read, {total_processed:,} records, {lines_skipped:,} skipped ({skip_rate:.1f}%)")
    
    # Guardar metadatos
    column_names = [col[0] for col in schema.columns]
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO databases (id, name, table_name, record_count, file_size_mb, columns_info)
            VALUES ($1, $1, $2, $3, $4, $5)
        """, database_id, table_name, total_processed, file_size_mb, json.dumps(column_names))
    
    # *** LIMPIEZA DESPUÉS de guardar - ahora que tenemos la nueva tabla ***
    await cleanup_old_tables()
    
    logger.info(f"SCNET COMPLETE: {total_processed:,} records in {processing_time:.2f}s ({speed:,.0f} rec/sec)")
    
    return total_processed, processing_time, table_name

async def insert_ultra_fast(pool, table_name: str, batch: List[tuple], columns: List[tuple]) -> int:
    """Inserción ultra-optimizada"""
    if not batch:
        return 0
    
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
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

# API Routes
@app.on_event("startup")
async def startup():
    await init_database()
    logger.info("Ultra-Optimized Log Analyzer started")

@app.get("/")
async def root():
    return {"message": "Ultra-Optimized Log Analyzer", "version": "4.0.0"}

@app.post("/upload", response_model=ProcessResponse)
async def upload_files(
    log_file: UploadFile = File(...), 
    config_file: UploadFile = File(...), 
    server_type: str = Query("eDS")
):
    start_time = time.time()
    
    # Detectar tipo de tecnología basado en extensiones de archivo
    log_extension = log_file.filename.split('.')[-1].lower()
    config_extension = config_file.filename.split('.')[-1].lower()
    
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
        tmp_config.write(await config_file.read())
        log_path, config_path = tmp_log.name, tmp_config.name
    
    try:
        # Procesar según el tipo de tecnología
        if server_type.lower() == "eds":
            logger.info(f"Processing eDS files: {log_file.filename} + {config_file.filename}")
            
            # Cargar configuración JSON
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Procesar con eDS
            records_processed, processing_time, table_name = await process_ultra_fast(
                log_path, config, database_id, log_file.filename
            )
            
            message = f"eDS: {records_processed:,} records processed -> {table_name}"
            
        elif server_type.lower() == "scnet":
            logger.info(f"Processing SCNET files: {log_file.filename} + {config_file.filename}")
            
            # Procesar con SCNET (XML + FSC)
            records_processed, processing_time, table_name = await process_scnet_ultra_fast(
                log_path, config_path, database_id, log_file.filename
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
            table_name=table_name
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
            SELECT id, name, table_name, created_at, record_count, file_size_mb
            FROM databases ORDER BY created_at DESC LIMIT 20
        """)
        return [DatabaseInfo(**dict(row)) for row in rows]

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
        
        conditions, params = [], []
        
        if query.message_id:
            conditions.append(f"message_id = ${len(params) + 1}")
            params.append(query.message_id)
        
        if query.search:
            text_cols = columns[2:7]  # Solo primeras 5 columnas para eficiencia
            search_conds = [f"{col}::text ILIKE ${len(params) + 1}" for col in text_cols]
            if search_conds:
                conditions.append(f"({' OR '.join(search_conds)})")
                params.append(f"%{query.search}%")
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        total_records = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name} {where_clause}", *params)
        
        offset = (query.page - 1) * query.limit
        rows = await conn.fetch(f"""
            SELECT * FROM {table_name} {where_clause}
            ORDER BY timestamp DESC LIMIT {query.limit} OFFSET {offset}
        """, *params)
        
        data = [{k: v.isoformat() if isinstance(v, datetime) else v 
                for k, v in row.items() if k != "id"} for row in rows]
        
        return LogResponse(
            data=data,
            total_records=total_records,
            page=query.page,
            total_pages=(total_records + query.limit - 1) // query.limit,
            processing_time=time.time() - start_time,
            columns=columns
        )

@app.get("/table/{table_name}/stats")
async def get_table_stats(table_name: str):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
        
        exists = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)", 
            safe_name
        )
        if not exists:
            raise HTTPException(404, "Table not found")
        
        total_records, message_dist = await asyncio.gather(
            conn.fetchval(f"SELECT COUNT(*) FROM {safe_name}"),
            conn.fetch(f"SELECT message_id, COUNT(*) as count FROM {safe_name} GROUP BY message_id ORDER BY message_id")
        )
        
        return {
            'total_records': total_records,
            'message_distribution': [dict(row) for row in message_dist]
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)