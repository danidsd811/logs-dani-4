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

# Configuración
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Log Analyzer - Ultra Optimized", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://loguser:logpass@postgres:5432/logs_analyzer")
db_pool = None

# Regex compilados para timestamp parsing ultra-rápido
TIMESTAMP_PATTERNS = [
    re.compile(r'^(\d{2})/(\d{2})/(\d{4}) (\d{2}):(\d{2}):(\d{2})\.(\d{3})$'),
    re.compile(r'^(\d{2})/(\d{2})/(\d{4}) (\d{2}):(\d{2}):(\d{2})$')
]

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
    
    logger.info(f"ULTRA-FAST COMPLETE: {total_processed:,} records in {processing_time:.2f}s ({speed:,.0f} rec/sec)")
    
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
async def upload_files(log_file: UploadFile = File(...), config_file: UploadFile = File(...), server_type: str = Query("eDS")):
    start_time = time.time()
    
    if not log_file.filename.endswith('.log'):
        raise HTTPException(400, "File must be .log")
    if not config_file.filename.endswith('.json'):
        raise HTTPException(400, "Config must be .json")
    
    database_id = f"{server_type.lower()}_{int(time.time())}_{log_file.filename.replace('.log', '')}"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.log') as tmp_log, \
         tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_config:
        
        tmp_log.write(await log_file.read())
        tmp_config.write(await config_file.read())
        log_path, config_path = tmp_log.name, tmp_config.name
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        records_processed, processing_time, table_name = await process_ultra_fast(
            log_path, config, database_id, log_file.filename
        )
        
        return ProcessResponse(
            success=True,
            message=f"Ultra-fast: {records_processed:,} records -> {table_name}",
            processing_time=time.time() - start_time,
            records_processed=records_processed,
            database_id=database_id,
            table_name=table_name
        )
    
    finally:
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