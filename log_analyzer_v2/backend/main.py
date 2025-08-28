from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import polars as pl
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Log Analyzer API", version="2.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://loguser:logpass@postgres:5432/logs_analyzer")
db_pool = None

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

# Core processing functions
def extract_all_fields_and_mappings(config: Dict) -> Tuple[Dict[int, Dict], List[str], List[str]]:
    """Extract ALL unique fields and create mappings per MessageID"""
    mappings = {}
    all_fields = set()
    display_order = []  # Order from MessageID 20
    
    for channel in config.get("ChannelsConfiguration", []):
        for msg in channel.get("MessageConfiguration", []):
            msg_id = int(msg.get("MessageId", -1))
            if 1 <= msg_id <= 100 and msg_id not in [30, 99]:
                fields = {}
                used_orders = set()  # Track used orders to detect duplicates
                
                for field in msg.get("EnabledFields", []):
                    if field_name := field.get("MessageField", ""):
                        order = field.get("Order", field.get("DefaultOrder", 9999))
                        
                        # If Order is already used, fall back to DefaultOrder
                        if order in used_orders:
                            default_order = field.get("DefaultOrder", 9999)
                            logger.warning(f"MessageID {msg_id}: Order {order} duplicated for {field_name}, using DefaultOrder {default_order}")
                            order = default_order
                        
                        used_orders.add(order)
                        fields[order] = field_name
                        all_fields.add(field_name)
                        
                        # Capture display order from MessageID 20
                        if msg_id == 20:
                            display_order.append((order, field_name))
                
                if fields:
                    mappings[msg_id] = fields
                    # Log the final mapping for this MessageID
                    logger.info(f"MessageID {msg_id} mapping: {sorted(fields.items())}")
    
    # Sort display order by Order value from MessageID 20
    display_order.sort(key=lambda x: x[0])
    ordered_fields = [field for order, field in display_order]
    
    # Add any remaining fields not in MessageID 20
    remaining_fields = sorted(all_fields - set(ordered_fields))
    ordered_fields.extend(remaining_fields)
    
    # Final columns: timestamp, message_id + all ordered fields
    all_columns = ["timestamp", "message_id"] + ordered_fields
    
    return mappings, all_columns, ordered_fields

def analyze_log_file_structure(log_path: str) -> Tuple[int, Dict[int, int], int]:
    """🎯 COMPLETE FILE ANALYSIS: Find max columns, MessageID distribution, and sample data"""
    max_cols = 0
    lines_processed = 0
    message_id_stats = {}
    message_id_max_cols = {}
    sample_lines = []
    
    logger.info("🔍 ANALYZING COMPLETE FILE STRUCTURE...")
    start_time = time.time()
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('|')
            current_cols = len(parts)
            
            # Store first 5 sample lines for debugging
            if len(sample_lines) < 5:
                sample_lines.append(f"Line {line_num}: {current_cols} cols -> {line[:100]}...")
            
            if len(parts) < 2:
                continue
                
            # Check if second column (index 1) is a valid message ID
            try:
                message_id_str = parts[1].strip()
                if not message_id_str.isdigit():
                    continue
                    
                message_id = int(message_id_str)
                if 1 <= message_id <= 100 and message_id not in [30, 99]:
                    # Track statistics
                    message_id_stats[message_id] = message_id_stats.get(message_id, 0) + 1
                    
                    # Update max columns for this message ID
                    if message_id not in message_id_max_cols or current_cols > message_id_max_cols[message_id]:
                        message_id_max_cols[message_id] = current_cols
                        logger.info(f"📊 MessageID {message_id}: NEW MAX {current_cols} columns (line {line_num})")
                    
                    # Update global max
                    if current_cols > max_cols:
                        max_cols = current_cols
                        logger.info(f"🏆 GLOBAL MAX: {max_cols} columns (MessageID {message_id}, line {line_num})")
                    
                    lines_processed += 1
                    
                    # Log progress every 50k lines
                    if lines_processed % 50000 == 0:
                        elapsed = time.time() - start_time
                        logger.info(f"⏳ Progress: {lines_processed:,} valid lines, {elapsed:.1f}s, current max: {max_cols}")
                        
            except (ValueError, IndexError):
                continue
    
    analysis_time = time.time() - start_time
    
    # Final summary
    logger.info("=" * 60)
    logger.info("📋 COMPLETE FILE ANALYSIS RESULTS:")
    logger.info(f"⏱️  Analysis time: {analysis_time:.2f} seconds")
    logger.info(f"📊 Total valid lines analyzed: {lines_processed:,}")
    logger.info(f"🏆 Maximum columns found: {max_cols}")
    logger.info(f"🎯 MessageIDs found: {sorted(message_id_stats.keys())}")
    
    for msg_id in sorted(message_id_max_cols.keys()):
        count = message_id_stats[msg_id]
        cols = message_id_max_cols[msg_id]
        logger.info(f"    MessageID {msg_id:2d}: {count:6,} lines, max {cols:2d} columns")
    
    logger.info("📄 Sample lines:")
    for sample in sample_lines[:3]:
        logger.info(f"    {sample}")
    logger.info("=" * 60)
    
    return max_cols, message_id_max_cols, lines_processed

async def get_unique_table_name(base_name: str) -> str:
    """Generate unique table name (dia_22, dia_22_1, etc.)"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        counter = 0
        table_name = base_name
        
        while True:
            exists = await conn.fetchval("""
                SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)
            """, table_name)
            
            if not exists:
                return table_name
            
            counter += 1
            table_name = f"{base_name}_{counter}"

async def create_dynamic_table(table_name: str, all_columns: List[str]):
    """Create table with ALL possible columns from JSON"""
    pool = await get_db_pool()
    
    # Safe column names for SQL
    safe_columns = ["timestamp", "message_id"]
    for col in all_columns[2:]:
        safe_col = re.sub(r'[^a-zA-Z0-9_]', '_', col.lower())
        safe_columns.append(safe_col)
    
    col_defs = [
        "id BIGSERIAL PRIMARY KEY",
        "timestamp TIMESTAMP",
        "message_id INTEGER"
    ] + [f"{col} TEXT DEFAULT '-'" for col in safe_columns[2:]]
    
    async with pool.acquire() as conn:
        await conn.execute(f"CREATE TABLE {table_name} ({', '.join(col_defs)})")
        await conn.execute(f"CREATE INDEX idx_{table_name}_msgid ON {table_name} (message_id)")
        await conn.execute(f"CREATE INDEX idx_{table_name}_ts ON {table_name} (timestamp)")

def extract_table_number(filename: str) -> str:
    """Extract number from filename safely"""
    number_match = re.search(r'(\d+)', filename)
    return number_match.group(1) if number_match else str(int(time.time()))

async def process_log_optimized(log_path: str, config: Dict, database_id: str, filename: str) -> Tuple[int, float, str]:
    """🎯 COMPLETELY REWRITTEN: Full file analysis + optimized processing"""
    start_time = time.time()
    
    # Extract complete configuration
    mappings, all_columns, display_order = extract_all_fields_and_mappings(config)
    if not mappings:
        raise HTTPException(400, "No valid message configurations found")
    
    # Generate unique table name
    table_number = extract_table_number(filename)
    base_table_name = f"dia_{table_number}"
    table_name = await get_unique_table_name(base_table_name)
    
    file_size_mb = os.path.getsize(log_path) / (1024 * 1024)
    logger.info("🚀" * 20)
    logger.info(f"🚀 PROCESSING: {filename} ({file_size_mb:.1f}MB) -> {table_name}")
    logger.info(f"📋 JSON configuration: {len(all_columns)} total columns expected")
    logger.info(f"🎯 MessageIDs in config: {sorted(mappings.keys())}")
    
    # 🎯 PHASE 1: COMPLETE FILE ANALYSIS
    max_cols, message_id_max_cols, valid_lines_found = analyze_log_file_structure(log_path)
    
    if max_cols == 0:
        raise HTTPException(400, "No valid message lines found in log file")
    
    # Create table with ALL columns from JSON
    await create_dynamic_table(table_name, all_columns)
    logger.info(f"✅ Database table created with {len(all_columns)} columns")
    
    # 🎯 PHASE 2: PRE-FILTER FILE TO CONTAIN ONLY VALID MESSAGE LINES
    logger.info(f"🔧 Pre-filtering file to contain only valid MessageID lines...")
    
    import tempfile
    filtered_file = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False, encoding='utf-8')
    lines_written = 0
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as input_file:
            for line in input_file:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('|')
                if len(parts) >= 2:
                    try:
                        message_id_str = parts[1].strip()
                        if message_id_str.isdigit():
                            message_id = int(message_id_str)
                            if 1 <= message_id <= 100 and message_id not in [30, 99]:
                                filtered_file.write(line + '\n')
                                lines_written += 1
                    except (ValueError, IndexError):
                        continue
        
        filtered_file.close()
        logger.info(f"✅ Filtered file created: {lines_written:,} valid lines (was ~{valid_lines_found})")
        
        # 🎯 PHASE 3: LOAD FILTERED DATA (NOW POLARS WILL SEE CORRECT STRUCTURE)
        logger.info(f"⚡ Loading pre-filtered data...")
        
        df = pl.read_csv(
            filtered_file.name,
            separator='|',
            has_header=False,
            ignore_errors=True,
            encoding='utf8-lossy',
            truncate_ragged_lines=True,
            dtypes={"column_1": pl.Utf8, "column_2": pl.Utf8}
        )
        
        current_cols = len(df.columns)
        logger.info(f"📊 Polars detected {current_cols} columns from filtered data (expected {max_cols})")
        
        # Now we should have the correct structure or very close to it
        if current_cols < max_cols:
            logger.info(f"🔧 Adding {max_cols - current_cols} missing columns...")
            for i in range(current_cols, max_cols):
                df = df.with_columns(pl.lit("").cast(pl.Utf8).alias(f"temp_col_{i}"))
        elif current_cols > max_cols:
            logger.info(f"🔧 Trimming to {max_cols} columns...")
            df = df.select([f"column_{i+1}" for i in range(max_cols)])
        
        # Rename all columns to f0, f1, f2, ... f{max_cols-1}
        new_column_names = [f"f{i}" for i in range(max_cols)]
        df = df.rename({old: new for old, new in zip(df.columns, new_column_names)})
        
        # Convert ALL columns to string to avoid dtype issues in mapping
        df = df.with_columns([pl.col(col).cast(pl.Utf8) for col in df.columns])
        
        logger.info(f"✅ Clean data loaded: {len(df)} rows, {len(df.columns)} columns (all Utf8)")
        
    finally:
        # Clean up the temporary filtered file
        try:
            os.unlink(filtered_file.name)
        except:
            pass
    
    # PHASE 3: Filter valid lines (MessageID validation)
    logger.info("🔍 Filtering valid message records...")
    
    df = df.filter(
        pl.col("f0").is_not_null() &
        pl.col("f1").is_not_null() &
        pl.col("f1").str.strip().str.contains(r"^\d{1,3}$") &
        pl.col("f1").str.strip().cast(pl.Int32, strict=False).is_between(1, 100) &
        (~pl.col("f1").str.strip().cast(pl.Int32, strict=False).is_in([30, 99]))
    )
    
    logger.info(f"✅ {len(df)} valid records after filtering (expected ~{valid_lines_found})")
    
    if len(df) == 0:
        return 0, time.time() - start_time, table_name
    
    # PHASE 4: Process timestamp and message_id
    logger.info("🕐 Processing timestamp and message_id...")
    
    try:
        df = df.with_columns([
            # Process timestamp with better error handling
            pl.col("f0").str.strip().str.strptime(pl.Datetime, "%d/%m/%Y %H:%M:%S%.f", strict=False)
            .fill_null(
                pl.col("f0").str.strip().str.strptime(pl.Datetime, "%d/%m/%Y %H:%M:%S", strict=False)
            )
            .alias("timestamp"),
            
            # Process message_id
            pl.col("f1").str.strip().cast(pl.Int32).alias("message_id")
        ])
        logger.info("✅ Timestamp and message_id processing completed")
        
    except Exception as e:
        logger.error(f"❌ Error processing timestamp/message_id: {e}")
        # Log sample data for debugging
        sample_data = df.select([pl.col("f0").head(5), pl.col("f1").head(5)])
        logger.error(f"Sample data: {sample_data}")
        raise HTTPException(500, f"Error processing datetime fields: {str(e)}")
    
    # PHASE 5: Dynamic field mapping according to JSON configuration  
    logger.info("🗺️ Applying dynamic field mapping...")
    logger.info(f"🎯 Mapping {len(all_columns)} JSON fields to {max_cols} file columns")
    
    # Prepare safe column names
    safe_columns = ["timestamp", "message_id"]
    for col in all_columns[2:]:
        safe_col = re.sub(r'[^a-zA-Z0-9_]', '_', col.lower())
        safe_columns.append(safe_col)
    
    # Build mapping expressions
    mapping_expressions = [pl.col("timestamp"), pl.col("message_id")]
    
    for target_col_name in all_columns[2:]:
        safe_target = re.sub(r'[^a-zA-Z0-9_]', '_', target_col_name.lower())
        expr = pl.lit("-").cast(pl.Utf8)  # Default value
        
        # Check each MessageID configuration
        for msg_id, field_mapping in mappings.items():
            for order, field_name in field_mapping.items():
                if field_name == target_col_name:
                    source_col_index = order + 2  # +2 for timestamp and message_id
                    source_col = f"f{source_col_index}"
                    
                    # Only map if source column exists in our detected structure
                    if source_col_index < max_cols:
                        expr = pl.when(pl.col("message_id") == msg_id).then(
                            pl.when(pl.col(source_col).is_null() | (pl.col(source_col).str.strip() == ""))
                            .then(pl.lit("-"))
                            .otherwise(pl.col(source_col).str.strip())
                        ).otherwise(expr)
                        logger.info(f"    📌 {target_col_name} <- f{source_col_index} (MessageID {msg_id}, Order {order})")
                        break
                    else:
                        logger.warning(f"    ⚠️  {target_col_name}: source f{source_col_index} > max_cols {max_cols}")
                        break
        
        mapping_expressions.append(expr.alias(safe_target))
    
    # Apply mapping
    try:
        df_mapped = df.select(mapping_expressions)
        logger.info(f"✅ Mapping complete: {len(df_mapped)} records with {len(safe_columns)} columns")
        
    except Exception as e:
        logger.error(f"❌ Error in field mapping: {e}")
        logger.error(f"Target columns: {len(all_columns)}, Detected max columns: {max_cols}")
        logger.error(f"MessageID mappings: {list(mappings.keys())}")
        raise HTTPException(500, f"Error in field mapping: {str(e)}")
    
    # PHASE 6: Bulk insert to database
    logger.info("💨 Bulk inserting to database...")
    
    # Convert to records for insertion
    records = df_mapped.to_dicts()
    
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # Smart chunking for large datasets
        chunk_size = 100000
        total_inserted = 0
        
        try:
            if len(records) > chunk_size:
                logger.info(f"📦 Processing {len(records)} records in chunks of {chunk_size}")
                
                for i in range(0, len(records), chunk_size):
                    chunk = records[i:i + chunk_size]
                    
                    # Prepare data for this chunk
                    insert_data = []
                    for record in chunk:
                        row = []
                        for col in safe_columns:
                            value = record.get(col)
                            if col == "message_id":
                                row.append(value if isinstance(value, int) else None)
                            elif col == "timestamp":
                                row.append(value)
                            else:
                                row.append(str(value) if value is not None else "-")
                        insert_data.append(tuple(row))
                    
                    # Insert chunk
                    await conn.copy_records_to_table(table_name, records=insert_data, columns=safe_columns)
                    total_inserted += len(insert_data)
                    
                    # Log progress every 5 chunks
                    if (i//chunk_size + 1) % 5 == 0:
                        logger.info(f"📦 Progress: {total_inserted:,} records inserted")
                    
            else:
                # Single insert for smaller datasets
                insert_data = []
                for record in records:
                    row = []
                    for col in safe_columns:
                        value = record.get(col)
                        if col == "message_id":
                            row.append(value if isinstance(value, int) else None)
                        elif col == "timestamp":
                            row.append(value)
                        else:
                            row.append(str(value) if value is not None else "-")
                    insert_data.append(tuple(row))
                
                await conn.copy_records_to_table(table_name, records=insert_data, columns=safe_columns)
                total_inserted = len(insert_data)
            
            # Save metadata
            await conn.execute("""
                INSERT INTO databases (id, name, table_name, record_count, file_size_mb, columns_info)
                VALUES ($1, $1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE SET
                    record_count = EXCLUDED.record_count,
                    file_size_mb = EXCLUDED.file_size_mb,
                    columns_info = EXCLUDED.columns_info
            """, database_id, table_name, total_inserted, file_size_mb, json.dumps(all_columns))
            
        except Exception as e:
            logger.error(f"❌ Database insertion error: {e}")
            raise HTTPException(500, f"Error inserting data: {str(e)}")
    
    processing_time = time.time() - start_time
    speed = total_inserted / processing_time if processing_time > 0 else 0
    
    logger.info("🎉" * 20)
    logger.info(f"🎉 PROCESSING COMPLETE: {total_inserted:,} records in {processing_time:.2f}s ({speed:,.0f} rec/sec)")
    logger.info("🎉" * 20)
    
    return total_inserted, processing_time, table_name

# API Routes
@app.on_event("startup")
async def startup():
    await init_database()
    logger.info("🚀 Log Analyzer API started - COMPLETE FILE ANALYSIS VERSION")

@app.get("/")
async def root():
    return {"message": "Log Analyzer API - Complete File Analysis", "status": "running", "version": "2.1.0"}

@app.post("/upload", response_model=ProcessResponse)
async def upload_files(log_file: UploadFile = File(...), config_file: UploadFile = File(...), server_type: str = Query("eDS")):
    """Process log files with JSON configuration"""
    start_time = time.time()
    
    if not log_file.filename.endswith('.log'):
        raise HTTPException(400, "File must be .log")
    if server_type == "eDS" and not config_file.filename.endswith('.json'):
        raise HTTPException(400, "eDS requires .json config")
    
    database_id = f"{server_type.lower()}_{int(time.time())}_{log_file.filename.replace('.log', '')}"
    
    # Save files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.log') as tmp_log, \
         tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_config:
        
        tmp_log.write(await log_file.read())
        tmp_config.write(await config_file.read())
        log_path, config_path = tmp_log.name, tmp_config.name
    
    try:
        # Parse JSON configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.loads(f.read())
        
        # Process with completely rewritten function
        records_processed, processing_time, table_name = await process_log_optimized(
            log_path, config, database_id, log_file.filename
        )
        
        return ProcessResponse(
            success=True,
            message=f"Successfully processed into table '{table_name}'",
            processing_time=time.time() - start_time,
            records_processed=records_processed,
            database_id=database_id,
            table_name=table_name
        )
    
    finally:
        # Cleanup
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
        db_info = await conn.fetchrow("SELECT table_name, columns_info FROM databases WHERE id = $1", database_id)
        if not db_info:
            raise HTTPException(404, "Database not found")
        
        table_name, columns = db_info['table_name'], json.loads(db_info['columns_info'] or '[]')
        
        # Build query conditions
        conditions, params = [], []
        
        if query.message_id:
            conditions.append(f"message_id = ${len(params) + 1}")
            params.append(query.message_id)
        
        if query.search:
            search_cols = [f"{re.sub(r'[^a-zA-Z0-9_]', '_', col.lower())}::text ILIKE ${len(params) + 1}" 
                          for col in columns[2:]]
            if search_cols:
                conditions.append(f"({' OR '.join(search_cols)})")
                params.append(f"%{query.search}%")
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        # Get data
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
        exists = await conn.fetchval("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)", table_name)
        if not exists:
            raise HTTPException(404, "Table not found")
        
        # Get statistics in parallel
        total_records, message_dist, hourly_dist, time_range = await asyncio.gather(
            conn.fetchval(f"SELECT COUNT(*) FROM {table_name}"),
            conn.fetch(f"SELECT message_id, COUNT(*) as count FROM {table_name} GROUP BY message_id ORDER BY message_id"),
            conn.fetch(f"SELECT DATE_TRUNC('hour', timestamp) as hour, COUNT(*) as count FROM {table_name} WHERE timestamp IS NOT NULL GROUP BY hour ORDER BY hour"),
            conn.fetchrow(f"SELECT MIN(timestamp) as min_time, MAX(timestamp) as max_time FROM {table_name} WHERE timestamp IS NOT NULL")
        )
        
        stats = {
            'total_records': total_records,
            'message_distribution': [dict(row) for row in message_dist],
            'hourly_distribution': [{'hour': row['hour'].isoformat() if row['hour'] else None, 'count': row['count']} for row in hourly_dist]
        }
        
        if time_range['min_time'] and time_range['max_time']:
            stats['time_range'] = {
                'start': time_range['min_time'].isoformat(),
                'end': time_range['max_time'].isoformat()
            }
        
        return stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)