import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Upload, Search, Database, FileText, X, BarChart3, Activity, ChevronLeft, ChevronRight } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const API_BASE = 'http://localhost:8000';

// Colores para gráficos

function LogAnalyzerApp() {
  const [activeTab, setActiveTab] = useState('upload');
  const [databases, setDatabases] = useState([]);
  const [selectedDatabase, setSelectedDatabase] = useState(null);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [inductionQuality, setInductionQuality] = useState(null);
  const [badHostpics, setBadHostpics] = useState(null);
  const [goodHostpics, setGoodHostpics] = useState(null);
  const [sortQuality, setSortQuality] = useState(null);
  const [customers, setCustomers] = useState([]);
  const analyticsAbortRef = useRef(null);

  // Estados para filtros
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalRecords, setTotalRecords] = useState(0);
  const [columns, setColumns] = useState([]);

  // Cargar bases de datos y clientes al iniciar
  useEffect(() => {
    fetchDatabases();
    fetch(`${API_BASE}/customers`)
      .then(r => r.json())
      .then(data => setCustomers([...data].sort((a, b) => a.name.localeCompare(b.name))))
      .catch(() => {});
  }, []);

  // CORREGIDO: Función global para refresh desde ViewLogsTab
  useEffect(() => {
    window.refreshDatabases = fetchDatabases;
  
    return () => {
      window.refreshDatabases = null;
    };
  }, []);

  const fetchDatabases = async (selectNewest = false) => {
    try {
      const response = await fetch(`${API_BASE}/databases`);
      const data = await response.json();
      setDatabases(data);
      if (data.length > 0 && (!selectedDatabase || selectNewest)) {
        setSelectedDatabase(data[0]);
      }
    } catch (error) {
      console.error('Error fetching databases:', error);
    }
  };

  const fetchLogs = async (page = 1, search = '') => {
    if (!selectedDatabase) return;

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/logs/${selectedDatabase.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          page,
          limit: 100,
          search: search || undefined
        })
      });

      const data = await response.json();
      setLogs(data.data);
      setTotalPages(data.total_pages);
      setTotalRecords(data.total_records);
      setCurrentPage(page);
      setColumns(data.columns || []);
    } catch (error) {
      console.error('Error fetching logs:', error);
      setLogs([]);
    } finally {
      setLoading(false);
    }
  };

  const fetchInductionQuality = async (databaseId) => {
    if (!databaseId) return;
    if (analyticsAbortRef.current) analyticsAbortRef.current.abort();
    const controller = new AbortController();
    analyticsAbortRef.current = controller;
    const { signal } = controller;

    setInductionQuality(null);
    setBadHostpics(null);
    setGoodHostpics(null);
    setSortQuality(null);
    try {
      const [iq, bad, good, sq] = await Promise.allSettled([
        fetch(`${API_BASE}/databases/${databaseId}/induction_quality`, { signal }).then(r => r.json()),
        fetch(`${API_BASE}/databases/${databaseId}/bad_hostpics`, { signal }).then(r => r.json()),
        fetch(`${API_BASE}/databases/${databaseId}/good_hostpics`, { signal }).then(r => r.json()),
        fetch(`${API_BASE}/databases/${databaseId}/sort_quality`, { signal }).then(r => r.json()),
      ]);
      if (signal.aborted) return;
      setInductionQuality(iq.status === 'fulfilled' ? iq.value : { data: [] });
      setBadHostpics(bad.status === 'fulfilled' ? bad.value : { data: [], total: 0 });
      setGoodHostpics(good.status === 'fulfilled' ? good.value : { data: [], total: 0 });
      setSortQuality(sq.status === 'fulfilled' ? sq.value : { data: [] });
    } catch (e) {
      if (!signal.aborted) throw e;
    }
  };

  // Efecto para cargar logs y estadísticas cuando cambia la base de datos
  useEffect(() => {
    if (selectedDatabase) {
      if (activeTab === 'view') {
        fetchLogs(1, searchTerm);
      } else if (activeTab === 'charts') {
        fetchInductionQuality(selectedDatabase.id);
      }
    }
  }, [selectedDatabase, activeTab]);

  const handleSearch = () => {
    setCurrentPage(1);
    fetchLogs(1, searchTerm);
  };

  const handlePageChange = (page) => {
    fetchLogs(page, searchTerm);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <FileText className="h-8 w-8 text-blue-600" />
              <h1 className="text-2xl font-bold text-gray-900">Log Analyzer v2.0</h1>
              <span className="px-2 py-1 text-xs bg-green-100 text-green-800 rounded-full">
                FastAPI + Polars + PostgreSQL
              </span>
            </div>
            {selectedDatabase && (
              <div className="text-sm text-gray-600">
                Tabla activa: <span className="font-semibold">{selectedDatabase.table_name}</span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            <button
              onClick={() => setActiveTab('upload')}
              className={
                'py-4 px-1 border-b-2 font-medium text-sm ' +
                (activeTab === 'upload'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300')
              }
            >
              <Upload className="w-4 h-4 inline mr-2" />
              Upload & Process
            </button>
            <button
              onClick={() => setActiveTab('view')}
              className={
                'py-4 px-1 border-b-2 font-medium text-sm ' +
                (activeTab === 'view'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300')
              }
            >
              <Database className="w-4 h-4 inline mr-2" />
              View Logs
            </button>
            <button
              onClick={() => setActiveTab('charts')}
              className={
                'py-4 px-1 border-b-2 font-medium text-sm ' +
                (activeTab === 'charts'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300')
              }
            >
              <BarChart3 className="w-4 h-4 inline mr-2" />
              Analytics & Charts
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {activeTab === 'upload' && (
          <UploadTab
            onUploadSuccess={(result) => {
              setUploadStatus(result);
              fetchDatabases(true);
            }}
            uploadStatus={uploadStatus}
            customers={customers}
          />
        )}

        {activeTab === 'view' && (
          <ViewLogsTab
            databases={databases}
            selectedDatabase={selectedDatabase}
            onDatabaseSelect={setSelectedDatabase}
            logs={logs}
            loading={loading}
            searchTerm={searchTerm}
            onSearchTermChange={setSearchTerm}
            onSearch={handleSearch}
            currentPage={currentPage}
            totalPages={totalPages}
            totalRecords={totalRecords}
            onPageChange={handlePageChange}
            columns={columns}
          />
        )}

        {activeTab === 'charts' && (
          <ChartsTab
            databases={databases}
            selectedDatabase={selectedDatabase}
            onDatabaseSelect={setSelectedDatabase}
            loading={loading}
            inductionQuality={inductionQuality}
            badHostpics={badHostpics}
            goodHostpics={goodHostpics}
            sortQuality={sortQuality}
          />
        )}
      </main>
    </div>
  );
}

// Componente de Upload con Temporizador
function UploadTab({ onUploadSuccess, uploadStatus, customers }) {
  const [serverType, setServerType] = useState('eDS');
  const [customerId, setCustomerId] = useState('');
  const [logFile, setLogFile] = useState(null);
  const [configFile, setConfigFile] = useState(null);
  const logInputRef = React.useRef();
  const configInputRef = React.useRef();

  // Limpiar archivos al cambiar de tecnología
  useEffect(() => {
    setLogFile(null);
    setConfigFile(null);
    if (logInputRef.current) logInputRef.current.value = '';
    if (configInputRef.current) configInputRef.current.value = '';
  }, [serverType]);
  const [uploading, setUploading] = useState(false);
  const [processingTime, setProcessingTime] = useState(0);
  const [timer, setTimer] = useState(null);
  const [finalProcessingTime, setFinalProcessingTime] = useState(null);

  // Temporizador: actualiza cada 100ms y muestra el tiempo real del backend al finalizar
  useEffect(() => {
    let startTime;
    if (uploading) {
      setFinalProcessingTime(null);
      startTime = performance.now();
      const interval = setInterval(() => {
        setProcessingTime((performance.now() - startTime) / 1000);
      }, 100); // 100ms para décimas de segundo
      setTimer(interval);
      return () => {
        clearInterval(interval);
        setTimer(null);
      };
    } else {
      if (timer) {
        clearInterval(timer);
        setTimer(null);
      }
    }
  }, [uploading]);

  // 🚀 FIX CRÍTICO: handleUpload con timeout para archivos de 500MB+
  const handleUpload = async (e) => {
  e.preventDefault();
  if (!logFile || !configFile) {
    alert('Please select both files');
    return;
  }
  setUploading(true);
  setProcessingTime(0);
  setFinalProcessingTime(null);
  
  const startTime = performance.now(); // ← Capturar tiempo de inicio aquí
  
  try {
    const formData = new FormData();
    formData.append('log_file', logFile);
    formData.append('config_file', configFile);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5 * 60 * 1000);

    const customerParam = customerId ? `&customer_id=${encodeURIComponent(customerId)}` : '';
    const response = await fetch(`${API_BASE}/upload?server_type=${serverType}${customerParam}`, {
      method: 'POST',
      body: formData,
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    const result = await response.json();
    if (response.ok) {
      const totalTime = (performance.now() - startTime) / 1000; // ← Calcular tiempo total
      const resultWithTotalTime = { ...result, total_time: totalTime }; // ← Agregar al resultado
      onUploadSuccess(resultWithTotalTime);
    } else {
      alert('Error: ' + result.detail);
    }
  } catch (error) {
    // ... resto del código de error
  } finally {
    setUploading(false);
  }
};

  return (
  <div className="max-w-lg mx-auto">
      <div className="bg-white shadow rounded-lg p-4">
        <h2 className="text-lg font-medium text-gray-900 mb-6">Upload Log Files</h2>

        <form onSubmit={handleUpload} className="space-y-6">
          {/* Server Type Selection */}
          <div>
            <label className="text-base font-medium text-gray-900">Server Type</label>
            <div className="mt-4 space-x-6">
              <label className="inline-flex items-center">
                <input
                  type="radio"
                  value="eDS"
                  checked={serverType === 'eDS'}
                  onChange={(e) => setServerType(e.target.value)}
                  className="form-radio h-4 w-4 text-blue-600"
                />
                <span className="ml-2 font-semibold">eDS</span>
              </label>
              <label className="inline-flex items-center">
                <input
                  type="radio"
                  value="SCNET"
                  checked={serverType === 'SCNET'}
                  onChange={(e) => setServerType(e.target.value)}
                  className="form-radio h-4 w-4 text-blue-600"
                />
                <span className="ml-2 font-semibold">SCNET</span>
              </label>
            </div>
          </div>

          {/* Customer Selection */}
          {customers.length > 0 && (
            <div>
              <label className="block text-sm font-medium text-gray-700">Customer</label>
              <select
                value={customerId}
                onChange={e => setCustomerId(e.target.value)}
                className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 text-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">— Select customer —</option>
                {customers.map(c => (
                  <option key={c.id} value={c.id}>{c.name}</option>
                ))}
              </select>
            </div>
          )}

          {/* Log File Upload */}
          <div>
            <label className="block text-sm font-medium text-gray-700">
              Log File ({serverType === 'eDS' ? '.log' : '.FSC'})
            </label>
            <input
              type="file"
              accept={serverType === 'eDS' ? '.log' : '.fsc'}
              onChange={(e) => setLogFile(e.target.files[0])}
              ref={logInputRef}
              className="mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
          </div>

          {/* Config File Upload */}
          <div>
            <label className="block text-sm font-medium text-gray-700">
              Config File ({serverType === 'eDS' ? '.json' : '.xml'})
            </label>
            <input
              type="file"
              accept={serverType === 'eDS' ? '.json' : '.xml'}
              onChange={(e) => setConfigFile(e.target.files[0])}
              ref={configInputRef}
              className="mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
          </div>

          {/* Submit Button with blue gradient and timer inside */}
          <div>
            <button
              type="submit"
              disabled={uploading || !logFile || !configFile}
              className={
                'w-full flex justify-center items-center py-2.5 px-8 rounded-lg text-base font-semibold transition-colors duration-200 ' +
                (uploading
                  ? 'bg-gradient-to-r from-blue-400 via-blue-500 to-indigo-500 border-2 border-blue-300 shadow-lg text-white'
                  : (!logFile || !configFile)
                    ? 'bg-gray-400 text-white cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700 text-white border border-transparent shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500')
              }
              style={uploading ? {transition: 'background 0.3s'} : {}}
            >
              {uploading ? (
                <>
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white mr-3"></div>
                  <span className="tracking-wide">Processing... {processingTime.toFixed(1)}s</span>
                </>
              ) : (
                'Upload & Process'
              )}
            </button>
          </div>
        </form>

  {/* Mostrar el tiempo real final del backend tras terminar */}
  {/* Eliminado: mensaje duplicado de Processing time */}

        {/* Upload Status */}
        {uploadStatus && (
          <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-md">
            <div className="flex">
              <div className="ml-3">
                <h3 className="text-sm font-medium text-green-800">
                  Processing Complete! 🎉
                </h3>
                <div className="mt-2 text-sm text-green-700">
                  <p>Records processed: <strong>{uploadStatus.records_processed?.toLocaleString()}</strong></p>
                  <p>Total time: <strong>{uploadStatus.total_time?.toFixed(2)}s</strong></p>
                  <p>Table created: <strong>{uploadStatus.table_name}</strong></p>
                  <p>Database ID: <strong>{uploadStatus.database_id}</strong></p>
                  <p className="text-xs mt-2 text-green-600">
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Componente de Visualización de Logs
function ViewLogsTab({
  databases,
  selectedDatabase,
  onDatabaseSelect,
  logs,
  loading,
  searchTerm,
  onSearchTermChange,
  onSearch,
  currentPage,
  totalPages,
  totalRecords,
  onPageChange,
  columns
}) {
  const [refreshing, setRefreshing] = useState(false);

  // Función para refrescar databases - CORREGIDA para evitar duplicados
  const handleRefreshDatabases = async () => {
    setRefreshing(true);
    try {
      const response = await fetch(`${API_BASE}/databases`);
      const data = await response.json();
      
      // Llamar directamente a fetchDatabases del componente padre para evitar duplicados
      if (window.refreshDatabases) {
        window.refreshDatabases();
      }
    } catch (error) {
      console.error('Error refreshing databases:', error);
    } finally {
      setRefreshing(false);
    }
  };

  // Función para resaltar coincidencias de búsqueda
  const highlightSearchTerm = (text, searchTerm) => {
    if (!searchTerm || !text) return text;
    
    const regex = new RegExp(`(${searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    const parts = text.toString().split(regex);
    
    return parts.map((part, index) => 
      regex.test(part) ? (
        <span key={index} className="bg-yellow-200 px-1 rounded">
          {part}
        </span>
      ) : part
    );
  };

  // Usar columnas del API o extraer dinámicamente de los logs
  const displayColumns = useMemo(() => {
    if (columns && columns.length > 0) {
      return columns;
    }
    
    if (logs.length === 0) return [];

    const allKeys = new Set();
    logs.forEach(log => {
      Object.keys(log).forEach(key => allKeys.add(key));
    });

    const sortedKeys = Array.from(allKeys).sort((a, b) => {
      if (a === 'timestamp') return -1;
      if (b === 'timestamp') return 1;
      if (a === 'message_id') return -1;
      if (b === 'message_id') return 1;
      return a.localeCompare(b);
    });

    return sortedKeys;
  }, [logs, columns]);

  const formatColumnName = useMemo(() => {
    const specialMappings = {
      'timestamp': 'Time',
      'message_id': 'Message ID',
      'Pic': 'PIC',
      'AlibiId': 'Alibi ID',
      'Hostpic': 'Host PIC',
      'ParcelLength': 'Parcel Length',
      'ParcelHeight': 'Parcel Height',
      'ParcelWidth': 'Parcel Width',
      'ParcelWeight': 'Parcel Weight'
    };

    return (columnName) => {
      if (specialMappings[columnName]) {
        return specialMappings[columnName];
      }
      return columnName.replace(/([a-z])([A-Z])/g, '$1 $2');
    };
  }, []);

  return (
    <div className="space-y-6">
      {/* Database & Search Controls - DISEÑO COMPACTO */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 shadow rounded-lg p-4">
        <div className="flex items-center justify-between">
          
          {/* Lado Izquierdo: Database Selection */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Database className="h-5 w-5 text-blue-600" />
              <label className="text-sm font-semibold text-gray-800">
                Active Database:
              </label>
            </div>
            
            <select
              value={selectedDatabase?.id || ''}
              onChange={(e) => {
                const db = databases.find(d => d.id === e.target.value);
                onDatabaseSelect(db);
              }}
              className="block w-72 pl-3 pr-8 py-2 text-sm border border-blue-300 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 rounded-md font-medium bg-white"
            >
              {databases.length === 0 ? (
                <option value="">No databases available</option>
              ) : (
                databases.map(db => (
                  <option key={db.id} value={db.id}>
                    {db.table_name} ({db.record_count?.toLocaleString() || '0'} records) - {new Date(db.created_at).toLocaleDateString()}{db.customer_name ? ` · ${db.customer_name}` : ''}
                  </option>
                ))
              )}
            </select>
            
            <button
              onClick={handleRefreshDatabases}
              disabled={refreshing}
              className="inline-flex items-center px-2 py-2 border border-blue-300 rounded-md text-sm font-medium text-blue-700 bg-white hover:bg-blue-50 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
              title="Refresh database list"
            >
              <svg className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          </div>

          {/* Lado Derecho: Search Controls */}
          <div className="flex items-center space-x-2">
            <label className="text-sm font-semibold text-gray-800">
              Search:
            </label>
            <div className="relative">
              <input
                type="text"
                placeholder="Search records..."
                value={searchTerm}
                onChange={(e) => onSearchTermChange(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && onSearch()}
                className="block w-64 pl-8 pr-3 py-2 border border-gray-300 rounded-md text-sm bg-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
              />
              <div className="absolute inset-y-0 left-0 pl-2 flex items-center pointer-events-none">
                <Search className="h-4 w-4 text-gray-400" />
              </div>
            </div>
            
            <button
              onClick={onSearch}
              className="inline-flex items-center px-3 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              Search
            </button>
            
            {searchTerm && (
              <button
                onClick={() => {
                  onSearchTermChange('');
                  onSearch();
                }}
                className="inline-flex items-center px-2 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>

        {/* Database Info - Posición inferior */}
        {selectedDatabase && (
          <div className="mt-3 flex items-center justify-between">
            <div className="bg-white rounded px-3 py-2 text-xs text-gray-700 border border-blue-100">
              <span className="font-semibold">{selectedDatabase.table_name}</span>
              <span className="mx-2">•</span>
              <span>{selectedDatabase.record_count?.toLocaleString() || '0'} total records</span>
              <span className="mx-2">•</span>
              <span>Created: {new Date(selectedDatabase.created_at).toLocaleDateString()}</span>
              <span className="mx-2">•</span>
              <span>Size: {selectedDatabase.file_size_mb?.toFixed(1) || '0'} MB</span>
            </div>

            {/* Search Results Info */}
            {searchTerm && (
              <div className="bg-yellow-100 border border-yellow-300 rounded px-3 py-1 text-xs text-yellow-800">
                Found <strong>{totalRecords.toLocaleString()}</strong> records matching "{searchTerm}"
              </div>
            )}
          </div>
        )}
      </div>

      {/* Data Table con búsqueda resaltada */}
      <div className="bg-white shadow rounded-lg overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            <span className="ml-4 text-gray-600">Loading logs...</span>
          </div>
        ) : logs.length === 0 ? (
          <div className="text-center py-12">
            <Database className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No logs found</h3>
            <p className="mt-1 text-sm text-gray-500">
              {selectedDatabase ? 'Try adjusting your search criteria.' : 'Select a database to view logs.'}
            </p>
          </div>
        ) : (
          <>
            {/* Table con resaltado de búsqueda */}
            <div className="overflow-x-auto max-h-96">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-blue-50 sticky top-0">
                  <tr>
                    {displayColumns.map(column => (
                      <th
                        key={column}
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b-2 border-blue-200"
                      >
                        {formatColumnName(column)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {logs.map((log, index) => (
                    <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      {displayColumns.map(column => {
                        let cellValue = '';
                        if (column === 'timestamp' && log[column]) {
                          cellValue = new Date(log[column]).toLocaleString();
                        } else {
                          cellValue = log[column] || '-';
                        }

                        return (
                          <td key={column} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {searchTerm && cellValue !== '-' ? 
                              highlightSearchTerm(cellValue, searchTerm) : 
                              cellValue
                            }
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="bg-white px-4 py-3 flex items-center justify-between border-t border-gray-200 sm:px-6">
                <div className="flex-1 flex justify-between sm:hidden">
                  <button
                    onClick={() => onPageChange(currentPage - 1)}
                    disabled={currentPage === 1}
                    className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Previous
                  </button>
                  <button
                    onClick={() => onPageChange(currentPage + 1)}
                    disabled={currentPage === totalPages}
                    className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Next
                  </button>
                </div>
                <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
                  <div>
                    <p className="text-sm text-gray-700">
                      Showing page <span className="font-medium">{currentPage}</span> of{' '}
                      <span className="font-medium">{totalPages}</span> ({totalRecords.toLocaleString()} total records)
                    </p>
                  </div>
                  <div>
                    <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
                      <button
                        onClick={() => onPageChange(currentPage - 1)}
                        disabled={currentPage === 1}
                        className="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <ChevronLeft className="h-5 w-5" />
                      </button>
                      {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                        let pageNum;
                        if (totalPages <= 5) {
                          pageNum = i + 1;
                        } else if (currentPage <= 3) {
                          pageNum = i + 1;
                        } else if (currentPage >= totalPages - 2) {
                          pageNum = totalPages - 4 + i;
                        } else {
                          pageNum = currentPage - 2 + i;
                        }
                        return (
                          <button
                            key={pageNum}
                            onClick={() => onPageChange(pageNum)}
                            className={
                              'relative inline-flex items-center px-4 py-2 border text-sm font-medium ' +
                              (currentPage === pageNum
                                ? 'z-10 bg-blue-50 border-blue-500 text-blue-600'
                                : 'bg-white border-gray-300 text-gray-500 hover:bg-gray-50')
                            }
                          >
                            {pageNum}
                          </button>
                        );
                      })}
                      <button
                        onClick={() => onPageChange(currentPage + 1)}
                        disabled={currentPage === totalPages}
                        className="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <ChevronRight className="h-5 w-5" />
                      </button>
                    </nav>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function aggregateByInfeed(rows) {
  const map = {};
  for (const r of rows) {
    if (!map[r.infeed]) {
      map[r.infeed] = { infeed: r.infeed, infeed_num: r.infeed_num, good: 0, bad: 0, total: 0 };
    }
    map[r.infeed].good += r.good;
    map[r.infeed].bad  += r.bad;
    map[r.infeed].total += r.total;
  }
  return Object.values(map)
    .map(d => ({
      ...d,
      good_pct: d.total > 0 ? Math.round((d.good / d.total) * 1000) / 10 : 0,
      bad_pct:  d.total > 0 ? Math.round((d.bad  / d.total) * 1000) / 10 : 0,
    }))
    .sort((a, b) => a.infeed_num - b.infeed_num);
}

// Componente de Gráficos y Análisis
const ODS_LABELS = {
  '1': 'Diverted',           '2': 'Dest. not reached',    '3': 'Dest. not available',
  '4': 'Failed to divert',   '5': 'Invalid destination',  '6': 'Unreachable dest.',
  '7': 'Dest. not received', '8': 'Condition invalid',    '9': 'Dest. not requested',
  'A': 'Flow restricted',    'B': 'Max. recirculation',   'C': 'Dest. override',
  '0': 'Unknown state',
};
const ODS_COLORS = {
  '1': '#10B981', '2': '#EF4444', '3': '#F97316', '4': '#F59E0B',
  '5': '#6366F1', '6': '#EC4899', '7': '#8B5CF6', '8': '#0EA5E9',
  '9': '#14B8A6', 'A': '#84CC16', 'B': '#991B1B', 'C': '#64748B', '0': '#9CA3AF',
};

function ChartsTab({ databases, selectedDatabase, onDatabaseSelect, loading, inductionQuality, badHostpics, goodHostpics, sortQuality }) {
  const [filterChartHour, setFilterChartHour] = useState('');

  const hours = useMemo(() => {
    if (!inductionQuality?.data?.length) return [];
    return [...new Set(inductionQuality.data.map(r => r.hour))].sort((a, b) => a - b);
  }, [inductionQuality]);

  const chartData = useMemo(() => {
    if (!inductionQuality?.data?.length) return [];
    const rows = filterChartHour === ''
      ? inductionQuality.data
      : inductionQuality.data.filter(r => r.hour === filterChartHour);
    return aggregateByInfeed(rows);
  }, [inductionQuality, filterChartHour]);

  const [sortMinimized, setSortMinimized] = useState(false);
  const [inductionMinimized, setInductionMinimized] = useState(false);

  const sortChartData = useMemo(() => {
    if (!sortQuality?.data?.length) return { rows: [], infeeds: [] };
    const data = sortQuality.data;

    // Infeeds ordenados por zone_id
    const infeedMap = {};
    data.forEach(r => { infeedMap[r.zone_id] = { zone_id: r.zone_id, zone_name: r.zone_name }; });
    const infeeds = Object.values(infeedMap).sort((a, b) => a.zone_id - b.zone_id).map(z => z.zone_name);

    // Estados ordenados: ODS=1 primero, resto alfanumérico
    const states = [...new Set(data.map(r => r.state))].sort((a, b) => {
      if (a === '1') return -1;
      if (b === '1') return 1;
      return a.localeCompare(b);
    });

    // Una fila por ODS, columnas = nombre de INFEED
    const odsMap = {};
    data.forEach(r => {
      if (!odsMap[r.state]) odsMap[r.state] = { state: r.state };
      odsMap[r.state][r.zone_name] = (odsMap[r.state][r.zone_name] || 0) + r.count;
    });
    const rows = states.map(s => odsMap[s]).filter(Boolean);
    return { rows, infeeds };
  }, [sortQuality]);

  useEffect(() => { setFilterChartHour(''); }, [selectedDatabase?.id]);

  if (!selectedDatabase) {
    return (
      <div className="text-center py-12">
        <BarChart3 className="mx-auto h-12 w-12 text-gray-400" />
        <h3 className="mt-2 text-sm font-medium text-gray-900">No database selected</h3>
        <p className="mt-1 text-sm text-gray-500">Select a database to view analytics</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        <span className="ml-4 text-gray-600">Loading statistics...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Selector de base de datos */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 shadow rounded-lg p-4">
        <div className="flex items-center space-x-4">
          <Database className="h-5 w-5 text-blue-600" />
          <label className="text-sm font-semibold text-gray-800">Active Database:</label>
          <select
            value={selectedDatabase?.id || ''}
            onChange={(e) => {
              const db = databases.find(d => d.id === e.target.value);
              onDatabaseSelect(db);
            }}
            className="block w-72 pl-3 pr-8 py-2 text-sm border border-blue-300 focus:outline-none focus:ring-1 focus:ring-blue-500 rounded-md font-medium bg-white"
          >
            {databases.map(db => (
              <option key={db.id} value={db.id}>
                {db.table_name} ({db.record_count?.toLocaleString() || '0'} records) - {new Date(db.created_at).toLocaleDateString()}{db.customer_name ? ` · ${db.customer_name}` : ''}
              </option>
            ))}
          </select>
          <div className="text-sm text-gray-600 bg-white rounded px-3 py-1 border border-blue-100">
            <span className="font-semibold">{selectedDatabase.table_name}</span>
            <span className="mx-2">•</span>
            <span>{selectedDatabase.record_count?.toLocaleString()} records</span>
            <span className="mx-2">•</span>
            <span>{selectedDatabase.file_size_mb?.toFixed(1)} MB</span>
          </div>
        </div>
      </div>

      {/* Calidad de Clasificación por ODS */}
      {sortQuality !== null && !sortQuality.error && (
        <div className="bg-white rounded-lg shadow overflow-hidden">
          {/* Cabecera siempre visible con botón minimizar */}
          <div
            className="flex items-center justify-between px-6 py-4 cursor-pointer select-none hover:bg-gray-50"
            onClick={() => setSortMinimized(v => !v)}
          >
            <div className="flex items-center gap-3">
              <h3 className="text-lg font-medium text-gray-900">
                Calidad de Clasificación por ODS
              </h3>
              {sortQuality?.customer && (
                <span className="text-sm font-normal text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
                  {sortQuality.customer}
                </span>
              )}
              {sortQuality.data?.length > 0 && (() => {
                const totalOk  = sortQuality.data.filter(r => r.state === '1').reduce((s, r) => s + r.count, 0);
                const totalAll = sortQuality.data.reduce((s, r) => s + r.count, 0);
                const pct      = totalAll > 0 ? ((totalOk / totalAll) * 100).toFixed(1) : null;
                return pct !== null && (
                  <span className={`text-sm font-semibold px-2 py-0.5 rounded ${parseFloat(pct) >= 90 ? 'bg-green-100 text-green-700' : parseFloat(pct) >= 70 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}`}>
                    {pct}% clasificados
                  </span>
                );
              })()}
            </div>
            <span className="text-gray-400 text-lg font-bold">{sortMinimized ? '▸' : '▾'}</span>
          </div>

          {/* Cuerpo colapsable */}
          {!sortMinimized && (
            <div className="px-6 pb-6">
              <p className="text-sm text-gray-500 mb-4">
                Paquetes únicos (HOSTPIC × ODS) agrupados por estado · cada color = punto de entrada
              </p>
              {sortQuality.data?.length > 0 ? (
                <>
                  {(() => {
                    const totalOk  = sortQuality.data.filter(r => r.state === '1').reduce((s, r) => s + r.count, 0);
                    const totalAll = sortQuality.data.reduce((s, r) => s + r.count, 0);
                    const pct      = totalAll > 0 ? ((totalOk / totalAll) * 100).toFixed(1) : null;
                    return (
                      <div className="grid grid-cols-3 gap-4 mb-6">
                        <div className="bg-green-50 border border-green-200 p-4 rounded-lg">
                          <p className="text-xs font-medium text-gray-500 uppercase">Clasificados (ODS=1)</p>
                          <p className="text-2xl font-bold text-green-700 mt-1">{totalOk.toLocaleString()}</p>
                        </div>
                        <div className="bg-red-50 border border-red-200 p-4 rounded-lg">
                          <p className="text-xs font-medium text-gray-500 uppercase">Otros estados</p>
                          <p className="text-2xl font-bold text-red-700 mt-1">{(totalAll - totalOk).toLocaleString()}</p>
                        </div>
                        <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg">
                          <p className="text-xs font-medium text-gray-500 uppercase">% Clasificados</p>
                          <p className={`text-2xl font-bold mt-1 ${parseFloat(pct) >= 90 ? 'text-green-700' : parseFloat(pct) >= 70 ? 'text-yellow-600' : 'text-red-700'}`}>
                            {pct !== null ? `${pct}%` : 'N/A'}
                          </p>
                        </div>
                      </div>
                    );
                  })()}

                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart data={sortChartData.rows} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="state"
                        tick={{ fontSize: 11 }}
                        tickFormatter={v => `${v} · ${ODS_LABELS[v] || v}`}
                        interval={0}
                        angle={-20}
                        textAnchor="end"
                        height={50}
                      />
                      <YAxis tick={{ fontSize: 12 }} />
                      <Tooltip
                        content={({ active, payload, label }) => {
                          if (!active || !payload?.length) return null;
                          const total = payload.reduce((s, p) => s + (p.value || 0), 0);
                          return (
                            <div className="bg-white border border-gray-200 rounded-lg p-3 shadow text-sm max-w-xs">
                              <p className="font-semibold text-gray-800 mb-1">ODS {label} — {ODS_LABELS[label] || label}</p>
                              <p className="text-xs text-gray-400 mb-2">Total: {total.toLocaleString()}</p>
                              {payload.filter(p => p.value > 0).map(p => (
                                <p key={p.dataKey} style={{ color: p.fill }}>
                                  {p.dataKey}: {p.value.toLocaleString()}
                                </p>
                              ))}
                            </div>
                          );
                        }}
                      />
                      <Legend />
                      {sortChartData.infeeds.map((infeed, i) => (
                        <Bar key={infeed} dataKey={infeed} stackId="s"
                          fill={['#3B82F6','#8B5CF6','#F97316','#10B981','#EC4899','#0EA5E9','#9CA3AF'][i % 7]} />
                      ))}
                    </BarChart>
                  </ResponsiveContainer>
                </>
              ) : (
                <div className="text-center py-10 text-gray-400">
                  <Activity className="mx-auto h-10 w-10 mb-2 text-gray-300" />
                  <p className="text-sm">No se encontraron datos de clasificación en esta base de datos.</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Calidad de Inducción por Infeed */}
      {inductionQuality !== null && !inductionQuality.error && (
        <div className="bg-white rounded-lg shadow overflow-hidden">
          {/* Cabecera siempre visible con botón minimizar */}
          <div
            className="flex items-center justify-between px-6 py-4 cursor-pointer select-none hover:bg-gray-50"
            onClick={() => setInductionMinimized(v => !v)}
          >
            <div className="flex items-center gap-3">
              <h3 className="text-lg font-medium text-gray-900">
                Calidad de Inducción por Infeed
              </h3>
              {inductionQuality?.customer && (
                <span className="text-sm font-normal text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
                  {inductionQuality.customer}
                </span>
              )}
              {inductionQuality.data?.length > 0 && (() => {
                const totalGood = inductionQuality.data.reduce((s, d) => s + d.good, 0);
                const totalBad  = inductionQuality.data.reduce((s, d) => s + d.bad,  0);
                const totalAll  = totalGood + totalBad;
                const pct       = totalAll > 0 ? ((totalGood / totalAll) * 100).toFixed(1) : null;
                return pct !== null && (
                  <span className={`text-sm font-semibold px-2 py-0.5 rounded ${parseFloat(pct) >= 90 ? 'bg-green-100 text-green-700' : parseFloat(pct) >= 70 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}`}>
                    {pct}% correctos
                  </span>
                );
              })()}
            </div>
            <span className="text-gray-400 text-lg font-bold">{inductionMinimized ? '▸' : '▾'}</span>
          </div>

          {/* Cuerpo colapsable */}
          {!inductionMinimized && (
            <div className="px-6 pb-6">
              {hours.length > 0 && (
                <div className="flex justify-end mb-3">
                  <select
                    value={filterChartHour}
                    onChange={e => setFilterChartHour(e.target.value === '' ? '' : parseInt(e.target.value))}
                    className="border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400 bg-white"
                  >
                    <option value="">Todas las horas</option>
                    {hours.map(h => (
                      <option key={h} value={h}>{String(h).padStart(2,'0')}:00 – {String(h).padStart(2,'0')}:59</option>
                    ))}
                  </select>
                </div>
              )}
              <p className="text-sm text-gray-500 mb-4">
                Correctos: msg20 con lastDest≠998 &nbsp;|&nbsp; Incorrectos: msg21 con lastDest=998
              </p>

              {inductionQuality.data && inductionQuality.data.length > 0 ? (
                <>
                  {(() => {
                    const totalGood = chartData.reduce((s, d) => s + d.good, 0);
                    const totalBad  = chartData.reduce((s, d) => s + d.bad,  0);
                    const totalAll  = totalGood + totalBad;
                    const globalPct = totalAll > 0 ? ((totalGood / totalAll) * 100).toFixed(1) : null;
                    return (
                      <div className="grid grid-cols-3 gap-4 mb-6">
                        <div className="bg-green-50 border border-green-200 p-4 rounded-lg">
                          <p className="text-xs font-medium text-gray-500 uppercase">Total Correctos</p>
                          <p className="text-2xl font-bold text-green-700 mt-1">{totalGood.toLocaleString()}</p>
                        </div>
                        <div className="bg-red-50 border border-red-200 p-4 rounded-lg">
                          <p className="text-xs font-medium text-gray-500 uppercase">Total Incorrectos</p>
                          <p className="text-2xl font-bold text-red-700 mt-1">{totalBad.toLocaleString()}</p>
                        </div>
                        <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg">
                          <p className="text-xs font-medium text-gray-500 uppercase">Calidad Global</p>
                          <p className={`text-2xl font-bold mt-1 ${parseFloat(globalPct) >= 90 ? 'text-green-700' : parseFloat(globalPct) >= 70 ? 'text-yellow-600' : 'text-red-700'}`}>
                            {globalPct !== null ? `${globalPct}%` : 'N/A'}
                          </p>
                        </div>
                      </div>
                    );
                  })()}

                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="infeed" tick={{ fontSize: 11 }} tickFormatter={(v) => v === 'Loop del Crossorter' ? 'Loop' : v} />
                      <YAxis tick={{ fontSize: 12 }} />
                      <Tooltip
                        content={({ active, payload, label }) => {
                          if (!active || !payload?.length) return null;
                          const good  = payload.find(p => p.dataKey === 'good')?.value || 0;
                          const bad   = payload.find(p => p.dataKey === 'bad')?.value  || 0;
                          const total = good + bad;
                          return (
                            <div className="bg-white border border-gray-200 rounded-lg p-3 shadow text-sm">
                              <p className="font-semibold text-gray-800 mb-2">{label}</p>
                              <p className="text-green-600">✓ Correctos: {good.toLocaleString()} ({total ? ((good / total) * 100).toFixed(1) : 0}%)</p>
                              <p className="text-red-600">✗ Incorrectos: {bad.toLocaleString()} ({total ? ((bad / total) * 100).toFixed(1) : 0}%)</p>
                              <p className="text-gray-500 mt-2 border-t pt-1">Total: {total.toLocaleString()}</p>
                            </div>
                          );
                        }}
                      />
                      <Legend formatter={(v) => v === 'good' ? 'Correctos' : 'Incorrectos'} />
                      <Bar dataKey="good" name="good" stackId="s" fill="#10B981" />
                      <Bar dataKey="bad"  name="bad"  stackId="s" fill="#EF4444" />
                    </BarChart>
                  </ResponsiveContainer>

                  <div className="mt-6 overflow-x-auto">
                    <table className="min-w-full text-sm divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Infeed</th>
                          <th className="px-4 py-2 text-right text-xs font-medium text-green-600 uppercase">Correctos</th>
                          <th className="px-4 py-2 text-right text-xs font-medium text-red-600 uppercase">Incorrectos</th>
                          <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">Total</th>
                          <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">% Correctos</th>
                          <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">% Incorrectos</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-100">
                        {chartData.map((row, idx) => (
                          <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                            <td className="px-4 py-2 font-medium text-gray-900">{row.infeed}</td>
                            <td className="px-4 py-2 text-right text-green-600 font-medium">{row.good.toLocaleString()}</td>
                            <td className="px-4 py-2 text-right text-red-600 font-medium">{row.bad.toLocaleString()}</td>
                            <td className="px-4 py-2 text-right font-semibold">{row.total.toLocaleString()}</td>
                            <td className="px-4 py-2 text-right">
                              <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                                row.good_pct >= 90 ? 'bg-green-100 text-green-800' :
                                row.good_pct >= 70 ? 'bg-yellow-100 text-yellow-800' :
                                                      'bg-red-100 text-red-800'
                              }`}>
                                {row.good_pct}%
                              </span>
                            </td>
                            <td className="px-4 py-2 text-right text-gray-500">{row.bad_pct}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <div className="text-center py-10 text-gray-400">
                  <Activity className="mx-auto h-10 w-10 mb-2 text-gray-300" />
                  {inductionQuality.error
                    ? <p className="text-sm text-orange-500">{inductionQuality.error}</p>
                    : <p className="text-sm">No se encontraron datos de inducción en esta base de datos.</p>
                  }
                  {inductionQuality.columns_found && (
                    <p className="text-xs mt-2">
                      hostpic: {inductionQuality.columns_found.hostpic ? '✓' : '✗'}&nbsp;
                      lastdestination: {inductionQuality.columns_found.last_destination ? '✓' : '✗'}&nbsp;
                      parcelentrypoint: {inductionQuality.columns_found.parcel_entry_point ? '✓' : '✗'}
                    </p>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* HOSTPICs incorrectos */}
      {badHostpics === null ? (
        <div className="bg-white p-6 rounded-lg shadow flex items-center justify-center h-24">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-red-500"></div>
          <span className="ml-3 text-gray-500 text-sm">Cargando HOSTPICs incorrectos...</span>
        </div>
      ) : badHostpics.data && badHostpics.data.length > 0 ? (
        <HostpicsTable data={badHostpics.data} total={badHostpics.total} variant="bad" />
      ) : null}

      {/* HOSTPICs correctos */}
      {goodHostpics === null ? (
        <div className="bg-white p-6 rounded-lg shadow flex items-center justify-center h-24">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-green-500"></div>
          <span className="ml-3 text-gray-500 text-sm">Cargando HOSTPICs correctos...</span>
        </div>
      ) : goodHostpics.data && goodHostpics.data.length > 0 ? (
        <HostpicsTable data={goodHostpics.data} total={goodHostpics.total} variant="good" />
      ) : null}

    </div>
  );
}

function HostpicsTable({ data, total, variant }) {
  const isBad = variant === 'bad';
  const [collapsed, setCollapsed] = useState(false);
  const [search, setSearch] = useState('');
  const [filterInfeed, setFilterInfeed] = useState('');
  const [filterEntryPoint, setFilterEntryPoint] = useState('');
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 50;

  const accentRing  = isBad ? 'focus:ring-red-400'   : 'focus:ring-green-400';
  const copyBtnCls  = isBad
    ? 'bg-red-50 border-red-200 text-red-700 hover:bg-red-100'
    : 'bg-green-50 border-green-200 text-green-700 hover:bg-green-100';
  const hostpicColor = isBad ? 'text-red-700' : 'text-green-700';
  const badgeCls    = isBad ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800';
  const title       = isBad ? 'HOSTPICs inducidos incorrectamente' : 'HOSTPICs inducidos correctamente';
  const subtitle    = isBad ? 'msg21 + lastDest=998' : 'msg20 + lastDest≠998';

  const infeeds = useMemo(() =>
    [...new Set(data.map(r => r.infeed))].sort((a, b) => {
      const numA = parseInt(a.replace('INFEED ', '')) || 999;
      const numB = parseInt(b.replace('INFEED ', '')) || 999;
      return numA - numB;
    }), [data]);

  const entryPoints = useMemo(() =>
    [...new Set(
      data.filter(r => !filterInfeed || r.infeed === filterInfeed).map(r => r.entry_point)
    )].sort(), [data, filterInfeed]);

  const filtered = useMemo(() =>
    data.filter(r => {
      const matchSearch = !search || r.hostpic.toLowerCase().includes(search.toLowerCase());
      const matchInfeed = !filterInfeed || r.infeed === filterInfeed;
      const matchEP = !filterEntryPoint || r.entry_point === filterEntryPoint;
      return matchSearch && matchInfeed && matchEP;
    }), [data, search, filterInfeed, filterEntryPoint]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const paged = useMemo(() =>
    filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE),
    [filtered, page]);

  const handleSearch = (val) => { setSearch(val); setPage(1); };
  const handleInfeed = (val) => { setFilterInfeed(val); setFilterEntryPoint(''); setPage(1); };
  const handleEntryPoint = (val) => { setFilterEntryPoint(val); setPage(1); };

  const copyAll = () => {
    navigator.clipboard.writeText(filtered.map(r => r.hostpic).join('\n'));
  };

  return (
    <div className="bg-white rounded-lg shadow">
      {/* Header */}
      <div className="px-6 py-4 border-b flex items-center justify-between flex-wrap gap-3">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
          <p className="text-xs text-gray-500 mt-0.5">
            {subtitle} — {total.toLocaleString()} únicos en total
            {filtered.length !== total && ` (${filtered.length.toLocaleString()} filtrados)`}
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {!collapsed && (
            <>
              <input
                type="text"
                placeholder="Buscar HOSTPIC..."
                value={search}
                onChange={e => handleSearch(e.target.value)}
                className={`border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-1 ${accentRing} w-48`}
              />
              <select
                value={filterInfeed}
                onChange={e => handleInfeed(e.target.value)}
                className={`border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-1 ${accentRing}`}
              >
                <option value="">Todos los infeeds</option>
                {infeeds.map(inf => <option key={inf} value={inf}>{inf}</option>)}
              </select>
              <select
                value={filterEntryPoint}
                onChange={e => handleEntryPoint(e.target.value)}
                className={`border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-1 ${accentRing} font-mono`}
              >
                <option value="">Todos los entry points</option>
                {entryPoints.map(ep => <option key={ep} value={ep}>{ep}</option>)}
              </select>
              <button
                onClick={copyAll}
                className={`px-3 py-1.5 text-sm border rounded font-medium ${copyBtnCls}`}
                title="Copiar todos los HOSTPICs filtrados al portapapeles"
              >
                Copiar lista
              </button>
            </>
          )}
          {/* Botón minimizar / expandir */}
          <button
            onClick={() => setCollapsed(c => !c)}
            className="ml-1 px-2 py-1.5 text-gray-400 hover:text-gray-700 hover:bg-gray-100 rounded border border-gray-200 text-xs font-mono"
            title={collapsed ? 'Expandir tabla' : 'Minimizar tabla'}
          >
            {collapsed ? '▼ Expandir' : '▲ Minimizar'}
          </button>
        </div>
      </div>

      {!collapsed && (
        <>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">#</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">HOSTPIC</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">INFEED</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Entry Point</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-100">
                {paged.map((row, idx) => (
                  <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-4 py-2 text-gray-400 text-xs">{(page - 1) * PAGE_SIZE + idx + 1}</td>
                    <td className={`px-4 py-2 font-mono font-semibold ${hostpicColor}`}>{row.hostpic}</td>
                    <td className="px-4 py-2">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${badgeCls}`}>{row.infeed}</span>
                    </td>
                    <td className="px-4 py-2 font-mono text-gray-500 text-xs">{row.entry_point}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {totalPages > 1 && (
            <div className="px-6 py-3 border-t flex items-center justify-between text-sm text-gray-600">
              <span>Página {page} de {totalPages}</span>
              <div className="flex gap-2">
                <button disabled={page === 1} onClick={() => setPage(p => p - 1)}
                  className="px-3 py-1 border rounded disabled:opacity-40 hover:bg-gray-50">Anterior</button>
                <button disabled={page === totalPages} onClick={() => setPage(p => p + 1)}
                  className="px-3 py-1 border rounded disabled:opacity-40 hover:bg-gray-50">Siguiente</button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default LogAnalyzerApp;