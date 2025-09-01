import React, { useState, useEffect, useMemo } from 'react';
import { Upload, Search, Database, FileText, X, BarChart3, Clock, Activity, ChevronLeft, ChevronRight } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const API_BASE = 'http://localhost:8000';

// Colores para gráficos
const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6', '#F97316'];

function LogAnalyzerApp() {
  const [activeTab, setActiveTab] = useState('upload');
  const [databases, setDatabases] = useState([]);
  const [selectedDatabase, setSelectedDatabase] = useState(null);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [tableStats, setTableStats] = useState(null);

  // Estados para filtros
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalRecords, setTotalRecords] = useState(0);
  const [columns, setColumns] = useState([]);

  // Cargar bases de datos al iniciar
  useEffect(() => {
    fetchDatabases();
  }, []);

  const fetchDatabases = async () => {
    try {
      const response = await fetch(`${API_BASE}/databases`);
      const data = await response.json();
      setDatabases(data);
      if (data.length > 0 && !selectedDatabase) {
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

  const fetchTableStats = async (tableName) => {
    if (!tableName) return;
    
    try {
      const response = await fetch(`${API_BASE}/table/${tableName}/stats`);
      const data = await response.json();
      setTableStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
      setTableStats(null);
    }
  };

  // Efecto para cargar logs y estadísticas cuando cambia la base de datos
  useEffect(() => {
    if (selectedDatabase) {
      if (activeTab === 'view') {
        fetchLogs(1, searchTerm);
      } else if (activeTab === 'charts') {
        fetchTableStats(selectedDatabase.table_name);
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
              fetchDatabases();
            }}
            uploadStatus={uploadStatus}
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
            tableStats={tableStats}
            loading={loading}
          />
        )}
      </main>
    </div>
  );
}

// Componente de Upload con Temporizador
function UploadTab({ onUploadSuccess, uploadStatus }) {
  const [serverType, setServerType] = useState('eDS');
  const [logFile, setLogFile] = useState(null);
  const [configFile, setConfigFile] = useState(null);
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

    const response = await fetch(`${API_BASE}/upload?server_type=${serverType}`, {
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

          {/* Log File Upload */}
          <div>
            <label className="block text-sm font-medium text-gray-700">
              Log File ({serverType === 'eDS' ? '.log' : '.FSC'})
            </label>
            <input
              type="file"
              accept={serverType === 'eDS' ? '.log' : '.fsc'}
              onChange={(e) => setLogFile(e.target.files[0])}
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
              className="mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
          </div>

          {/* Submit Button with blue gradient and timer inside */}
          <div>
            <button
              type="submit"
              disabled={uploading || !logFile || !configFile}
              className={
                'w-full flex justify-center items-center py-1.5 px-4 rounded-md text-sm font-semibold transition-colors duration-200 ' +
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
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-3"></div>
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
      {/* Database Selection & Search */}
  <div className="bg-white shadow p-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0">
          {/* Database Selector */}
          <div className="flex items-center space-x-4">
            <label className="text-sm font-medium text-gray-700">Database:</label>
            <select
              value={selectedDatabase?.id || ''}
              onChange={(e) => {
                const db = databases.find(d => d.id === e.target.value);
                onDatabaseSelect(db);
              }}
              className="block w-64 pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
            >
              {databases.map(db => (
                <option key={db.id} value={db.id}>
                  {db.table_name} ({db.record_count?.toLocaleString()} records)
                </option>
              ))}
            </select>
          </div>

          {/* Search */}
          <div className="flex items-center space-x-2">
            <div className="relative">
              <input
                type="text"
                placeholder="Search in all fields..."
                value={searchTerm}
                onChange={(e) => onSearchTermChange(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && onSearch()}
                className="block w-80 pl-10 pr-4 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
              />
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Search className="h-5 w-5 text-gray-400" />
              </div>
            </div>
            <button
              onClick={onSearch}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              Search
            </button>
            {searchTerm && (
              <button
                onClick={() => {
                  onSearchTermChange('');
                  onSearch();
                }}
                className="inline-flex items-center px-3 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>

        {/* Search Results Info */}
        {searchTerm && (
          <div className="mt-4 text-sm text-gray-600">
            Found <strong>{totalRecords.toLocaleString()}</strong> records for "{searchTerm}"
          </div>
        )}
      </div>

      {/* Data Table */}
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
            {/* Table */}
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
                      {displayColumns.map(column => (
                        <td key={column} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {column === 'timestamp' && log[column]
                            ? new Date(log[column]).toLocaleString()
                            : log[column] || '-'
                          }
                        </td>
                      ))}
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

// Componente de Gráficos y Análisis
function ChartsTab({ databases, selectedDatabase, onDatabaseSelect, tableStats, loading }) {
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

  if (!tableStats) {
    return (
      <div className="text-center py-12">
        <Activity className="mx-auto h-12 w-12 text-gray-400" />
        <h3 className="mt-2 text-sm font-medium text-gray-900">No statistics available</h3>
        <p className="mt-1 text-sm text-gray-500">Statistics will appear after processing data</p>
      </div>
    );
  }

  // Preparar datos para gráficos
  const messageDistData = tableStats.message_distribution || [];
  const hourlyData = tableStats.hourly_distribution?.map(item => ({
    ...item,
    hour: item.hour ? new Date(item.hour).getHours() + ':00' : 'Unknown'
  })) || [];

  // Datos para el gráfico de pastel - Top 10 message IDs
  const pieData = messageDistData
    .sort((a, b) => b.count - a.count)
    .slice(0, 10)
    .map(item => ({
      name: `Message ${item.message_id}`,
      value: item.count
    }));

  return (
    <div className="space-y-6">
      {/* Database Selector */}
      <div className="bg-white shadow rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-4">
            <label className="text-sm font-medium text-gray-700">Database:</label>
            <select
              value={selectedDatabase?.id || ''}
              onChange={(e)=> {
                const db = databases.find(d => d.id === e.target.value);
                onDatabaseSelect(db);
              }}
              className="block w-64 pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
            >
              {databases.map(db => (
                <option key={db.id} value={db.id}>
                  {db.table_name} ({db.record_count?.toLocaleString()} records)
                </option>
              ))}
            </select>
          </div>
          <div className="text-sm text-gray-600">
            Total Records: <span className="font-semibold text-lg">{tableStats.total_records?.toLocaleString()}</span>
          </div>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <div className="bg-blue-50 p-4 rounded-lg">
            <div className="flex items-center">
              <Database className="h-8 w-8 text-blue-600" />
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-600">Total Records</p>
                <p className="text-2xl font-bold text-gray-900">
                  {tableStats.total_records?.toLocaleString()}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-green-50 p-4 rounded-lg">
            <div className="flex items-center">
              <Activity className="h-8 w-8 text-green-600" />
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-600">Unique Messages</p>
                <p className="text-2xl font-bold text-gray-900">
                  {messageDistData.length}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-purple-50 p-4 rounded-lg">
            <div className="flex items-center">
              <Clock className="h-8 w-8 text-purple-600" />
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-600">Time Range</p>
                <p className="text-sm font-bold text-gray-900">
                  {tableStats.time_range ? 
                    `${new Date(tableStats.time_range.start).toLocaleDateString()}` :
                    'N/A'
                  }
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Message Distribution Bar Chart */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Message Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={messageDistData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="message_id" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" fill="#3B82F6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Top 10 Messages Pie Chart */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Top 10 Message Types</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={(entry) => `${entry.name}: ${entry.value}`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Hourly Distribution Line Chart */}
        <div className="bg-white p-6 rounded-lg shadow lg:col-span-2">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Hourly Activity</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={hourlyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="count" stroke="#10B981" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Statistics Table */}
      <div className="bg-white shadow rounded-lg overflow-hidden">
        <div className="px-6 py-4 bg-gray-50 border-b">
          <h3 className="text-lg font-medium text-gray-900">Message Type Details</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Message ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Count
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Percentage
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Visual
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {messageDistData.map((item, index) => {
                const percentage = ((item.count / tableStats.total_records) * 100).toFixed(2);
                return (
                  <tr key={index}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      Message {item.message_id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {item.count.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {percentage}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div 
                            className="bg-blue-600 h-2.5 rounded-full" 
                            style={{width: `${Math.min(parseFloat(percentage) * 2, 100)}%`}}
                          ></div>
                        </div>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default LogAnalyzerApp;