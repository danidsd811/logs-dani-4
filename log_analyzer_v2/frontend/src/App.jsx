import React, { useState, useEffect, useRef } from 'react';
import { Upload, Database, FileText, BarChart3 } from 'lucide-react';
import { API_BASE } from './constants';
import UploadTab from './components/UploadTab';
import ViewLogsTab from './components/ViewLogsTab';
import ChartsTab from './components/ChartsTab';

function LogAnalyzerApp() {
  const [activeTab, setActiveTab] = useState('upload');
  const [databases, setDatabases] = useState([]);
  const [selectedDatabase, setSelectedDatabase] = useState(null);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [inductionQuality, setInductionQuality] = useState(null);
  const [sortQuality, setSortQuality] = useState(null);
  const [scaleQuality, setScaleQuality] = useState(null);
  const [blockedStatus, setBlockedStatus] = useState(null);
  const [trackingLosses, setTrackingLosses] = useState(null);
  const [customers, setCustomers] = useState([]);
  const [analyticsLoading, setAnalyticsLoading] = useState(false);
  const analyticsAbortRef = useRef(null);
  const [isUploading, setIsUploading] = useState(false);

  const [searchTerm, setSearchTerm] = useState('');
  const [appliedSearchTerm, setAppliedSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalRecords, setTotalRecords] = useState(0);
  const [columns, setColumns] = useState([]);

  useEffect(() => {
    fetchDatabases();
    fetch(`${API_BASE}/customers`)
      .then(r => r.json())
      .then(data => setCustomers([...data].sort((a, b) => a.name.localeCompare(b.name))))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (activeTab === 'upload') {
      fetch(`${API_BASE}/customers`)
        .then(r => r.json())
        .then(data => setCustomers([...data].sort((a, b) => a.name.localeCompare(b.name))))
        .catch(() => {});
    }
  }, [activeTab]);

  useEffect(() => {
    window.refreshDatabases = fetchDatabases;
    return () => { window.refreshDatabases = null; };
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
        body: JSON.stringify({ page, limit: 100, search: search || undefined })
      });
      const data = await response.json();
      setLogs(data.data);
      setTotalPages(data.total_pages);
      setTotalRecords(data.total_records);
      setCurrentPage(page);
      setColumns(data.columns || []);
      setAppliedSearchTerm(search || '');
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
    setSortQuality(null);
    setScaleQuality(null);
    setBlockedStatus(null);
    setTrackingLosses(null);
    setAnalyticsLoading(true);
    try {
      const [iq, sq, scale, bs, tl] = await Promise.allSettled([
        fetch(`${API_BASE}/databases/${databaseId}/induction_quality`, { signal }).then(r => r.json()),
        fetch(`${API_BASE}/databases/${databaseId}/sort_quality`, { signal }).then(r => r.json()),
        fetch(`${API_BASE}/databases/${databaseId}/scale_quality`, { signal }).then(r => r.json()),
        fetch(`${API_BASE}/databases/${databaseId}/blocked_status`, { signal }).then(r => r.json()),
        fetch(`${API_BASE}/databases/${databaseId}/tracking_losses`, { signal }).then(r => r.json()),
      ]);
      if (signal.aborted) return;
      setInductionQuality(iq.status === 'fulfilled' ? iq.value : { data: [] });
      setSortQuality(sq.status === 'fulfilled' ? sq.value : { data: [] });
      setScaleQuality(scale.status === 'fulfilled' ? scale.value : { data: [] });
      setBlockedStatus(bs.status === 'fulfilled' ? bs.value : { data: [] });
      setTrackingLosses(tl.status === 'fulfilled' ? tl.value : { data: [] });
    } catch (e) {
      if (!signal.aborted) throw e;
    } finally {
      if (!signal.aborted) setAnalyticsLoading(false);
    }
  };

  useEffect(() => {
    if (selectedDatabase) {
      if (activeTab === 'view') fetchLogs(1, searchTerm);
      else if (activeTab === 'charts') fetchInductionQuality(selectedDatabase.id);
    }
  }, [selectedDatabase, activeTab]);

  const handleSearch = (forceTerm) => { setCurrentPage(1); fetchLogs(1, typeof forceTerm === 'string' ? forceTerm : searchTerm); };
  const handlePageChange = (page) => fetchLogs(page, searchTerm);

  return (
    <div className="min-h-screen bg-gray-50">
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

      <nav className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {[
              { key: 'upload', label: 'Upload & Process', Icon: Upload },
              { key: 'view',   label: 'View Logs',        Icon: Database },
              { key: 'charts', label: 'Analytics & Charts', Icon: BarChart3 },
            ].map(({ key, label, Icon }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                className={
                  'py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 ' +
                  (activeTab === key
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300')
                }
              >
                <Icon className="w-4 h-4" />
                {label}
                {key === 'upload' && isUploading && activeTab !== 'upload' && (
                  <span className="flex items-center gap-1 px-1.5 py-0.5 bg-blue-100 text-blue-700 text-xs rounded-full font-semibold">
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
                    procesando
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {/* UploadTab siempre montado para que un upload en curso no se interrumpa al cambiar de pestaña */}
        <div className={activeTab !== 'upload' ? 'hidden' : ''}>
          <UploadTab
            onUploadSuccess={(result) => { setUploadStatus(result); fetchDatabases(true); }}
            onUploadingChange={setIsUploading}
            uploadStatus={uploadStatus}
            customers={customers}
          />
        </div>

        {activeTab === 'view' && (
          <ViewLogsTab
            databases={databases}
            selectedDatabase={selectedDatabase}
            onDatabaseSelect={setSelectedDatabase}
            logs={logs}
            loading={loading}
            searchTerm={searchTerm}
            appliedSearchTerm={appliedSearchTerm}
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
            onRefreshDatabases={fetchDatabases}
            analyticsLoading={analyticsLoading}
            inductionQuality={inductionQuality}
            sortQuality={sortQuality}
            scaleQuality={scaleQuality}
            blockedStatus={blockedStatus}
            trackingLosses={trackingLosses}
          />
        )}
      </main>
    </div>
  );
}

export default LogAnalyzerApp;
