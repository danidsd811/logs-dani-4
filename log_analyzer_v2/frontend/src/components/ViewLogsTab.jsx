import React, { useState, useMemo } from 'react';
import { Search, Database, X, ChevronLeft, ChevronRight } from 'lucide-react';

const SPECIAL_COLUMN_NAMES = {
  'timestamp': 'Time',
  'message_id': 'Message ID',
  'Pic': 'PIC',
  'AlibiId': 'Alibi ID',
  'Hostpic': 'Host PIC',
  'ParcelLength': 'Parcel Length',
  'ParcelHeight': 'Parcel Height',
  'ParcelWidth': 'Parcel Width',
  'ParcelWeight': 'Parcel Weight',
};

function formatColumnName(col) {
  return SPECIAL_COLUMN_NAMES[col] || col.replace(/([a-z])([A-Z])/g, '$1 $2');
}

function highlightSearchTerm(text, searchTerm) {
  if (!searchTerm || !text) return text;
  const regex = new RegExp(`(${searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
  const parts = text.toString().split(regex);
  return parts.map((part, index) =>
    regex.test(part) ? (
      <span key={index} className="bg-yellow-200 px-1 rounded">{part}</span>
    ) : part
  );
}

export default function ViewLogsTab({
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
  columns,
}) {
  const [refreshing, setRefreshing] = useState(false);

  const handleRefreshDatabases = async () => {
    setRefreshing(true);
    try {
      if (window.refreshDatabases) await window.refreshDatabases();
    } catch (error) {
      console.error('Error refreshing databases:', error);
    } finally {
      setRefreshing(false);
    }
  };

  const displayColumns = useMemo(() => {
    if (columns && columns.length > 0) return columns;
    if (logs.length === 0) return [];
    const allKeys = new Set();
    logs.forEach(log => Object.keys(log).forEach(key => allKeys.add(key)));
    return Array.from(allKeys).sort((a, b) => {
      if (a === 'timestamp') return -1;
      if (b === 'timestamp') return 1;
      if (a === 'message_id') return -1;
      if (b === 'message_id') return 1;
      return a.localeCompare(b);
    });
  }, [logs, columns]);

  return (
    <div className="space-y-6">
      {/* Database & Search Controls */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 shadow rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Database className="h-5 w-5 text-blue-600" />
              <label className="text-sm font-semibold text-gray-800">Active Database:</label>
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

          <div className="flex items-center space-x-2">
            <label className="text-sm font-semibold text-gray-800">Search:</label>
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
                onClick={() => { onSearchTermChange(''); onSearch(); }}
                className="inline-flex items-center px-2 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>

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
            {searchTerm && (
              <div className="bg-yellow-100 border border-yellow-300 rounded px-3 py-1 text-xs text-yellow-800">
                Found <strong>{totalRecords.toLocaleString()}</strong> records matching "{searchTerm}"
              </div>
            )}
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
                            {searchTerm && cellValue !== '-'
                              ? highlightSearchTerm(cellValue, searchTerm)
                              : cellValue}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

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
                  <p className="text-sm text-gray-700">
                    Showing page <span className="font-medium">{currentPage}</span> of{' '}
                    <span className="font-medium">{totalPages}</span> ({totalRecords.toLocaleString()} total records)
                  </p>
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
                      if (totalPages <= 5) pageNum = i + 1;
                      else if (currentPage <= 3) pageNum = i + 1;
                      else if (currentPage >= totalPages - 2) pageNum = totalPages - 4 + i;
                      else pageNum = currentPage - 2 + i;
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
            )}
          </>
        )}
      </div>
    </div>
  );
}
