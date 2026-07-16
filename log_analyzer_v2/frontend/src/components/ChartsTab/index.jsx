import React from 'react';
import { Database, BarChart3 } from 'lucide-react';
import SortQualityChart from './SortQualityChart';
import InductionQualityChart from './InductionQualityChart';
import ScaleQualityChart from './ScaleQualityChart';
import BlockedStatusChart from './BlockedStatusChart';
import TrackingLossesChart from './TrackingLossesChart';

export default function ChartsTab({
  databases,
  selectedDatabase,
  onDatabaseSelect,
  onRefreshDatabases,
  analyticsLoading,
  inductionQuality,
  sortQuality,
  scaleQuality,
  blockedStatus,
  trackingLosses,
}) {
  return (
    <div className="space-y-6">
      {/* Selector de base de datos — siempre visible */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 shadow rounded-lg p-4">
        <div className="flex items-center space-x-4">
          <Database className="h-5 w-5 text-blue-600" />
          <label className="text-sm font-semibold text-gray-800">Active Database:</label>
          <select
            value={selectedDatabase?.id || ''}
            onChange={(e) => {
              const db = databases.find(d => d.id === e.target.value);
              if (db) onDatabaseSelect(db);
            }}
            className="block w-72 pl-3 pr-8 py-2 text-sm border border-blue-300 focus:outline-none focus:ring-1 focus:ring-blue-500 rounded-md font-medium bg-white"
          >
            {databases.length === 0 && (
              <option value="" disabled>— No hay bases de datos —</option>
            )}
            {databases.map(db => (
              <option key={db.id} value={db.id}>
                {db.table_name} ({db.record_count?.toLocaleString() || '0'} records) - {new Date(db.created_at).toLocaleDateString()}{db.customer_name ? ` · ${db.customer_name}` : ''}
              </option>
            ))}
          </select>
          <button
            onClick={() => onRefreshDatabases()}
            title="Refrescar lista de bases de datos"
            className="p-1.5 rounded-md text-blue-500 hover:bg-blue-100 transition-colors"
          >
            ↻
          </button>
          {selectedDatabase && (
            <div className="text-sm text-gray-600 bg-white rounded px-3 py-1 border border-blue-100">
              <span className="font-semibold">{selectedDatabase.table_name}</span>
              <span className="mx-2">•</span>
              <span>{selectedDatabase.record_count?.toLocaleString()} records</span>
              <span className="mx-2">•</span>
              <span>{selectedDatabase.file_size_mb?.toFixed(1)} MB</span>
            </div>
          )}
        </div>
      </div>

      {/* Sin base de datos */}
      {!selectedDatabase && (
        <div className="text-center py-12">
          <BarChart3 className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">
            {databases.length === 0 ? 'No hay bases de datos disponibles' : 'Selecciona una base de datos'}
          </h3>
          <p className="mt-1 text-sm text-gray-500">
            {databases.length === 0
              ? 'Procesa un archivo de log primero para generar datos.'
              : 'Elige una base de datos del selector para ver los analytics.'}
          </p>
        </div>
      )}

      {/* Cargando analytics */}
      {selectedDatabase && analyticsLoading && (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          <span className="ml-4 text-gray-600">Cargando estadísticas...</span>
        </div>
      )}

      {/* Gráficos */}
      {selectedDatabase && !analyticsLoading && (
        <>
          <SortQualityChart sortQuality={sortQuality} />
          <InductionQualityChart
            inductionQuality={inductionQuality}
            selectedDatabaseId={selectedDatabase?.id}
          />
          <ScaleQualityChart
            scaleQuality={scaleQuality}
            selectedDatabaseId={selectedDatabase?.id}
          />
          <BlockedStatusChart blockedStatus={blockedStatus} />
          <TrackingLossesChart trackingLosses={trackingLosses} />
        </>
      )}
    </div>
  );
}
