import React, { useState, useMemo, useEffect, useCallback } from 'react';
import { Activity } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import HostpicsTable from '../HostpicsTable';
import { API_BASE } from '../../constants';

function aggregateByInfeed(rows) {
  const map = {};
  for (const r of rows) {
    if (!map[r.infeed]) {
      map[r.infeed] = { infeed: r.infeed, infeed_num: r.infeed_num, good: 0, bad: 0, total: 0 };
    }
    map[r.infeed].good  += r.good;
    map[r.infeed].bad   += r.bad;
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

export default function InductionQualityChart({ inductionQuality, selectedDatabaseId }) {
  const [minimized, setMinimized] = useState(false);
  const [filterChartHour, setFilterChartHour] = useState('');
  const [hostpicsExpanded, setHostpicsExpanded] = useState(false);
  const [badHostpics, setBadHostpics] = useState(null);
  const [goodHostpics, setGoodHostpics] = useState(null);
  const [hostpicsLoading, setHostpicsLoading] = useState(false);

  useEffect(() => {
    setFilterChartHour('');
    setHostpicsExpanded(false);
    setBadHostpics(null);
    setGoodHostpics(null);
    setHostpicsLoading(false);
  }, [selectedDatabaseId]);

  const fetchHostpics = useCallback(async () => {
    if (!selectedDatabaseId || hostpicsLoading) return;
    setHostpicsLoading(true);
    try {
      const [bad, good] = await Promise.all([
        fetch(`${API_BASE}/databases/${selectedDatabaseId}/bad_hostpics`).then(r => r.json()),
        fetch(`${API_BASE}/databases/${selectedDatabaseId}/good_hostpics`).then(r => r.json()),
      ]);
      setBadHostpics(bad);
      setGoodHostpics(good);
    } catch {
      setBadHostpics({ data: [], total: 0 });
      setGoodHostpics({ data: [], total: 0 });
    } finally {
      setHostpicsLoading(false);
    }
  }, [selectedDatabaseId, hostpicsLoading]);

  const handleToggleHostpics = () => {
    const next = !hostpicsExpanded;
    setHostpicsExpanded(next);
    if (next && badHostpics === null && !hostpicsLoading) fetchHostpics();
  };

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

  if (!inductionQuality || inductionQuality.error) return null;

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div
        className="flex items-center justify-between px-6 py-4 cursor-pointer select-none hover:bg-gray-50"
        onClick={() => setMinimized(v => !v)}
      >
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-medium text-gray-900">Calidad de Inducción por Infeed</h3>
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
        <span className="text-gray-400 text-lg font-bold">{minimized ? '▸' : '▾'}</span>
      </div>

      {!minimized && (
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
                  <XAxis
                    dataKey="infeed"
                    tick={{ fontSize: 11 }}
                    interval={0}
                    angle={-20}
                    textAnchor="end"
                    height={55}
                    tickFormatter={(v) => {
                      if (v === 'Loop del Crossorter') return 'Loop';
                      if (v === 'Infeeds 1-2-3-4-9-10') return 'I. 1-4,9-10';
                      if (v === 'Infeeds 5-6-7-8') return 'I. 5-8';
                      return v;
                    }}
                  />
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

          {/* Detalle HOSTPICs — carga lazy al expandir */}
          {inductionQuality.data?.length > 0 && (
            <div className="mt-4 border-t border-gray-100 pt-4">
              <button
                onClick={handleToggleHostpics}
                className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-700 select-none"
              >
                <span className="font-bold">{hostpicsExpanded ? '▾' : '▸'}</span>
                <span>{hostpicsExpanded ? 'Ocultar detalle HOSTPICs' : 'Ver detalle HOSTPICs'}</span>
              </button>

              {hostpicsExpanded && (
                <div className="mt-4 space-y-4">
                  {hostpicsLoading ? (
                    <div className="bg-white p-6 rounded-lg shadow flex items-center justify-center h-24">
                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
                      <span className="ml-3 text-gray-500 text-sm">Cargando HOSTPICs...</span>
                    </div>
                  ) : (
                    <>
                      {badHostpics?.data?.length > 0 && (
                        <HostpicsTable data={badHostpics.data} total={badHostpics.total} variant="bad" />
                      )}
                      {goodHostpics?.data?.length > 0 && (
                        <HostpicsTable data={goodHostpics.data} total={goodHostpics.total} variant="good" />
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
