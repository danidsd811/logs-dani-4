import React, { useState, useMemo } from 'react';
import { Activity } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { ODS_LABELS, ODS_COLORS } from '../../constants';

export default function SortQualityChart({ sortQuality }) {
  const [minimized, setMinimized] = useState(false);

  const sortKpis = useMemo(() => {
    if (!sortQuality?.data?.length) return null;
    const totalOk  = sortQuality.data.filter(r => r.state === '1').reduce((s, r) => s + r.count, 0);
    const totalAll = sortQuality.data.reduce((s, r) => s + r.count, 0);
    const pct      = totalAll > 0 ? ((totalOk / totalAll) * 100).toFixed(1) : null;
    return { totalOk, totalAll, pct };
  }, [sortQuality]);

  const sortChartData = useMemo(() => {
    if (!sortQuality?.data?.length) return { rows: [], states: [] };
    const data = sortQuality.data;
    const infeedMap = {};
    data.forEach(r => { infeedMap[r.zone_id] = { zone_id: r.zone_id, zone_name: r.zone_name }; });
    const infeeds = Object.values(infeedMap).sort((a, b) => a.zone_id - b.zone_id);
    const states = [...new Set(data.map(r => r.state))].sort((a, b) => {
      if (a === '1') return -1;
      if (b === '1') return 1;
      return a.localeCompare(b);
    });
    const rows = infeeds.map(({ zone_id, zone_name }) => {
      const row = { zone_name };
      data.filter(r => r.zone_id === zone_id).forEach(r => {
        row[r.state] = (row[r.state] || 0) + r.count;
      });
      return row;
    });
    return { rows, states };
  }, [sortQuality]);

  if (!sortQuality || sortQuality.error) return null;

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div
        className="flex items-center justify-between px-6 py-4 cursor-pointer select-none hover:bg-gray-50"
        onClick={() => setMinimized(v => !v)}
      >
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-medium text-gray-900">Calidad de Clasificación por ODS</h3>
          {sortQuality?.customer && (
            <span className="text-sm font-normal text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
              {sortQuality.customer}
            </span>
          )}
          {sortKpis?.pct && (
            <span className={`text-sm font-semibold px-2 py-0.5 rounded ${parseFloat(sortKpis.pct) >= 90 ? 'bg-green-100 text-green-700' : parseFloat(sortKpis.pct) >= 70 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}`}>
              {sortKpis.pct}% clasificados
            </span>
          )}
        </div>
        <span className="text-gray-400 text-lg font-bold">{minimized ? '▸' : '▾'}</span>
      </div>

      {!minimized && (
        <div className="px-6 pb-6">
          <p className="text-sm text-gray-500 mb-4">
            Paquetes únicos (HOSTPIC × ODS) por infeed · cada color = motivo ODS
          </p>
          {sortKpis ? (
            <>
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-green-50 border border-green-200 p-4 rounded-lg">
                  <p className="text-xs font-medium text-gray-500 uppercase">Clasificados (ODS=1)</p>
                  <p className="text-2xl font-bold text-green-700 mt-1">{sortKpis.totalOk.toLocaleString()}</p>
                </div>
                <div className="bg-red-50 border border-red-200 p-4 rounded-lg">
                  <p className="text-xs font-medium text-gray-500 uppercase">Otros estados</p>
                  <p className="text-2xl font-bold text-red-700 mt-1">{(sortKpis.totalAll - sortKpis.totalOk).toLocaleString()}</p>
                </div>
                <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg">
                  <p className="text-xs font-medium text-gray-500 uppercase">% Clasificados</p>
                  <p className={`text-2xl font-bold mt-1 ${parseFloat(sortKpis.pct) >= 90 ? 'text-green-700' : parseFloat(sortKpis.pct) >= 70 ? 'text-yellow-600' : 'text-red-700'}`}>
                    {sortKpis.pct !== null ? `${sortKpis.pct}%` : 'N/A'}
                  </p>
                </div>
              </div>

              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={sortChartData.rows} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="zone_name"
                    tick={{ fontSize: 11 }}
                    interval={0}
                    angle={-15}
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
                          <p className="font-semibold text-gray-800 mb-1">{label}</p>
                          <p className="text-xs text-gray-400 mb-2">Total: {total.toLocaleString()}</p>
                          {payload.filter(p => p.value > 0).map(p => (
                            <p key={p.dataKey} style={{ color: p.fill }}>
                              ODS {p.dataKey} — {ODS_LABELS[p.dataKey] || p.dataKey}: {p.value.toLocaleString()}
                            </p>
                          ))}
                        </div>
                      );
                    }}
                  />
                  <Legend formatter={v => `ODS ${v} — ${ODS_LABELS[v] || v}`} />
                  {sortChartData.states.map(state => (
                    <Bar key={state} dataKey={state} stackId="s" fill={ODS_COLORS[state] || '#9CA3AF'} />
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
  );
}
