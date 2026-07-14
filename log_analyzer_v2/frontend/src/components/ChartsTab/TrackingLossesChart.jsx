import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const ZONE_COLOR = '#6366F1';

function shortZoneName(v) {
  if (v === 'Loop del Crossorter') return 'Loop';
  if (v === 'Infeeds 1-2-3-4-9-10') return 'I. 1-4,9-10';
  if (v === 'Infeeds 5-6-7-8') return 'I. 5-8';
  return v;
}

function pctBadge(pct) {
  const cls = pct >= 30 ? 'bg-red-100 text-red-800' : pct >= 10 ? 'bg-yellow-100 text-yellow-800' : 'bg-gray-100 text-gray-700';
  return <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${cls}`}>{pct}%</span>;
}

export default function TrackingLossesChart({ trackingLosses }) {
  const [minimized, setMinimized] = useState(false);
  const [expandedZones, setExpandedZones] = useState(new Set());

  if (!trackingLosses || trackingLosses.error || !trackingLosses.data?.length) return null;

  const total   = trackingLosses.total ?? 0;
  const topZone = trackingLosses.data.reduce((max, row) => row.count > max.count ? row : max, trackingLosses.data[0]);

  function toggleZone(zoneId) {
    setExpandedZones(prev => {
      const next = new Set(prev);
      next.has(zoneId) ? next.delete(zoneId) : next.add(zoneId);
      return next;
    });
  }

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div
        className="flex items-center justify-between px-6 py-4 cursor-pointer select-none hover:bg-gray-50"
        onClick={() => setMinimized(v => !v)}
      >
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-medium text-gray-900">Pérdidas de tracking</h3>
          {trackingLosses.customer && (
            <span className="text-sm font-normal text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
              {trackingLosses.customer}
            </span>
          )}
          <span className="text-sm font-semibold px-2 py-0.5 rounded bg-indigo-100 text-indigo-700">
            {total.toLocaleString()} PICs perdidos
          </span>
        </div>
        <span className="text-gray-400 text-lg font-bold">{minimized ? '▸' : '▾'}</span>
      </div>

      {!minimized && (
        <div className="px-6 pb-6">
          <p className="text-sm text-gray-500 mb-4">
            Eventos msg20 con parcelexitstate=2 · zona por parcelexitpoint
          </p>

          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="bg-indigo-50 border border-indigo-200 p-4 rounded-lg">
              <p className="text-xs font-medium text-gray-500 uppercase">Total perdidos</p>
              <p className="text-2xl font-bold text-indigo-700 mt-1">{total.toLocaleString()}</p>
            </div>
            <div className="bg-gray-50 border border-gray-200 p-4 rounded-lg">
              <p className="text-xs font-medium text-gray-500 uppercase">Zona con más pérdidas</p>
              <p className="text-lg font-bold text-gray-800 mt-1 truncate">{topZone?.zone_name ?? '—'}</p>
            </div>
            <div className="bg-gray-50 border border-gray-200 p-4 rounded-lg">
              <p className="text-xs font-medium text-gray-500 uppercase">% sobre total perdidos</p>
              <p className="text-2xl font-bold text-gray-700 mt-1">{topZone?.pct ?? 0}%</p>
            </div>
          </div>

          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={trackingLosses.data} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="zone_name"
                tick={{ fontSize: 11 }}
                interval={0}
                angle={-20}
                textAnchor="end"
                height={55}
                tickFormatter={shortZoneName}
              />
              <YAxis tick={{ fontSize: 12 }} allowDecimals={false} />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null;
                  const d = payload[0]?.payload;
                  return (
                    <div className="bg-white border border-gray-200 rounded-lg p-3 shadow text-sm">
                      <p className="font-semibold text-gray-800 mb-1">{label}</p>
                      <p className="text-indigo-600">{d?.count?.toLocaleString()} PICs perdidos</p>
                      <p className="text-gray-400 text-xs mt-1">{d?.pct}% del total</p>
                    </div>
                  );
                }}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {trackingLosses.data.map((_, i) => (
                  <Cell key={i} fill={ZONE_COLOR} fillOpacity={0.85} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          {/* Tabla con detalle expandible por zona */}
          <div className="mt-6 overflow-x-auto">
            <table className="min-w-full text-sm divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase w-6"></th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Zona / PEC</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-indigo-600 uppercase">PICs perdidos</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">% del total</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-400 uppercase">% de zona</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-100">
                {trackingLosses.data.map((row, idx) => {
                  const isExpanded = expandedZones.has(row.zone_id);
                  const hasPoints  = row.points?.length > 0;
                  return (
                    <React.Fragment key={idx}>
                      {/* Fila de zona */}
                      <tr
                        className={`${idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'} ${hasPoints ? 'cursor-pointer hover:bg-indigo-50' : ''}`}
                        onClick={() => hasPoints && toggleZone(row.zone_id)}
                      >
                        <td className="px-4 py-2 text-gray-400 text-xs font-bold select-none">
                          {hasPoints ? (isExpanded ? '▾' : '▸') : ''}
                        </td>
                        <td className="px-4 py-2 font-semibold text-gray-900">{row.zone_name}</td>
                        <td className="px-4 py-2 text-right text-indigo-600 font-bold">{row.count.toLocaleString()}</td>
                        <td className="px-4 py-2 text-right">{pctBadge(row.pct)}</td>
                        <td className="px-4 py-2 text-right text-gray-300 text-xs">—</td>
                      </tr>

                      {/* Filas de puntos individuales */}
                      {isExpanded && row.points?.map((pt, pidx) => (
                        <tr key={pidx} className="bg-indigo-50 border-l-2 border-indigo-300">
                          <td className="px-4 py-1.5"></td>
                          <td className="px-4 py-1.5 text-gray-600 font-mono text-xs pl-8">{pt.exit_point}</td>
                          <td className="px-4 py-1.5 text-right text-indigo-500 font-medium text-xs">{pt.count.toLocaleString()}</td>
                          <td className="px-4 py-1.5 text-right">
                            <span className="text-xs text-gray-400">{Math.round(pt.count / total * 100 * 10) / 10}%</span>
                          </td>
                          <td className="px-4 py-1.5 text-right">
                            <span className={`px-1.5 py-0.5 rounded text-xs font-semibold ${
                              pt.pct_of_zone >= 30 ? 'bg-red-100 text-red-700' :
                              pt.pct_of_zone >= 15 ? 'bg-yellow-100 text-yellow-700' :
                                                     'bg-gray-100 text-gray-500'
                            }`}>{pt.pct_of_zone}%</span>
                          </td>
                        </tr>
                      ))}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
