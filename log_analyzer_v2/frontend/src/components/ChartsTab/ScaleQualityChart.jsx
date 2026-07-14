import React, { useState, useMemo, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { SCALE_COLORS } from '../../constants';

export default function ScaleQualityChart({ scaleQuality, selectedDatabaseId }) {
  const [minimized, setMinimized] = useState(false);
  const [otrosExpanded, setOtrosExpanded] = useState(false);

  useEffect(() => {
    setOtrosExpanded(false);
  }, [selectedDatabaseId]);

  const scalePreRows = useMemo(
    () => (scaleQuality?.data || []).filter(r => !r.is_other && !r.is_other_header),
    [scaleQuality]
  );

  const scaleErrorCodes = useMemo(
    () => Object.keys(scaleQuality?.error_codes || {}),
    [scaleQuality]
  );

  const scaleKpis = useMemo(() => {
    if (!scalePreRows.length) return null;
    const excl    = scaleQuality?.exclude_from_pct || [];
    const totalOk = scalePreRows.reduce((s, r) => s + r.ok, 0);
    const totalAll = scalePreRows.reduce((s, r) => s + r.total, 0);
    const excluded = scalePreRows.reduce(
      (s, r) => s + excl.reduce((x, c) => x + (r[`err_${c}`] || 0), 0), 0
    );
    const pctBase = totalAll - excluded;
    const pct     = pctBase > 0 ? ((totalOk / pctBase) * 100).toFixed(1) : null;
    return { totalOk, pctBase, totalErr: pctBase - totalOk, pct };
  }, [scalePreRows, scaleQuality]);

  if (!scaleQuality || scaleQuality.error || !scaleQuality.data?.length) return null;

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div
        className="flex items-center justify-between px-6 py-4 cursor-pointer select-none hover:bg-gray-50"
        onClick={() => setMinimized(v => !v)}
      >
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-medium text-gray-900">Calidad de Básculas por Infeed</h3>
          {scaleQuality?.customer && (
            <span className="text-sm font-normal text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
              {scaleQuality.customer}
            </span>
          )}
          {scaleKpis?.pct && (
            <span className={`text-sm font-semibold px-2 py-0.5 rounded ${parseFloat(scaleKpis.pct) >= 90 ? 'bg-green-100 text-green-700' : parseFloat(scaleKpis.pct) >= 70 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}`}>
              {scaleKpis.pct}% OK
            </span>
          )}
        </div>
        <span className="text-gray-400 text-lg font-bold">{minimized ? '▸' : '▾'}</span>
      </div>

      {!minimized && scaleKpis && (
        <div className="px-6 pb-6">
          <div className="grid grid-cols-3 gap-4 mb-6 mt-2">
            <div className="bg-gray-50 border border-gray-200 p-4 rounded-lg">
              <p className="text-xs font-medium text-gray-500 uppercase">Total Pesajes</p>
              <p className="text-2xl font-bold text-gray-800 mt-1">{scaleKpis.pctBase.toLocaleString()}</p>
            </div>
            <div className="bg-green-50 border border-green-200 p-4 rounded-lg">
              <p className="text-xs font-medium text-gray-500 uppercase">Data OK</p>
              <p className="text-2xl font-bold text-green-700 mt-1">{scaleKpis.totalOk.toLocaleString()}</p>
              {scaleKpis.pct && <p className="text-sm text-green-600 mt-1">{scaleKpis.pct}%</p>}
            </div>
            <div className="bg-red-50 border border-red-200 p-4 rounded-lg">
              <p className="text-xs font-medium text-gray-500 uppercase">Errores</p>
              <p className="text-2xl font-bold text-red-700 mt-1">{scaleKpis.totalErr.toLocaleString()}</p>
              {scaleKpis.pct && <p className="text-sm text-red-600 mt-1">{(100 - parseFloat(scaleKpis.pct)).toFixed(1)}%</p>}
            </div>
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={scalePreRows} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="infeed" tick={{ fontSize: 11 }} tickFormatter={v => v === 'Loop del Crossorter' ? 'Loop' : v} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null;
                  const total = payload.reduce((s, p) => s + (p.value || 0), 0);
                  return (
                    <div className="bg-white border border-gray-200 rounded-lg p-3 shadow text-sm">
                      <p className="font-semibold text-gray-800 mb-2">{label}</p>
                      {payload.filter(p => p.value > 0).map(p => (
                        <p key={p.dataKey} style={{ color: p.fill }}>{p.name}: {p.value.toLocaleString()}</p>
                      ))}
                      <p className="text-gray-500 mt-2 border-t pt-1">Total: {total.toLocaleString()}</p>
                    </div>
                  );
                }}
              />
              <Bar dataKey="ok"      name="Data OK"          stackId="s" fill={SCALE_COLORS.ok} />
              <Bar dataKey="noscan"  name="No Scan"          stackId="s" fill={SCALE_COLORS.noscan} />
              <Bar dataKey="unknown" name="Unknown (6 sin g)" stackId="s" fill={SCALE_COLORS.unknown} />
              {scaleErrorCodes.map(code => (
                <Bar key={code} dataKey={`err_${code}`} name={scaleQuality.error_codes[code]} stackId="s" fill={SCALE_COLORS[`err_${code}`] || '#9CA3AF'} />
              ))}
            </BarChart>
          </ResponsiveContainer>

          <div className="mt-6">
            <table className="w-full text-sm divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-3 py-2 text-left font-medium text-gray-500 uppercase">Infeed</th>
                  <th className="px-3 py-2 text-center font-medium text-green-600 uppercase">Data OK</th>
                  <th className="px-3 py-2 text-center font-medium text-slate-500 uppercase">No Scan</th>
                  <th className="px-3 py-2 text-center font-medium text-yellow-600 uppercase">Unknown</th>
                  {scaleErrorCodes.map(code => (
                    <th key={code} className="px-2 py-2 text-center font-medium text-red-500 uppercase leading-tight max-w-[72px]">
                      {scaleQuality.error_codes[code]}
                    </th>
                  ))}
                  <th className="px-3 py-2 text-center font-medium text-gray-500 uppercase">Total</th>
                  <th className="px-3 py-2 text-center font-medium text-gray-500 uppercase">% OK</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-100">
                {scaleQuality.data.map((row, idx) => {
                  if (row.is_other_header) return (
                    <tr key={idx} className="bg-gray-200 border-t-2 border-gray-400 cursor-pointer select-none hover:bg-gray-300"
                        onClick={() => setOtrosExpanded(x => !x)}>
                      <td className="px-3 py-2 font-semibold text-gray-600 italic">
                        <span className="mr-1 text-gray-500">{otrosExpanded ? '▾' : '▸'}</span>
                        {row.infeed}
                      </td>
                      <td className="px-3 py-2 text-center font-semibold text-gray-600">{row.ok.toLocaleString()}</td>
                      <td className="px-3 py-2 text-center font-semibold text-gray-600">{row.noscan.toLocaleString()}</td>
                      <td className="px-3 py-2 text-center font-semibold text-gray-600">{row.unknown.toLocaleString()}</td>
                      {scaleErrorCodes.map(code => (
                        <td key={code} className="px-2 py-2 text-center font-semibold text-gray-600">{(row[`err_${code}`] || 0).toLocaleString()}</td>
                      ))}
                      <td className="px-3 py-2 text-center font-bold text-gray-700">{row.total.toLocaleString()}</td>
                      <td className="px-3 py-2 text-center text-gray-400">—</td>
                    </tr>
                  );
                  if (row.is_other) return otrosExpanded ? (
                    <tr key={idx} className="bg-gray-50 text-gray-400 italic">
                      <td className="px-3 py-2 font-medium text-gray-500 pl-7">↳ {row.infeed}</td>
                      <td className="px-3 py-2 text-center">{row.ok.toLocaleString()}</td>
                      <td className="px-3 py-2 text-center">{row.noscan.toLocaleString()}</td>
                      <td className="px-3 py-2 text-center">{row.unknown.toLocaleString()}</td>
                      {scaleErrorCodes.map(code => (
                        <td key={code} className="px-2 py-2 text-center">{(row[`err_${code}`] || 0).toLocaleString()}</td>
                      ))}
                      <td className="px-3 py-2 text-center font-semibold text-gray-500">{row.total.toLocaleString()}</td>
                      <td className="px-3 py-2 text-center">—</td>
                    </tr>
                  ) : null;
                  return (
                    <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      <td className="px-3 py-2 font-medium text-gray-900">{row.infeed}</td>
                      <td className="px-3 py-2 text-center text-green-600 font-medium">{row.ok.toLocaleString()}</td>
                      <td className="px-3 py-2 text-center text-slate-500">{row.noscan.toLocaleString()}</td>
                      <td className="px-3 py-2 text-center text-yellow-600">{row.unknown.toLocaleString()}</td>
                      {scaleErrorCodes.map(code => (
                        <td key={code} className="px-2 py-2 text-center text-red-500">{(row[`err_${code}`] || 0).toLocaleString()}</td>
                      ))}
                      <td className="px-3 py-2 text-center font-semibold">{row.total.toLocaleString()}</td>
                      <td className="px-3 py-2 text-center">
                        <span className={`px-1.5 py-0.5 rounded-full text-xs font-semibold ${row.ok_pct >= 90 ? 'bg-green-100 text-green-800' : row.ok_pct >= 70 ? 'bg-yellow-100 text-yellow-800' : 'bg-red-100 text-red-800'}`}>
                          {row.ok_pct}%
                        </span>
                      </td>
                    </tr>
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
