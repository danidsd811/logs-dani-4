import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { BLOCKED_FLAG_LABELS, BLOCKED_FLAG_COLORS } from '../../constants';

export default function BlockedStatusChart({ blockedStatus }) {
  const [minimized, setMinimized] = useState(false);

  if (!blockedStatus || blockedStatus.error || !blockedStatus.data?.length) return null;

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div
        className="flex items-center justify-between px-6 py-4 cursor-pointer select-none hover:bg-gray-50"
        onClick={() => setMinimized(v => !v)}
      >
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-medium text-gray-900">Motivos de paquete bloqueado entre carriers</h3>
          <span className="text-sm text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
            Parcel Blocked Status · {blockedStatus.total?.toLocaleString()} HOSTPICs únicos
          </span>
        </div>
        <span className="text-gray-400 text-lg font-bold">{minimized ? '▸' : '▾'}</span>
      </div>

      {!minimized && (
        <div className="px-6 pb-5">
          <p className="text-sm text-gray-500 mb-4">
            Primera ocurrencia por HOSTPIC (msg 21, dest 998) · un HOSTPIC puede contabilizarse en varios motivos simultáneos
          </p>
          <div className="flex gap-8 flex-wrap">
            <div className="flex-1 min-w-[320px]">
              <ResponsiveContainer width="100%" height={Math.max(180, blockedStatus.data.length * 44)}>
                <BarChart
                  layout="vertical"
                  data={blockedStatus.data.map(d => ({ ...d, label: BLOCKED_FLAG_LABELS[d.flag] || d.flag }))}
                  margin={{ top: 4, right: 60, left: 8, bottom: 4 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 12 }} allowDecimals={false} />
                  <YAxis type="category" dataKey="label" width={155} tick={{ fontSize: 12 }} />
                  <Tooltip
                    formatter={(value, name, props) => [
                      `${value.toLocaleString()} HOSTPICs (${props.payload.pct}%)`,
                      BLOCKED_FLAG_LABELS[props.payload.flag] || props.payload.flag,
                    ]}
                  />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                    {blockedStatus.data.map((entry, i) => (
                      <Cell key={i} fill={BLOCKED_FLAG_COLORS[entry.flag] || '#94A3B8'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="w-60 flex-shrink-0">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-gray-500 border-b">
                    <th className="pb-2">Flag</th>
                    <th className="pb-2 text-right">HOSTPICs</th>
                    <th className="pb-2 text-right">%</th>
                  </tr>
                </thead>
                <tbody>
                  {blockedStatus.data.map((row, i) => (
                    <tr key={i} className="border-b border-gray-50">
                      <td className="py-1.5 flex items-center gap-1.5">
                        <span className="inline-block w-2.5 h-2.5 rounded-sm flex-shrink-0"
                          style={{ background: BLOCKED_FLAG_COLORS[row.flag] || '#94A3B8' }} />
                        <span className="text-xs text-gray-700">{BLOCKED_FLAG_LABELS[row.flag] || row.flag}</span>
                      </td>
                      <td className="py-1.5 text-right font-mono text-xs">{row.count.toLocaleString()}</td>
                      <td className="py-1.5 text-right">
                        <span className={`px-1.5 py-0.5 rounded text-xs font-semibold ${row.pct >= 30 ? 'bg-red-100 text-red-800' : row.pct >= 10 ? 'bg-yellow-100 text-yellow-800' : 'bg-gray-100 text-gray-700'}`}>
                          {row.pct}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
