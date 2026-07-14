import React, { useState, useMemo } from 'react';

export default function HostpicsTable({ data, total, variant }) {
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
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div
        className="flex items-center justify-between px-6 py-4 cursor-pointer select-none hover:bg-gray-50"
        onClick={() => setCollapsed(c => !c)}
      >
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-medium text-gray-900">{title}</h3>
          <span className={`px-2 py-0.5 text-sm font-semibold rounded ${isBad ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
            {total.toLocaleString()} únicos
          </span>
          <span className="text-xs text-gray-400">{subtitle}</span>
        </div>
        <span className="text-gray-400 text-lg font-bold">{collapsed ? '▸' : '▾'}</span>
      </div>

      {!collapsed && (
        <>
          <div className="px-6 py-3 border-t border-b border-gray-100 flex items-center gap-2 flex-wrap">
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
              onClick={e => { e.stopPropagation(); copyAll(); }}
              className={`px-3 py-1.5 text-sm border rounded font-medium ${copyBtnCls}`}
              title="Copiar todos los HOSTPICs filtrados al portapapeles"
            >
              Copiar lista
            </button>
            {filtered.length !== total && (
              <span className="text-xs text-gray-400">{filtered.length.toLocaleString()} filtrados</span>
            )}
          </div>

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
