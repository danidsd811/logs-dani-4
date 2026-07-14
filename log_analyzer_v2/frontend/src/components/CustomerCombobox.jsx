import React, { useState, useEffect, useRef } from 'react';

const fmtName = name => {
  if (!name) return name;
  return name.split(' ').map(word =>
    /^[A-Z]{2,3}$/.test(word)
      ? word
      : word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
  ).join(' ');
};

export default function CustomerCombobox({ value, onChange, customers }) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const ref = useRef(null);

  useEffect(() => {
    const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const configured = customers.filter(c => c.charts?.length > 0);
  const others     = customers.filter(c => !c.charts?.length);
  const filterFn   = (list) => list.filter(c => !search || c.name.toLowerCase().includes(search.toLowerCase()));
  const filtCfg    = filterFn(configured);
  const filtOther  = filterFn(others);
  const selected   = customers.find(c => c.id === value);

  const pick = (id) => { onChange(id); setOpen(false); setSearch(''); };

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen(v => !v)}
        className="mt-1 w-full flex items-center justify-between border border-gray-300 rounded-md shadow-sm py-2 px-3 text-sm bg-white focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 hover:border-gray-400"
      >
        <span className="flex items-center gap-2 truncate min-w-0">
          {selected ? (
            <>
              <span className={`flex-shrink-0 ${selected.charts?.length ? 'text-green-500' : 'text-gray-300'}`}>●</span>
              <span className="truncate">{fmtName(selected.name)}</span>
            </>
          ) : (
            <span className="text-gray-400">— Select customer —</span>
          )}
        </span>
        <svg className="h-4 w-4 text-gray-400 flex-shrink-0 ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="absolute z-50 mt-1 w-full bg-white border border-gray-200 rounded-md shadow-lg flex flex-col" style={{ maxHeight: '300px' }}>
          <div className="p-2 border-b border-gray-100 flex-shrink-0">
            <input
              autoFocus
              type="text"
              placeholder="Buscar cliente..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-full px-2 py-1.5 text-sm border border-gray-200 rounded focus:outline-none focus:ring-1 focus:ring-blue-400"
            />
          </div>
          <div className="overflow-y-auto">
            <div onClick={() => pick('')} className="px-3 py-2 text-sm text-gray-400 cursor-pointer hover:bg-gray-50 italic">
              — Sin selección —
            </div>
            {filtCfg.length > 0 && (
              <>
                <div className="px-3 py-1 text-xs font-semibold text-green-700 uppercase tracking-wide bg-green-50 border-y border-green-100">
                  ● Con analytics
                </div>
                {filtCfg.map(c => (
                  <div key={c.id} onClick={() => pick(c.id)}
                    className={`flex items-center gap-2 px-3 py-2 text-sm cursor-pointer hover:bg-green-50 ${value === c.id ? 'bg-green-50 font-semibold' : ''}`}>
                    <span className="text-green-500 flex-shrink-0 text-xs">●</span>
                    <span className="truncate">{fmtName(c.name)}</span>
                  </div>
                ))}
              </>
            )}
            {filtOther.length > 0 && (
              <>
                <div className="px-3 py-1 text-xs font-semibold text-gray-400 uppercase tracking-wide bg-gray-50 border-y border-gray-100">
                  ○ Sin analytics
                </div>
                {filtOther.map(c => (
                  <div key={c.id} onClick={() => pick(c.id)}
                    className={`flex items-center gap-2 px-3 py-2 text-sm cursor-pointer hover:bg-gray-50 ${value === c.id ? 'bg-gray-100 font-semibold' : ''}`}>
                    <span className="text-gray-300 flex-shrink-0 text-xs">●</span>
                    <span className="truncate text-gray-600">{fmtName(c.name)}</span>
                  </div>
                ))}
              </>
            )}
            {filtCfg.length === 0 && filtOther.length === 0 && (
              <div className="px-3 py-6 text-sm text-gray-400 text-center">No se encontraron clientes</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
