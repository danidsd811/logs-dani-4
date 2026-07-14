import React, { useState, useEffect, useRef } from 'react';
import CustomerCombobox from './CustomerCombobox';
import { API_BASE } from '../constants';

export default function UploadTab({ onUploadSuccess, onUploadingChange, uploadStatus, customers }) {
  const [serverType, setServerType] = useState('eDS');
  const [customerId, setCustomerId] = useState('');
  const [crossorterType, setCrossorterType] = useState('standard');
  const [logFile, setLogFile] = useState(null);
  const [configFile, setConfigFile] = useState(null);
  const [updateConfig, setUpdateConfig] = useState(false);
  const logInputRef = useRef();
  const configInputRef = useRef();

  useEffect(() => {
    const customer = customers.find(c => c.id === customerId);
    setCrossorterType(customer?.crossorter_type || 'standard');
    setServerType(customer?.server_type || 'eDS');
    setUpdateConfig(false);
  }, [customerId, customers]);

  useEffect(() => {
    setLogFile(null);
    setConfigFile(null);
    if (logInputRef.current) logInputRef.current.value = '';
    if (configInputRef.current) configInputRef.current.value = '';
  }, [serverType]);

  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    onUploadingChange?.(uploading);
  }, [uploading]);
  const [processingTime, setProcessingTime] = useState(0);
  const [timer, setTimer] = useState(null);
  const [finalProcessingTime, setFinalProcessingTime] = useState(null);

  useEffect(() => {
    let startTime;
    if (uploading) {
      setFinalProcessingTime(null);
      startTime = performance.now();
      const interval = setInterval(() => {
        setProcessingTime((performance.now() - startTime) / 1000);
      }, 100);
      setTimer(interval);
      return () => {
        clearInterval(interval);
        setTimer(null);
      };
    } else {
      if (timer) {
        clearInterval(timer);
        setTimer(null);
      }
    }
  }, [uploading]);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!customerId) {
      alert('Debes seleccionar un cliente antes de procesar.');
      return;
    }
    const selCustomer = customers.find(c => c.id === customerId);
    const hasAnalytics = selCustomer?.charts?.length > 0;
    const savedFilename = selCustomer?.log_config_filename;
    const configOk = (hasAnalytics && savedFilename && !updateConfig) || !!configFile;
    if (!logFile || !configOk) {
      alert('Please select both files');
      return;
    }
    setUploading(true);
    setProcessingTime(0);
    setFinalProcessingTime(null);

    const startTime = performance.now();

    try {
      const formData = new FormData();
      formData.append('log_file', logFile);
      if (configFile) formData.append('config_file', configFile);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5 * 60 * 1000);

      const customerParam = customerId ? `&customer_id=${encodeURIComponent(customerId)}` : '';
      const crossorterParam = `&crossorter_type=${crossorterType}`;
      const response = await fetch(`${API_BASE}/upload?server_type=${serverType}${customerParam}${crossorterParam}`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      const result = await response.json();
      if (response.ok) {
        const totalTime = (performance.now() - startTime) / 1000;
        onUploadSuccess({ ...result, total_time: totalTime });
      } else {
        alert('Error: ' + result.detail);
      }
    } catch (error) {
      // fetch aborted or network error
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-lg mx-auto">
      <div className="bg-white shadow rounded-lg p-4">
        <h2 className="text-lg font-medium text-gray-900 mb-6">Upload Log Files</h2>

        <form onSubmit={handleUpload} className="space-y-6">
          {/* Customer Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700">
              Customer <span className="text-red-500">*</span>
            </label>
            <CustomerCombobox
              value={customerId}
              onChange={setCustomerId}
              customers={customers}
            />
            {!customerId && (
              <p className="mt-1 text-xs text-red-500">Selecciona un cliente para continuar.</p>
            )}
          </div>

          {/* Server Type + Crossorter Type: solo para clientes sin analytics */}
          {customerId && !customers.find(c => c.id === customerId)?.charts?.length && (
            <div className="flex gap-10 pt-1">
              <div>
                <label className="text-base font-medium text-gray-900">Server Type</label>
                <div className="mt-4 space-x-6">
                  <label className="inline-flex items-center">
                    <input
                      type="radio"
                      value="eDS"
                      checked={serverType === 'eDS'}
                      onChange={(e) => setServerType(e.target.value)}
                      className="form-radio h-4 w-4 text-blue-600"
                    />
                    <span className="ml-2 font-semibold">eDS</span>
                  </label>
                  <label className="inline-flex items-center">
                    <input
                      type="radio"
                      value="SCNET"
                      checked={serverType === 'SCNET'}
                      onChange={(e) => setServerType(e.target.value)}
                      className="form-radio h-4 w-4 text-blue-600"
                    />
                    <span className="ml-2 font-semibold">SCNET</span>
                  </label>
                </div>
              </div>

              <div className="border-l border-gray-200 pl-10">
                <label className="text-base font-medium text-gray-900">Crossorter Type</label>
                <div className="mt-4 space-x-6">
                  <label className="inline-flex items-center">
                    <input
                      type="radio"
                      value="standard"
                      checked={crossorterType === 'standard'}
                      onChange={(e) => setCrossorterType(e.target.value)}
                      className="form-radio h-4 w-4 text-blue-600"
                    />
                    <span className="ml-2 font-semibold">Standard</span>
                  </label>
                  <label className="inline-flex items-center">
                    <input
                      type="radio"
                      value="xxl"
                      checked={crossorterType === 'xxl'}
                      onChange={(e) => setCrossorterType(e.target.value)}
                      className="form-radio h-4 w-4 text-blue-600"
                    />
                    <span className="ml-2 font-semibold">XXL</span>
                  </label>
                </div>
              </div>
            </div>
          )}

          {/* Log File Upload */}
          <div>
            <label className="block text-sm font-medium text-gray-700">
              Log File ({serverType === 'eDS' ? '.log' : '.FSC'})
            </label>
            <input
              type="file"
              accept={serverType === 'eDS' ? '.log' : '.fsc'}
              onChange={(e) => setLogFile(e.target.files[0])}
              ref={logInputRef}
              className="mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
          </div>

          {/* Config File */}
          {(() => {
            const selCustomer = customers.find(c => c.id === customerId);
            const hasAnalytics = selCustomer?.charts?.length > 0;
            const savedFilename = selCustomer?.log_config_filename;

            if (hasAnalytics && savedFilename && !updateConfig) {
              return (
                <div className="flex items-center gap-3 p-3 bg-green-50 border border-green-200 rounded-lg">
                  <span className="text-green-600 text-lg">✓</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-green-800">Config guardada</p>
                    <p className="text-xs text-green-600 truncate">{savedFilename}</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => { setUpdateConfig(true); setConfigFile(null); if (configInputRef.current) configInputRef.current.value = ''; }}
                    className="text-xs text-blue-600 hover:underline whitespace-nowrap"
                  >
                    Cambiar
                  </button>
                </div>
              );
            }

            return (
              <div>
                <div className="flex items-center justify-between">
                  <label className="block text-sm font-medium text-gray-700">
                    Config File ({serverType === 'eDS' ? '.json' : '.xml'})
                    {hasAnalytics && !savedFilename && (
                      <span className="ml-2 text-xs text-blue-600 font-normal">— se guardará para usos futuros</span>
                    )}
                  </label>
                  {hasAnalytics && savedFilename && updateConfig && (
                    <button
                      type="button"
                      onClick={() => { setUpdateConfig(false); setConfigFile(null); if (configInputRef.current) configInputRef.current.value = ''; }}
                      className="text-xs text-gray-500 hover:underline"
                    >
                      Cancelar
                    </button>
                  )}
                </div>
                <input
                  type="file"
                  accept={serverType === 'eDS' ? '.json' : '.xml'}
                  onChange={(e) => setConfigFile(e.target.files[0])}
                  ref={configInputRef}
                  className="mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                />
              </div>
            );
          })()}

          {/* Submit Button */}
          {(() => {
            const selCustomer = customers.find(c => c.id === customerId);
            const hasAnalytics = selCustomer?.charts?.length > 0;
            const savedFilename = selCustomer?.log_config_filename;
            const configOk = (hasAnalytics && savedFilename && !updateConfig) || !!configFile;
            return (
              <div>
                <button
                  type="submit"
                  disabled={uploading || !logFile || !configOk || !customerId}
                  className={
                    'w-full flex justify-center items-center py-2.5 px-8 rounded-lg text-base font-semibold transition-colors duration-200 ' +
                    (uploading
                      ? 'bg-gradient-to-r from-blue-400 via-blue-500 to-indigo-500 border-2 border-blue-300 shadow-lg text-white'
                      : (!logFile || !configOk || !customerId)
                        ? 'bg-gray-400 text-white cursor-not-allowed'
                        : 'bg-blue-600 hover:bg-blue-700 text-white border border-transparent shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500')
                  }
                  style={uploading ? {transition: 'background 0.3s'} : {}}
                >
                  {uploading ? (
                    <>
                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white mr-3"></div>
                      <span className="tracking-wide">Processing... {processingTime.toFixed(1)}s</span>
                    </>
                  ) : (
                    'Upload & Process'
                  )}
                </button>
              </div>
            );
          })()}
        </form>

        {/* Upload Status */}
        {uploadStatus && (
          <div className={`mt-6 p-4 rounded-md border ${uploadStatus.records_processed === 0 ? 'bg-amber-50 border-amber-300' : 'bg-green-50 border-green-200'}`}>
            <div className="flex">
              <div className="ml-3 w-full">
                <h3 className={`text-sm font-medium ${uploadStatus.records_processed === 0 ? 'text-amber-800' : 'text-green-800'}`}>
                  {uploadStatus.records_processed === 0 ? 'Procesamiento completado — 0 registros insertados' : 'Processing Complete!'}
                </h3>
                <div className={`mt-2 text-sm ${uploadStatus.records_processed === 0 ? 'text-amber-700' : 'text-green-700'}`}>
                  <p>Records processed: <strong>{uploadStatus.records_processed?.toLocaleString()}</strong></p>
                  <p>Total time: <strong>{uploadStatus.total_time?.toFixed(2)}s</strong></p>
                  <p>Table created: <strong>{uploadStatus.table_name}</strong></p>
                  <p>Database ID: <strong>{uploadStatus.database_id}</strong></p>
                </div>
                {uploadStatus.debug_info && (
                  <div className="mt-3 p-3 bg-amber-100 border border-amber-300 rounded text-xs font-mono text-amber-900 whitespace-pre-wrap break-all">
                    <p className="font-bold mb-1 font-sans">Diagnóstico del parser:</p>
                    {uploadStatus.debug_info}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
