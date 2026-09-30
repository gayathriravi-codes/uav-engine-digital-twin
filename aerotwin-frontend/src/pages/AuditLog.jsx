import React, { useState } from 'react';
import { useStore } from '../store/useStore';
import { MISSION_ACTIONS } from '../config';
import { Download, ChevronDown, ChevronUp, Search, FileX } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const AuditLog = () => {
  const { auditTrail } = useStore();
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('All');
  const [expandedRows, setExpandedRows] = useState(new Set());

  const toggleRow = (id) => {
    const newSet = new Set(expandedRows);
    if (newSet.has(id)) newSet.delete(id);
    else newSet.add(id);
    setExpandedRows(newSet);
  };

  const filteredData = auditTrail.filter(log => {
    const matchFilter = filter === 'All' || log.action === filter;
    const matchSearch = JSON.stringify(log).toLowerCase().includes(search.toLowerCase());
    return matchFilter && matchSearch;
  });

  const handleExport = () => {
    const lines = filteredData.map(log => JSON.stringify(log)).join('\n');
    const blob = new Blob([lines], { type: 'application/jsonl' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `nidan-audit-log-${new Date().toISOString()}.jsonl`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-6 space-y-6 flex flex-col h-full"
    >
      <header className="flex justify-between items-end">
        <div>
          <h1 className="text-2xl font-bold mb-2">Audit Log</h1>
          <p className="text-gray-400">Immutable record of all system state changes and operator confirmations.</p>
        </div>
        <button className="glass-btn flex items-center gap-2" onClick={handleExport}>
          <Download size={16} /> Export JSONL
        </button>
      </header>

      <div className="glass-card p-4 flex flex-col md:flex-row gap-4 items-center">
        <div className="relative flex-1 w-full">
          <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input 
            type="text"
            className="glass-input pl-10 w-full"
            placeholder="Search audit logs..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <div className="flex flex-wrap gap-2">
          {['All', ...Object.keys(MISSION_ACTIONS)].map(act => (
            <button
              key={act}
              className={`glass-chip ${filter === act ? 'bg-white/20 ring-1 ring-white/30' : ''}`}
              onClick={() => setFilter(act)}
            >
              {MISSION_ACTIONS[act]?.label || act}
            </button>
          ))}
        </div>
      </div>

      <div className="glass-card overflow-hidden flex-1 flex flex-col">
        <div className="overflow-x-auto">
          <table className="glass-table w-full text-left border-collapse">
            <thead>
              <tr className="bg-white/5 border-b border-white/10 text-gray-400 text-sm">
                <th className="p-4 font-medium">Tick</th>
                <th className="p-4 font-medium">Time</th>
                <th className="p-4 font-medium">Action</th>
                <th className="p-4 font-medium">Fault</th>
                <th className="p-4 font-medium">Operator</th>
                <th className="p-4 font-medium w-8"></th>
              </tr>
            </thead>
            <tbody>
              <AnimatePresence>
                {filteredData.length > 0 ? (
                  filteredData.map(log => {
                    const isExpanded = expandedRows.has(log.id);
                    const color = MISSION_ACTIONS[log.action]?.color || '#fff';
                    return (
                      <React.Fragment key={log.id}>
                        <motion.tr 
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                          className="border-b border-white/5 hover:bg-white/5 cursor-pointer transition-colors"
                          onClick={() => toggleRow(log.id)}
                        >
                          <td className="p-4 mono text-sm">{log.tick}</td>
                          <td className="p-4 mono text-sm text-gray-400">{new Date(log.timestamp).toLocaleTimeString()}</td>
                          <td className="p-4 font-medium" style={{ color }}>{log.action}</td>
                          <td className="p-4">{log.fault} <span className="mono text-xs text-gray-400">({(log.confidence*100).toFixed(0)}%)</span></td>
                          <td className="p-4">
                            {log.actionChanged ? (
                              <span className="text-purple-400 text-xs border border-purple-500/30 bg-purple-500/10 px-2 py-1 rounded">OVERRIDE</span>
                            ) : (
                              <span className="text-gray-400 text-xs border border-white/10 bg-white/5 px-2 py-1 rounded">AUTO</span>
                            )}
                          </td>
                          <td className="p-4">
                            {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                          </td>
                        </motion.tr>
                        {isExpanded && (
                          <tr>
                            <td colSpan={6} className="p-0 border-b border-white/5">
                              <motion.div 
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                className="bg-black/20 p-4 pl-12 space-y-2 overflow-hidden"
                              >
                                <div className="text-sm font-semibold mb-2">Decision Reasons:</div>
                                <ul className="list-disc pl-4 text-sm text-gray-300 space-y-1">
                                  {log.reasons.map((r, i) => <li key={i}>{r}</li>)}
                                </ul>
                                <div className="mt-4 text-xs text-gray-500 mono">
                                  Log ID: {log.id} | Mission Clock: {log.missionClock}s
                                </div>
                              </motion.div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan={6} className="p-12 text-center text-gray-500">
                      <FileX size={48} className="mx-auto mb-4 opacity-50" />
                      <p>No audit logs found.</p>
                    </td>
                  </tr>
                )}
              </AnimatePresence>
            </tbody>
          </table>
        </div>
      </div>
    </motion.div>
  );
};

export default AuditLog;
