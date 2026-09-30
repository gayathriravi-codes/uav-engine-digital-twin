import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '../store/useStore';
import { SENSOR_METADATA } from '../config';
import GlassCard from '../components/GlassCard';
import RadialGauge from '../components/RadialGauge';
import Sparkline from '../components/Sparkline';
import TrustBadge from '../components/TrustBadge';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { X } from 'lucide-react';

export default function LiveMonitor() {
  const { currentTick, tickHistory, selectedSensorModal, setSelectedSensorModal } = useStore();
  const [visibleSensors, setVisibleSensors] = useState(() => {
    return Object.keys(SENSOR_METADATA).reduce((acc, key) => ({ ...acc, [key]: true }), {});
  });

  if (!currentTick) return null;

  const toggleSensor = (key) => {
    setVisibleSensors(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const selectedSensorMeta = selectedSensorModal ? SENSOR_METADATA[selectedSensorModal] : null;

  const getHistoryData = (sensorKey, limit = 60) => {
    return tickHistory.slice(-limit).map(t => ({
      time: t.t,
      value: t.sensors[sensorKey]
    }));
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-6 space-y-8"
    >
      <header>
        <h1 className="text-2xl font-bold tracking-wider mb-2">Live Monitor</h1>
        <p className="text-slate-400">Near real-time sensor telemetry.</p>
      </header>

      {/* Sensor Gauges Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {Object.entries(currentTick.sensors).map(([key, value]) => {
          const meta = SENSOR_METADATA[key];
          const isTrusted = currentTick.trust ? currentTick.trust[key] : true;
          const history = getHistoryData(key);

          return (
            <GlassCard 
              key={key} 
              className="p-4 cursor-pointer hover:bg-white/10 transition-colors"
              onClick={() => setSelectedSensorModal(key)}
            >
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="font-semibold text-lg">{meta?.name || key}</h3>
                  <p className="text-xs text-slate-400">{meta?.unit}</p>
                </div>
                <TrustBadge trusted={isTrusted} />
              </div>
              
              <div className="flex justify-between items-center">
                <RadialGauge value={value} min={meta?.min || 0} max={meta?.max || 100} />
                <div className="w-1/2 h-16">
                  <Sparkline data={history} dataKey="value" color={isTrusted ? "#14B8A6" : "#F59E0B"} />
                </div>
              </div>
            </GlassCard>
          );
        })}
      </div>

      {/* Multi-sensor Chart */}
      <GlassCard className="p-6">
        <h3 className="text-lg font-semibold mb-4">Combined Telemetry</h3>
        <div className="flex flex-wrap gap-2 mb-6">
          {Object.entries(SENSOR_METADATA).map(([key, meta]) => (
            <button
              key={key}
              onClick={() => toggleSensor(key)}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-colors border ${
                visibleSensors[key] 
                  ? 'bg-purple-500/20 border-purple-500/50 text-purple-200' 
                  : 'bg-white/5 border-white/10 text-slate-400'
              }`}
            >
              {meta.shortName}
            </button>
          ))}
        </div>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={tickHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="t" stroke="rgba(255,255,255,0.5)" tick={{ fontSize: 12, fill: '#94a3b8' }} />
              <YAxis stroke="rgba(255,255,255,0.5)" tick={{ fontSize: 12, fill: '#94a3b8' }} />
              <Tooltip 
                contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                itemStyle={{ fontFamily: 'JetBrains Mono', fontSize: '12px' }}
              />
              {Object.keys(SENSOR_METADATA).map((key, i) => visibleSensors[key] && (
                <Line 
                  key={key}
                  type="monotone" 
                  dataKey={`sensors.${key}`} 
                  stroke={`hsl(${i * 50}, 70%, 60%)`} 
                  dot={false}
                  strokeWidth={2}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </GlassCard>

      {/* Sensor Modal */}
      <AnimatePresence>
        {selectedSensorModal && selectedSensorMeta && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSelectedSensorModal(null)}
              className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            />
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="relative w-full max-w-3xl glass-card rounded-2xl p-6 shadow-2xl"
            >
              <button 
                onClick={() => setSelectedSensorModal(null)}
                className="absolute top-4 right-4 text-slate-400 hover:text-white"
              >
                <X className="w-6 h-6" />
              </button>
              
              <div className="mb-6">
                <h2 className="text-2xl font-bold">{selectedSensorMeta.name}</h2>
                <p className="text-slate-400">{selectedSensorMeta.description}</p>
                <div className="mt-4 flex items-end gap-4">
                  <span className="mono text-4xl font-bold text-teal-400">
                    {currentTick.sensors[selectedSensorModal].toFixed(2)}
                  </span>
                  <span className="text-slate-400 mb-1">{selectedSensorMeta.unit}</span>
                  <TrustBadge trusted={currentTick.trust ? currentTick.trust[selectedSensorModal] : true} />
                </div>
              </div>

              <div className="h-64 w-full mb-6">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={getHistoryData(selectedSensorModal, 120)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" tick={{ fontSize: 12 }} />
                    <YAxis domain={['auto', 'auto']} stroke="rgba(255,255,255,0.5)" tick={{ fontSize: 12, fontFamily: 'JetBrains Mono' }} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(255,255,255,0.1)' }}
                      labelStyle={{ color: '#94a3b8' }}
                    />
                    <Line type="monotone" dataKey="value" stroke="#14B8A6" strokeWidth={3} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="grid grid-cols-3 gap-4 text-sm">
                <div className="bg-white/5 p-3 rounded-lg border border-white/10">
                  <div className="text-slate-400 mb-1">Normal Range</div>
                  <div className="mono font-semibold">{selectedSensorMeta.normalRange?.[0] ?? 'N/A'} - {selectedSensorMeta.normalRange?.[1] ?? 'N/A'}</div>
                </div>
                <div className="bg-white/5 p-3 rounded-lg border border-white/10">
                  <div className="text-slate-400 mb-1">Warning Limits</div>
                  <div className="mono font-semibold text-amber-400">{selectedSensorMeta.warningRange?.[0] ?? 'N/A'} - {selectedSensorMeta.warningRange?.[1] ?? 'N/A'}</div>
                </div>
                <div className="bg-white/5 p-3 rounded-lg border border-white/10">
                  <div className="text-slate-400 mb-1">Physical Limits</div>
                  <div className="mono font-semibold text-red-400">{selectedSensorMeta.min} - {selectedSensorMeta.max}</div>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
