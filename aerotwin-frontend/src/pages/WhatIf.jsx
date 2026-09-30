import React from 'react';
import { useStore } from '../store/useStore';
import { FAULT_TYPES, SENSOR_KEYS, SENSOR_METADATA, MISSION_ACTIONS } from '../config';
import { generateFlight } from '../data/simSource';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts';
import { motion } from 'framer-motion';
import { Play } from 'lucide-react';

const WhatIf = () => {
  const {
    flightTicks,
    whatIfScenario,
    setWhatIfScenario,
    whatIfData,
    setWhatIfData,
    whatIfSensor,
    setWhatIfSensor,
    whatIfLoading,
    setWhatIfLoading,
    addToast
  } = useStore();

  const handleRun = async () => {
    setWhatIfLoading(true);
    try {
      // Small artificial delay for effect
      await new Promise(resolve => setTimeout(resolve, 800));
      const newData = generateFlight(whatIfScenario);
      setWhatIfData(newData);
      addToast('Counterfactual flight generated', 'success');
    } catch (e) {
      addToast('Error generating flight', 'error');
    } finally {
      setWhatIfLoading(false);
    }
  };

  const getCombinedData = (sensorKey) => {
    if (!flightTicks || flightTicks.length === 0 || !whatIfData) return [];
    const len = Math.max(flightTicks.length, whatIfData.length);
    const combined = [];
    for (let i = 0; i < len; i++) {
      const currentTick = flightTicks[i];
      const counterTick = whatIfData[i];
      const dataPoint = { tick: i };
      
      if (currentTick) {
        dataPoint.currentSensor = currentTick.sensors[sensorKey];
        dataPoint.currentHealth = currentTick.health_score;
        dataPoint.currentRul = currentTick.rul_estimate_timesteps;
      }
      if (counterTick) {
        dataPoint.counterSensor = counterTick.sensors[sensorKey];
        dataPoint.counterHealth = counterTick.health_score;
        dataPoint.counterRul = counterTick.rul_estimate_timesteps;
      }
      combined.push(dataPoint);
    }
    return combined;
  };

  const combinedData = getCombinedData(whatIfSensor);
  const currentFinal = flightTicks && flightTicks.length > 0 ? flightTicks[flightTicks.length - 1] : null;
  const counterFinal = whatIfData && whatIfData.length > 0 ? whatIfData[whatIfData.length - 1] : null;

  const tooltipStyle = { background: '#111A33', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#fff' };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-6 space-y-6"
    >
      <header>
        <h1 className="text-2xl font-bold mb-2">What-If Analysis</h1>
        <p className="text-gray-400">Simulate counterfactual scenarios and compare outcomes against the current live flight.</p>
      </header>

      <div className="glass-card p-6">
        <h2 className="text-lg font-semibold mb-4">Scenario Configuration</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 items-end">
          <div>
            <label className="block text-sm text-gray-400 mb-2">Injected Fault</label>
            <select 
              className="glass-select w-full"
              value={whatIfScenario.faultType}
              onChange={(e) => setWhatIfScenario({ ...whatIfScenario, faultType: e.target.value })}
            >
              <option value="none">None (Healthy)</option>
              {Object.entries(FAULT_TYPES).map(([k, v]) => (
                <option key={k} value={k}>{v.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">Severity ({whatIfScenario.severity})</label>
            <input 
              type="range"
              className="glass-slider w-full"
              min="0"
              max="1"
              step="0.05"
              value={whatIfScenario.severity}
              onChange={(e) => setWhatIfScenario({ ...whatIfScenario, severity: parseFloat(e.target.value) })}
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">Seed (Optional)</label>
            <input 
              type="text"
              className="glass-input w-full"
              placeholder="Random"
              value={whatIfScenario.seed || ''}
              onChange={(e) => setWhatIfScenario({ ...whatIfScenario, seed: e.target.value })}
            />
          </div>
          <div>
            <button 
              className="glass-btn glass-btn--primary w-full flex items-center justify-center gap-2"
              onClick={handleRun}
              disabled={whatIfLoading}
            >
              {whatIfLoading ? 'Simulating...' : <><Play size={16} /> Run Simulation</>}
            </button>
          </div>
        </div>
      </div>

      {whatIfData && (
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="space-y-6"
        >
          <div className="glass-card p-6">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-lg font-semibold">Sensor Comparison</h2>
              <select 
                className="glass-select w-64"
                value={whatIfSensor}
                onChange={(e) => setWhatIfSensor(e.target.value)}
              >
                {SENSOR_KEYS.map(key => (
                  <option key={key} value={key}>{SENSOR_METADATA[key].label} ({SENSOR_METADATA[key].unit})</option>
                ))}
              </select>
            </div>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={combinedData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="tick" stroke="rgba(255,255,255,0.3)" tick={{fill: '#9CA3AF', fontSize: 12}} />
                  <YAxis stroke="rgba(255,255,255,0.3)" tick={{fill: '#9CA3AF', fontSize: 12, fontFamily: 'JetBrains Mono'}} domain={['auto', 'auto']} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend />
                  <Line type="monotone" name="Current Live" dataKey="currentSensor" stroke="#14B8A6" strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Line type="monotone" name="Counterfactual" dataKey="counterSensor" stroke="#F59E0B" strokeWidth={2} strokeDasharray="5 5" dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="glass-card p-6">
              <h2 className="text-lg font-semibold mb-4">Health Score Comparison</h2>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={combinedData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="tick" stroke="rgba(255,255,255,0.3)" tick={{fill: '#9CA3AF', fontSize: 12}} />
                    <YAxis stroke="rgba(255,255,255,0.3)" tick={{fill: '#9CA3AF', fontSize: 12, fontFamily: 'JetBrains Mono'}} domain={[0, 100]} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Legend />
                    <Line type="monotone" name="Current Live" dataKey="currentHealth" stroke="#14B8A6" strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line type="monotone" name="Counterfactual" dataKey="counterHealth" stroke="#F59E0B" strokeWidth={2} strokeDasharray="5 5" dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="glass-card p-6">
              <h2 className="text-lg font-semibold mb-4">RUL Comparison (timesteps)</h2>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={combinedData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="tick" stroke="rgba(255,255,255,0.3)" tick={{fill: '#9CA3AF', fontSize: 12}} />
                    <YAxis stroke="rgba(255,255,255,0.3)" tick={{fill: '#9CA3AF', fontSize: 12, fontFamily: 'JetBrains Mono'}} domain={[0, 'auto']} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Legend />
                    <Line type="monotone" name="Current Live" dataKey="currentRul" stroke="#14B8A6" strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line type="monotone" name="Counterfactual" dataKey="counterRul" stroke="#F59E0B" strokeWidth={2} strokeDasharray="5 5" dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="glass-card p-6 relative overflow-hidden">
              <h3 className="text-sm text-gray-400 mb-2 uppercase tracking-wider">Current Live Outcome</h3>
              <div className="text-2xl font-bold" style={{ color: currentFinal ? MISSION_ACTIONS[currentFinal.action]?.color : '#fff' }}>
                {currentFinal?.action || 'UNKNOWN'}
              </div>
              <div className="mt-4">
                <div className="text-sm text-gray-400">Final RUL</div>
                <div className="text-xl mono">{currentFinal?.rul_estimate_timesteps || 0} timesteps</div>
              </div>
            </div>
            <div className="glass-card p-6 relative overflow-hidden border border-orange-500/20">
              <h3 className="text-sm text-gray-400 mb-2 uppercase tracking-wider">Counterfactual Outcome</h3>
              <div className="text-2xl font-bold" style={{ color: counterFinal ? MISSION_ACTIONS[counterFinal.action]?.color : '#fff' }}>
                {counterFinal?.action || 'UNKNOWN'}
              </div>
              <div className="mt-4">
                <div className="text-sm text-gray-400">Final RUL</div>
                <div className="text-xl mono">{counterFinal?.rul_estimate_timesteps || 0} timesteps</div>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default WhatIf;
