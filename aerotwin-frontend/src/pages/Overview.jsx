import React from 'react';
import { motion } from 'framer-motion';
import { useStore } from '../store/useStore';
import { PROJECT_NAME, QUICK_SCENARIOS, SENSOR_METADATA } from '../config';
import GlassCard from '../components/GlassCard';
import HealthOrb from '../components/HealthOrb';
import MissionBanner from '../components/MissionBanner';
import AnimatedNumber from '../components/AnimatedNumber';
import { Activity, Clock, ShieldAlert } from 'lucide-react';

export default function Overview() {
  const { currentTick, auditTrail, setScenario } = useStore();

  if (!currentTick) return null;

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-6 space-y-6"
    >
      <header className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-wider">{PROJECT_NAME} Command Centre</h1>
          <p className="text-slate-400 mt-1">Live simulated stream & health monitoring</p>
        </div>
      </header>

      {/* Hero Section */}
      <section className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mb-8">
        <GlassCard className="flex flex-col items-center justify-center p-8 min-h-[350px]">
          <h2 className="text-xl font-semibold mb-6 text-slate-300">Engine Health</h2>
          <HealthOrb 
            healthScore={currentTick.health_score} 
            severity={currentTick.severity} 
            action={currentTick.action} 
          />
        </GlassCard>
        
        <div className="flex flex-col gap-6 xl:col-span-2">
          <MissionBanner tick={currentTick} />
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 h-full">
            <GlassCard className="flex flex-col justify-center p-6 h-full border-t-4 border-t-purple-500/50">
              <div className="flex items-center gap-2 mb-2">
                <ShieldAlert className="w-5 h-5 text-purple-400" />
                <h3 className="text-sm font-medium text-slate-300 uppercase tracking-widest">Diagnosis</h3>
              </div>
              <div className="text-2xl font-bold mb-1 truncate" title={currentTick.predicted_fault || 'None'}>
                {currentTick.predicted_fault || 'None'}
              </div>
              <div className="flex justify-between items-end mt-auto">
                <span className="text-xs text-slate-400">Confidence</span>
                <span className="mono text-lg text-purple-300">{(currentTick.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="mt-4 inline-flex px-2 py-1 rounded text-xs border border-white/10 bg-white/5 w-fit">
                {currentTick.condition_status}
              </div>
            </GlassCard>

            <GlassCard className="flex flex-col justify-center p-6 h-full border-t-4 border-t-teal-500/50">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="w-5 h-5 text-teal-400" />
                <h3 className="text-sm font-medium text-slate-300 uppercase tracking-widest">Remaining Life</h3>
              </div>
              <div className="flex items-baseline gap-2 mb-1 mt-auto">
                <AnimatedNumber value={currentTick.rul_estimate_timesteps} className="text-5xl font-bold mono" />
                <span className="text-sm text-slate-400">timesteps</span>
              </div>
              <div className="text-xs text-slate-400 mt-2">
                Lower bound: <span className="mono text-slate-300">{currentTick.rul_lower_bound_timesteps}</span> timesteps
              </div>
            </GlassCard>
          </div>
        </div>
      </section>

      {/* Sensor Row */}
      <section className="mb-8">
        <h3 className="text-lg font-medium text-slate-300 mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5" /> Live Sensors
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
          {Object.entries(currentTick.sensors).map(([key, value]) => {
            const meta = SENSOR_METADATA[key];
            const isTrusted = currentTick.trust ? currentTick.trust[key] : true;
            return (
              <GlassCard key={key} className="p-3 flex flex-col justify-between">
                <div className="flex justify-between items-start mb-2">
                  <span className="text-xs font-medium text-slate-400">{meta?.shortName || key}</span>
                  <div className={`w-2 h-2 rounded-full ${isTrusted ? 'bg-teal-500' : 'bg-amber-500'}`} title={isTrusted ? 'Trusted' : 'Untrusted'} />
                </div>
                <div className="mono text-lg font-semibold">{value.toFixed(1)}</div>
              </GlassCard>
            );
          })}
        </div>
      </section>

      {/* Quick Scenarios */}
      <section className="mb-8">
        <h3 className="text-lg font-medium text-slate-300 mb-4">Quick Scenarios</h3>
        <div className="flex flex-wrap gap-3">
          {QUICK_SCENARIOS.map((scenario, idx) => (
            <button
              key={idx}
              onClick={() => setScenario(scenario)}
              className="px-4 py-2 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 transition-colors text-sm font-medium"
            >
              {scenario.fault} ({scenario.severity})
            </button>
          ))}
        </div>
      </section>

      {/* Activity Ticker */}
      <section>
        <h3 className="text-lg font-medium text-slate-300 mb-4">Activity Log</h3>
        <div className="flex flex-col gap-2">
          {auditTrail.slice(0, 5).map((log, idx) => (
            <GlassCard key={idx} className="p-3 text-sm flex items-center gap-4">
              <span className="mono text-slate-400 w-24">Tick {log.tick}</span>
              <span className="flex-1">{log.message}</span>
              <span className="text-xs text-slate-500">{new Date(log.timestamp).toLocaleTimeString()}</span>
            </GlassCard>
          ))}
          {auditTrail.length === 0 && (
            <div className="text-slate-500 text-sm italic">No recent activity.</div>
          )}
        </div>
      </section>
    </motion.div>
  );
}
