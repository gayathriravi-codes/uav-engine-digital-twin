import React from 'react';
import { motion } from 'framer-motion';
import { useStore } from '../store/useStore';
import GlassCard from '../components/GlassCard';
import RulBand from '../components/RulBand';
import AnimatedNumber from '../components/AnimatedNumber';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { AlertTriangle, Clock, TrendingUp } from 'lucide-react';

export default function RemainingLife() {
  const { currentTick, tickHistory } = useStore();

  if (!currentTick) return null;

  const trendData = tickHistory.map(t => ({
    tick: t.t,
    estimate: t.rul_estimate_timesteps,
    lowerBound: t.rul_lower_bound_timesteps,
    range: [t.rul_lower_bound_timesteps, t.rul_estimate_timesteps]
  }));

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-6 space-y-8"
    >
      <header>
        <h1 className="text-2xl font-bold tracking-wider mb-2">Remaining Useful Life (RUL)</h1>
        <p className="text-slate-400">Prognostics and failure horizon estimation.</p>
      </header>

      {/* Main RUL Display */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <GlassCard className="col-span-2 p-8 flex flex-col justify-center items-center relative overflow-hidden">
          <div className="absolute top-4 left-4 flex items-center gap-2 text-teal-400">
            <Clock className="w-5 h-5" />
            <span className="font-semibold tracking-wider">ESTIMATE</span>
          </div>
          
          <div className="flex items-baseline gap-4 mt-6">
            <AnimatedNumber 
              value={currentTick.rul_estimate_timesteps} 
              className="text-7xl font-bold mono text-white drop-shadow-[0_0_15px_rgba(255,255,255,0.2)]" 
            />
            <span className="text-xl text-slate-400">timesteps</span>
          </div>
          
          <div className="mt-6 flex items-center gap-8 text-sm">
            <div className="flex flex-col items-center">
              <span className="text-slate-500 mb-1">Lower Bound</span>
              <span className="mono font-semibold text-slate-300 text-lg">
                {currentTick.rul_lower_bound_timesteps}
              </span>
            </div>
            <div className="flex flex-col items-center">
              <span className="text-slate-500 mb-1">Uncertainty</span>
              <span className="mono font-semibold text-slate-300 text-lg">
                ±{currentTick.rul_uncertainty_timesteps}
              </span>
            </div>
          </div>
        </GlassCard>

        <div className="flex flex-col gap-6">
          <GlassCard className="p-6 flex-1 flex flex-col justify-center">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="w-5 h-5 text-purple-400" />
              <h3 className="font-semibold text-slate-200">Recovery Status</h3>
            </div>
            <div className="flex justify-between items-end mb-2">
              <span className="text-sm text-slate-400">Recovery Rate</span>
              <span className="mono text-lg text-purple-300">{(currentTick.recovery_rate * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-white/10 h-2 rounded-full overflow-hidden">
              <div 
                className="bg-purple-500 h-full rounded-full transition-all duration-500" 
                style={{ width: `${Math.min(100, Math.max(0, currentTick.recovery_rate * 100))}%` }}
              />
            </div>
            {currentTick.recovered && (
              <div className="mt-4 px-3 py-2 bg-teal-500/20 border border-teal-500/30 rounded text-teal-300 text-sm font-medium text-center">
                System Recovered
              </div>
            )}
          </GlassCard>

          <GlassCard className="p-4 bg-amber-500/10 border-amber-500/30">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-amber-200/80 leading-relaxed">
                RUL estimates become more volatile and potentially less reliable near the absolute end of life. Maintain conservative operating margins.
              </p>
            </div>
          </GlassCard>
        </div>
      </div>

      {/* RUL Band Visualization */}
      <GlassCard className="p-6">
        <h3 className="text-lg font-semibold mb-6">Prognostic Horizon</h3>
        <RulBand tick={currentTick} history={tickHistory} />
      </GlassCard>

      {/* Trend Chart */}
      <GlassCard className="p-6">
        <h3 className="text-lg font-semibold mb-6">Historical RUL Trend</h3>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={trendData}>
              <defs>
                <linearGradient id="colorEstimate" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#14B8A6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#14B8A6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="tick" stroke="rgba(255,255,255,0.4)" tick={{ fontSize: 12, fill: '#94a3b8' }} />
              <YAxis stroke="rgba(255,255,255,0.4)" tick={{ fontSize: 12, fill: '#94a3b8', fontFamily: 'JetBrains Mono' }} />
              <Tooltip 
                contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                itemStyle={{ fontFamily: 'JetBrains Mono' }}
                labelStyle={{ color: '#94a3b8' }}
              />
              <Area 
                type="monotone" 
                dataKey="range" 
                stroke="none" 
                fill="#14B8A6" 
                fillOpacity={0.1} 
              />
              <Area 
                type="monotone" 
                dataKey="estimate" 
                stroke="#14B8A6" 
                strokeWidth={2}
                fillOpacity={1} 
                fill="url(#colorEstimate)" 
              />
              <Area 
                type="monotone" 
                dataKey="lowerBound" 
                stroke="#F59E0B" 
                strokeDasharray="5 5"
                fill="none" 
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </GlassCard>
    </motion.div>
  );
}
