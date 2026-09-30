import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';

const AnimatedNumber = ({ value }) => (
  <motion.span 
    key={value}
    initial={{ opacity: 0.5, y: -5 }}
    animate={{ opacity: 1, y: 0 }}
    style={{ display: 'inline-block' }}
  >
    {Math.round(value)}
  </motion.span>
);

export default function RulBand({ tick, history }) {
  if (!tick) return null;

  const rul = tick.rul_estimate_timesteps ?? 0;
  const lower = tick.rul_lower_bound ?? (rul * 0.8);
  const upper = tick.rul_upper_bound ?? (rul * 1.2);
  const recovered = tick.recovered || false;
  const recoveryRate = tick.recovery_rate || 0;

  const data = history?.map(h => ({
    t: h.t,
    rul: h.rul_estimate_timesteps,
    lower: h.rul_lower_bound || (h.rul_estimate_timesteps * 0.8),
    upper: h.rul_upper_bound || (h.rul_estimate_timesteps * 1.2)
  })) || [];

  return (
    <div className="glass-card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h3 style={{ margin: '0 0 0.5rem 0', opacity: 0.8, fontSize: '1rem' }}>Remaining Useful Life</h3>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '2.5rem', fontWeight: 700, color: 'var(--healthy)' }}>
              <AnimatedNumber value={rul} />
            </span>
            <span style={{ opacity: 0.6 }}>timesteps</span>
          </div>
          <div style={{ fontSize: '0.85rem', opacity: 0.7, marginTop: '0.25rem' }}>
            Lower bound: <span style={{ fontFamily: 'var(--font-mono)' }}>{Math.round(lower)}</span>
          </div>
        </div>
        
        {recovered && (
          <div style={{ background: 'rgba(20, 184, 166, 0.1)', border: '1px solid var(--healthy)', padding: '0.5rem 1rem', borderRadius: '8px', color: 'var(--healthy)' }}>
            Recovering (+{recoveryRate.toFixed(2)} / step)
          </div>
        )}
      </div>

      <div style={{ height: '200px', width: '100%', marginTop: '1rem' }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
            <XAxis dataKey="t" stroke="rgba(255,255,255,0.3)" tick={{ fontSize: 12, fill: 'rgba(255,255,255,0.5)' }} />
            <YAxis stroke="rgba(255,255,255,0.3)" tick={{ fontSize: 12, fill: 'rgba(255,255,255,0.5)', fontFamily: 'var(--font-mono)' }} />
            <Tooltip
              contentStyle={{ background: 'var(--glass-bg)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', backdropFilter: 'blur(10px)' }}
              itemStyle={{ fontFamily: 'var(--font-mono)' }}
            />
            <Area type="monotone" dataKey="upper" stroke="none" fill="rgba(255,255,255,0.05)" />
            <Area type="monotone" dataKey="lower" stroke="none" fill="#0B1020" />
            <Area type="monotone" dataKey="rul" stroke="var(--healthy)" strokeWidth={2} fill="url(#colorRul)" />
            <defs>
              <linearGradient id="colorRul" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--healthy)" stopOpacity={0.3} />
                <stop offset="95%" stopColor="var(--healthy)" stopOpacity={0} />
              </linearGradient>
            </defs>
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div style={{ fontSize: '0.75rem', opacity: 0.5, fontStyle: 'italic' }}>
        Note: RUL is least reliable near end of life.
      </div>
    </div>
  );
}
