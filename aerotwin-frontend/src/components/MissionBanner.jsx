import React from 'react';
import { motion } from 'framer-motion';
import { MISSION_ACTIONS } from '../config';
import { ShieldAlert, ShieldCheck, AlertTriangle, XCircle, Zap } from 'lucide-react';

const icons = {
  CONTINUE: ShieldCheck,
  REDUCE_LOAD: Zap,
  RTB: AlertTriangle,
  ABORT: XCircle
};

const colors = {
  CONTINUE: 'var(--healthy)',
  REDUCE_LOAD: 'var(--watch)',
  RTB: '#F97316', // orange
  ABORT: 'var(--critical)'
};

export default function MissionBanner({ tick }) {
  if (!tick || !tick.recommended_action) return null;
  
  const actionKey = tick.recommended_action.action || 'CONTINUE';
  const config = MISSION_ACTIONS ? MISSION_ACTIONS[actionKey] : null;
  
  const label = config?.label || actionKey;
  const description = tick.recommended_action.description || '';
  const reasons = tick.recommended_action.reasons || [];
  const risk = tick.recommended_action.risk_score || 0;
  
  const Icon = icons[actionKey] || ShieldAlert;
  const color = colors[actionKey] || colors.CONTINUE;
  const isAlert = actionKey === 'ABORT' || actionKey === 'RTB';

  return (
    <motion.div 
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      key={tick.action_changed ? `changed-${tick.t}` : 'banner'}
      className="glass-card"
      style={{
        padding: '1.5rem',
        border: `1px solid ${color}`,
        background: isAlert ? `linear-gradient(135deg, rgba(255,255,255,0.03), ${color}22)` : 'var(--glass-bg)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
        <motion.div
          animate={tick.action_changed ? { scale: [1, 1.2, 1], rotate: [0, -10, 10, 0] } : {}}
          transition={{ duration: 0.5 }}
        >
          <Icon size={32} color={color} />
        </motion.div>
        <div>
          <h2 style={{ color, margin: 0, fontSize: '1.5rem', fontWeight: 600 }}>{label}</h2>
          <p style={{ margin: 0, opacity: 0.8, fontSize: '0.9rem' }}>{description}</p>
        </div>
        <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
          <div style={{ fontSize: '0.8rem', opacity: 0.7, textTransform: 'uppercase' }}>Risk Score</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.5rem', fontWeight: 700, color }}>
            {risk.toFixed(2)}
          </div>
        </div>
      </div>
      
      {reasons.length > 0 && (
        <ul style={{ margin: 0, paddingLeft: '1.5rem', opacity: 0.8, fontSize: '0.9rem' }}>
          {reasons.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
      )}
    </motion.div>
  );
}
