import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Check, X, ShieldAlert } from 'lucide-react';
import { useStore } from '../store/useStore';

export default function ConfirmGate({ tick }) {
  const updateAuditConfirmation = useStore(state => state.updateAuditConfirmation);
  const addToast = useStore(state => state.addToast);
  
  const [decision, setDecision] = useState(null);
  
  useEffect(() => {
    setDecision(null);
  }, [tick?.t]);

  if (!tick || !tick.requires_confirmation) return null;
  
  const actionStr = tick.recommended_action?.action || 'Unknown Action';
  const conf = (tick.recommended_action?.confidence || 0) * 100;
  const reasons = tick.recommended_action?.reasons || [];

  const handleDecision = (type) => {
    setDecision(type);
    if (updateAuditConfirmation) updateAuditConfirmation(tick.t, type);
    if (addToast) {
      addToast({
        title: type === 'confirm' ? 'Action Confirmed' : 'Action Rejected',
        message: `Operator ${type}ed ${actionStr} at t=${tick.t}`,
        type: type === 'confirm' ? 'info' : 'warning'
      });
    }
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 50, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95, transition: { duration: 0.2 } }}
        className="glass-card"
        style={{
          padding: '1.5rem',
          border: '1px solid var(--operator, #6D5EF5)',
          background: 'linear-gradient(135deg, rgba(109, 94, 245, 0.1), rgba(109, 94, 245, 0.02))',
          boxShadow: '0 8px 32px rgba(109, 94, 245, 0.15)'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
          <ShieldAlert color="var(--operator, #6D5EF5)" size={24} />
          <h3 style={{ margin: 0, color: 'var(--operator, #6D5EF5)', fontSize: '1.1rem' }}>
            Operator Authorization Required
          </h3>
        </div>
        
        <div style={{ marginBottom: '1.5rem' }}>
          <p style={{ margin: '0 0 0.5rem 0', fontWeight: 600 }}>Action: {actionStr}</p>
          <p style={{ margin: '0 0 0.5rem 0', fontFamily: 'var(--font-mono)', fontSize: '0.9rem', opacity: 0.8 }}>
            AI Confidence: {conf.toFixed(1)}%
          </p>
          {reasons.length > 0 && (
            <ul style={{ margin: '0 0 0.5rem 0', paddingLeft: '1.25rem', fontSize: '0.9rem', opacity: 0.8 }}>
              {reasons.map((r, idx) => <li key={idx}>{r}</li>)}
            </ul>
          )}
        </div>

        {!decision ? (
          <div style={{ display: 'flex', gap: '1rem' }}>
            <button
              onClick={() => handleDecision('confirm')}
              style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem',
                padding: '0.75rem', background: 'var(--operator, #6D5EF5)', color: '#fff', border: 'none',
                borderRadius: '8px', cursor: 'pointer', fontWeight: 600
              }}
            >
              <Check size={18} /> Confirm
            </button>
            <button
              onClick={() => handleDecision('reject')}
              style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem',
                padding: '0.75rem', background: 'transparent', color: 'var(--critical)', 
                border: '1px solid var(--critical)', borderRadius: '8px', cursor: 'pointer', fontWeight: 600
              }}
            >
              <X size={18} /> Reject
            </button>
          </div>
        ) : (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            style={{ textAlign: 'center', padding: '0.5rem', color: decision === 'confirm' ? 'var(--healthy)' : 'var(--critical)' }}
          >
            Decision recorded: {decision.toUpperCase()}
          </motion.div>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
