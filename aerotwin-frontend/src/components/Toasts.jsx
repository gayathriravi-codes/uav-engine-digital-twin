import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Info, AlertTriangle, AlertCircle, X } from 'lucide-react';
import { useStore } from '../store/useStore';

const icons = {
  info: Info,
  warning: AlertTriangle,
  critical: AlertCircle
};

const colors = {
  info: 'var(--healthy)',
  warning: 'var(--watch)',
  critical: 'var(--critical)'
};

const Toast = ({ toast, onRemove }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      if (onRemove) onRemove(toast.id);
    }, 5000);
    return () => clearTimeout(timer);
  }, [toast.id, onRemove]);

  const Icon = icons[toast.type] || Info;
  const color = colors[toast.type] || colors.info;

  return (
    <motion.div
      initial={{ opacity: 0, x: 50, scale: 0.95 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 50, scale: 0.95, transition: { duration: 0.2 } }}
      layout
      className="glass-card"
      style={{
        width: '320px', padding: '1rem', marginBottom: '0.75rem',
        borderLeft: `4px solid ${color}`,
        background: 'var(--glass-bg)',
        boxShadow: '0 8px 32px rgba(0,0,0,0.2)',
        position: 'relative'
      }}
    >
      <button 
        onClick={() => onRemove && onRemove(toast.id)}
        style={{ position: 'absolute', top: '0.5rem', right: '0.5rem', background: 'none', border: 'none', color: '#fff', opacity: 0.5, cursor: 'pointer' }}
      >
        <X size={14} />
      </button>
      <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
        <Icon size={20} color={color} style={{ marginTop: '0.1rem' }} />
        <div>
          <h4 style={{ margin: '0 0 0.25rem 0', fontSize: '0.95rem' }}>{toast.title}</h4>
          <p style={{ margin: 0, fontSize: '0.85rem', opacity: 0.8, lineHeight: 1.4 }}>{toast.message}</p>
        </div>
      </div>
    </motion.div>
  );
};

export default function Toasts() {
  const toasts = useStore(state => state.toasts) || [];
  const removeToast = useStore(state => state.removeToast);

  const visibleToasts = toasts.slice(-5);

  return (
    <div style={{ position: 'fixed', top: '1.5rem', right: '1.5rem', zIndex: 9999, display: 'flex', flexDirection: 'column-reverse' }}>
      <AnimatePresence>
        {visibleToasts.map(toast => (
          <Toast key={toast.id} toast={toast} onRemove={removeToast} />
        ))}
      </AnimatePresence>
    </div>
  );
}
