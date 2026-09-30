import React from 'react';
import { motion } from 'framer-motion';

export default function ProbBars({ classProbs, predictedFault }) {
  if (!classProbs) return null;
  
  const entries = Object.entries(classProbs).sort((a, b) => b[1] - a[1]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      {entries.map(([className, prob]) => {
        const isActive = className === predictedFault;
        const color = isActive ? 'var(--watch)' : 'rgba(255, 255, 255, 0.4)';
        const width = `${prob * 100}%`;
        
        return (
          <div key={className} style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ width: '80px', fontSize: '0.8rem', textAlign: 'right', opacity: isActive ? 1 : 0.7, fontWeight: isActive ? 600 : 400 }}>
              {className}
            </div>
            <div style={{ flex: 1, height: '12px', background: 'rgba(255,255,255,0.05)', borderRadius: '6px', overflow: 'hidden' }}>
              <motion.div
                initial={{ width: 0 }}
                animate={{ width }}
                transition={{ type: 'spring', stiffness: 50, damping: 15 }}
                style={{ height: '100%', background: color, borderRadius: '6px' }}
              />
            </div>
            <div style={{ width: '50px', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color }}>
              {(prob * 100).toFixed(1)}%
            </div>
          </div>
        );
      })}
    </div>
  );
}
