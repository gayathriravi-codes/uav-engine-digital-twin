import React from 'react';
import { motion } from 'framer-motion';

export default function ShapBars({ explanation }) {
  if (!explanation || !Array.isArray(explanation)) return null;
  
  const sorted = [...explanation].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  const maxVal = Math.max(...sorted.map(e => Math.abs(e.contribution)), 0.001);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      {sorted.map((item, idx) => {
        const { feature, contribution } = item;
        const isPos = contribution >= 0;
        const color = isPos ? 'var(--healthy)' : 'var(--critical)';
        const widthPercent = (Math.abs(contribution) / maxVal) * 100;
        
        return (
          <div key={`${feature}-${idx}`} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem' }}>
            <div style={{ width: '100px', textAlign: 'right', opacity: 0.8, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {feature}
            </div>
            
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', position: 'relative', height: '16px' }}>
              <div style={{ position: 'absolute', left: '50%', width: '1px', height: '100%', background: 'rgba(255,255,255,0.2)' }} />
              
              <div style={{ flex: 1, display: 'flex', justifyContent: 'flex-end', paddingRight: '2px' }}>
                {!isPos && (
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${widthPercent}%` }}
                    style={{ height: '8px', background: color, borderRadius: '4px 0 0 4px' }}
                  />
                )}
              </div>
              <div style={{ flex: 1, display: 'flex', justifyContent: 'flex-start', paddingLeft: '2px' }}>
                {isPos && (
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${widthPercent}%` }}
                    style={{ height: '8px', background: color, borderRadius: '0 4px 4px 0' }}
                  />
                )}
              </div>
            </div>
            
            <div style={{ width: '50px', fontFamily: 'var(--font-mono)', color, textAlign: 'left' }}>
              {contribution > 0 ? '+' : ''}{contribution.toFixed(3)}
            </div>
          </div>
        );
      })}
    </div>
  );
}
