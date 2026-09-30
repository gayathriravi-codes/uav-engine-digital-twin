import React from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import AnimatedNumber from './AnimatedNumber';

export default function HealthOrb({ healthScore, severity, action, size = 280 }) {
  const prefersReducedMotion = useReducedMotion();
  
  // Determine color and timings
  let color = '#14B8A6'; // teal
  let pulseDuration = 4; // slow
  let spinDuration = 20;
  
  if (healthScore !== null && !isNaN(healthScore)) {
    if (healthScore < 40) {
      color = '#EF4444'; // red
      pulseDuration = 1; // fast
      spinDuration = 5;
    } else if (healthScore <= 80) {
      color = '#F59E0B'; // amber
      pulseDuration = 2; // medium
      spinDuration = 10;
    }
  }

  const ring1Animation = prefersReducedMotion ? {} : {
    rotate: [0, 360],
    transition: { duration: spinDuration, repeat: Infinity, ease: "linear" }
  };

  const ring2Animation = prefersReducedMotion ? {} : {
    rotate: [360, 0],
    scale: [0.9, 1.05, 0.9],
    transition: { duration: spinDuration * 1.5, repeat: Infinity, ease: "linear" }
  };

  const pulseAnimation = prefersReducedMotion ? {} : {
    scale: [1, 1.05, 1],
    boxShadow: [
      `0 0 30px 0px ${color}60`,
      `0 0 60px 20px ${color}80`,
      `0 0 30px 0px ${color}60`
    ],
    transition: { duration: pulseDuration, repeat: Infinity, ease: "easeInOut" }
  };

  return (
    <div 
      style={{ 
        width: size, 
        height: size, 
        position: 'relative', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        margin: '0 auto'
      }}
    >
      {/* Outer spinning dashed ring */}
      <motion.div
        animate={ring1Animation}
        style={{
          position: 'absolute', width: '100%', height: '100%',
          borderRadius: '50%', 
          border: `2px dashed ${color}80`,
          borderLeftColor: 'transparent',
          borderRightColor: 'transparent'
        }}
      />
      
      {/* Inner counter-spinning dotted ring */}
      <motion.div
        animate={ring2Animation}
        style={{
          position: 'absolute', width: '85%', height: '85%',
          borderRadius: '50%', 
          border: `4px dotted ${color}50`
        }}
      />
      
      {/* Glowing core pulse */}
      <motion.div
        animate={pulseAnimation}
        style={{
          width: '60%', height: '60%', borderRadius: '50%',
          background: `radial-gradient(circle at 30% 30%, ${color}ff, ${color}60 60%, transparent)`,
          backdropFilter: 'blur(20px)',
          border: `1px solid ${color}aa`,
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          color: 'white', textShadow: '0 2px 4px rgba(0,0,0,0.8)',
          boxShadow: `inset 0 0 20px ${color}`
        }}
      >
        <div style={{ fontSize: '3.5rem', fontWeight: '800', lineHeight: 1 }}>
          <AnimatedNumber value={healthScore} precision={0} />
        </div>
        <div style={{ fontSize: '0.875rem', opacity: 0.9, marginTop: '4px', textTransform: 'uppercase', letterSpacing: '2px', fontWeight: 600 }}>
          {action || 'Health Score'}
        </div>
      </motion.div>
    </div>
  );
}
