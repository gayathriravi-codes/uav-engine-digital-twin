import React from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import AnimatedNumber from './AnimatedNumber';

export default function RadialGauge({ 
  value, 
  min = 0, 
  max = 100, 
  healthyRange = [0, 80], 
  warningRange = [80, 90], 
  unit = '', 
  label = '', 
  size = 140 
}) {
  const prefersReducedMotion = useReducedMotion();
  const radius = size * 0.4;
  const strokeWidth = size * 0.08;
  const center = size / 2;
  
  const circumference = 2 * Math.PI * radius;
  // 270 degrees arc = 0.75 of circumference
  const arcLength = circumference * 0.75;
  
  const clampedValue = Math.min(Math.max(value, min), max);
  const percentage = (clampedValue - min) / (max - min);
  const valueArcLength = arcLength * percentage;
  
  let color = '#14B8A6'; // teal
  if (clampedValue >= warningRange[0] && clampedValue <= warningRange[1]) {
    color = '#F59E0B'; // amber
  } else if (clampedValue > warningRange[1] || clampedValue < healthyRange[0]) {
    color = '#EF4444'; // red
  }

  // Rotate 135deg so that 0deg of the circle starts at the bottom-left
  const svgTransform = "rotate(135)";

  return (
    <div style={{ width: size, height: size, position: 'relative', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <g transform={`translate(${center}, ${center}) ${svgTransform}`}>
          <circle
            cx="0"
            cy="0"
            r={radius}
            fill="none"
            stroke="rgba(255, 255, 255, 0.1)"
            strokeWidth={strokeWidth}
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeLinecap="round"
          />
          <motion.circle
            cx="0"
            cy="0"
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            initial={prefersReducedMotion ? false : { strokeDasharray: `0 ${circumference}` }}
            animate={{ strokeDasharray: `${valueArcLength} ${circumference}`, stroke: color }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            style={{ strokeDasharray: `${valueArcLength} ${circumference}` }}
          />
        </g>
      </svg>
      
      <div style={{ 
        position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', 
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' 
      }}>
        <div style={{ fontSize: size * 0.18, fontWeight: 'bold', fontFamily: 'var(--font-mono, "JetBrains Mono", monospace)', color: 'white', marginTop: size * 0.1 }}>
          <AnimatedNumber value={value} precision={1} />
          <span style={{ fontSize: '0.6em', opacity: 0.7, marginLeft: '2px' }}>{unit}</span>
        </div>
        <div style={{ fontSize: size * 0.09, opacity: 0.7, color: 'white', marginTop: size * 0.15 }}>
          {label}
        </div>
      </div>
    </div>
  );
}
