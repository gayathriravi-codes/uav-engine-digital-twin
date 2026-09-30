import React from 'react';

export default function Sparkline({ 
  data = [], 
  color = '#14B8A6', 
  width = 120, 
  height = 32 
}) {
  if (!data || data.length === 0) {
    return (
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <line x1="0" y1={height / 2} x2={width} y2={height / 2} stroke={color} strokeWidth="2" opacity="0.5" />
      </svg>
    );
  }

  const validData = data.filter(d => typeof d === 'number' && !isNaN(d));
  if (validData.length < 2) {
    return (
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <line x1="0" y1={height / 2} x2={width} y2={height / 2} stroke={color} strokeWidth="2" opacity="0.5" />
      </svg>
    );
  }

  const min = Math.min(...validData);
  const max = Math.max(...validData);
  const range = max - min || 1; // avoid division by zero
  
  const points = validData.map((d, i) => {
    const x = (i / (validData.length - 1)) * width;
    const y = height - ((d - min) / range) * (height - 4) - 2; // 2px padding
    return `${x},${y}`;
  }).join(' ');

  const fillPoints = `0,${height} ${points} ${width},${height}`;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <defs>
        <linearGradient id={`sparkline-gradient-${color}`} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={fillPoints} fill={`url(#sparkline-gradient-${color})`} />
      <polyline points={points} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
