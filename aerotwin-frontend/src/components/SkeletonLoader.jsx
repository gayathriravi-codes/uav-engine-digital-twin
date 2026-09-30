import React, { useState } from 'react';

export default function SkeletonLoader({ variant = 'text', width, height, count = 1 }) {
  const getStyle = () => {
    let style = {};
    if (width) style.width = width;
    if (height) style.height = height;

    if (variant === 'card' && !height) style.height = '150px';
    if (variant === 'gauge' && !width) {
      style.width = '140px';
      style.height = '140px';
      style.borderRadius = '50%';
    }
    if (variant === 'chart' && !height) style.height = '200px';
    if (variant === 'text' && !height) style.height = '20px';
    
    return style;
  };

  const baseClass = `skeleton skeleton--${variant}`;

  const skeletons = Array.from({ length: count }).map((_, i) => (
    <div key={i} className={baseClass} style={getStyle()} />
  ));

  return (
    <>
      <style>{`
        .skeleton {
          background-color: rgba(255, 255, 255, 0.05);
          position: relative;
          overflow: hidden;
          border-radius: 4px;
        }
        .skeleton--card { border-radius: 16px; }
        .skeleton--gauge { border-radius: 50%; }
        .skeleton--chart { border-radius: 8px; }
        .skeleton::after {
          content: '';
          position: absolute;
          top: 0; left: 0; right: 0; bottom: 0;
          background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
          animation: shimmer 2s infinite;
          transform: translateX(-100%);
        }
        @keyframes shimmer {
          100% { transform: translateX(100%); }
        }
        @media (prefers-reduced-motion: reduce) {
          .skeleton::after { animation: none; }
        }
      `}</style>
      {count === 1 ? skeletons[0] : <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>{skeletons}</div>}
    </>
  );
}
