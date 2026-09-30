import React, { useRef } from 'react';
import { motion } from 'framer-motion';

export default function GlassCard({
  children,
  className = '',
  gradient = false,
  tilt = false,
  onClick,
  style,
  ...rest
}) {
  const cardRef = useRef(null);

  const handleMouseMove = (e) => {
    if (!tilt || !cardRef.current) return;
    
    // Check for prefers-reduced-motion
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    const rect = cardRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    
    const rotateX = ((y - centerY) / centerY) * -3; // max ±3deg
    const rotateY = ((x - centerX) / centerX) * 3;  // max ±3deg
    
    const glowX = (x / rect.width) * 100;
    const glowY = (y / rect.height) * 100;

    cardRef.current.style.setProperty('--tilt-x', `${rotateX}deg`);
    cardRef.current.style.setProperty('--tilt-y', `${rotateY}deg`);
    cardRef.current.style.setProperty('--glow-x', `${glowX}%`);
    cardRef.current.style.setProperty('--glow-y', `${glowY}%`);
  };

  const handleMouseLeave = () => {
    if (!tilt || !cardRef.current) return;
    cardRef.current.style.setProperty('--tilt-x', `0deg`);
    cardRef.current.style.setProperty('--tilt-y', `0deg`);
    cardRef.current.style.setProperty('--glow-x', `50%`);
    cardRef.current.style.setProperty('--glow-y', `50%`);
  };

  const classNames = [
    'glass-card',
    gradient ? 'glass-card--gradient-border' : '',
    tilt ? 'glass-card--tilt' : '',
    className
  ].filter(Boolean).join(' ');

  const tiltStyle = tilt ? {
    transform: 'perspective(1000px) rotateX(var(--tilt-x, 0deg)) rotateY(var(--tilt-y, 0deg))',
    transition: 'transform 0.1s ease',
    background: 'radial-gradient(circle at var(--glow-x, 50%) var(--glow-y, 50%), rgba(255,255,255,0.06), transparent 60%)'
  } : {};

  const baseStyle = {
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
    backdropFilter: 'blur(20px)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    borderRadius: '16px',
    padding: '1.5rem',
    ...tiltStyle,
    ...style
  };

  return (
    <div
      ref={cardRef}
      className={classNames}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onClick={onClick}
      style={baseStyle}
      {...rest}
    >
      {children}
    </div>
  );
}
