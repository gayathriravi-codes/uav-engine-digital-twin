import React, { useEffect, useState, useRef } from 'react';

export default function AnimatedNumber({
  value,
  precision = 0,
  duration = 400,
  className = '',
  prefix = '',
  suffix = ''
}) {
  const [displayValue, setDisplayValue] = useState(value);
  const requestRef = useRef(null);
  const previousTimeRef = useRef(null);
  const startValueRef = useRef(value);
  const targetValueRef = useRef(value);
  const durationRef = useRef(duration);

  useEffect(() => {
    if (value === null || isNaN(value)) {
      setDisplayValue(value);
      return;
    }

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion || duration === 0) {
      setDisplayValue(value);
      return;
    }

    startValueRef.current = displayValue !== null && !isNaN(displayValue) ? displayValue : 0;
    targetValueRef.current = value;
    durationRef.current = duration;
    
    if (startValueRef.current !== targetValueRef.current) {
      const animate = time => {
        if (previousTimeRef.current !== undefined && previousTimeRef.current !== null) {
          const deltaTime = time - previousTimeRef.current;
          
          if (deltaTime < durationRef.current) {
            const progress = deltaTime / durationRef.current;
            const easeProgress = 1 - (1 - progress) * (1 - progress); // easeOutQuad
            const nextVal = startValueRef.current + (targetValueRef.current - startValueRef.current) * easeProgress;
            setDisplayValue(nextVal);
            requestRef.current = requestAnimationFrame(animate);
          } else {
            setDisplayValue(targetValueRef.current);
            previousTimeRef.current = null;
          }
        } else {
          previousTimeRef.current = time;
          requestRef.current = requestAnimationFrame(animate);
        }
      };
      
      requestRef.current = requestAnimationFrame(animate);
      
      return () => {
        if (requestRef.current) cancelAnimationFrame(requestRef.current);
      };
    } else {
      setDisplayValue(value);
    }
  }, [value, duration]);

  const formattedValue = (value === null || isNaN(value)) 
    ? '--' 
    : (typeof displayValue === 'number' ? displayValue.toFixed(precision) : '--');

  return (
    <span className={className} style={{ fontFamily: 'var(--font-mono, "JetBrains Mono", monospace)' }}>
      {prefix}{formattedValue}{suffix}
    </span>
  );
}
