import { useRef, useEffect, useCallback } from 'react';

export default function AuroraBackground() {
  const containerRef = useRef(null);
  
  // Mouse parallax: move blobs slightly based on mouse position
  const handleMouseMove = useCallback((e) => {
    if (!containerRef.current) return;
    const { clientX, clientY } = e;
    const cx = (clientX / window.innerWidth - 0.5) * 2; // -1 to 1
    const cy = (clientY / window.innerHeight - 0.5) * 2;
    const blobs = containerRef.current.querySelectorAll('.aurora-bg__blob');
    blobs.forEach((blob, i) => {
      const factor = (i + 1) * 8; // Different parallax depth per blob
      blob.style.transform = `translate(${cx * factor}px, ${cy * factor}px)`;
    });
  }, []);

  useEffect(() => {
    // Check reduced motion preference
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (mq.matches) return;
    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [handleMouseMove]);

  return (
    <div className="aurora-bg" ref={containerRef}>
      <div className="aurora-bg__blob aurora-bg__blob--teal" />
      <div className="aurora-bg__blob aurora-bg__blob--indigo" />
      <div className="aurora-bg__blob aurora-bg__blob--amber" />
      <div className="aurora-bg__blob aurora-bg__blob--blue" />
      <div className="aurora-bg__noise" />
      <div className="aurora-bg__grid" />
    </div>
  );
}
