import React, { useState } from 'react';
import { ShieldCheck, ShieldAlert } from 'lucide-react';

export default function TrustBadge({ trusted, sensorName }) {
  const [showTooltip, setShowTooltip] = useState(false);

  const badgeStyle = trusted
    ? { backgroundColor: 'rgba(20, 184, 166, 0.1)', borderColor: 'rgba(20, 184, 166, 0.3)', color: '#14B8A6' }
    : { backgroundColor: 'rgba(245, 158, 11, 0.1)', borderColor: 'rgba(245, 158, 11, 0.3)', color: '#F59E0B' };

  return (
    <div 
      style={{ position: 'relative', display: 'inline-block' }}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <div 
        className={trusted ? "glass-badge--teal" : "glass-badge--amber"}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '4px 8px',
          borderRadius: '12px',
          border: '1px solid',
          fontSize: '0.75rem',
          fontWeight: 500,
          cursor: 'help',
          backdropFilter: 'blur(10px)',
          ...badgeStyle
        }}
      >
        {trusted ? (
          <>
            <ShieldCheck size={14} color="#14B8A6" />
            <span>Trusted</span>
          </>
        ) : (
          <>
            <ShieldAlert size={14} color="#F59E0B" />
            <span>Untrusted</span>
          </>
        )}
      </div>

      {showTooltip && (
        <div style={{
          position: 'absolute',
          bottom: '100%',
          left: '50%',
          transform: 'translateX(-50%)',
          marginBottom: '8px',
          padding: '6px 10px',
          backgroundColor: '#0B1020',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: '6px',
          fontSize: '0.75rem',
          color: 'white',
          whiteSpace: 'nowrap',
          zIndex: 10,
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.5)'
        }}>
          {trusted 
            ? `${sensorName ? sensorName + ' ' : ''}Sensor data is verified and operating normally.` 
            : `${sensorName ? sensorName + ' ' : ''}Sensor data may be unreliable or degraded.`}
        </div>
      )}
    </div>
  );
}
