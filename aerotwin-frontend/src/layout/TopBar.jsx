import React from 'react';
import { useLocation } from 'react-router-dom';
import { Play, Pause, Activity } from 'lucide-react';
import { useStore } from '../store/useStore';
import { PROJECT_NAME, PAGES } from '../config';

export default function TopBar() {
  const location = useLocation();
  const { 
    source, 
    setSource, 
    connectionStatus, 
    fallbackNotice,
    currentTick,
    isPlaying,
    setIsPlaying
  } = useStore();

  const currentPage = PAGES.find(p => p.path === location.pathname) || { label: 'Dashboard' };

  return (
    <>
      <style>{`
        .topbar {
          position: fixed;
          top: 0;
          left: var(--sidebar-width, 260px);
          right: 0;
          height: 64px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0 1.5rem;
          background: rgba(11, 16, 32, 0.6);
          backdrop-filter: blur(20px);
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
          z-index: 40;
          transition: left 0.3s ease;
        }

        @media (max-width: 768px) {
          .topbar {
            left: 0;
          }
        }

        .topbar-left {
          display: flex;
          align-items: center;
          gap: 1.5rem;
        }

        .wordmark {
          font-weight: 700;
          font-size: 1.25rem;
          letter-spacing: 0.05em;
          background: linear-gradient(90deg, #fff, #14B8A6);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .page-title {
          font-size: 1rem;
          color: rgba(255, 255, 255, 0.9);
          font-weight: 500;
          border-left: 1px solid rgba(255, 255, 255, 0.2);
          padding-left: 1.5rem;
        }

        .topbar-right {
          display: flex;
          align-items: center;
          gap: 1rem;
        }

        .status-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }

        .status-dot.connected { background: #14B8A6; box-shadow: 0 0 8px #14B8A6; }
        .status-dot.connecting { background: #F59E0B; box-shadow: 0 0 8px #F59E0B; }
        .status-dot.unreachable { background: #EF4444; box-shadow: 0 0 8px #EF4444; }

        .source-toggle {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.85rem;
          color: rgba(255, 255, 255, 0.8);
          background: rgba(255, 255, 255, 0.05);
          padding: 0.35rem 0.75rem;
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.1);
          cursor: pointer;
        }
        
        .sim-badge {
          font-size: 0.75rem;
          font-weight: 600;
          color: #F59E0B;
          background: rgba(245, 158, 11, 0.15);
          border: 1px solid rgba(245, 158, 11, 0.3);
          padding: 0.25rem 0.5rem;
          border-radius: 8px;
          display: flex;
          align-items: center;
          gap: 0.35rem;
        }

        .mission-clock {
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.9rem;
          color: rgba(255, 255, 255, 0.9);
          background: rgba(0, 0, 0, 0.3);
          padding: 0.35rem 0.75rem;
          border-radius: 8px;
          border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .play-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid rgba(255, 255, 255, 0.2);
          color: #fff;
          width: 32px;
          height: 32px;
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .play-btn:hover {
          background: rgba(255, 255, 255, 0.2);
        }

        .fallback-notice {
          font-size: 0.8rem;
          color: #EF4444;
          margin-right: 0.5rem;
        }
      `}</style>
      <div className="topbar glass-card">
        <div className="topbar-left">
          <div className="wordmark">{PROJECT_NAME}</div>
          <div className="page-title">{currentPage.label}</div>
        </div>
        <div className="topbar-right">
          {fallbackNotice && <div className="fallback-notice">{fallbackNotice}</div>}
          
          <button 
            className="source-toggle" 
            onClick={() => setSource(source === 'sim' ? 'api' : 'sim')}
          >
            <div className={`status-dot ${connectionStatus}`}></div>
            {source === 'sim' ? 'Simulator' : 'Live API'}
          </button>

          {source === 'sim' && (
            <div className="sim-badge">
              <Activity size={14} /> Simulated Data
            </div>
          )}

          <div className="mission-clock">
            {currentTick?.mission_clock || '00:00:00'}
          </div>

          <button className="play-btn" onClick={() => setIsPlaying(!isPlaying)}>
            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
          </button>
        </div>
      </div>
    </>
  );
}
