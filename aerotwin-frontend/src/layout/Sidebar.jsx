import React, { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import * as LucideIcons from 'lucide-react';
import { PAGES } from '../config';

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  // Sync sidebar width with CSS variables for layout adjustments
  useEffect(() => {
    document.documentElement.style.setProperty('--sidebar-width', collapsed ? '80px' : '260px');
  }, [collapsed]);

  return (
    <>
      <style>{`
        :root {
          --sidebar-width: 260px;
        }
        .sidebar {
          width: var(--sidebar-width);
          transition: width 0.3s ease;
          display: flex;
          flex-direction: column;
          height: 100vh;
          border-right: 1px solid rgba(255, 255, 255, 0.1);
          background: rgba(11, 16, 32, 0.6);
          backdrop-filter: blur(20px);
          position: fixed;
          top: 0;
          left: 0;
          z-index: 50;
        }
        
        .sidebar-toggle {
          padding: 1rem;
          display: flex;
          justify-content: ${collapsed ? 'center' : 'flex-end'};
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
          color: rgba(255, 255, 255, 0.6);
          background: transparent;
          border: none;
          cursor: pointer;
        }
        
        .sidebar-toggle:hover {
          color: #fff;
        }

        .nav-list {
          flex: 1;
          display: flex;
          flex-direction: column;
          padding: 1rem 0;
          gap: 0.25rem;
          overflow-y: auto;
        }

        .nav-item {
          display: flex;
          align-items: center;
          padding: 0.75rem 1.25rem;
          color: rgba(255, 255, 255, 0.6);
          text-decoration: none;
          transition: all 0.2s;
          position: relative;
          gap: 1rem;
          overflow: hidden;
          white-space: nowrap;
        }

        .nav-item:hover {
          background: rgba(255, 255, 255, 0.05);
          color: #fff;
        }

        .nav-item.active {
          background: rgba(255, 255, 255, 0.1);
          color: #fff;
        }

        .nav-item.active::before {
          content: '';
          position: absolute;
          left: 0;
          top: 0;
          bottom: 0;
          width: 3px;
          background: #14B8A6;
          box-shadow: 0 0 10px #14B8A6;
        }

        .nav-icon {
          flex-shrink: 0;
        }
        
        .nav-label {
          opacity: ${collapsed ? '0' : '1'};
          transition: opacity 0.2s ease;
        }

        /* Mobile Bottom Nav */
        @media (max-width: 768px) {
          .sidebar {
            width: 100%;
            height: 64px;
            top: auto;
            bottom: 0;
            flex-direction: row;
            border-right: none;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
          }
          
          .sidebar-toggle {
            display: none;
          }

          .nav-list {
            flex-direction: row;
            padding: 0;
            gap: 0;
            overflow-x: auto;
            overflow-y: hidden;
          }

          .nav-item {
            flex-direction: column;
            gap: 0.25rem;
            padding: 0.5rem;
            min-width: 72px;
            justify-content: center;
          }

          .nav-item.active::before {
            width: 100%;
            height: 3px;
            bottom: 0;
            top: auto;
            left: 0;
          }

          .nav-label {
            opacity: 1;
            font-size: 0.65rem;
          }
        }
      `}</style>
      <nav className="sidebar glass-sidebar">
        <button className="sidebar-toggle" onClick={() => setCollapsed(!collapsed)}>
          {collapsed ? <LucideIcons.ChevronRight size={20} /> : <LucideIcons.ChevronLeft size={20} />}
        </button>
        <div className="nav-list">
          {PAGES.map(page => {
            const Icon = LucideIcons[page.icon] || LucideIcons.Circle;
            const isActive = location.pathname === page.path;
            
            return (
              <NavLink 
                key={page.id} 
                to={page.path} 
                className={`nav-item ${isActive ? 'active' : ''}`}
                title={collapsed ? page.label : ''}
              >
                <Icon size={20} className="nav-icon" />
                <span className="nav-label">{page.label}</span>
              </NavLink>
            );
          })}
        </div>
      </nav>
    </>
  );
}
