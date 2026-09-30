import React, { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';

export default function PageShell({ children }) {
  const location = useLocation();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [location.pathname]);

  return (
    <>
      <style>{`
        .page-shell {
          margin-left: var(--sidebar-width, 260px);
          margin-top: 64px;
          min-height: calc(100vh - 64px);
          padding: 2rem;
          display: flex;
          flex-direction: column;
          transition: margin-left 0.3s ease;
        }

        .page-content {
          flex: 1;
          max-width: 1400px;
          margin: 0 auto;
          width: 100%;
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        .page-footer {
          margin-top: 3rem;
          padding-top: 1.5rem;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
          text-align: center;
          color: rgba(255, 255, 255, 0.4);
          font-size: 0.85rem;
        }

        @media (max-width: 768px) {
          .page-shell {
            margin-left: 0;
            margin-bottom: 64px; /* for bottom bar */
            padding: 1rem;
          }
        }
      `}</style>
      <div className="page-shell">
        <motion.div 
          className="page-content"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.4, staggerChildren: 0.1 }}
        >
          {children}
        </motion.div>
        
        <footer className="page-footer">
          <p>All data is synthetic. Not validated on real engines.</p>
        </footer>
      </div>
    </>
  );
}
