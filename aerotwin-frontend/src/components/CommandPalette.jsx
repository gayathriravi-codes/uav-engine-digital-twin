import React, { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Command } from 'lucide-react';
import { useStore } from '../store/useStore';
import { PAGES, QUICK_SCENARIOS } from '../config';

export default function CommandPalette() {
  const open = useStore(state => state.commandPaletteOpen);
  const setOpen = useStore(state => state.setCommandPaletteOpen);
  const setScenario = useStore(state => state.setScenario);
  
  const [search, setSearch] = useState('');
  const inputRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        if (setOpen) setOpen(!open);
      }
      if (e.key === 'Escape' && open) {
        if (setOpen) setOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, setOpen]);

  useEffect(() => {
    if (open && inputRef.current) {
      inputRef.current.focus();
      setSearch('');
    }
  }, [open]);

  if (!open) return null;

  const filteredPages = (PAGES || []).filter(p => p.name.toLowerCase().includes(search.toLowerCase()));
  const filteredScenarios = (QUICK_SCENARIOS || []).filter(s => s.name.toLowerCase().includes(search.toLowerCase()));

  const handleNavigate = (path) => {
    navigate(path);
    if (setOpen) setOpen(false);
  };

  const handleScenario = (scenario) => {
    if (setScenario) setScenario(scenario.id);
    if (setOpen) setOpen(false);
  };

  return (
    <AnimatePresence>
      <div 
        style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(11, 16, 32, 0.8)', backdropFilter: 'blur(4px)',
          zIndex: 99999, display: 'flex', justifyContent: 'center', paddingTop: '10vh'
        }}
        onClick={() => setOpen && setOpen(false)}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: -20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95 }}
          onClick={(e) => e.stopPropagation()}
          className="glass-card"
          style={{
            width: '100%', maxWidth: '600px', maxHeight: '80vh', display: 'flex', flexDirection: 'column',
            background: 'var(--glass-bg)', border: '1px solid rgba(255,255,255,0.1)', overflow: 'hidden'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', padding: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
            <Search size={20} style={{ opacity: 0.5, marginRight: '1rem' }} />
            <input
              ref={inputRef}
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search pages and scenarios..."
              style={{
                flex: 1, background: 'transparent', border: 'none', color: '#fff', fontSize: '1.1rem', outline: 'none', fontFamily: 'inherit'
              }}
            />
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', opacity: 0.5, fontSize: '0.8rem' }}>
              <Command size={14} /> K to close
            </div>
          </div>
          
          <div style={{ padding: '1rem', overflowY: 'auto', flex: 1 }}>
            {filteredPages.length > 0 && (
              <div style={{ marginBottom: '1.5rem' }}>
                <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', opacity: 0.5, marginBottom: '0.5rem', fontWeight: 600 }}>Navigate</div>
                {filteredPages.map(page => (
                  <div
                    key={page.path}
                    onClick={() => handleNavigate(page.path)}
                    style={{
                      padding: '0.75rem 1rem', cursor: 'pointer', borderRadius: '6px',
                      display: 'flex', alignItems: 'center', gap: '0.75rem',
                      transition: 'background 0.2s'
                    }}
                    onMouseOver={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}
                    onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}
                  >
                    {page.name}
                  </div>
                ))}
              </div>
            )}
            
            {filteredScenarios.length > 0 && (
              <div>
                <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', opacity: 0.5, marginBottom: '0.5rem', fontWeight: 600 }}>Scenarios</div>
                {filteredScenarios.map(scenario => (
                  <div
                    key={scenario.id}
                    onClick={() => handleScenario(scenario)}
                    style={{
                      padding: '0.75rem 1rem', cursor: 'pointer', borderRadius: '6px',
                      display: 'flex', alignItems: 'center', gap: '0.75rem',
                      transition: 'background 0.2s'
                    }}
                    onMouseOver={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}
                    onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}
                  >
                    {scenario.name}
                  </div>
                ))}
              </div>
            )}
            
            {filteredPages.length === 0 && filteredScenarios.length === 0 && (
              <div style={{ textAlign: 'center', padding: '2rem', opacity: 0.5 }}>
                No results found for "{search}"
              </div>
            )}
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
