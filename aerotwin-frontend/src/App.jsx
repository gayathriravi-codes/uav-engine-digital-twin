/**
 * src/App.jsx
 * Root application component for NIDAN.
 * Sets up the layout shell (aurora + sidebar + topbar) and page routing.
 * Telemetry hook runs at this level so the stream persists across page navigation.
 */
import { Routes, Route } from 'react-router-dom';
import { useTelemetry } from './data/useTelemetry';

/* Layout */
import AuroraBackground from './layout/AuroraBackground';
import Sidebar from './layout/Sidebar';
import TopBar from './layout/TopBar';
import PageShell from './layout/PageShell';

/* Global overlays */
import CommandPalette from './components/CommandPalette';
import Toasts from './components/Toasts';

/* Pages (lazy loading could be added later for performance) */
import Overview from './pages/Overview';
import LiveMonitor from './pages/LiveMonitor';
import Diagnosis from './pages/Diagnosis';
import RemainingLife from './pages/RemainingLife';
import Mission from './pages/Mission';
import WhatIf from './pages/WhatIf';
import Replay from './pages/Replay';
import AuditLog from './pages/AuditLog';
import About from './pages/About';

export default function App() {
  /* Start telemetry stream — persists across all page navigations */
  useTelemetry();

  return (
    <>
      {/* Animated aurora background behind everything */}
      <AuroraBackground />

      {/* Glass sidebar navigation */}
      <Sidebar />

      {/* Main content area (shifted right by sidebar width) */}
      <div
        style={{
          marginLeft: 'var(--sidebar-width, 240px)',
          minHeight: '100vh',
          position: 'relative',
          zIndex: 1,
          display: 'flex',
          flexDirection: 'column',
          transition: 'margin-left 0.3s ease',
        }}
      >
        {/* Glass top bar */}
        <TopBar />

        {/* Page content wrapped in PageShell for animations + footer */}
        <main style={{ flex: 1, paddingTop: 'var(--topbar-height, 64px)' }}>
          <PageShell>
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/live-monitor" element={<LiveMonitor />} />
              <Route path="/diagnosis" element={<Diagnosis />} />
              <Route path="/remaining-life" element={<RemainingLife />} />
              <Route path="/mission" element={<Mission />} />
              <Route path="/what-if" element={<WhatIf />} />
              <Route path="/replay" element={<Replay />} />
              <Route path="/audit-log" element={<AuditLog />} />
              <Route path="/about" element={<About />} />
            </Routes>
          </PageShell>
        </main>
      </div>

      {/* Global overlays (above everything) */}
      <CommandPalette />
      <Toasts />
    </>
  );
}