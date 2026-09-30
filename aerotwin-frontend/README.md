# NIDAN — UAV Engine Digital Twin Operator Console

> **AI-Enabled Digital Twin for Aero Piston Engine Health Monitoring**
> Smart India Hackathon 2026 · PS 26054 · DRDO

NIDAN (Sanskrit/Hindi for "diagnosis") is a multi-page operator console for monitoring the health of aero piston engines in MALE UAVs. It watches 7 engine sensors, checks sensor trust, diagnoses 6 fault types, estimates remaining useful life (RUL), and recommends mission actions — with a human operator in the loop for high-stakes decisions.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

The app runs at **http://localhost:5173** by default.

## Connecting to the Real API

By default, NIDAN runs with a built-in **browser simulator** (SimSource). To connect to the FastAPI backend:

1. Set the environment variable before starting:
   ```bash
   VITE_API_URL=http://your-backend:8000 npm run dev
   ```
   Or create a `.env` file:
   ```
   VITE_API_URL=http://localhost:8000
   ```

2. Toggle the data source in the top bar: **Simulator ↔ Live API**

If the API is unreachable, NIDAN falls back to the simulator automatically with a toast notification.

## Tick Data Mapping

The adapter at `src/data/normalizeTick.js` maps incoming ticks (from either source) into a standard shape. It accepts various key aliases and never crashes on missing fields. To add new mappings, edit the alias lookups in that file.

## Pages

| Page | Route | Description |
|------|-------|-------------|
| Overview | `/` | Command centre with health orb, mission action, sensor summary |
| Live Monitor | `/live-monitor` | 7 sensor gauges with sparklines, multi-sensor chart |
| Diagnosis | `/diagnosis` | Fault classification, probability bars, SHAP attribution, fault library |
| Remaining Life | `/remaining-life` | RUL estimate with uncertainty band and trend chart |
| Mission & Operator | `/mission` | Mission banner, confirmation gate, action timeline |
| What-If | `/what-if` | Counterfactual flight comparison |
| Replay | `/replay` | Full flight scrubbing with synchronized panels |
| Audit Log | `/audit-log` | Searchable, filterable decision log with JSONL export |
| About & Limits | `/about` | System flow diagram, pillars, known limits, team |

## Tech Stack

- **React 19** + **Vite** (build tool)
- **react-router-dom** (multi-page navigation)
- **framer-motion** (animations)
- **recharts** (charts)
- **zustand** (state management)
- **lucide-react** (icons)
- Plain CSS with CSS variables (glassmorphism, dark theme)

## Configuration

All project-level constants live in `src/config.js`:

- `PROJECT_NAME` — product name ("NIDAN")
- `TEAM_NAME`, `TEAM_MEMBERS` — easy to edit
- `PAGES` — route definitions
- `SENSOR_METADATA` — 7 sensor definitions with ranges
- `FAULT_LIBRARY` — fault descriptions
- `MISSION_ACTIONS` — action colours and labels
- `PILLAR_CARDS`, `KNOWN_LIMITS` — About page content

## Design

- **Dark theme only** — deep navy base, animated aurora background
- **Glassmorphism** — translucent cards with backdrop blur
- **Typography** — Inter (UI) + JetBrains Mono (numeric values)
- **Responsive** — sidebar collapses on small widths
- **Accessibility** — `prefers-reduced-motion` disables heavy animations
- **Ctrl+K** — command palette for quick navigation

## Honesty Rules

- RUL is always in **timesteps** (never minutes/hours)
- The phrase "real-time" never appears in the UI
- A persistent "Simulated data" badge shows when the simulator is the source
- Sensor trust is display-only — it does not claim to drive mission decisions
