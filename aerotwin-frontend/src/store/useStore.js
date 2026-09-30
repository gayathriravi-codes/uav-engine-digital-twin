/**
 * src/store/useStore.js
 * Central Zustand store for telemetry, controls, scrubbing, What-If, and UI states.
 */
import { create } from "zustand";
import { generateFlight } from "../data/simSource";

const INITIAL_SCENARIO = {
  fault: "none",
  severity: 0.5,
  seed: 42,
};

// Initial baseline ticks
const initialFlight = generateFlight({
  fault: INITIAL_SCENARIO.fault,
  severity: INITIAL_SCENARIO.severity,
  seed: INITIAL_SCENARIO.seed,
  totalTicks: 90,
});
const initialTick = initialFlight[0];

export const useStore = create((set, get) => ({
  // Telemetry Source
  source: "sim", // 'sim' | 'api'
  connectionStatus: "connected", // 'connected' | 'connecting' | 'unreachable' | 'fallback'
  fallbackNotice: null,

  setSource: (source) => set({ source }),
  setConnectionStatus: (connectionStatus) => set({ connectionStatus }),
  setFallbackNotice: (fallbackNotice) => set({ fallbackNotice }),

  // Toasts
  toasts: [],
  addToast: (toast) => set(state => ({
    toasts: [...state.toasts, { id: Date.now() + Math.random(), createdAt: Date.now(), ...toast }].slice(-5)
  })),
  removeToast: (id) => set(state => ({ toasts: state.toasts.filter(t => t.id !== id) })),

  // UI Modes & Modals
  commandPaletteOpen: false,
  setCommandPaletteOpen: (v) => set({ commandPaletteOpen: v }),
  
  selectedSensorModal: null,
  setSelectedSensorModal: (key) => set({ selectedSensorModal: key }),
  
  replayMode: false,
  setReplayMode: (v) => set({ replayMode: v }),

  selectedSensor: null,
  setSelectedSensor: (selectedSensor) => set({ selectedSensor }),

  // Playback & Scenario Controls
  isPlaying: true,
  playbackSpeed: 1.0, // Multiplier: 0.25, 0.5, 1, 2, 4
  scenario: INITIAL_SCENARIO,
  remainingMissionTime: 180, // In TIMESTEPS

  setIsPlaying: (isPlaying) => set({ isPlaying }),
  setPlaybackSpeed: (playbackSpeed) => set({ playbackSpeed }),
  setScenario: (patch) => {
    const nextScenario = { ...get().scenario, ...patch };
    const currentT = get().currentTick?.t || 0;
    const targetT = nextScenario.fault !== "none" ? Math.max(currentT, 4) : 0;
    // Regenerate flight data for scrub timeline when scenario changes
    const newFlight = generateFlight({
      fault: nextScenario.fault,
      severity: nextScenario.severity,
      seed: nextScenario.seed,
      totalTicks: 90,
      onsetTick: 2,
    });
    const effectiveT = Math.min(targetT, newFlight.length - 1);
    set({
      scenario: nextScenario,
      flightTicks: newFlight,
      scrubberIndex: effectiveT,
      currentTick: newFlight[effectiveT],
      tickHistory: newFlight.slice(0, effectiveT + 1),
    });
  },
  setRemainingMissionTime: (remainingMissionTime) =>
    set({ remainingMissionTime: Math.max(1, Number(remainingMissionTime) || 1) }),

  // Active Telemetry State
  currentTick: initialTick,
  tickHistory: [initialTick], // Rolling buffer of recent ticks
  flightTicks: initialFlight, // Full flight ticks for scrubbing
  scrubberIndex: 0,
  isScrubbing: false,

  setCurrentTick: (tick) => {
    const history = get().tickHistory;
    const nextHistory = [...history.slice(-299), tick];
    
    // Check if action changed or confirmation triggered to log in audit trail
    const prevTick = get().currentTick;
    if (prevTick && (prevTick.action !== tick.action || tick.requires_confirmation)) {
      get().addAuditRecord({
        id: `${tick.t}-${Date.now()}`,
        timestamp: new Date().toLocaleTimeString(),
        tick: tick.t,
        missionClock: tick.mission_clock,
        action: tick.action,
        fault: tick.predicted_fault,
        confidence: tick.confidence,
        reasons: tick.reasons,
        actionChanged: prevTick.action !== tick.action,
        confirmationStatus: tick.requires_confirmation ? "Pending Operator Gate" : "Autonomous",
      });
    }
    
    if (prevTick && prevTick.action !== tick.action) {
      get().addToast({
        type: tick.action === 'ABORT' || tick.action === 'RTB' ? 'critical' : tick.action === 'REDUCE_LOAD' ? 'warning' : 'info',
        title: 'Mission action changed',
        message: `${prevTick.action} → ${tick.action}`,
      });
    }
    // Check for newly untrusted sensors
    if (tick.untrusted_sensors && tick.untrusted_sensors.length > 0 && prevTick) {
      const newUntrusted = tick.untrusted_sensors.filter(s => !prevTick.untrusted_sensors?.includes(s));
      newUntrusted.forEach(s => {
        get().addToast({ type: 'warning', title: 'Sensor untrusted', message: `${s} flagged as untrusted` });
      });
    }

    set({
      currentTick: tick,
      tickHistory: nextHistory,
      scrubberIndex: tick.t,
    });
  },

  setFlightTicks: (flightTicks) => set({ flightTicks }),
  setScrubberIndex: (idx) => {
    const ticks = get().flightTicks;
    if (ticks && ticks[idx]) {
      set({
        scrubberIndex: idx,
        currentTick: ticks[idx],
      });
    }
  },
  setIsScrubbing: (isScrubbing) => set({ isScrubbing }),

  resetSimulation: () => {
    const { scenario } = get();
    const newFlight = generateFlight({
      fault: scenario.fault,
      severity: scenario.severity,
      seed: scenario.seed,
      totalTicks: 90,
    });
    set({
      flightTicks: newFlight,
      scrubberIndex: 0,
      currentTick: newFlight[0],
      tickHistory: [newFlight[0]],
      isPlaying: true,
    });
  },

  // What-If Counterfactual Comparison
  whatIfScenario: {
    fault: "cooling_degradation",
    severity: 0.8,
    seed: 42,
  },
  whatIfData: null,
  whatIfSensor: "cht",
  whatIfLoading: false,

  setWhatIfScenario: (patch) =>
    set({ whatIfScenario: { ...get().whatIfScenario, ...patch } }),
  setWhatIfData: (whatIfData) => set({ whatIfData }),
  setWhatIfSensor: (whatIfSensor) => set({ whatIfSensor }),
  setWhatIfLoading: (whatIfLoading) => set({ whatIfLoading }),

  // Audit Trail
  auditTrail: [
    {
      id: "init-0",
      timestamp: new Date().toLocaleTimeString(),
      tick: 0,
      missionClock: "00:00:00",
      action: "CONTINUE",
      fault: "none",
      confidence: 0.98,
      reasons: ["Telemetry parameters within certified flight limits."],
      actionChanged: false,
      confirmationStatus: "Autonomous Initialized",
    },
  ],

  addAuditRecord: (record) => {
    // Newest first
    set((state) => ({
      auditTrail: [record, ...state.auditTrail.slice(0, 99)],
    }));
  },

  updateAuditConfirmation: (t, decision) => {
    set((state) => ({
      auditTrail: state.auditTrail.map((item) =>
        item.tick === t
          ? {
              ...item,
              confirmationStatus: decision === "confirm" ? "Confirmed by Operator" : "Rejected by Operator",
            }
          : item
      ),
    }));
  },
}));
