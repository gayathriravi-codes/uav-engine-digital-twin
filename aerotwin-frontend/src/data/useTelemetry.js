/**
 * src/data/useTelemetry.js
 * Central telemetry lifecycle hook coordinating simulator pacing or WebSocket streaming.
 */

import { useEffect, useRef } from "react";
import { useStore } from "../store/useStore";
import { generateSimTick } from "./simSource";
import { apiSource } from "./apiSource";

export function useTelemetry() {
  const {
    source,
    isPlaying,
    playbackSpeed,
    scenario,
    currentTick,
    isScrubbing,
    setSource,
    setConnectionStatus,
    setFallbackNotice,
    setCurrentTick,
  } = useStore();

  const simTickIndexRef = useRef(0);
  const isPlayingRef = useRef(isPlaying);
  const isScrubbingRef = useRef(isScrubbing);
  const scenarioRef = useRef(scenario);
  const currentTickRef = useRef(currentTick);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    isScrubbingRef.current = isScrubbing;
  }, [isScrubbing]);

  useEffect(() => {
    scenarioRef.current = scenario;
    simTickIndexRef.current = currentTick?.t || 0;
  }, [scenario, currentTick?.t]);

  // Telemetry loop for SimSource
  useEffect(() => {
    if (source !== "sim") return;

    setConnectionStatus("connected");

    // Interval time in ms adjusted by playbackSpeed (e.g. 1000ms / 2 = 500ms)
    const intervalMs = Math.max(100, Math.round(1000 / playbackSpeed));

    const timer = setInterval(() => {
      if (!isPlayingRef.current || isScrubbingRef.current) return;

      const nextT = (simTickIndexRef.current + 1) % 90;
      simTickIndexRef.current = nextT;

      const nextTick = generateSimTick({
        t: nextT,
        fault: scenarioRef.current.fault,
        severity: scenarioRef.current.severity,
        seed: scenarioRef.current.seed,
        onsetTick: 2,
        prevTick: currentTickRef.current,
        tickSeconds: 1.0,
      });

      currentTickRef.current = nextTick;
      setCurrentTick(nextTick);
    }, intervalMs);

    return () => clearInterval(timer);
  }, [source, isPlaying, playbackSpeed, setCurrentTick, setConnectionStatus]);

  // Telemetry connection for ApiSource
  useEffect(() => {
    if (source !== "api") {
      apiSource.disconnectStream();
      return;
    }

    setConnectionStatus("connecting");
    setFallbackNotice(null);

    // First do a health check
    apiSource.checkHealth().then((isHealthy) => {
      if (!isHealthy) {
        setConnectionStatus("unreachable");
        setFallbackNotice(
          "Backend API at " + apiSource.baseUrl + " is unreachable. Reverted automatically to simulator."
        );
        // Clean fallback to sim
        setSource("sim");
        return;
      }

      // Connect to WebSocket stream
      apiSource.connectStream({
        fault: scenario.fault,
        severity: scenario.severity,
        seed: scenario.seed,
        tickSeconds: 1.0 / playbackSpeed,
        session: "console-session",
        onStatusChange: (status) => setConnectionStatus(status),
        onTick: (tick) => {
          if (!isScrubbingRef.current) {
            currentTickRef.current = tick;
            setCurrentTick(tick);
          }
        },
        onError: () => {
          setConnectionStatus("unreachable");
          setFallbackNotice(
            "WebSocket connection failed. Falling back smoothly to simulator."
          );
          setSource("sim");
        },
        onClose: () => {
          setConnectionStatus("unreachable");
        },
      });
    });

    return () => {
      apiSource.disconnectStream();
    };
  }, [source, scenario, playbackSpeed, setCurrentTick, setConnectionStatus, setFallbackNotice, setSource]);
}
