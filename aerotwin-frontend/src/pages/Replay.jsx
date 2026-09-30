import React, { useState, useEffect } from 'react';
import { useStore } from '../store/useStore';
import { generateFlight } from '../data/simSource';
import { SENSOR_KEYS, SENSOR_METADATA, MISSION_ACTIONS, FAULT_TYPES } from '../config';
import { Play, Pause, ChevronLeft, ChevronRight } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import HealthOrb from '../components/HealthOrb';

const Replay = () => {
  const { scenario } = useStore();
  const [ticks, setTicks] = useState([]);
  const [replayTick, setReplayTick] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  useEffect(() => {
    // Generate a full flight for replay based on current scenario settings
    const data = generateFlight(scenario, 200);
    setTicks(data);
    setReplayTick(0);
    setIsPlaying(false);
  }, [scenario]);

  useEffect(() => {
    let interval;
    if (isPlaying && ticks.length > 0) {
      interval = setInterval(() => {
        setReplayTick((prev) => {
          if (prev >= ticks.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 1000 / playbackSpeed);
    }
    return () => clearInterval(interval);
  }, [isPlaying, ticks, playbackSpeed]);

  const handleSliderChange = (e) => {
    setReplayTick(parseInt(e.target.value));
  };

  const handleStepBack = () => setReplayTick(prev => Math.max(0, prev - 1));
  const handleStepForward = () => setReplayTick(prev => Math.min(ticks.length - 1, prev + 1));
  const togglePlay = () => setIsPlaying(prev => !prev);

  if (ticks.length === 0) return null;

  const currentData = ticks[replayTick];
  const actionColor = MISSION_ACTIONS[currentData.action]?.color || '#fff';

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-6 flex flex-col h-full space-y-6"
    >
      <header>
        <h1 className="text-2xl font-bold mb-2">Flight Replay</h1>
        <p className="text-gray-400">Review historical flights or simulated complete missions.</p>
      </header>

      <div className="glass-card p-6 flex flex-col space-y-6">
        <div className="flex items-center space-x-4">
          <input 
            type="range"
            className="glass-slider flex-1"
            min="0"
            max={ticks.length - 1}
            value={replayTick}
            onChange={handleSliderChange}
          />
          <div className="mono text-sm whitespace-nowrap min-w-[120px] text-right">
            Tick {replayTick} / {ticks.length - 1}
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <button className="glass-btn p-2" onClick={handleStepBack} disabled={replayTick === 0}>
              <ChevronLeft size={20} />
            </button>
            <button className="glass-btn p-2 glass-btn--primary" onClick={togglePlay}>
              {isPlaying ? <Pause size={20} /> : <Play size={20} />}
            </button>
            <button className="glass-btn p-2" onClick={handleStepForward} disabled={replayTick === ticks.length - 1}>
              <ChevronRight size={20} />
            </button>
          </div>

          <div className="flex items-center space-x-2">
            {[0.5, 1, 2, 4].map(speed => (
              <button 
                key={speed}
                className={`glass-chip ${playbackSpeed === speed ? 'bg-white/20' : ''}`}
                onClick={() => setPlaybackSpeed(speed)}
              >
                {speed}x
              </button>
            ))}
          </div>
        </div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div 
          key={replayTick}
          initial={{ opacity: 0.8, filter: 'blur(4px)' }}
          animate={{ opacity: 1, filter: 'blur(0px)' }}
          exit={{ opacity: 0.8, filter: 'blur(4px)' }}
          transition={{ duration: 0.2 }}
          className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1"
        >
          <div className="glass-card p-6 flex flex-col items-center justify-center space-y-6">
            <HealthOrb healthScore={currentData.health_score} size={160} />
            <div className="text-center mt-4">
              <div className="text-sm text-gray-400">Estimated RUL</div>
              <div className="text-3xl font-bold mono mt-1">{currentData.rul_estimate_timesteps} timesteps</div>
            </div>
          </div>

          <div className="glass-card p-6 lg:col-span-2 flex flex-col space-y-6">
            <h2 className="text-lg font-semibold border-b border-white/10 pb-2">Sensors</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {SENSOR_KEYS.slice(0, 8).map(key => {
                const meta = SENSOR_METADATA[key];
                const val = currentData.sensors[key];
                return (
                  <div key={key} className="bg-white/5 p-3 rounded-lg border border-white/10">
                    <div className="text-xs text-gray-400 mb-1">{meta.label}</div>
                    <div className="text-lg mono">{val.toFixed(2)} <span className="text-xs text-gray-500">{meta.unit}</span></div>
                  </div>
                );
              })}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 flex-1">
              <div className="bg-white/5 p-4 rounded-lg border border-white/10">
                <h3 className="text-sm font-semibold mb-2">Diagnosis</h3>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-400">Fault:</span>
                  <span className="font-medium text-red-400">{FAULT_TYPES[currentData.predicted_fault]?.label || 'Healthy'}</span>
                </div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-400">Confidence:</span>
                  <span className="mono">{(currentData.fault_confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Condition:</span>
                  <span>{currentData.condition_status}</span>
                </div>
              </div>

              <div className="bg-white/5 p-4 rounded-lg border border-white/10 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-1 h-full" style={{ backgroundColor: actionColor }} />
                <h3 className="text-sm font-semibold mb-2 pl-2">Mission Action</h3>
                <div className="text-xl font-bold mb-2 pl-2" style={{ color: actionColor }}>
                  {MISSION_ACTIONS[currentData.action]?.label || currentData.action}
                </div>
                <ul className="text-xs text-gray-400 list-disc pl-6 space-y-1">
                  {currentData.reasons.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

    </motion.div>
  );
};

export default Replay;
