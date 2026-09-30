import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '../store/useStore';
import { FAULT_LIBRARY } from '../config';
import GlassCard from '../components/GlassCard';
import ProbBars from '../components/ProbBars';
import ShapBars from '../components/ShapBars';
import RadialGauge from '../components/RadialGauge';
import { ShieldAlert, ChevronDown, ChevronUp } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export default function Diagnosis() {
  const { currentTick, tickHistory } = useStore();
  const [expandedCard, setExpandedCard] = useState(null);

  if (!currentTick) return null;

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-6 space-y-8"
    >
      <header>
        <h1 className="text-2xl font-bold tracking-wider mb-2">Diagnostics</h1>
        <p className="text-slate-400">AI-driven fault prediction & feature attribution.</p>
      </header>

      {/* Top Row: Prediction & Health */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <GlassCard className="col-span-2 p-6 flex flex-col justify-center">
          <div className="flex items-center gap-3 mb-4">
            <ShieldAlert className="w-6 h-6 text-purple-400" />
            <h2 className="text-xl font-semibold">Current Prediction</h2>
          </div>
          <div className="text-3xl font-bold mb-2 text-white">
            {currentTick.predicted_fault || 'Healthy'}
          </div>
          <div className="flex items-center gap-4 mt-4">
            <div className="px-3 py-1 bg-white/10 rounded border border-white/20 text-sm">
              {currentTick.condition_status}
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm text-slate-400">Confidence:</span>
              <span className="mono text-lg text-purple-400">
                {(currentTick.confidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </GlassCard>

        <GlassCard className="p-6 flex flex-col items-center justify-center">
          <h3 className="text-sm font-medium text-slate-400 mb-4">System Health</h3>
          <RadialGauge 
            value={currentTick.health_score} 
            min={0} max={100} 
            size={160} 
            strokeWidth={12} 
            color={currentTick.health_score > 70 ? "#14B8A6" : currentTick.health_score > 40 ? "#F59E0B" : "#EF4444"}
          />
        </GlassCard>
      </div>

      {/* Middle Row: Probabilities & Explanations */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        <GlassCard className="p-6 lg:col-span-1">
          <h3 className="text-lg font-semibold mb-4 text-slate-300">Confidence History</h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={tickHistory.slice(-50)}>
                <defs>
                  <linearGradient id="colorConf" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#B026FF" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#B026FF" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="tick" hide />
                <YAxis domain={[0, 1]} hide />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'rgba(27, 15, 46, 0.9)', border: '1px solid rgba(176,38,255,0.3)', borderRadius: '8px' }}
                  labelStyle={{ color: '#ccc' }}
                />
                <Area type="monotone" dataKey="confidence" stroke="#B026FF" strokeWidth={3} fillOpacity={1} fill="url(#colorConf)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-slate-400 mt-2 text-center">Past 50 timesteps</p>
        </GlassCard>

        <GlassCard className="p-6 lg:col-span-1">
          <h3 className="text-lg font-semibold mb-4">Class Probabilities</h3>
          <ProbBars 
            classProbs={currentTick.class_probs} 
            predictedFault={currentTick.predicted_fault} 
          />
        </GlassCard>

        <GlassCard className="p-6 lg:col-span-1">
          <h3 className="text-lg font-semibold mb-4">Feature Attribution (SHAP)</h3>
          <ShapBars explanation={currentTick.explanation} />
        </GlassCard>
      </div>

      {/* Bottom: Fault Library */}
      <section>
        <h3 className="text-xl font-bold mb-6">Fault Library Reference</h3>
        <motion.div 
          className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6"
          initial="hidden"
          animate="visible"
          variants={{
            hidden: { opacity: 0 },
            visible: { opacity: 1, transition: { staggerChildren: 0.1 } }
          }}
        >
          {Object.values(FAULT_LIBRARY).map((fault, idx) => {
            const isPredicted = fault.id === currentTick.predicted_fault;
            const isExpanded = expandedCard === idx;
            return (
              <motion.div key={idx} variants={{ hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } }}>
                <GlassCard 
                  className={`p-5 transition-all duration-300 h-full ${isPredicted ? 'border-purple-500/50 bg-purple-900/20 shadow-[0_0_15px_rgba(176,38,255,0.3)]' : ''}`}
                >
                  <div 
                    className="flex justify-between items-start cursor-pointer"
                    onClick={() => setExpandedCard(isExpanded ? null : idx)}
                  >
                    <div>
                      <h4 className="text-lg font-semibold flex items-center gap-2">
                        {fault.title}
                        {isPredicted && <span className="w-3 h-3 rounded-full bg-purple-500 animate-pulse shadow-[0_0_10px_#B026FF]" />}
                      </h4>
                      <p className="text-sm text-slate-400 mt-1">{fault.subtitle}</p>
                    </div>
                    {isExpanded ? <ChevronUp className="w-5 h-5 text-slate-400" /> : <ChevronDown className="w-5 h-5 text-slate-400" />}
                  </div>
                  
                  <div className="flex flex-wrap gap-2 mt-4">
                    {fault.affectedSensors.map(s => (
                      <span key={s} className="px-2 py-1 text-xs rounded bg-white/5 border border-white/10 text-slate-300">
                        {s}
                      </span>
                    ))}
                  </div>

                  <AnimatePresence>
                    {isExpanded && (
                      <motion.div 
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-4 pt-4 border-t border-white/10 text-sm text-slate-300 space-y-3 overflow-hidden"
                      >
                        <p><strong>Description:</strong> {fault.description}</p>
                        <p><strong>Signature:</strong> {fault.signature}</p>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </GlassCard>
              </motion.div>
            );
          })}
        </motion.div>
      </section>
    </motion.div>
  );
}
