import React from 'react';
import { motion } from 'framer-motion';
import { useStore } from '../store/useStore';
import GlassCard from '../components/GlassCard';
import MissionBanner from '../components/MissionBanner';
import ConfirmGate from '../components/ConfirmGate';
import { Crosshair, Shield, ActivitySquare, AlertTriangle } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.15 } }
};

const itemVariants = {
  hidden: { opacity: 0, scale: 0.95, y: 20 },
  visible: { opacity: 1, scale: 1, y: 0, transition: { type: "spring", stiffness: 100 } }
};

export default function Mission() {
  const { currentTick, tickHistory, auditTrail, remainingMissionTime, setRemainingMissionTime } = useStore();

  if (!currentTick) return null;

  const actionChanges = auditTrail.filter(log => log.message?.includes('Action updated to') || log.actionChanged);
  const elapsed = currentTick.tick || 0;
  const totalMissionTime = elapsed + (remainingMissionTime || 0);
  const timeData = [
    { name: 'Elapsed', value: elapsed },
    { name: 'Remaining', value: remainingMissionTime > 0 ? remainingMissionTime : 0 }
  ];
  const pieColors = ['#B026FF', '#1B0F2E'];

  return (
    <motion.div 
      initial="hidden"
      animate="visible"
      variants={containerVariants}
      className="p-4 md:p-8 space-y-8 w-full"
    >
      <motion.header variants={itemVariants}>
        <h1 className="text-3xl font-bold tracking-wider mb-2 text-white drop-shadow-md">Mission & Operator</h1>
        <p className="text-slate-400">Action recommendations, confirmations, and dynamic risk tracking.</p>
      </motion.header>

      {/* Top Banner & Confirm Gate */}
      <motion.section variants={itemVariants} className="space-y-6">
        <MissionBanner tick={currentTick} />
        {currentTick.requires_confirmation && (
          <ConfirmGate tick={currentTick} />
        )}
      </motion.section>

      {/* 3-Column Spread Layout */}
      <motion.div variants={containerVariants} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 w-full items-stretch">
        
        {/* Col 1: Mission Parameters & Time Graph */}
        <motion.div variants={itemVariants} className="flex flex-col h-full">
          <GlassCard className="p-6 flex flex-col h-full hover:shadow-[0_0_20px_rgba(0,240,255,0.2)] transition-shadow">
            <div className="flex items-center gap-2 mb-6">
              <Crosshair className="w-6 h-6 text-cyan-400" />
              <h3 className="text-xl font-semibold text-white">Mission Clock</h3>
            </div>
            
            <div className="flex-1 flex flex-col justify-center items-center relative min-h-[200px] mb-6">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={timeData}
                    innerRadius="70%"
                    outerRadius="90%"
                    startAngle={90}
                    endAngle={-270}
                    dataKey="value"
                    stroke="none"
                  >
                    {timeData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={pieColors[index % pieColors.length]} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'rgba(27, 15, 46, 0.9)', border: '1px solid rgba(176,38,255,0.3)', borderRadius: '8px' }}
                    itemStyle={{ color: '#fff' }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                <span className="text-3xl font-bold mono text-white">{elapsed}</span>
                <span className="text-xs text-slate-400 uppercase tracking-widest">Elapsed</span>
              </div>
            </div>

            <div className="space-y-4 mt-auto">
              <div>
                <label className="block text-sm text-slate-400 mb-2">Adjust Remaining Time</label>
                <input 
                  type="number" 
                  value={remainingMissionTime}
                  onChange={(e) => setRemainingMissionTime(parseInt(e.target.value) || 0)}
                  className="w-full bg-black/40 border border-purple-500/30 rounded-lg px-4 py-3 text-white mono focus:outline-none focus:border-cyan-400 transition-colors shadow-inner"
                />
              </div>
              <div className="pt-4 border-t border-white/10 flex justify-between items-center text-sm">
                <span className="text-slate-400">Total Est. Mission Time</span>
                <span className="mono text-cyan-300 font-bold">{totalMissionTime}</span>
              </div>
            </div>
          </GlassCard>
        </motion.div>

        {/* Col 2: Risk Assessment & History */}
        <motion.div variants={itemVariants} className="flex flex-col h-full">
          <GlassCard className="p-6 flex flex-col h-full hover:shadow-[0_0_20px_rgba(255,0,85,0.2)] transition-shadow">
            <div className="flex items-center gap-2 mb-6">
              <Shield className="w-6 h-6 text-pink-500" />
              <h3 className="text-xl font-semibold text-white">Risk Tracker</h3>
            </div>
            
            <div className="flex items-end justify-between mb-4">
              <span className="text-5xl font-bold mono text-white drop-shadow-[0_0_10px_rgba(255,0,85,0.5)]">{currentTick.risk_score}</span>
              <span className="text-sm text-slate-400 uppercase tracking-widest mb-2">/ 100 Risk Index</span>
            </div>
            
            <div className="flex-1 min-h-[150px] mb-6 border border-white/5 rounded-xl bg-black/20 p-2 relative overflow-hidden">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={tickHistory.slice(-100)}>
                  <defs>
                    <linearGradient id="colorRisk" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#FF0055" stopOpacity={0.6}/>
                      <stop offset="95%" stopColor="#FF0055" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="tick" hide />
                  <YAxis domain={[0, 100]} hide />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'rgba(27, 15, 46, 0.9)', border: '1px solid rgba(255,0,85,0.3)', borderRadius: '8px' }}
                    labelStyle={{ color: '#ccc' }}
                  />
                  <Area type="monotone" dataKey="risk_score" stroke="#FF0055" strokeWidth={3} fillOpacity={1} fill="url(#colorRisk)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            
            <div className="mt-auto pt-4 border-t border-white/10">
              <h4 className="text-sm font-semibold text-slate-300 flex items-center gap-2 mb-3">
                <AlertTriangle className="w-4 h-4 text-amber-500" /> Active Risk Factors
              </h4>
              {currentTick.reasons && currentTick.reasons.length > 0 ? (
                <ul className="text-sm text-slate-400 space-y-2 pl-2">
                  {currentTick.reasons.map((reason, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-pink-500 mt-1">•</span> {reason}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-slate-500 italic">No significant risk factors.</p>
              )}
            </div>
          </GlassCard>
        </motion.div>

        {/* Col 3: Action Timeline */}
        <motion.div variants={itemVariants} className="flex flex-col h-full">
          <GlassCard className="p-6 flex flex-col h-full relative overflow-hidden hover:shadow-[0_0_20px_rgba(176,38,255,0.2)] transition-shadow">
            <div className="flex items-center gap-2 mb-6">
              <ActivitySquare className="w-6 h-6 text-purple-400" />
              <h3 className="text-xl font-semibold text-white">Action Ledger</h3>
            </div>
            
            <div className="flex-1 overflow-y-auto pr-2 space-y-4 max-h-[500px] custom-scrollbar">
              {actionChanges.length > 0 ? (
                actionChanges.map((log, idx) => (
                  <motion.div 
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    key={idx} 
                    className="flex gap-4 relative group"
                  >
                    <div className="absolute left-[11px] top-8 bottom-[-16px] w-0.5 bg-purple-500/20 group-last:hidden" />
                    
                    <div className="w-6 h-6 rounded-full bg-purple-900/50 border border-purple-500 flex-shrink-0 mt-1 flex items-center justify-center relative z-10 shadow-[0_0_10px_rgba(176,38,255,0.4)]">
                      <div className="w-2 h-2 rounded-full bg-purple-400" />
                    </div>
                    
                    <div className="flex-1 p-3 rounded-lg bg-black/30 border border-white/5 group-hover:border-purple-500/30 transition-colors">
                      <div className="flex justify-between items-start mb-1">
                        <span className="mono text-purple-300 text-xs font-bold uppercase tracking-wider">T-{log.tick}</span>
                        <span className="text-xs text-slate-500">{new Date(log.timestamp).toLocaleTimeString()}</span>
                      </div>
                      <p className="text-sm text-slate-200">{log.message}</p>
                    </div>
                  </motion.div>
                ))
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-slate-500 italic opacity-50">
                  <ActivitySquare className="w-12 h-12 mb-2 text-slate-600" />
                  No major action changes recorded yet.
                </div>
              )}
            </div>
          </GlassCard>
        </motion.div>

      </motion.div>
    </motion.div>
  );
}
