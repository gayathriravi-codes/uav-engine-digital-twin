import React from 'react';
import { motion } from 'framer-motion';
import { PILLAR_CARDS, KNOWN_LIMITS, TEAM_NAME, TEAM_MEMBERS, HACKATHON_TAG } from '../config';
import { Activity, ShieldAlert, Cpu, Crosshair, UserCheck } from 'lucide-react';

const FlowConnector = () => (
  <div style={{
    width: 40, height: 2,
    background: 'linear-gradient(90deg, var(--color-healthy), var(--color-operator))',
    position: 'relative',
    alignSelf: 'center',
    flexShrink: 0,
    opacity: 0.5
  }}>
    <div style={{
      position: 'absolute', top: -3, width: 8, height: 8,
      borderRadius: '50%', background: 'var(--color-operator)',
      animation: 'flowDot 2s ease-in-out infinite',
    }} />
    <style>{`
      @keyframes flowDot {
        0% { left: 0; opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { left: calc(100% - 8px); opacity: 0; }
      }
    `}</style>
  </div>
);

const FlowNode = ({ icon: Icon, label }) => (
  <div className="glass-card p-4 flex flex-col items-center justify-center space-y-2 min-w-[120px] bg-white/5 border border-white/10">
    <Icon size={24} className="text-teal-400" />
    <span className="text-sm font-medium text-center">{label}</span>
  </div>
);

const About = () => {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-6 space-y-12 max-w-5xl mx-auto"
    >
      <header className="text-center space-y-4">
        <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-teal-400 to-purple-500">
          NIDAN Operator Console
        </h1>
        <p className="text-gray-400 text-lg max-w-2xl mx-auto">
          AI-enabled digital twin for aero piston engine health monitoring.
        </p>
        <div className="inline-block glass-chip text-teal-400 border-teal-500/30">
          {HACKATHON_TAG}
        </div>
      </header>

      <section>
        <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
          <Activity className="text-teal-400" /> System Workflow
        </h2>
        <div className="glass-card p-8 overflow-x-auto">
          <div className="flex items-center justify-between min-w-[800px]">
            <FlowNode icon={Activity} label="Sensor Stream" />
            <FlowConnector />
            <FlowNode icon={ShieldAlert} label="Sensor Trust" />
            <FlowConnector />
            <FlowNode icon={Cpu} label="Diagnosis & RUL" />
            <FlowConnector />
            <FlowNode icon={Crosshair} label="Mission Action" />
            <FlowConnector />
            <FlowNode icon={UserCheck} label="Operator Confirmation" />
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
          <Cpu className="text-purple-400" /> Core Pillars
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {PILLAR_CARDS.map((pillar, idx) => (
            <div key={idx} className="glass-card p-6 hover:bg-white/5 transition-colors">
              <h3 className="text-lg font-semibold mb-2 text-teal-300">{pillar.title}</h3>
              <p className="text-sm text-gray-400 leading-relaxed">{pillar.description}</p>
            </div>
          ))}
        </div>
      </section>

      <section>
        <div className="glass-card p-6 border-amber-500/30 bg-amber-500/5">
          <h2 className="text-lg font-semibold mb-4 text-amber-400 flex items-center gap-2">
            <ShieldAlert /> Known Limitations
          </h2>
          <ul className="list-disc pl-6 space-y-2 text-sm text-amber-200/80">
            {KNOWN_LIMITS.map((limit, idx) => (
              <li key={idx}>{limit}</li>
            ))}
          </ul>
        </div>
      </section>

      <section className="glass-card p-8 border-t-4 border-t-purple-500">
        <h2 className="text-xl font-semibold mb-6">Team {TEAM_NAME}</h2>
        <div className="flex flex-wrap gap-4 mb-8">
          {TEAM_MEMBERS.map((member, idx) => (
            <div key={idx} className="glass-chip bg-white/5 border border-white/10 px-4 py-2 flex items-center gap-2">
              <div className="w-6 h-6 rounded-full bg-gradient-to-br from-purple-500 to-teal-500 flex items-center justify-center text-xs font-bold">
                {member.charAt(0)}
              </div>
              <span>{member}</span>
            </div>
          ))}
        </div>
        <div>
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">Problem Statement</h3>
          <p className="text-lg">PS 26054 — AI-enabled digital twin for aero piston engine health monitoring</p>
        </div>
      </section>
    </motion.div>
  );
};

export default About;
