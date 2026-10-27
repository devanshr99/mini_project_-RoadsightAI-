import React, { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { 
  Eye, 
  BarChart3, 
  Database, 
  Cpu, 
  History, 
  Info, 
  ShieldCheck, 
  AlertCircle,
  Menu,
  X
} from 'lucide-react';
import { api } from '../services/api';
import { HealthStatus } from '../types';

export const Navbar: React.FC = () => {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    let isMounted = true;
    api.getHealth()
      .then(data => { if (isMounted) setHealth(data); })
      .catch(() => { if (isMounted) setHealth(null); });
    return () => { isMounted = false; };
  }, []);

  const navLinks = [
    { to: '/', label: 'Overview', icon: ShieldCheck },
    { to: '/recognize', label: 'Recognize', icon: Eye },
    { to: '/evaluation', label: 'Evaluation', icon: BarChart3 },
    { to: '/dataset', label: 'Dataset', icon: Database },
    { to: '/models', label: 'Models', icon: Cpu },
    { to: '/experiments', label: 'Experiments', icon: History },
    { to: '/about', label: 'About', icon: Info },
  ];

  return (
    <header className="sticky top-0 z-50 bg-[#fafaf8]/90 backdrop-blur-md border-b border-stone-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Brand */}
          <NavLink to="/" className="flex items-center space-x-3 group">
            <div className="w-9 h-9 rounded-lg bg-stone-900 text-white flex items-center justify-center font-bold shadow-sm group-hover:bg-[#e04a2f] transition-colors">
              <svg className="w-5 h-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="12 2 2 22 22 22 12 2"></polygon>
                <line x1="12" y1="9" x2="12" y2="13"></line>
                <line x1="12" y1="17" x2="12.01" y2="17"></line>
              </svg>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-semibold text-base text-stone-900 tracking-tight">RoadSight AI</span>
                <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-stone-100 text-stone-600 border border-stone-200">
                  GTSRB
                </span>
              </div>
              <p className="text-[11px] text-stone-500 hidden sm:block">Road Sign Recognition &amp; Explainability</p>
            </div>
          </NavLink>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-1">
            {navLinks.map((link) => {
              const Icon = link.icon;
              return (
                <NavLink
                  key={link.to}
                  to={link.to}
                  className={({ isActive }) =>
                    `flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                      isActive
                        ? 'bg-stone-900 text-white shadow-xs'
                        : 'text-stone-600 hover:text-stone-900 hover:bg-stone-100'
                    }`
                  }
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{link.label}</span>
                </NavLink>
              );
            })}
          </nav>

          {/* Device & Status Badge */}
          <div className="hidden sm:flex items-center space-x-2">
            {health ? (
              <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-800 text-[11px] font-mono">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span className="font-medium">Backend Ready ({health.device.toUpperCase()})</span>
              </div>
            ) : (
              <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-amber-50 border border-amber-200 text-amber-800 text-[11px] font-mono">
                <AlertCircle className="w-3 h-3 text-amber-600" />
                <span>Checking Server...</span>
              </div>
            )}
          </div>

          {/* Mobile menu toggle */}
          <div className="flex md:hidden">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-md text-stone-700 hover:bg-stone-100"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu dropdown */}
      {mobileMenuOpen && (
        <div className="md:hidden border-b border-stone-200 bg-[#fafaf8] px-4 pt-2 pb-4 space-y-1">
          {navLinks.map((link) => {
            const Icon = link.icon;
            return (
              <NavLink
                key={link.to}
                to={link.to}
                onClick={() => setMobileMenuOpen(false)}
                className={({ isActive }) =>
                  `flex items-center space-x-2.5 px-3 py-2 rounded-md text-sm font-medium ${
                    isActive
                      ? 'bg-stone-900 text-white'
                      : 'text-stone-700 hover:bg-stone-100'
                  }`
                }
              >
                <Icon className="w-4 h-4" />
                <span>{link.label}</span>
              </NavLink>
            );
          })}
          <div className="pt-2">
            {health && (
              <div className="px-3 py-1.5 rounded bg-emerald-50 text-emerald-800 text-xs font-mono">
                ● Device: {health.device.toUpperCase()} | Torch {health.torch_version}
              </div>
            )}
          </div>
        </div>
      )}
    </header>
  );
};
