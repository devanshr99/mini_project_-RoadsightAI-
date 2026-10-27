import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import { OverviewPage } from './pages/OverviewPage';
import { RecognizePage } from './pages/RecognizePage';
import { EvaluationPage } from './pages/EvaluationPage';
import { DatasetPage } from './pages/DatasetPage';
import { ModelsPage } from './pages/ModelsPage';
import { ExperimentsPage } from './pages/ExperimentsPage';
import { AboutPage } from './pages/AboutPage';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-[#fafaf8] text-[#1c2128] flex flex-col font-sans selection:bg-[#e04a2f] selection:text-white">
        <Navbar />

        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/recognize" element={<RecognizePage />} />
            <Route path="/evaluation" element={<EvaluationPage />} />
            <Route path="/dataset" element={<DatasetPage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/experiments" element={<ExperimentsPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>

        <footer className="border-t border-stone-200 bg-white py-6 mt-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-stone-500 font-mono">
            <div>
              <strong className="text-stone-800 font-sans">RoadSight AI</strong> &bull; B.Tech Computer Science Final Project
            </div>
            <div className="flex items-center space-x-3">
              <span>Local-First Execution</span>
              <span>&bull;</span>
              <span>GTSRB 43-Class Benchmark</span>
              <span>&bull;</span>
              <span>Grad-CAM Explainability</span>
            </div>
          </div>
        </footer>
      </div>
    </BrowserRouter>
  );
};

export default App;
