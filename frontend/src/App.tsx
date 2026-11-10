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
      <div className="min-h-screen bg-[#F7F6F3] text-[#171717] flex flex-col font-sans selection:bg-[#FAF0EE] selection:text-[#C94A3A]">
        <Navbar />

        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
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

        <footer className="border-t border-[#E5E2DC] bg-white py-6 mt-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-[#66645F]">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-[#171717]">RoadSight AI</span>
              <span className="text-[#8F8C85]">&bull;</span>
              <span>B.Tech Computer Science Engineering Mini-Project</span>
            </div>
            <div className="flex items-center space-x-3 font-mono text-[11px] text-[#66645F]">
              <span>GTSRB 43-Class Benchmark</span>
              <span className="text-[#8F8C85]">&bull;</span>
              <span>Grad-CAM Saliency</span>
              <span className="text-[#8F8C85]">&bull;</span>
              <span className="text-[#1E6B37]">100% Local Inference</span>
            </div>
          </div>
        </footer>
      </div>
    </BrowserRouter>
  );
};

export default App;
