import React, { useState, useRef, useCallback } from 'react';
import { Sliders } from 'lucide-react';

interface Props {
  originalImage: string;
  overlayImage: string;
  heatmapImage?: string;
}

export const ImageComparisonSlider: React.FC<Props> = ({
  originalImage,
  overlayImage,
  heatmapImage,
}) => {
  const [sliderPos, setSliderPos] = useState(50);
  const [viewMode, setViewMode] = useState<'slider' | 'side-by-side' | 'heatmap'>('slider');
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);

  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const percent = Math.min(Math.max((x / rect.width) * 100, 0), 100);
    setSliderPos(percent);
  }, []);

  const handleMouseDown = () => {
    isDragging.current = true;
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging.current) return;
    handleMove(e.clientX);
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (e.touches.length > 0) {
      handleMove(e.touches[0].clientX);
    }
  };

  return (
    <div className="space-y-3">
      {/* View mode toggle controls */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-stone-600 flex items-center gap-1.5">
          <Sliders className="w-3.5 h-3.5 text-stone-500" />
          Explainability Visualizer
        </span>
        <div className="inline-flex rounded-lg border border-stone-200 bg-stone-100 p-0.5 text-xs font-medium">
          <button
            type="button"
            onClick={() => setViewMode('slider')}
            className={`px-2.5 py-1 rounded-md transition-all ${
              viewMode === 'slider'
                ? 'bg-white text-stone-900 shadow-xs font-semibold'
                : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            Split Slider
          </button>
          <button
            type="button"
            onClick={() => setViewMode('side-by-side')}
            className={`px-2.5 py-1 rounded-md transition-all ${
              viewMode === 'side-by-side'
                ? 'bg-white text-stone-900 shadow-xs font-semibold'
                : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            Side-by-Side
          </button>
          {heatmapImage && (
            <button
              type="button"
              onClick={() => setViewMode('heatmap')}
              className={`px-2.5 py-1 rounded-md transition-all ${
                viewMode === 'heatmap'
                  ? 'bg-white text-stone-900 shadow-xs font-semibold'
                  : 'text-stone-600 hover:text-stone-900'
              }`}
            >
              Jet Heatmap
            </button>
          )}
        </div>
      </div>

      {viewMode === 'slider' && (
        <div
          ref={containerRef}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onMouseMove={handleMouseMove}
          onTouchMove={handleTouchMove}
          className="relative w-full aspect-square max-w-[340px] mx-auto overflow-hidden rounded-xl border border-stone-300 shadow-sm select-none cursor-ew-resize bg-stone-950"
        >
          {/* Background: Grad-CAM Overlay */}
          <img
            src={overlayImage}
            alt="Grad-CAM Overlay"
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
          />

          {/* Foreground: Original Image clipped by slider position */}
          <div
            className="absolute inset-0 overflow-hidden pointer-events-none"
            style={{ width: `${sliderPos}%` }}
          >
            <img
              src={originalImage}
              alt="Original Input"
              className="absolute inset-0 w-full h-full object-contain max-w-none"
              style={{ width: containerRef.current ? `${containerRef.current.clientWidth}px` : '100%' }}
            />
          </div>

          {/* Divider Line */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-white shadow-md pointer-events-none"
            style={{ left: `${sliderPos}%` }}
          >
            <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-6 h-6 rounded-full bg-stone-900 text-white border-2 border-white flex items-center justify-center text-[10px] shadow-sm">
              ↔
            </div>
          </div>

          {/* Labels */}
          <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/60 backdrop-blur-xs text-[10px] text-white font-mono pointer-events-none">
            Original
          </div>
          <div className="absolute top-2 right-2 px-2 py-0.5 rounded bg-black/60 backdrop-blur-xs text-[10px] text-white font-mono pointer-events-none">
            Grad-CAM Heatmap
          </div>
          <div className="absolute bottom-2 inset-x-0 text-center pointer-events-none">
            <span className="px-2 py-0.5 rounded bg-black/50 text-[10px] text-stone-300">
              Drag slider left / right to inspect attention
            </span>
          </div>
        </div>
      )}

      {viewMode === 'side-by-side' && (
        <div className="grid grid-cols-2 gap-3 max-w-[420px] mx-auto">
          <div className="space-y-1.5 text-center">
            <div className="w-full aspect-square rounded-xl border border-stone-200 overflow-hidden bg-stone-950 flex items-center justify-center shadow-xs">
              <img
                src={originalImage}
                alt="Original Sign"
                className="w-full h-full object-contain"
              />
            </div>
            <span className="text-[11px] font-medium text-stone-600">Original Sign</span>
          </div>
          <div className="space-y-1.5 text-center">
            <div className="w-full aspect-square rounded-xl border border-stone-200 overflow-hidden bg-stone-950 flex items-center justify-center shadow-xs">
              <img
                src={overlayImage}
                alt="Grad-CAM Overlay"
                className="w-full h-full object-contain"
              />
            </div>
            <span className="text-[11px] font-medium text-stone-600">Grad-CAM Overlay</span>
          </div>
        </div>
      )}

      {viewMode === 'heatmap' && heatmapImage && (
        <div className="max-w-[280px] mx-auto space-y-1.5 text-center">
          <div className="w-full aspect-square rounded-xl border border-stone-200 overflow-hidden bg-stone-950 flex items-center justify-center shadow-xs">
            <img
              src={heatmapImage}
              alt="Jet Colormap Heatmap"
              className="w-full h-full object-contain"
            />
          </div>
          <span className="text-[11px] font-medium text-stone-600">Raw Jet Activation Map (Red = High Attention)</span>
        </div>
      )}
    </div>
  );
};
