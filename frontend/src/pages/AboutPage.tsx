import React from 'react';
import { 
  Code2, 
  BookOpen, 
  Terminal
} from 'lucide-react';

export const AboutPage: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-10 py-2">
      {/* Title */}
      <div className="border-b border-[#E5E2DC] pb-5">
        <h1 className="text-2xl font-bold text-[#171717] tracking-tight">
          System Documentation &amp; Academic Methodology
        </h1>
        <p className="text-xs text-[#66645F] mt-1">
          Technical specifications, neural pipeline mechanics, mathematical Grad-CAM formulation, and verification guidelines for the B.Tech Computer Science Road Sign Recognition &amp; Explainability System.
        </p>
      </div>

      {/* Project Objective */}
      <section className="bg-white p-6 rounded-xl border border-[#E5E2DC] shadow-xs space-y-3">
        <h2 className="text-sm font-semibold text-[#171717] flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-[#C94A3A]" />
          1. Research Objective &amp; Problem Formulation
        </h2>
        <p className="text-xs text-[#66645F] leading-relaxed">
          Autonomous vehicle navigation and Advanced Driver Assistance Systems (ADAS) depend on rapid, deterministic recognition of road signs under extreme real-world challenges: variable lighting, weather degradation, motion blur, physical occlusion, and viewpoint variance. Standard black-box convolutional neural networks can attain high classification accuracy while exploiting spurious dataset artifacts rather than authentic sign pictograms.
        </p>
        <p className="text-xs text-[#66645F] leading-relaxed">
          <strong>RoadSight AI</strong> addresses both classification robustness and model interpretability by pairing high-performance convolutional and residual networks with mathematical <strong>Grad-CAM (Gradient-Weighted Class Activation Mapping)</strong>. This provides safety engineers with visual saliency heatmaps that verify the network is attending directly to the sign's semantic core.
        </p>
      </section>

      {/* End-to-End Pipeline Architecture */}
      <section className="bg-white p-6 rounded-xl border border-[#E5E2DC] shadow-xs space-y-4">
        <h2 className="text-sm font-semibold text-[#171717] flex items-center gap-2">
          <Code2 className="w-4 h-4 text-[#171717]" />
          2. End-to-End Inference &amp; Saliency Dataflow
        </h2>

        <div className="p-4 rounded-lg bg-[#FAF9F6] border border-[#E5E2DC] font-mono text-[11px] text-[#171717] space-y-2 overflow-x-auto">
          <div className="font-semibold text-[#171717]">Sequential Execution Flow:</div>
          <div>[Arbitrary Input Image (PNG/JPG/WEBP, Max 10MB)]</div>
          <div className="text-[#8F8C85] pl-4">&darr; 3-channel RGB conversion &bull; Aspect-preserving bilinear resize to 64&times;64</div>
          <div>[Strict Input Tensor Shape Assert: [1, 3, 64, 64] &bull; GTSRB Mean/Std Normalization]</div>
          <div className="text-[#8F8C85] pl-4">&darr; Forward pass through instantiated PyTorch model (Strict state_dict verified)</div>
          <div>[Logits Vector (43 Classes)] &rarr; Softmax Activation &rarr; Top-3 Predicted Confidences</div>
          <div className="text-[#8F8C85] pl-4">&darr; Backward pass from target score to target convolutional layer feature maps</div>
          <div>[Pooled Gradients &alpha;_k^c] &times; [Feature Maps A^k] &rarr; ReLU &rarr; Jet Colormap Overlay (&alpha; = 0.5)</div>
          <div className="text-[#8F8C85] pl-4">&darr; Safe PyTorch hook cleanup: handle.remove() prevents memory leaks</div>
          <div>[REST API Response: JSON metadata + Base64 PNG visualizations (&lt; 30ms latency)]</div>
          <div>[Interactive React Comparison Slider / Side-by-Side View]</div>
        </div>
      </section>

      {/* Mathematical Formulation of Grad-CAM */}
      <section className="bg-white p-6 rounded-xl border border-[#E5E2DC] shadow-xs space-y-3">
        <h2 className="text-sm font-semibold text-[#171717]">
          3. Mathematical Formulation of Grad-CAM
        </h2>
        <p className="text-xs text-[#66645F] leading-relaxed">
          Grad-CAM computes the gradient of the unnormalized score <code className="bg-[#FAF9F6] border border-[#E5E2DC] px-1 py-0.5 rounded font-mono text-[11px] text-[#171717]">y^c</code> for class <code className="bg-[#FAF9F6] border border-[#E5E2DC] px-1 py-0.5 rounded font-mono text-[11px] text-[#171717]">c</code> with respect to the feature activation maps <code className="bg-[#FAF9F6] border border-[#E5E2DC] px-1 py-0.5 rounded font-mono text-[11px] text-[#171717]">A^k</code> of the target convolutional layer:
        </p>

        <div className="p-3 bg-[#FAF9F6] rounded-lg text-center font-mono text-xs text-[#171717] border border-[#E5E2DC]">
          &alpha;_k^c = (1 / Z) &sum;_i &sum;_j (&part;y^c / &part;A_ij^k)
        </div>

        <p className="text-xs text-[#66645F] leading-relaxed">
          The importance weights <code className="bg-[#FAF9F6] border border-[#E5E2DC] px-1 py-0.5 rounded font-mono text-[11px] text-[#171717]">&alpha;_k^c</code> capture the significance of feature map <code className="bg-[#FAF9F6] border border-[#E5E2DC] px-1 py-0.5 rounded font-mono text-[11px] text-[#171717]">k</code> for target category <code className="bg-[#FAF9F6] border border-[#E5E2DC] px-1 py-0.5 rounded font-mono text-[11px] text-[#171717]">c</code>. A weighted sum of activation maps followed by a rectified linear unit (ReLU) generates the localized saliency map:
        </p>

        <div className="p-3 bg-[#FAF9F6] rounded-lg text-center font-mono text-xs text-[#171717] border border-[#E5E2DC]">
          L_Grad-CAM^c = ReLU( &sum;_k &alpha;_k^c A^k )
        </div>

        <p className="text-[11px] text-[#8F8C85] leading-relaxed">
          The ReLU filter isolates positive contributions that increase the target class score, filtering out features that correlate with other classes.
        </p>
      </section>

      {/* Local Execution & Startup Commands */}
      <section className="bg-white p-6 rounded-xl border border-[#E5E2DC] shadow-xs space-y-4">
        <h2 className="text-sm font-semibold text-[#171717] flex items-center gap-2">
          <Terminal className="w-4 h-4 text-[#1E6B37]" />
          4. Local Execution &amp; Reproducibility
        </h2>

        <div className="space-y-3 text-xs">
          <div>
            <span className="font-semibold text-[#171717] block mb-1">A. Flask Backend Server:</span>
            <pre className="p-3 rounded-lg bg-[#171717] text-white font-mono text-[11px] overflow-x-auto">
python app.py
# Server starts on http://127.0.0.1:5000
            </pre>
          </div>

          <div>
            <span className="font-semibold text-[#171717] block mb-1">B. React + Vite Frontend:</span>
            <pre className="p-3 rounded-lg bg-[#171717] text-white font-mono text-[11px] overflow-x-auto">
cd frontend
npm run dev
# Vite development interface on http://localhost:5173
            </pre>
          </div>

          <div>
            <span className="font-semibold text-[#171717] block mb-1">C. Run Automated Validation Suite:</span>
            <pre className="p-3 rounded-lg bg-[#171717] text-white font-mono text-[11px] overflow-x-auto">
pytest tests/
# Executes 29 automated tests across models, tensor shapes, Grad-CAM and REST endpoints
            </pre>
          </div>
        </div>
      </section>
    </div>
  );
};
