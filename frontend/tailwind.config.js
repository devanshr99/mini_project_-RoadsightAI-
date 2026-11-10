/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#F7F6F3',
        surface: {
          DEFAULT: '#FFFFFF',
          subtle: '#FAF9F6',
          muted: '#F1EFEA',
          hover: '#EDEAE4',
        },
        border: {
          DEFAULT: '#E5E2DC',
          subtle: '#ECE9E3',
          strong: '#D0CDC5',
        },
        content: {
          primary: '#171717',
          secondary: '#66645F',
          tertiary: '#8F8C85',
        },
        brand: {
          DEFAULT: '#C94A3A',
          hover: '#B33D2F',
          subtle: '#FAF0EE',
          border: '#F0C4BD',
        },
        status: {
          success: '#1E6B37',
          'success-bg': '#EBF5EE',
          'success-border': '#C3E6CD',
          warning: '#92400E',
          'warning-bg': '#FEF3C7',
          'warning-border': '#FDE68A',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
      boxShadow: {
        'card': '0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02)',
        'card-hover': '0 4px 12px rgba(0, 0, 0, 0.06), 0 1px 3px rgba(0, 0, 0, 0.04)',
      }
    },
  },
  plugins: [],
}
