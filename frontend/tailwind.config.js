/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#f9f9f7',
        surface: '#ffffff',
        'surface-muted': '#f4f4f0',
        primary: {
          DEFAULT: '#e04a2f',
          hover: '#c83b22',
          light: '#fdf3f0',
        },
        slate: {
          950: '#0d1117',
          900: '#161b22',
          800: '#21262d',
          700: '#30363d',
          600: '#484f58',
          500: '#6e7681',
          400: '#8b949e',
          300: '#b1bac4',
          200: '#d0d7de',
          100: '#eaeef2',
          50: '#f6f8fa',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      }
    },
  },
  plugins: [],
}
