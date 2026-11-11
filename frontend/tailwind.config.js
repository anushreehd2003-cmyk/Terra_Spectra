/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        terra: {
          dark: '#0f172a',
          card: '#1e293b',
          border: '#334155',
          emerald: '#10b981',
          danger: '#ef4444',
          warning: '#f59e0b'
        }
      }
    },
  },
  plugins: [],
}
