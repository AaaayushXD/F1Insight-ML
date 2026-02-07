/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#2C64DD',
        accent: '#A1BAF0',
        surface: '#E8EEFF',
        satellite: '#5E3FEF',
        h2: '#3C3C43',
      },
    },
  },
  plugins: [],
}
