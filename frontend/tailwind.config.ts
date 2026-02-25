import type { Config } from 'tailwindcss'

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // TSX 파일까지 감시
  ],
  theme: {
    extend: {},
  },
  plugins: [],
} satisfies Config