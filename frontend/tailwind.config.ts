import type { Config } from 'tailwindcss'

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    // 아래와 같이 상대 경로를 더 구체적으로 적어주면 인식이 더 잘 됩니다.
    "./src/pages/**/*.{js,ts,jsx,tsx}",
    "./src/components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'main-blue': '#60A5FA',
        'main-gray': '#D9D9D9',
        'main-sky': '#E9F3FF',
        'main-red': '#FF0000',
      },
      backgroundImage: {
      'aria-gradient': 'radial-gradient(circle at center, #ffffff 0%, #e0ebff 100%)',
    }
    },
  },
  plugins: [],
} satisfies Config