// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import remarkGfm from 'remark-gfm';
import remarkGithubAlerts from 'remark-github-alerts';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
  site: 'https://jade-ray.github.io',
  base: '/',
  markdown: {
    remarkPlugins: [remarkGfm, remarkGithubAlerts, remarkMath],
    rehypePlugins: [[rehypeKatex, {
      /** @param {string} errorCode */
      strict: (errorCode) => {
        if (errorCode === 'unicodeTextInMathMode') {
          return 'ignore';
        }

        return 'warn';
      },
    }]],
  },
  vite: {
    plugins: [tailwindcss()],
  },
});