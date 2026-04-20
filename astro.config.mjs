// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import remarkGithubAlerts from 'remark-github-alerts';

// https://astro.build/config
export default defineConfig({
  site: 'https://jade-ray.github.io',
  base: '/',
  markdown: {
    remarkPlugins: [remarkGithubAlerts],
  },
  vite: {
    plugins: [tailwindcss()],
  },
});