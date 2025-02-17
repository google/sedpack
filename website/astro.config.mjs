// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeMathJax from 'rehype-mathjax';

// https://astro.build/config
export default defineConfig({
  site: 'https://google.github.io/sedpack/',
  base: '/sedpack',

  // Configure `remark-math` and `rehype-mathjax` plugins:
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeMathJax],
  },

  integrations: [
    starlight({
      title: 'Sedpack Documentation',
      social: {
        github: 'https://github.com/google/sedpack',
      },
      // Custom CSS to style MathJax equations
      customCss: ['./src/mathjax.css'],
      sidebar: [
        {
          label: 'Start Here',
          items: [
            // Each item here is one entry in the navigation menu.
            { label: 'Getting Started', slug: 'start_here/intro' },
            { label: 'Installation', slug: 'start_here/install' },
          ],
        },
      ],
    }),
  ],
});
