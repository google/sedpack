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
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/google/sedpack',
        }
      ],
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
        {
          label: 'Tutorials',
          items: [
            { label: 'MNIST', slug: 'tutorials/mnist' },
            {
              label: 'Side Channel Attacks',
              items: [
                { label: 'SCA Overview', slug: 'tutorials/sca/overview' },
                { label: 'Dataset Preparation', slug: 'tutorials/sca/dataset' },
                {
			label: 'Classical Attacks',
			items: [
				{ label: 'Signal to Noise Ratio', slug: 'tutorials/sca/snr' },
				{ label: 'GPU Acceleration of CPA and Template Attacks', slug: 'tutorials/sca/gpu_cpa_template' },
			],
		},
                { label: 'Deep Learning (GPAM)', slug: 'tutorials/sca/gpam' },
              ],
            },
          ],
        },
      ],
    }),
  ],
});
