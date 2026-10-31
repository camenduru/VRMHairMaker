import { defineConfig } from 'vite';

export default defineConfig({
  base: '/',
  build: { outDir: 'dist', emptyOutDir: true, target: 'es2020' },
  server: {
    port: 5174,
    proxy: {
      '/api': 'http://127.0.0.1:8300',
      '/results': 'http://127.0.0.1:8300',
      '/models': 'http://127.0.0.1:8300',
      '/examples': 'http://127.0.0.1:8300',
    },
  },
});
