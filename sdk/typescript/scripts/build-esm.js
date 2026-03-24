/**
 * Builds a true ESM output from the TypeScript source using tsconfig.esm.json,
 * then renames .js files to .mjs for Node ESM resolution.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const esmDir = path.join(__dirname, '..', 'dist', 'esm');

// Clean previous ESM output.
fs.rmSync(esmDir, { recursive: true, force: true });

// Compile ESM build via the dedicated tsconfig.
execSync('npx tsc -p tsconfig.esm.json', {
  cwd: path.join(__dirname, '..'),
  stdio: 'inherit',
});

// Rename .js → .mjs and rewrite import specifiers (.js → .mjs) in emitted files.
function rewriteDir(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      rewriteDir(fullPath);
      continue;
    }
    if (!entry.name.endsWith('.js')) continue;

    let content = fs.readFileSync(fullPath, 'utf8');

    // Rewrite relative import/export specifiers: './foo.js' → './foo.mjs'
    content = content.replace(
      /(from\s+['"])(\.\.?\/[^'"]+)(\.js)(['"])/g,
      '$1$2.mjs$4'
    );
    // Rewrite dynamic import() specifiers: import('./foo.js') → import('./foo.mjs')
    content = content.replace(
      /(import\s*\(\s*['"])(\.\.?\/[^'"]+)(\.js)(['"]\s*\))/g,
      '$1$2.mjs$4'
    );

    const mjsPath = fullPath.replace(/\.js$/, '.mjs');
    fs.writeFileSync(mjsPath, content);
    fs.unlinkSync(fullPath);
  }
}

rewriteDir(esmDir);
console.log('ESM build compiled at dist/esm/ (.mjs files)');
