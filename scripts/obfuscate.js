/**
 * Post-build obfuscation step.
 * Processes all .js and .mjs files in dist/, leaving .d.ts untouched.
 */

const fs = require('fs');
const path = require('path');
const JavaScriptObfuscator = require('javascript-obfuscator');

const distDir = path.join(__dirname, '..', 'dist');

const obfuscatorConfig = {
  // ── High-level preset ──────────────────────────────────
  compact: true,
  controlFlowFlattening: true,
  controlFlowFlatteningThreshold: 1,
  deadCodeInjection: true,
  deadCodeInjectionThreshold: 1,
  debugProtection: true,
  debugProtectionInterval: 2000,
  disableConsoleOutput: false,
  identifierNamesGenerator: 'hexadecimal',
  log: false,
  numbersToExpressions: true,
  renameGlobals: false,       // keep exports accessible
  selfDefending: true,
  simplify: true,
  splitStrings: true,
  splitStringsChunkLength: 5,
  stringArray: true,
  stringArrayCallsTransform: true,
  stringArrayCallsTransformThreshold: 1,
  stringArrayEncoding: ['rc4'],
  stringArrayIndexShift: true,
  stringArrayRotate: true,
  stringArrayShuffle: true,
  stringArrayWrappersCount: 5,
  stringArrayWrappersChainedCalls: true,
  stringArrayWrappersParametersMaxCount: 5,
  stringArrayWrappersType: 'function',
  stringArrayThreshold: 1,
  transformObjectKeys: true,
  unicodeEscapeSequence: false,

  // ── Target ─────────────────────────────────────────────
  target: 'node',
  seed: 0,
};

let fileCount = 0;

function processDir(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      processDir(fullPath);
      continue;
    }

    // Only obfuscate .js and .mjs files, skip .d.ts and .d.mts
    if (entry.name.endsWith('.d.ts') || entry.name.endsWith('.d.mts')) continue;
    if (!entry.name.endsWith('.js') && !entry.name.endsWith('.mjs')) continue;

    const source = fs.readFileSync(fullPath, 'utf8');
    const result = JavaScriptObfuscator.obfuscate(source, obfuscatorConfig);
    fs.writeFileSync(fullPath, result.getObfuscatedCode());
    fileCount++;

    const rel = path.relative(distDir, fullPath);
    console.log(`  obfuscated: ${rel}`);
  }
}

console.log('Obfuscating dist/ ...');
processDir(distDir);
console.log(`Done — ${fileCount} file(s) obfuscated.`);
