// Resolve a distil binary for this platform, downloading the pinned release
// asset when the cache does not already hold it.
//
// Why an npm package for a program that is not JavaScript: every MCP directory
// installs servers the way npm installs them. The official registry validates an
// npm version, Smithery runs `npx`, mcp.so lists the same command. A compiled
// server with no npm name is invisible to all three, so this package is the thin
// shim that makes `npx -y @munhq/distil` work.
//
// Write NOTHING to stdout. Stdout is the MCP JSON-RPC channel; one stray line
// there makes the server look broken with no error that explains why. Every
// diagnostic goes to stderr, which the client logs.
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const https = require('https');
const crypto = require('crypto');

const REPO = 'munhq/distil';
const ASSET_PREFIX = 'distil-mcp';
const CHECKSUM_FILE = 'checksums.txt';
const { version: VERSION } = require('../package.json');

const log = (msg) => process.stderr.write(`distil: ${msg}\n`);

// Map Node's platform/arch onto the asset names the release actually publishes.
// Node says darwin/win32/x64; releases are named by arch and platform. The
// mapping is spelled out rather than assembled, because assembling it is how a
// Mac ends up asking for a name that does not exist.
function assetFor(platform, arch) {
  const a = { x64: 'x86_64', arm64: 'aarch64' }[arch];
  const pl = { linux: 'linux', darwin: 'macos', win32: 'windows' }[platform];
  if (!a || !pl) return null;
  return `${ASSET_PREFIX}-${a}-${pl}${pl === 'windows' ? '.exe' : ''}`;
}

const assetName = () => assetFor(process.platform, process.arch);

function cacheDir() {
  const base = process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache');
  return path.join(base, 'distil', 'bin');
}

// The cached file carries the version, so an upgrade is a cache miss rather than
// a silent downgrade to whatever was fetched first.
function cachedBinary() {
  const exe = process.platform === 'win32' ? '.exe' : '';
  return path.join(cacheDir(), `distil-${VERSION}${exe}`);
}

function get(url, redirects = 0) {
  return new Promise((resolve, reject) => {
    if (redirects > 5) return reject(new Error(`too many redirects for ${url}`));
    https
      .get(url, { headers: { 'user-agent': `distil-npm/${VERSION}` } }, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          return resolve(get(res.headers.location, redirects + 1));
        }
        if (res.statusCode !== 200) {
          res.resume();
          return reject(new Error(`GET ${url} -> HTTP ${res.statusCode}`));
        }
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => resolve(Buffer.concat(chunks)));
        res.on('error', reject);
      })
      .on('error', reject);
  });
}

// The checksum published beside the binary is not proof on its own — whoever can
// replace one can replace the other. It catches a truncated download and a
// mismatched tag, which are the failures that actually happen.
async function verifiedDownload(asset) {
  const base = `https://github.com/${REPO}/releases/download/v${VERSION}`;
  log(`fetching ${asset} for v${VERSION} (once per version)`);

  const sums = (await get(`${base}/${CHECKSUM_FILE}`)).toString('utf8');
  const line = sums.split('\n').find((l) => l.trim().endsWith(` ${asset}`));
  if (!line) throw new Error(`${CHECKSUM_FILE} for v${VERSION} does not list ${asset}`);
  const want = line.trim().split(/\s+/)[0];

  const body = await get(`${base}/${asset}`);
  const got = crypto.createHash('sha256').update(body).digest('hex');
  if (got !== want) throw new Error(`checksum mismatch for ${asset}: want ${want}, got ${got}`);

  const dest = cachedBinary();
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const tmp = `${dest}.tmp-${process.pid}`;
  fs.writeFileSync(tmp, body, { mode: 0o755 });
  fs.renameSync(tmp, dest);
  log(`installed ${dest}`);
  return dest;
}

// An explicit override wins, for a local build. PATH is deliberately NOT
// consulted: this package declares one version to the registry, and running
// whatever happens to be on PATH would make that declaration a lie.
async function resolveBinary() {
  const override = process.env.DISTIL_BIN;
  if (override && fs.existsSync(override)) return override;

  const cached = cachedBinary();
  if (fs.existsSync(cached)) return cached;

  const asset = assetName();
  if (!asset) {
    throw new Error(
      `no release build for ${process.platform}/${process.arch}. ` +
        `See https://github.com/${REPO}`
    );
  }
  return verifiedDownload(asset);
}

module.exports = { resolveBinary, cachedBinary, assetName, assetFor, VERSION, log };
