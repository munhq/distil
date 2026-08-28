#!/usr/bin/env node
// Publish a Smithery stdio release: the listing, its config schema, and the full
// tool card.
//
// This exists because the first listing was published with a handful of curl
// commands, and a listing published by hand is a listing that describes an old
// version forever. Everything the API needs is derived here from the same files
// and the same binary the release ships.
//
// Three things the API requires that its error messages do not say plainly, each
// of which cost an attempt:
//   - `bundle` is required for a stdio release. Without it: 400 "Missing
//     required part: bundle".
//   - `payload` is a form field holding JSON *as a string*, not a file upload.
//   - `serverCard.serverInfo` requires BOTH name and version. Omitting version
//     returns 400 "Invalid input: expected string, received undefined", which
//     names no field at all.
//
// Usage:
//   SMITHERY_API_KEY=… node npm/publish-smithery.mjs \
//     --card card.json --bundle distil-0.3.3.mcpb
// Optional: --namespace (default munhq), --server (default distil), --dry-run.
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const pkg = JSON.parse(readFileSync(path.join(here, 'package.json'), 'utf8'));

const argv = process.argv.slice(2);
const argOf = (n, d = null) => {
  const i = argv.indexOf(n);
  return i === -1 ? d : argv[i + 1];
};
const has = (n) => argv.includes(n);

const namespace = argOf('--namespace', 'munhq');
const server = argOf('--server', 'distil');
const cardPath = argOf('--card');
const bundlePath = argOf('--bundle');
const dryRun = has('--dry-run');

const die = (msg) => {
  process.stderr.write(`publish-smithery: ${msg}\n`);
  process.exit(1);
};

if (!cardPath || !bundlePath) die('need --card <tools.json> and --bundle <file.mcpb>');
const key = process.env.SMITHERY_API_KEY;
if (!key && !dryRun) {
  die('SMITHERY_API_KEY is not set. It is the bearer token from smithery.ai; nothing else authenticates this API.');
}

const card = JSON.parse(readFileSync(cardPath, 'utf8'));
const tools = card.tools || card.result?.tools || [];
if (!tools.length) die(`${cardPath} declares no tools — refusing to publish a listing with an empty tool card`);

// Mirrors smithery.yaml. Nothing is required, because distil needs no account,
// key or configuration; the one option is for a client that starts the server
// outside the repository it should index.
const configSchema = {
  // distil needs no configuration: no account, no key, nothing to fill in. An
  // empty properties object is the honest value, and Smithery renders it as
  // "No configuration required".
  type: 'object',
  properties: {},
  required: [],
};

const payload = {
  type: 'stdio',
  runtime: 'node',
  configSchema,
  serverCard: {
    serverInfo: {
      name: 'distil',
      title: 'distil',
      version: pkg.version,
      description:
        "Context optimization for LLM agents: measure where a session's tokens actually go, and compress context only where compression pays for the prompt cache it invalidates.",
      websiteUrl: 'https://github.com/munhq/distil',
    },
    tools,
  },
};

const qualified = `${encodeURIComponent(`${namespace}/${server}`)}`;
const url = `https://api.smithery.ai/servers/${qualified}/releases`;

if (dryRun) {
  process.stdout.write(
    `dry run: PUT ${url}\n  version ${pkg.version}, ${tools.length} tools, bundle ${path.basename(bundlePath)}\n`
  );
  process.exit(0);
}

// The bundle is read once. A release is retried after the listing is created,
// and re-reading the file for the retry would let the two attempts disagree.
const bundle = readFileSync(bundlePath);

const release = () => {
  const form = new FormData();
  // A string field. Sent as a file part, the API answers "expected string,
  // received undefined" and names nothing.
  form.append('payload', JSON.stringify(payload));
  form.append('bundle', new Blob([bundle], { type: 'application/zip' }), path.basename(bundlePath));
  return fetch(url, { method: 'PUT', headers: { authorization: `Bearer ${key}` }, body: form });
};

// PUT .../releases UPDATES a listing; it does not create one, and a server that
// has never been listed answers 404 "Server not found". codeindex's listing was
// created by hand, so this script had never met that case and distil's first
// release failed on it.
//
// Creating one is its own call: PUT /servers/<qualified> with a JSON body. Do
// NOT reach for `smithery mcp publish` here — it builds configSchema only when
// the bundle manifest carries a non-empty user_config, and distil needs no
// configuration at all, so it sends undefined and the API rejects the payload.
const serverUrl = `https://api.smithery.ai/servers/${qualified}`;

// The listing's own record: what a directory page shows above the tool list.
// The release payload does NOT populate it — distil's first listing came up
// with an empty description because this call was missing. The description
// comes from npm/package.json so one edit moves every surface.
// Field names are the Platform API's, from https://api.smithery.ai/openapi:
// PUT accepts displayName and description only; PATCH accepts those plus
// homepage, repositoryUrl, backlinkUrl, license, iconUrl and unlisted.
const identity = { displayName: server, description: pkg.description };
const listingRecord = {
  ...identity,
  // Where the org lists what it builds, not the repository — repositoryUrl is
  // the field for that, and pointing both at the same place wastes a row.
  homepage: 'https://munhq.com/products',
  // npm records this as `git+https://….git`, which is a clone URL. A listing
  // links it for a human to click, so hand it a browsable one.
  repositoryUrl: (typeof pkg.repository === 'string' ? pkg.repository : pkg.repository?.url || '')
    .replace(/^git\+/, '')
    .replace(/\.git$/, ''),
  license: pkg.license,
  // The project's own mark, served from the default branch. A directory tile
  // is the first thing anyone sees, and a generic GitHub favicon says nothing
  // about which of the servers on that page this one is.
  iconUrl: 'https://raw.githubusercontent.com/munhq/distil/main/docs/brand/icon-512.png',
};

const createListing = async () => {
  const r = await fetch(serverUrl, {
    method: 'PUT',
    headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
    // PUT takes the two identity fields; everything else lands in the PATCH.
    body: JSON.stringify(identity),
  });
  if (!r.ok) die(`PUT ${serverUrl} -> HTTP ${r.status}: ${(await r.text()).slice(0, 300)}`);
  process.stdout.write(`created the ${namespace}/${server} listing\n`);
};

// Keep the record current on every publish, so a description edit ships with
// the next release instead of needing someone to remember. A failure here is
// reported but does not fail the run: the release itself already landed, and a
// red release for a stale subtitle would be the wrong trade.
const updateListing = async () => {
  const r = await fetch(serverUrl, {
    method: 'PATCH',
    headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
    body: JSON.stringify(listingRecord),
  });
  if (!r.ok) {
    process.stderr.write(
      `::warning::PATCH ${serverUrl} -> HTTP ${r.status}: ${(await r.text()).slice(0, 200)}\n`
    );
    return;
  }
  process.stdout.write(`listing record updated (description, homepage, repo, license, icon)\n`);
};

let res = await release();
if (res.status === 404) {
  process.stdout.write(`${namespace}/${server} is not listed yet; creating it\n`);
  await createListing();
  res = await release();
}
const text = await res.text();
if (!res.ok) die(`PUT ${url} -> HTTP ${res.status}: ${text}`);

let body;
try {
  body = JSON.parse(text);
} catch {
  die(`unparseable response: ${text.slice(0, 200)}`);
}
process.stdout.write(
  `published ${namespace}/${server} ${pkg.version}: status=${body.status} ` +
    `deployment=${body.deploymentId} url=${body.mcpUrl}\n` +
    `  ${tools.length} tools on the card\n`
);
for (const w of body.warnings || []) process.stderr.write(`  warning: ${w}\n`);
if (body.status && !['SUCCESS', 'QUEUED', 'WORKING'].includes(body.status)) {
  die(`release status is ${body.status}`);
}

await updateListing();
