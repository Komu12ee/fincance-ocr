#!/usr/bin/env node
/**
 * scripts/add-user.js
 *
 * Generates a bcrypt-hashed .env entry for a new user.
 * Run interactively:   node scripts/add-user.js
 *
 * Then paste the printed line into your .env file.
 * The server reads USER_<username>=<hash> on startup via config.js.
 */

'use strict';

const bcrypt   = require('bcryptjs');
const readline = require('readline');

const rl = readline.createInterface({
  input:  process.stdin,
  output: process.stdout,
});

function ask(question) {
  return new Promise(resolve => rl.question(question, resolve));
}

// Hide password input
function askPassword(prompt) {
  return new Promise(resolve => {
    process.stdout.write(prompt);
    const stdin = process.openStdin();
    process.stdin.setRawMode(true);
    process.stdin.resume();
    process.stdin.setEncoding('utf8');

    let password = '';
    process.stdin.on('data', function handler(ch) {
      ch = ch.toString();
      if (ch === '\n' || ch === '\r' || ch === '\u0003') {
        process.stdin.setRawMode(false);
        process.stdin.pause();
        process.stdin.removeListener('data', handler);
        process.stdout.write('\n');
        resolve(password);
      } else if (ch === '\u007f') {  // backspace
        password = password.slice(0, -1);
      } else {
        password += ch;
        process.stdout.write('*');
      }
    });
  });
}

async function main() {
  console.log('\n──────────────────────────────────────');
  console.log('  CHiPS-RAG  –  Add User');
  console.log('──────────────────────────────────────\n');

  const username = (await ask('Username: ')).trim().toLowerCase();
  if (!username || /\s/.test(username)) {
    console.error('Error: username cannot be empty or contain spaces.');
    process.exit(1);
  }

  let password;
  try {
    password = await askPassword('Password: ');
  } catch (_) {
    // Fallback if raw mode unavailable (e.g. piped input)
    password = (await ask('Password (visible): ')).trim();
  }

  if (!password || password.length < 6) {
    console.error('Error: password must be at least 6 characters.');
    process.exit(1);
  }

  process.stdout.write('\nHashing password...');
  const hash = await bcrypt.hash(password, 10);
  process.stdout.write(' done.\n\n');

  const envKey  = `USER_${username.toUpperCase()}`;
  const envLine = `${envKey}=${hash}`;

  console.log('──────────────────────────────────────');
  console.log('Add this line to your .env file:\n');
  console.log(envLine);
  console.log('\n──────────────────────────────────────');
  console.log('Done. Restart the server for the change to take effect.\n');

  rl.close();
  process.exit(0);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
