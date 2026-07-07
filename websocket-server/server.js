'use strict';

/**
 * ProctorSense relay server
 * ──────────────────────────
 * Minimal 1:1 WebSocket relay between exactly one "candidate" socket and
 * exactly one "proctor" socket. No database, no auth, no multi-session
 * support — this is intentionally a single global session.
 *
 * Protocol:
 *   1. Client connects.
 *   2. At some point (not necessarily the very first frame — see
 *      handleMessage below) it must send:
 *         { "type": "hello", "role": "candidate" | "proctor" }
 *      Anything received before a valid hello is silently dropped.
 *   3. Once identified, candidate -> proctor messages are relayed verbatim
 *      (as raw text/bytes, unmodified) as long as a proctor is connected.
 *   4. If a role slot is already taken, the new connecting socket is sent
 *      a JSON error message and then closed with code 4409 (custom
 *      "conflict" code in the private-use range 4000-4999).
 *
 * ── Extending this later ────────────────────────────────────────────────
 * - Real auth/session support: replace the single `session` object below
 *   with a Map<sessionId, session>, and derive sessionId from the hello
 *   message (or from a URL param / auth token) instead of assuming one
 *   global pair of sockets.
 * - Proctor -> candidate commands (e.g. "recalibrate"): the relay is
 *   already wired for this, it's just unused. See the commented-out
 *   block inside handleMessage's PROCTOR branch.
 */

const { WebSocket, WebSocketServer } = require('ws');

const PORT = process.env.PORT || 8080;

// The single active session. Exactly one candidate + one proctor at a time.
const session = {
  candidate: null, // WebSocket | null
  proctor: null,   // WebSocket | null
};

const CLOSE_CODE_CONFLICT = 4409; // private-use range; "role already occupied"

function log(...args) {
  console.log(`[${new Date().toISOString()}]`, ...args);
}

function sendJSON(ws, obj) {
  try {
    ws.send(JSON.stringify(obj));
  } catch (err) {
    log('Failed to send JSON to socket:', err.message);
  }
}

const wss = new WebSocketServer({ port: PORT });

wss.on('listening', () => {
  log(`ProctorSense relay listening on port ${PORT}`);
});

wss.on('connection', (ws, req) => {
  const remote = req.socket.remoteAddress;
  log('Connection opened from', remote);

  // Role is unknown until we see a valid hello message.
  ws.role = null;

  ws.on('message', (data, isBinary) => {
    handleMessage(ws, data, isBinary);
  });

  ws.on('close', (code, reason) => {
    handleDisconnect(ws, code, reason);
  });

  ws.on('error', (err) => {
    log('Socket error', ws.role ? `(${ws.role})` : '(unidentified)', '-', err.message);
  });
});

wss.on('error', (err) => {
  log('Server error:', err.message);
});

/**
 * Handles every inbound message. Before a socket has identified itself via
 * "hello", every message is treated as a handshake attempt; anything that
 * isn't a valid hello is silently dropped (lenient handshake — no error is
 * sent back, per spec). After identification, messages are relayed.
 */
function handleMessage(ws, data, isBinary) {
  if (!ws.role) {
    tryHandleHello(ws, data);
    return;
  }

  if (ws.role === 'candidate') {
    // Relay verbatim, candidate -> proctor.
    if (session.proctor && session.proctor.readyState === WebSocket.OPEN) {
      session.proctor.send(data, { binary: isBinary });
    }
    // No proctor connected yet: message is simply dropped. The candidate
    // side doesn't need an ack for this minimal version.
    return;
  }

  if (ws.role === 'proctor') {
    // Proctor -> candidate is not required for this version. Extension
    // point for future commands (e.g. "recalibrate") lives here:
    //
    //   if (session.candidate && session.candidate.readyState === session.candidate.OPEN) {
    //     session.candidate.send(data, { binary: isBinary });
    //   }
    //
    // Left disabled for now since only candidate -> proctor is required.
    return;
  }
}

/**
 * Attempts to parse an unidentified socket's message as a hello frame.
 * Any parse failure or non-hello message is dropped without feedback
 * (lenient handshake). A valid hello either claims the requested role slot
 * or rejects the connection if that slot is already occupied.
 */
function tryHandleHello(ws, data) {
  let msg;
  try {
    msg = JSON.parse(data.toString());
  } catch {
    return; // not JSON — ignore, keep waiting for a real hello
  }

  if (!msg || msg.type !== 'hello') return; // ignore anything pre-hello
  if (msg.role !== 'candidate' && msg.role !== 'proctor') return; // invalid role — ignore

  const role = msg.role;

  if (session[role]) {
    log(`Rejected duplicate ${role} connection — slot already occupied`);
    sendJSON(ws, {
      type: 'error',
      error: 'role_occupied',
      message: `A ${role} is already connected. Only one ${role} is allowed at a time.`,
    });
    ws.close(CLOSE_CODE_CONFLICT, `${role} slot already occupied`);
    return;
  }

  session[role] = ws;
  ws.role = role;
  log(`${role} connected and identified`);

  sendJSON(ws, { type: 'hello_ack', role });
}

/**
 * Clears the disconnecting socket's slot (if it held one) so a reconnect
 * can take its place. The other side is left untouched — it simply stops
 * receiving relayed messages until the slot is refilled.
 */
function handleDisconnect(ws, code, reason) {
  const reasonStr = reason && reason.length ? reason.toString() : '(no reason given)';

  if (!ws.role) {
    log('Unidentified connection closed', `code=${code}`, reasonStr);
    return;
  }

  log(`${ws.role} disconnected`, `code=${code}`, reasonStr);

  // Only clear the slot if this socket is still the one occupying it
  // (guards against a stale close event racing a newer connection).
  if (session[ws.role] === ws) {
    session[ws.role] = null;
  }
}