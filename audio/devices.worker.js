/**
 * devices.worker.js
 *
 * Messages IN:
 *   { type: 'mic-list',   devices: [ { deviceId, label, groupId } ] }
 *   { type: 'mic-active', deviceId, label }
 *
 * Messages OUT:
 *   { type: 'state', micDevices, activeMicId }
 *   { type: 'log',   message }
 */

let micDevices  = [];   // [{ deviceId, label, groupId, micType }]
let activeMicId = null; // deviceId string

function classifyMic(label = '') {
  if (/bluetooth|\(bluetooth\)|/i.test(label))
    return 'bluetooth';
  if (/usb|focusrite|scarlett|steinberg|rode|yeti|blue mic|elgato|hyperx|razer|shure|audio.technica|at2020/i.test(label))
    return 'external';
  return 'system';
}

function broadcast() {
  self.postMessage({
    type:        'state',
    micDevices:  micDevices.map(d => ({ ...d })),
    activeMicId,
  });
}

function log(msg) { self.postMessage({ type: 'log', message: msg }); }

self.onmessage = ({ data: msg }) => {
  switch (msg.type) {

    case 'mic-list': {
      micDevices = (msg.devices || []).map(d => ({
        deviceId: d.deviceId,
        label:    d.label || `Microphone (${d.deviceId.slice(0, 8)}…)`,
        groupId:  d.groupId,
        micType:  classifyMic(d.label || ''),
      }));
      log(`Mic list: ${micDevices.length} device(s)`);
      broadcast();
      break;
    }

    case 'mic-active': {
      activeMicId = msg.deviceId;
      log(`Active mic deviceId: ${activeMicId}`);
      broadcast();
      break;
    }
    default:
      log(`Unknown message type: "${msg.type}"`);
  }
};

log('devices.worker.js ready');