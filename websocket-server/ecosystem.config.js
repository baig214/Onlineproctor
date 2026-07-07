module.exports = {
  apps: [{
    name: 'proctors1',
    script: 'server.js',
    instances: 1,
    autorestart: true,
    max_memory_restart: '250M',
    exp_backoff_restart_delay: 100,
    env: {
      PORT: 8080
    }
  }]
};