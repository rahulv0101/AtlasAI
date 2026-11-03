const http = require('http');
const { spawn } = require('child_process');
const fs = require('fs');
const chrome = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const out = 'storage\\ai_core_panel_element.png';
const proc = spawn(chrome, ['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-port=9333','--user-data-dir=storage\\chrome-ai-core-profile-2','http://127.0.0.1:8765/'], {stdio: 'ignore'});
function sleep(ms){ return new Promise(r => setTimeout(r, ms)); }
function getJson(path) { return new Promise((resolve, reject) => {
  http.get({host:'127.0.0.1', port:9333, path}, res => {
    let data=''; res.on('data', c => data += c); res.on('end', () => { try { resolve(JSON.parse(data)); } catch(e) { reject(e); } });
  }).on('error', reject);
}); }
async function waitJson(path) { let last; for (let i=0;i<30;i++){ try { return await getJson(path); } catch(e){ last=e; await sleep(500); } } throw last; }
(async () => {
  try {
    const pages = await waitJson('/json');
    const page = pages.find(p => p.type === 'page');
    const WebSocket = require('ws');
    const ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise(resolve => ws.on('open', resolve));
    let id = 1;
    const send = (method, params={}) => new Promise(resolve => {
      const current = id++;
      const onMsg = raw => { const msg = JSON.parse(raw); if (msg.id === current) { ws.off('message', onMsg); resolve(msg); } };
      ws.on('message', onMsg);
      ws.send(JSON.stringify({id: current, method, params}));
    });
    await send('Runtime.evaluate', {expression: `document.getElementById('ai-core-panel')?.scrollIntoView({block:'center'});`, awaitPromise: true});
    await sleep(1500);
    const metrics = await send('Runtime.evaluate', {expression: `(() => { const el=document.getElementById('ai-core-panel'); const r=el.getBoundingClientRect(); return {x:r.x,y:r.y,width:r.width,height:r.height, text: el.innerText.slice(0,500)}; })()`});
    const rect = metrics.result.result.value;
    const shot = await send('Page.captureScreenshot', {format:'png', clip:{x:Math.max(0, rect.x-20), y:Math.max(0, rect.y-80), width:Math.min(1400, rect.width+40), height:Math.min(950, rect.height+140), scale:1}});
    fs.writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
    console.log(JSON.stringify(rect));
    ws.close();
  } finally { proc.kill(); }
})().catch(err => { console.error(err); proc.kill(); process.exit(1); });
