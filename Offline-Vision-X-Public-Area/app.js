(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const API = { status: '/api/status', tracks: '/api/tracks', events: '/api/events', stream: '/video.mjpg' };
  const video = $('video');
  let streamTimer;
  let assistantPending = false;
  const expandedReviewDrafts = new Set();
  const value = (obj, ...keys) => { for (const k of keys) if (obj?.[k] !== undefined && obj[k] !== null) return obj[k]; return null; };
  const listFrom = data => Array.isArray(data) ? data : (data?.items || data?.tracks || data?.events || data?.detections || []);
  const fmt = (v, digits=2) => Number.isFinite(Number(v)) ? Number(v).toFixed(digits) : '—';
  const safe = v => String(v ?? '—');
  async function get(path) {
    const response = await fetch(path, { cache: 'no-store', headers: { Accept: 'application/json' } });
    if (!response.ok) throw new Error(`${path} returned HTTP ${response.status}`);
    return response.json();
  }
  function renderStatus(s) {
    const cam = value(s, 'camera_status', 'cameraState', 'camera');
    const simulated = value(s, 'demo_mode') === true;
    const cameraOnline = typeof cam === 'object' ? !!value(cam, 'online', 'connected', 'available') : ['online','connected','ready'].includes(String(cam).toLowerCase());
    $('camera-status').textContent = simulated ? 'Simulated' : cameraOnline ? 'Online' : (cam ? 'Unavailable' : 'Unknown');
    $('camera-dot').className = `status-dot ${simulated ? 'simulated' : cameraOnline ? 'online' : 'offline'}`;
    $('camera-detail').textContent = simulated ? 'Generated frames · not camera input' : typeof cam === 'object' ? safe(value(cam,'name','source','message') || (cameraOnline ? 'Frame source connected' : 'No frame source')) : (safe(value(s,'camera_message','camera_detail') || (cameraOnline ? 'Frame source connected' : 'Camera unavailable')));
    const feedState=$('feed-state');
    feedState.classList.toggle('simulated',simulated);
    feedState.replaceChildren(document.createElement('i'),document.createTextNode(` ${simulated?'SIMULATED':cameraOnline?'LIVE':'OFFLINE'}`));
    $('fps').textContent = fmt(value(s,'fps','processing_fps'),1);
    $('latency').textContent = fmt(value(s,'latency_ms','processing_latency_ms','latency'),0);
    $('active-count').textContent = safe(value(s,'active_tracks','track_count') ?? '—');
    $('detection-count').textContent = safe(value(s,'detection_count','detections_count') ?? '—');
    const mode = String(value(s,'hud_mode','mode','application_mode') || '').toLowerCase();
    const selectedMode = mode.includes('public') ? 'public_area' : (mode.includes('industrial') || mode.includes('factory')) ? 'industrial' : 'human';
    document.querySelectorAll('[data-mode]').forEach(b => b.classList.toggle('active', b.dataset.mode===selectedMode));
    const width=value(s,'frame_width','width'), height=value(s,'frame_height','height');
    if(width && height) $('stream-resolution').textContent=`${width} × ${height}`;
    $('stream-caption').textContent = simulated ? 'Generated demo feed · not live video' : cameraOnline ? 'Live feed with backend overlays' : 'Camera unavailable';
    $('map-caption').textContent = value(s,'localization_status','calibration_status') || 'Calibrated camera plane';
    const banner=$('connection-banner');
    const error=value(s,'pipeline_error')||value(s,'tracker_error')||value(s,'detector_error');
    banner.className=`connection-banner ${error ? 'error' : 'online'}`;
    banner.textContent=simulated ? 'SIMULATION ONLY · generated person detections and zones · not a camera feed or safety signal' : error ? `Perception status · ${safe(error)}` : `Local service connected · ${safe(value(s,'hud_mode') || mode.toUpperCase())} · ${safe(value(s,'tracker'))} · visual-only advisory`;
    $('demo-indicator').hidden=!simulated;
    const publicArea=value(s,'public_area') || {};
    $('public-area-status').textContent = publicArea.configuration_error
      ? `Monitor configuration error · ${safe(publicArea.configuration_error)}`
      : publicArea.monitor_available
        ? (publicArea.demo_mode ? 'SIMULATION · sample zones/events only · not site geometry' : publicArea.enabled ? 'Active · local event review only · no notifications or emergency calls' : 'Ready · select PUBLIC AREA to enable local event rules')
        : 'Monitor unavailable';
    $('public-area-zones').textContent = `${fmt(value(publicArea,'configured_zones'),0)} ZONES`;
    $('gesture-status').textContent = safe(value(publicArea,'gesture_status') || 'Gesture capability unavailable.');
    renderAssistantStatus(value(s, 'assistant') || {});
  }
  function renderAssistantStatus(assistant) {
    const consent = $('assistant-consent');
    const frameConsent = $('assistant-frame-consent');
    const submit = $('assistant-submit');
    const available = assistant.available === true;
    $('assistant-status').textContent = assistant.message || (available ? 'Ready · requests are sent only when you click Analyze.' : 'Optional scene summaries are disabled.');
    $('assistant-model').textContent = assistant.model || (assistant.enabled ? 'CONFIGURATION NEEDED' : 'DISABLED');
    const localLabel = document.querySelector('.offline-label');
    if (localLabel?.lastChild) localLabel.lastChild.textContent = available ? ' OPTIONAL GEMINI API' : ' LOCAL SYSTEM';
    frameConsent.disabled = !available || assistant.frame_upload_allowed !== true;
    if (frameConsent.disabled) frameConsent.checked = false;
    submit.disabled = !available || !consent.checked || assistantPending;
  }
  function renderTracks(payload) {
    const tracks=listFrom(payload); const tbody=$('tracks'); tbody.replaceChildren();
    $('track-updated').textContent=`Updated ${new Date().toLocaleTimeString()}`;
    if(!tracks.length){tbody.innerHTML='<tr><td colspan="5" class="empty-cell">No active tracks reported</td></tr>';}
    tracks.forEach(t=>{
      const id=value(t,'id','track_id'), cls=value(t,'class_name','class','label'), state=value(t,'state','status')||'TRACKING';
      const x=value(t,'world_x','x'), y=value(t,'world_y','y'), speed=value(t,'speed','speed_mps');
      const current=x!==null&&y!==null?`${fmt(x)} m, ${fmt(y)} m`:'—';
      const leadX=value(t,'lead_x_m'), leadY=value(t,'lead_y_m');
      const lead=(leadX!==null&&leadY!==null)?`${fmt(leadX)} m, ${fmt(leadY)} m`:
        (value(t,'lead_x_px')!==null?`${fmt(value(t,'lead_x_px'),0)}, ${fmt(value(t,'lead_y_px'),0)} px`:'—');
      const pos=`${current} → ${lead} (${fmt(value(t,'projection_horizon_ms'),0)} ms)`;
      const direction=value(t,'direction','heading');
      const tr=document.createElement('tr'); tr.innerHTML=`<td><div class="object-id"></div><div class="object-class"></div></td><td></td><td></td><td></td><td><span class="state-pill"></span></td>`;
      tr.children[0].querySelector('.object-id').textContent=safe(id); tr.children[0].querySelector('.object-class').textContent=safe(cls);
      const pixelSpeed=value(t,'speed_px_s');
      tr.children[1].textContent=pos; tr.children[2].textContent=speed!==null?`${fmt(speed,1)} m/s${direction?` · ${direction}`:''}`:(pixelSpeed!==null?`${fmt(pixelSpeed,0)} px/s${direction?` · ${direction}`:''}`:'—'); tr.children[3].textContent=safe(value(t,'zone','zone_name'));
      const pill=tr.children[4].querySelector('.state-pill'); pill.textContent=String(state).toUpperCase(); if(/lost|removed|missing/i.test(state))pill.classList.add('lost'); tbody.append(tr);
    });
    renderMap(tracks);
  }
  function renderMap(tracks) {
    const points=$('map-points'); points.replaceChildren(); let located=0;
    const statusWindow=window.__vxStatus||{}; const width=Number(value(statusWindow,'map_width_m','room_width_m'))||10; const height=Number(value(statusWindow,'map_height_m','room_height_m'))||8;
    tracks.forEach(t=>{const x=Number(value(t,'world_x','x')), y=Number(value(t,'world_y','y')); if(!Number.isFinite(x)||!Number.isFinite(y))return; located++;
      const dot=document.createElement('div'); dot.className='map-dotpoint'+(/component|object|product|part/i.test(String(value(t,'class_name','class','label')||''))?' component':''); dot.style.left=`${Math.max(2,Math.min(98,x/width*100))}%`; dot.style.top=`${100-Math.max(2,Math.min(98,y/height*100))}%`;
      const label=document.createElement('span'); label.textContent=safe(value(t,'id','track_id')); dot.append(label); points.append(dot);
      const leadX=Number(value(t,'lead_x_m')), leadY=Number(value(t,'lead_y_m'));
      if(Number.isFinite(leadX)&&Number.isFinite(leadY)){const lead=document.createElement('div');lead.className='map-leadpoint';lead.style.left=`${Math.max(2,Math.min(98,leadX/width*100))}%`;lead.style.top=`${100-Math.max(2,Math.min(98,leadY/height*100))}%`;points.append(lead);}
    }); $('map-empty').hidden=located>0; $('map-unit').textContent=String(value(statusWindow,'coordinate_unit')||'METERS').toUpperCase();
  }
  function renderEvents(payload) {
    const events=listFrom(payload); $('event-count').textContent=String(events.length); const box=$('events'); box.replaceChildren();
    if(!events.length){box.innerHTML='<div class="empty-cell">No events reported</div>';return;}
    events.slice(0,40).forEach(e=>{
      const row=document.createElement('div'), timeNode=document.createElement('div'), mark=document.createElement('div'), body=document.createElement('div');
      row.className='event-row'; timeNode.className='event-time'; mark.className='event-mark';
      const timestamp=value(e,'timestamp','time','created_at');
      const parsedTime=timestamp?new Date(timestamp):null;
      timeNode.textContent=parsedTime&&!Number.isNaN(parsedTime.getTime())?parsedTime.toLocaleTimeString():'—';
      const level=String(value(e,'severity','level','type')||'info').toLowerCase();
      mark.classList.toggle('warn',/warn|anomaly/i.test(level)); mark.classList.toggle('error',/error|critical/i.test(level));
      const message=document.createElement('div'); message.className='event-text'; message.textContent=safe(value(e,'message','description','name'));
      const details=value(e,'details')||{}, context=[value(e,'track_id')&&`Temporary track ${value(e,'track_id')}`,value(e,'zone')&&`Zone ${value(e,'zone')}`];
      if(Number.isFinite(Number(details.duration_seconds)))context.push(`Observed duration ${fmt(details.duration_seconds,1)} s`);
      if(Number.isFinite(Number(details.detector_score)))context.push(`Detector score ${fmt(details.detector_score,2)} (not a probability)`);
      if(details.pattern)context.push(`Configured pattern ${safe(details.pattern)}`);
      if(details.previous_zone)context.push(`Previous zone ${safe(details.previous_zone)}`);
      const meta=document.createElement('div'); meta.className='event-meta'; meta.textContent=[safe(value(e,'rule')),level.toUpperCase(),...context.filter(Boolean)].filter(item=>item!=='—').join(' · ');
      body.append(message,meta);
      const draft=value(e,'operator_review_draft');
      if(draft){
        const review=document.createElement('details'), summary=document.createElement('summary'), draftText=document.createElement('p'), copy=document.createElement('button'), copyStatus=document.createElement('span');
        const draftKey=[timestamp,value(e,'rule'),value(e,'track_id')].join('|');
        review.className='review-draft'; review.open=expandedReviewDrafts.has(draftKey);
        review.addEventListener('toggle',()=>{if(review.open)expandedReviewDrafts.add(draftKey);else expandedReviewDrafts.delete(draftKey);});
        summary.textContent='Operator incident draft · review before response · not sent'; draftText.textContent=draft;
        copy.type='button'; copy.className='review-copy'; copy.textContent='Copy draft'; copyStatus.className='copy-status';
        copy.addEventListener('click',async()=>{
          try { await navigator.clipboard.writeText(draft); copyStatus.textContent='Copied locally for review.'; }
          catch { copyStatus.textContent='Clipboard unavailable; select and copy the draft text.'; }
        });
        review.append(summary,draftText,copy,copyStatus); body.append(review);
      }
      row.append(timeNode,mark,body); box.append(row);
    });
  }
  async function poll() {
    const results=await Promise.allSettled([get(API.status),get(API.tracks),get(API.events)]);
    if(results[0].status==='fulfilled'){window.__vxStatus=results[0].value;renderStatus(results[0].value);} else {const b=$('connection-banner');b.className='connection-banner error';b.textContent=`Local service unavailable · ${results[0].reason.message}`;$('camera-status').textContent='Unavailable';$('camera-dot').className='status-dot offline';$('camera-detail').textContent='Could not read /api/status';}
    if(results[1].status==='fulfilled')renderTracks(results[1].value); else {$('track-updated').textContent='Track feed unavailable';}
    if(results[2].status==='fulfilled')renderEvents(results[2].value);
  }
  $('assistant-consent').addEventListener('change',()=>renderAssistantStatus(window.__vxStatus?.assistant || {}));
  $('assistant-frame-consent').addEventListener('change',()=>renderAssistantStatus(window.__vxStatus?.assistant || {}));
  $('assistant-submit').addEventListener('click',async()=>{
    const button=$('assistant-submit'), state=$('assistant-result-status'), output=$('assistant-result');
    assistantPending=true; button.disabled=true; state.textContent='Submitting the explicitly approved request…'; output.textContent='';
    try {
      const response=await fetch('/api/assistant/scene',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
        confirm_external_processing:$('assistant-consent').checked,
        include_frame:$('assistant-frame-consent').checked
      })});
      const result=await response.json();
      if(!response.ok) throw new Error(result.error || `Assistant returned HTTP ${response.status}`);
      output.textContent=result.text;
      state.textContent=`${safe(result.model)} · ${result.frame_included?'frame and scene summary':'scene summary only'} · informational only`;
    } catch(error) {
      state.textContent=`No summary returned · ${error.message}`;
    } finally {
      $('assistant-consent').checked=false;
      $('assistant-frame-consent').checked=false;
      assistantPending=false;
      renderAssistantStatus(window.__vxStatus?.assistant || {});
    }
  });
  video.addEventListener('load',()=>{video.classList.add('loaded');$('video-message').hidden=true;clearTimeout(streamTimer);});
  video.addEventListener('error',()=>{video.classList.remove('loaded');$('video-message').hidden=false;$('video-message').querySelector('strong').textContent='CAMERA UNAVAILABLE';$('video-message').querySelector('small').textContent='No live stream at /video.mjpg';clearTimeout(streamTimer);streamTimer=setTimeout(()=>{video.src=`${API.stream}?t=${Date.now()}`;},5000);});
  $('refresh').addEventListener('click',()=>{poll();video.src=`${API.stream}?t=${Date.now()}`;});
  document.querySelectorAll('.mode-button').forEach(button=>button.addEventListener('click',async()=>{try{await fetch('/api/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:button.dataset.mode})});poll();}catch(e){const b=$('connection-banner');b.className='connection-banner error';b.textContent=`Could not change mode · ${e.message}`;}}));
  setInterval(()=>{$('clock').textContent=new Date().toLocaleTimeString();},1000); $('clock').textContent=new Date().toLocaleTimeString(); poll();setInterval(poll,1500);
})();
