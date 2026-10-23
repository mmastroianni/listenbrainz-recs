const plot = document.getElementById("artist-map");
const search = document.getElementById("map-search");
const suggest = document.getElementById("map-suggestions");
const status = document.getElementById("map-status");
const heading = document.getElementById("map-selected");
const detail = document.getElementById("map-detail");
const neighborsEl = document.getElementById("map-neighbors");
const showLabels = document.getElementById("show-labels");
const guideStatus = document.getElementById("guide-status");
const guideArtists = document.getElementById("guide-artists");
const guideGenres = document.getElementById("guide-genres");
let guideRequest = 0;
function fmt(n) { return Number(n || 0).toLocaleString(); }
async function inspectNeighborhood(x, y) {
  const seq = ++guideRequest;
  guideStatus.textContent = "Finding recognizable artists in this region…";
  try {
    const res = await fetch(`/api/map/neighborhood?x=${encodeURIComponent(x)}&y=${encodeURIComponent(y)}&count=250`);
    if (!res.ok) throw new Error(`Neighborhood lookup failed (${res.status})`);
    const data = await res.json();
    if (seq !== guideRequest) return;
    guideStatus.textContent = `Closest ${fmt(data.sample_size)} artists in the 2D projection. Genres reflect their strongest available MusicBrainz tag.`;
    guideArtists.replaceChildren();
    data.artists.forEach(a => {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.textContent = a.name;
      b.addEventListener("click", () => choose(a.mbid));
      const count = document.createElement("span");
      count.className = "guide-count";
      count.textContent = `${fmt(a.listeners)} listeners`;
      li.append(b, count);
      guideArtists.append(li);
    });
    guideGenres.replaceChildren();
    if (!data.genres.length) {
      guideGenres.textContent = "No genre tags available for this region.";
    } else {
      const max = data.genres[0].count;
      data.genres.forEach(g => {
        const row = document.createElement("div"); row.className = "genre-row";
        const name = document.createElement("span"); name.className = "genre-name"; name.textContent = g.name;
        const bar = document.createElement("div"); bar.className = "genre-bar";
        const fill = document.createElement("div"); fill.className = "genre-fill";
        fill.style.width = `${100 * g.count / max}%`;
        bar.append(fill);
        const count = document.createElement("span"); count.className = "genre-count"; count.textContent = g.count;
        row.append(name, bar, count); guideGenres.append(row);
      });
    }
  } catch (e) { if (seq === guideRequest) guideStatus.textContent = e.message; }
}

let points = [], suggestionTimer, currentMbid, requestId = 0;
const colors = {"jazz":"#317ca3","classical":"#a66dba","world":"#d49b35",
  "electronic":"#43a49b","rock":"#d36a65","metal":"#656575",
  "hip-hop":"#9a8a32","pop":"#e88ea8","other":"#aab2bc"};
const groups = Object.keys(colors);
let overlays = [];
function traceFor(group) {
  const p = points.filter(p => p.genre === group);
  return {type:"scattergl",mode:"markers",name:group,
    x:p.map(p=>p.x),y:p.map(p=>p.y),
    text:p.map(p=>p.name),customdata:p.map(p=>p.mbid),
    hovertemplate:"%{text}<extra>"+group+"</extra>",
    marker:{size:5,color:colors[group],opacity:.55}};
}
function draw() {
  const traces = groups.map(traceFor).concat(overlays);
  Plotly.react(plot,traces,{
    margin:{l:20,r:20,t:15,b:20},paper_bgcolor:"#ffffff",plot_bgcolor:"#ffffff",
    xaxis:{visible:false},yaxis:{visible:false,scaleanchor:"x",scaleratio:1},
    legend:{orientation:"h",y:-.03,x:0,font:{size:11}},
    hovermode:"closest",dragmode:"pan",uirevision:"keep",
  },{responsive:true,displaylogo:false,scrollZoom:true});
}
function overlayPoint(p,color,size,name,showText=false) {
  return {type:"scattergl",mode:showText?"markers+text":"markers",name,
    x:[p.x],y:[p.y],text:[p.name],textposition:"top center",
    customdata:[p.mbid],hovertemplate:"%{text}<extra>"+name+"</extra>",
    marker:{size,color,line:{color:"white",width:1.5}}};
}
async function choose(mbid, push=true) {
  const seq=++requestId;
  currentMbid=mbid;
  if(push)history.pushState(null,"","?artist="+encodeURIComponent(mbid));
  status.textContent="Finding artist and neighbors…";
  try {
    const response=await fetch("/api/map/artist/"+encodeURIComponent(mbid)+"?limit=20");
    if(!response.ok)throw new Error("Artist lookup failed: "+response.status);
    const result=await response.json();
    if(seq!==requestId)return;
    heading.textContent=result.seed.name;
    inspectNeighborhood(result.seed.x, result.seed.y);
    search.value=result.seed.name;
    detail.textContent="The highlighted dots are this artist and their 20 closest ALS neighbors. The map is an approximation.";
    neighborsEl.replaceChildren();
    result.neighbors.forEach(n=>{
      const b=document.createElement("button");
      b.textContent=n.name+" · "+n.als_score.toFixed(3);
      b.addEventListener("click",()=>choose(n.mbid));
      neighborsEl.append(b);
    });
    const label=showLabels.checked;
    overlays=[{type:"scattergl",mode:label?"markers+text":"markers",
      name:"Nearest neighbors",x:result.neighbors.map(n=>n.x),
      y:result.neighbors.map(n=>n.y),text:result.neighbors.map(n=>n.name),
      customdata:result.neighbors.map(n=>n.mbid),textposition:"top center",
      hovertemplate:"%{text}<extra>ALS neighbor</extra>",
      marker:{size:10,color:"#ff8c42",line:{color:"white",width:1}}},overlayPoint(result.seed,"#172f49",17,"Selected",label)];
    draw();
    // Zoom to seed and neighbors, avoiding extreme outliers.
    const xs=[result.seed.x,...result.neighbors.map(n=>n.x)];
    const ys=[result.seed.y,...result.neighbors.map(n=>n.y)];
    const span=Math.max(3,(Math.max(...xs)-Math.min(...xs))*1.5,(Math.max(...ys)-Math.min(...ys))*1.5);
    await Plotly.relayout(plot,{"xaxis.range":[result.seed.x-span/2,result.seed.x+span/2],
      "yaxis.range":[result.seed.y-span/2,result.seed.y+span/2]});
    status.textContent="Showing "+points.length.toLocaleString()+" background artists plus the selected artist and neighbors.";
  }catch(e){if(seq===requestId)status.textContent=e.message;}
}
showLabels.addEventListener("change",()=>{if(currentMbid)choose(currentMbid,false);});
search.addEventListener("input",()=>{
  clearTimeout(suggestionTimer);
  suggest.hidden=true;
  const q=search.value.trim();
  if(!q)return;
  suggestionTimer=setTimeout(async()=>{
    const res=await fetch("/api/suggest?q="+encodeURIComponent(q)+"&limit=10");
    if(!res.ok)return;
    const rows=await res.json();
    if(search.value.trim()!==q)return;
    suggest.replaceChildren();
    rows.forEach(a=>{
      const li=document.createElement("li"),b=document.createElement("button");
      b.textContent=a.name;b.addEventListener("click",()=>{suggest.hidden=true;choose(a.mbid);});
      li.append(b);suggest.append(li);
    });
    suggest.hidden=!rows.length;
  },180);
});
document.addEventListener("click",e=>{if(!e.target.closest(".map-search-wrap"))suggest.hidden=true;});
function onPlotClick(e) {
  const point = e.points?.[0];
  if (!point) return;
  const mbid = point.customdata;
  // Every plotted artist has its MusicBrainz ID in customdata.
  // Selecting it also refreshes the neighborhood guide via choose().
  if (typeof mbid === "string" && mbid.length > 0) {
    choose(mbid);
  } else if (Number.isFinite(point.x) && Number.isFinite(point.y)) {
    inspectNeighborhood(point.x, point.y);
  }
}

window.addEventListener("popstate",()=>{
  const mbid=new URLSearchParams(location.search).get("artist");
  if(mbid)choose(mbid,false);
});
(async()=>{
  try {
    const res=await fetch("/api/map");
    if(!res.ok)throw new Error("Map not built yet. Run python build_umap.py.");
    points=await res.json();
    await Plotly.newPlot(plot, groups.map(traceFor), {margin:{l:20,r:20,t:15,b:20},xaxis:{visible:false},yaxis:{visible:false,scaleanchor:"x"},legend:{orientation:"h",y:-.03}}, {responsive:true,displaylogo:false,scrollZoom:true});
    plot.on("plotly_click",onPlotClick);
    status.textContent=points.length.toLocaleString()+" artists on the map. Search for any trained artist.";
    const mbid=new URLSearchParams(location.search).get("artist");
    if(mbid)choose(mbid,false);
  }catch(e){status.textContent=e.message;}
})();
