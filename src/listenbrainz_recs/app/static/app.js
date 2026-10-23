const searchInput = document.querySelector("#artist-search");
const suggestionsEl = document.querySelector("#suggestions");
const recommendationsEl = document.querySelector("#recommendations");
const selectedEl = document.querySelector("#selected-artist");
const seedGenresEl = document.querySelector("#seed-genres");
const statusEl = document.querySelector("#status");
const limitEl = document.querySelector("#limit");
const savedEl = document.querySelector("#saved-artists");
const savedCountEl = document.querySelector("#saved-count");
const STORAGE_KEY = "artist-explorer-saved-v1";
let saved = readSaved();

function readSaved() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  } catch { return {}; }
}
function storeSaved() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(saved)); }
  catch { alert("Unable to save bookmarks in this browser."); }
  renderSaved();
  document.querySelectorAll(".bookmark-button[data-mbid]").forEach(button => {
    setBookmarkAppearance(button, button.dataset.mbid);
  });
}
function setBookmarkAppearance(button, mbid) {
  const isSaved = Object.hasOwn(saved, mbid);
  button.textContent = isSaved ? "★" : "☆";
  button.title = isSaved ? "Remove from saved artists" : "Save artist for later";
  button.setAttribute("aria-label", button.title);
  button.setAttribute("aria-pressed", String(isSaved));
}
function bookmarkButton(artist) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "bookmark-button";
  button.dataset.mbid = artist.mbid;
  setBookmarkAppearance(button, artist.mbid);
  button.addEventListener("click", () => {
    if (Object.hasOwn(saved, artist.mbid)) delete saved[artist.mbid];
    else saved[artist.mbid] = {name: artist.name, mbid: artist.mbid};
    storeSaved();
  });
  return button;
}
function externalLink(label, url) {
  const link = document.createElement("a");
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = label;
  link.className = "discovery-link";
  link.setAttribute("aria-label", label + " (opens in a new tab)");
  return link;
}
function artistActions(artist) {
  const actions = document.createElement("div");
  actions.className = "artist-actions";
  actions.append(
    externalLink("Listen ↗", "https://www.youtube.com/results?search_query=" + encodeURIComponent(artist.name + " music")),
    externalLink("MusicBrainz ↗", "https://musicbrainz.org/artist/" + encodeURIComponent(artist.mbid)),
    bookmarkButton(artist)
  );
  return actions;
}
function renderSaved() {
  savedEl.replaceChildren();
  const artists = Object.values(saved).sort((a,b) => a.name.localeCompare(b.name));
  savedCountEl.textContent = String(artists.length);
  for (const artist of artists) {
    const item = document.createElement("li");
    const go = document.createElement("button");
    go.className = "saved-name";
    go.textContent = artist.name;
    go.type = "button";
    go.addEventListener("click", () => selectArtist(artist));
    item.append(go, artistActions(artist));
    savedEl.append(item);
  }
  document.querySelector("#saved-empty").hidden = artists.length !== 0;
}
renderSaved();

let suggestions = [];
let activeIndex = -1;
let currentMbid = null;
let debounceTimer;
let suggestController;
let similarController;

function renderGenres(container, genres) {
  container.replaceChildren();
  for (const genre of genres) {
    const tag = document.createElement("span");
    tag.className = "genre-tag";
    tag.textContent = genre;
    container.append(tag);
  }
}

function hideSuggestions() {
  suggestionsEl.hidden = true;
  suggestionsEl.replaceChildren();
  searchInput.setAttribute("aria-expanded", "false");
  activeIndex = -1;
}

function renderSuggestions() {
  suggestionsEl.replaceChildren();
  if (!suggestions.length) {
    hideSuggestions();
    return;
  }
  suggestions.forEach((artist, index) => {
    const li = document.createElement("li");
    li.setAttribute("role", "option");
    li.setAttribute("aria-selected", String(index === activeIndex));
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = artist.name;
    button.classList.toggle("active", index === activeIndex);
    button.addEventListener("mousedown", (event) => event.preventDefault());
    button.addEventListener("click", () => selectArtist(artist));
    li.append(button);
    suggestionsEl.append(li);
  });
  suggestionsEl.hidden = false;
  searchInput.setAttribute("aria-expanded", "true");
}

async function fetchSuggestions() {
  const query = searchInput.value.trim();
  if (!query) {
    suggestions = [];
    hideSuggestions();
    return;
  }
  suggestController?.abort();
  const controller = new AbortController();
  suggestController = controller;
  try {
    const response = await fetch(`/api/suggest?q=${encodeURIComponent(query)}&limit=10`, {signal: controller.signal});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const result = await response.json();
    if (controller !== suggestController) return;
    suggestions = result;
    activeIndex = -1;
    renderSuggestions();
  } catch (error) {
    if (error.name !== "AbortError") {
      suggestions = [];
      hideSuggestions();
      statusEl.textContent = `Autocomplete error: ${error.message}`;
    }
  }
}

searchInput.addEventListener("input", () => {
  clearTimeout(debounceTimer);
  suggestController?.abort();
  suggestions = [];
  hideSuggestions();
  debounceTimer = setTimeout(fetchSuggestions, 180);
});

searchInput.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    hideSuggestions();
    return;
  }
  if (!suggestions.length || suggestionsEl.hidden) return;
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    activeIndex = event.key === "ArrowDown"
      ? (activeIndex + 1) % suggestions.length
      : (activeIndex - 1 + suggestions.length) % suggestions.length;
    renderSuggestions();
  } else if (event.key === "Enter") {
    event.preventDefault();
    selectArtist(suggestions[activeIndex >= 0 ? activeIndex : 0]);
  }
});

document.addEventListener("click", (event) => {
  if (!event.target.closest(".search-wrap")) hideSuggestions();
});

async function selectArtist(artist, updateHistory = true) {
  currentMbid = artist.mbid;
  searchInput.value = artist.name;
  suggestions = [];
  hideSuggestions();
  if (updateHistory) history.pushState({mbid: artist.mbid}, "", `?artist=${encodeURIComponent(artist.mbid)}`);
  await loadRecommendations(artist.mbid);
}

async function loadRecommendations(mbid) {
  similarController?.abort();
  const controller = new AbortController();
  similarController = controller;
  recommendationsEl.replaceChildren();
  selectedEl.textContent = "Loading…";
  seedGenresEl.replaceChildren();
  statusEl.textContent = "Finding nearest neighbors…";
  try {
    const response = await fetch(`/api/similar/${encodeURIComponent(mbid)}?limit=${limitEl.value}`, {signal: controller.signal});
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `HTTP ${response.status}`);
    }
    const payload = await response.json();
    if (controller !== similarController) return;
    selectedEl.textContent = payload.seed.name;
    renderGenres(seedGenresEl, payload.seed.genres || []);
    const seedActions = document.querySelector("#seed-actions");
    seedActions.replaceChildren(artistActions(payload.seed));
    searchInput.value = payload.seed.name;
    statusEl.textContent = `${payload.results.length} artists ranked by ALS cosine similarity. Click an artist to explore.`;
    payload.results.forEach((artist, index) => {
      const li = document.createElement("li");
      const rank = document.createElement("span");
      rank.className = "rank";
      rank.textContent = `${index + 1}.`;
      const button = document.createElement("button");
      button.className = "artist-link";
      button.type = "button";
      button.textContent = artist.name;
      button.addEventListener("click", () => selectArtist(artist));
      const artistInfo = document.createElement("div");
      artistInfo.className = "artist-info";
      artistInfo.append(button);
      const genreLabels = document.createElement("div");
      genreLabels.className = "genres";
      renderGenres(genreLabels, artist.genres || []);
      artistInfo.append(genreLabels);
      const score = document.createElement("span");
      score.className = "score";
      score.textContent = Number(artist.als_score).toFixed(4);
      score.title = "ALS cosine similarity";
      const right = document.createElement("div");
      right.className = "recommendation-right";
      right.append(score, artistActions(artist));
      li.append(rank, artistInfo, right);
      recommendationsEl.append(li);
    });
  } catch (error) {
    if (error.name !== "AbortError") {
      selectedEl.textContent = "Unable to load artist";
      statusEl.textContent = error.message;
    }
  }
}

limitEl.addEventListener("change", () => {
  if (currentMbid) loadRecommendations(currentMbid);
});

window.addEventListener("popstate", () => {
  const mbid = new URLSearchParams(location.search).get("artist");
  currentMbid = mbid;
  if (mbid) loadRecommendations(mbid);
  else {
    similarController?.abort();
    selectedEl.textContent = "Select an artist to explore";
    seedGenresEl.replaceChildren();
    document.querySelector("#seed-actions").replaceChildren();
    statusEl.textContent = "Recommendations use exact cosine similarity over the ALS factors.";
    recommendationsEl.replaceChildren();
    searchInput.value = "";
  }
});

const initialMbid = new URLSearchParams(location.search).get("artist");
if (initialMbid) {
  currentMbid = initialMbid;
  loadRecommendations(initialMbid);
}
