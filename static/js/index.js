(() => {
  "use strict";

  const status = document.querySelector("#media-status");
  const menu = document.querySelector("#main-nav");
  const menuToggle = document.querySelector("#menu-toggle");
  function closeMenu() {
    menu.classList.remove("is-open");
    menuToggle.setAttribute("aria-expanded", "false");
    menuToggle.setAttribute("aria-label", "Open navigation");
  }
  menuToggle.addEventListener("click", () => {
    const open = menuToggle.getAttribute("aria-expanded") !== "true";
    menu.classList.toggle("is-open", open);
    menuToggle.setAttribute("aria-expanded", String(open));
    menuToggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
  });
  menu.querySelectorAll("a").forEach(link => link.addEventListener("click", closeMenu));
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && menu.classList.contains("is-open")) {
      closeMenu();
      menuToggle.focus();
    }
  });

  const cover = document.querySelector("#cover-video");
  const coverToggle = document.querySelector("#cover-toggle");
  const film = document.querySelector("#overview-film");
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  let coverRequested = !reducedMotion.matches;
  let coverVisible = true;
  function syncCover() {
    const label = cover.paused ? "Play background video" : "Pause background video";
    coverToggle.setAttribute("aria-label", label);
    coverToggle.title = label;
    coverToggle.innerHTML = '<i class="fas fa-' + (cover.paused ? "play" : "pause") + '" aria-hidden="true"></i>';
  }
  async function playCover() {
    if (!coverRequested || !coverVisible || document.hidden || !film.paused || document.querySelector("#figure-dialog").open) return;
    try { await cover.play(); } catch { syncCover(); }
  }
  if (reducedMotion.matches) {
    cover.autoplay = false;
    cover.pause();
  }
  cover.addEventListener("play", syncCover);
  cover.addEventListener("pause", syncCover);
  coverToggle.addEventListener("click", () => {
    coverRequested = cover.paused;
    if (coverRequested) playCover();
    else cover.pause();
  });
  reducedMotion.addEventListener("change", event => {
    coverRequested = !event.matches;
    if (event.matches) cover.pause();
    else playCover();
  });
  new IntersectionObserver(entries => {
    coverVisible = entries[0].isIntersecting;
    if (coverVisible) playCover();
    else cover.pause();
  }, { threshold: 0.1 }).observe(cover);
  syncCover();

  const demoVideos = [...document.querySelectorAll(".demo video")];
  const groupPlay = document.querySelector("#group-play");
  const groupRestart = document.querySelector("#group-restart");
  let playGeneration = 0;
  let groupStarting = false;
  const currentVideos = () => [...document.querySelectorAll('.task-panel:not([hidden]) video')];
  function syncGroup() {
    const playing = currentVideos().some(video => !video.paused && !video.ended);
    groupPlay.querySelector("span").textContent = playing ? "Pause all four" : "Play all four";
    const icon = groupPlay.querySelector("svg, i");
    if (icon) icon.outerHTML = '<i class="fas fa-' + (playing ? "pause" : "play") + '" aria-hidden="true"></i>';
    groupPlay.disabled = groupStarting;
    groupRestart.disabled = groupStarting;
  }
  function pauseDemos() {
    playGeneration += 1;
    groupStarting = false;
    demoVideos.forEach(video => video.pause());
    syncGroup();
  }
  // Each clip owns its end state. An ended clip never stops the other players.
  demoVideos.forEach(video => {
    ["play", "pause", "ended"].forEach(event => video.addEventListener(event, syncGroup));
    video.addEventListener("error", () => {
      status.textContent = "A demonstration could not be loaded. Please retry using its video controls.";
    });
  });
  async function startGroup(restart = false) {
    const generation = ++playGeneration;
    const videos = currentVideos();
    groupStarting = true;
    syncGroup();
    film.pause();
    const results = await Promise.allSettled(videos.map(async video => {
      if (restart || video.ended) video.currentTime = 0;
      await video.play();
      // A tab switch or visibility change may occur while play() is pending.
      if (generation !== playGeneration) video.pause();
    }));
    if (generation !== playGeneration) return;
    groupStarting = false;
    syncGroup();
    status.textContent = results.some(result => result.status === "rejected")
      ? "Some videos could not start. Use their individual play controls."
      : "Playing four held-out-object demonstrations.";
  }
  groupPlay.addEventListener("click", () => {
    if (currentVideos().some(video => !video.paused && !video.ended)) pauseDemos();
    else startGroup();
  });
  groupRestart.addEventListener("click", () => startGroup(true));
  film.addEventListener("play", () => {
    cover.pause();
    pauseDemos();
  });
  film.addEventListener("pause", playCover);
  film.addEventListener("error", () => {
    status.textContent = "The overview video could not be loaded. Please reload the page and retry.";
  });

  document.querySelectorAll('[role="tablist"]').forEach(tablist => {
    const tabs = [...tablist.querySelectorAll('[role="tab"]')];
    function select(tab, focus = false) {
      if (tab.getAttribute("aria-selected") === "true") {
        if (focus) tab.focus();
        return;
      }
      if (tablist.classList.contains("task-tabs")) pauseDemos();
      tabs.forEach(item => {
        const selected = item === tab;
        item.setAttribute("aria-selected", String(selected));
        item.tabIndex = selected ? 0 : -1;
        document.getElementById(item.getAttribute("aria-controls")).hidden = !selected;
      });
      syncGroup();
      if (focus) tab.focus();
    }
    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => select(tab));
      tab.addEventListener("keydown", event => {
        let next;
        if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
        if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = tabs.length - 1;
        if (next !== undefined) {
          event.preventDefault();
          select(tabs[next], true);
        }
      });
    });
  });
  new IntersectionObserver(entries => {
    if (!entries[0].isIntersecting) pauseDemos();
  }, { threshold: 0 }).observe(document.querySelector("#real-world-videos"));
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      cover.pause();
      film.pause();
      pauseDemos();
    } else {
      playCover();
    }
  });

  const dialog = document.querySelector("#figure-dialog");
  const dialogImage = dialog.querySelector("img");
  const dialogTitle = document.querySelector("#figure-dialog-title");
  const sizeButton = document.querySelector("#figure-size");
  const imageContainer = dialog.querySelector(".dialog-image");
  let figureOpener;
  function resetZoom() {
    imageContainer.classList.remove("original");
    sizeButton.setAttribute("aria-pressed", "false");
    sizeButton.setAttribute("aria-label", "View figure at original size");
    sizeButton.title = "View at original size";
    sizeButton.innerHTML = '<i class="fas fa-search-plus" aria-hidden="true"></i>';
  }
  document.querySelectorAll("[data-figure]").forEach(button => {
    button.addEventListener("click", () => {
      const image = button.querySelector("img");
      figureOpener = button;
      dialogImage.src = image.src;
      dialogImage.alt = image.alt;
      dialogTitle.textContent = button.getAttribute("aria-label").replace(/^Enlarge /, "");
      resetZoom();
      pauseDemos();
      cover.pause();
      film.pause();
      dialog.showModal();
      document.body.classList.add("dialog-open");
    });
  });
  document.querySelector("#figure-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", event => {
    if (event.target !== dialog) return;
    const box = dialog.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
  });
  dialog.addEventListener("close", () => {
    document.body.classList.remove("dialog-open");
    resetZoom();
    figureOpener?.focus({ preventScroll: true });
    playCover();
  });
  sizeButton.addEventListener("click", () => {
    const original = imageContainer.classList.toggle("original");
    sizeButton.setAttribute("aria-pressed", String(original));
    sizeButton.setAttribute("aria-label", original ? "Fit figure to window" : "View figure at original size");
    sizeButton.title = original ? "Fit to window" : "View at original size";
    sizeButton.innerHTML = '<i class="fas fa-search-' + (original ? "minus" : "plus") + '" aria-hidden="true"></i>';
  });

  const navLinks = [...menu.querySelectorAll("a")];
  // Observe only sections represented in the top navigation.
  const navObserver = new IntersectionObserver(entries => {
    const active = entries.find(entry => entry.isIntersecting);
    if (!active) return;
    navLinks.forEach(link => {
      if (link.hash === "#" + active.target.id) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  }, { rootMargin: "-15% 0px -65% 0px" });
  navLinks.forEach(link => navObserver.observe(document.querySelector(link.hash)));
})();
