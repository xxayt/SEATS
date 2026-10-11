"use strict";

const root = document.documentElement;
const body = document.body;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const themeButton = document.querySelector("#theme-toggle");

// 应用配色并更新切换按钮的说明。
function applyTheme(dark) {
  body.classList.toggle("dark-mode", dark);
  themeButton.setAttribute(
    "aria-label",
    dark ? "Switch to light theme" : "Switch to dark theme",
  );
  themeButton.title = dark ? "Switch to light theme" : "Switch to dark theme";
}
applyTheme(localStorage.getItem("seats-theme") === "dark");
// 切换并记住用户选择的配色。
themeButton.addEventListener("click", () => {
  const dark = !body.classList.contains("dark-mode");
  applyTheme(dark);
  localStorage.setItem("seats-theme", dark ? "dark" : "light");
});

// 根据系统的减少动态效果设置控制页面动效。
function applyMotion() {
  root.classList.toggle("motion-paused", reducedMotion.matches);
}
applyMotion();
reducedMotion.addEventListener("change", applyMotion);

const slideStates = new WeakMap();

// 在独立标签组内切换内容，并同步键盘焦点状态。
function activateTab(tab, tabs) {
  const slideGroup = tab.closest("[data-slide-tabs]");
  const slideState = slideGroup && slideStates.get(slideGroup);
  if (slideState?.sliding) {
    slideState.queued = { tab, tabs };
    return;
  }
  const previousIndex = tabs.findIndex(
    (item) => item.getAttribute("aria-selected") === "true",
  );
  const nextIndex = tabs.indexOf(tab);
  tabs.forEach((item) => {
    const selected = item === tab;
    item.setAttribute("aria-selected", String(selected));
    item.tabIndex = selected ? 0 : -1;
    document.getElementById(item.getAttribute("aria-controls")).hidden =
      !selected;
  });
  // 新旧图片同时平移，连续点击时排队切换，避免打断动画造成闪现。
  if (slideState && previousIndex !== nextIndex && !reducedMotion.matches) {
    const previous = document.getElementById(tabs[previousIndex].getAttribute("aria-controls"));
    const panel = document.getElementById(tab.getAttribute("aria-controls"));
    const direction = nextIndex > previousIndex ? -1 : 1;
    const timing = { duration: 800, easing: "cubic-bezier(0.65, 0, 0.35, 1)" };
    slideState.sliding = true;
    previous.hidden = false;
    previous.inert = true;
    previous.setAttribute("aria-hidden", "true");
    const outgoing = previous.animate(
      [{ transform: "translateX(0)" }, { transform: `translateX(${direction * 100}%)` }],
      { ...timing, fill: "forwards" },
    );
    const incoming = panel.animate(
      [{ transform: `translateX(${-direction * 100}%)` }, { transform: "translateX(0)" }],
      timing,
    );
    // 滑动结束后收起旧图，再处理最后一次待切换的选择。
    incoming.onfinish = () => {
      previous.hidden = true;
      previous.inert = false;
      previous.removeAttribute("aria-hidden");
      outgoing.cancel();
      slideState.sliding = false;
      const queued = slideState.queued;
      slideState.queued = null;
      if (queued) activateTab(queued.tab, queued.tabs);
    };
  }
}
document.querySelectorAll("[data-tabs]").forEach((group) => {
  if (group.hasAttribute("data-slide-tabs")) {
    slideStates.set(group, { sliding: false, queued: null });
  }
  const tabs = [...group.querySelectorAll('[role="tab"]')];
  tabs.forEach((tab, index) => {
    // 点击时显示对应面板。
    tab.addEventListener("click", () => activateTab(tab, tabs));
    // 支持方向键与首尾键浏览标签。
    tab.addEventListener("keydown", (event) => {
      const vertical =
        tab.closest('[role="tablist"]').getAttribute("aria-orientation") ===
        "vertical";
      const next = vertical ? "ArrowDown" : "ArrowRight";
      const prev = vertical ? "ArrowUp" : "ArrowLeft";
      let target;
      if (event.key === next) target = (index + 1) % tabs.length;
      if (event.key === prev) target = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") target = 0;
      if (event.key === "End") target = tabs.length - 1;
      if (target === undefined) return;
      event.preventDefault();
      activateTab(tabs[target], tabs);
      tabs[target].focus();
    });
  });
});

// 只为尚未进入视口的内容添加一次渐入效果。
const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.remove("pending");
        revealObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.08 },
);
document.querySelectorAll(".reveal").forEach((element) => {
  if (
    element.getBoundingClientRect().top > window.innerHeight &&
    !reducedMotion.matches
  ) {
    element.classList.add("pending");
    revealObserver.observe(element);
  }
});

const progress = document.querySelector(".reading-progress");
const navigation = [...document.querySelectorAll(".site-header nav a")];
// 更新阅读进度和当前章节导航。
function updateReadingPosition() {
  const range = root.scrollHeight - window.innerHeight;
  progress.style.width = `${range > 0 ? (window.scrollY / range) * 100 : 0}%`;
  let active = "";
  navigation.forEach((link) => {
    if (document.querySelector(link.hash).getBoundingClientRect().top <= 180)
      active = link.hash;
  });
  navigation.forEach((link) => {
    link.classList.toggle("active", link.hash === active);
    if (link.hash === active) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  });
}
window.addEventListener("scroll", updateReadingPosition, { passive: true });
window.addEventListener("resize", updateReadingPosition);
updateReadingPosition();

const viewer = document.querySelector("#image-viewer");
const viewerImage = document.querySelector("#viewer-image");
let lastFigure;

document.querySelectorAll("[data-lightbox]").forEach((link) => {
  // 点击论文图片时铺满视口展示，保持原图比例。
  link.addEventListener("click", (event) => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    lastFigure = link;
    viewerImage.alt = link.querySelector("img").alt;
    viewerImage.src = link.href;
    body.classList.add("modal-open");
    viewer.showModal();
  });
});
// 点击图片或任意空白区域即可关闭全屏展示。
viewer.addEventListener("click", () => viewer.close());
// 关闭后恢复页面滚动和原图片的键盘焦点。
viewer.addEventListener("close", () => {
  body.classList.remove("modal-open");
  lastFigure?.focus({ preventScroll: true });
});

// 复制论文引用，并在剪贴板不可用时显示真实状态。
document.querySelector("#copy-citation").addEventListener("click", async () => {
  const button = document.querySelector("#copy-citation");
  const status = document.querySelector("#copy-status");
  try {
    await navigator.clipboard.writeText(
      document.querySelector("#bibtex-content").textContent,
    );
    button.textContent = "Copied";
    status.textContent = "Citation copied to clipboard.";
    // 恢复复制按钮文字。
    setTimeout(() => {
      button.textContent = "Copy citation ⧉";
    }, 2000);
  } catch {
    status.textContent =
      "Clipboard unavailable. Select and copy the citation below.";
    button.textContent = "Select text to copy";
  }
});
