document.addEventListener("DOMContentLoaded", function () {
  const slideshow = document.getElementById("attention-slideshow");
  if (!slideshow) return;

  const slides = Array.from(slideshow.querySelectorAll(".slide"));
  const dots = Array.from(slideshow.querySelectorAll(".dot"));
  const prevBtn = slideshow.querySelector(".prev-btn");
  const nextBtn = slideshow.querySelector(".next-btn");

  let current = 0;
  const words = ["The", "cat", "sat", "on", "the", "mat"];

  function buildTokenRow(containerId, labels, highlightIndex) {
    const row = document.getElementById(containerId);
    if (!row) return [];

    row.innerHTML = "";
    const tokens = [];

    labels.forEach((label, i) => {
      const token = document.createElement("div");
      token.className = "token word-token";
      if (i === highlightIndex) token.classList.add("highlight-token");
      token.textContent = label;
      row.appendChild(token);
      tokens.push(token);
    });

    return tokens;
  }

  function lineBetween(layer, a, b, className) {
    const layerBox = layer.getBoundingClientRect();
    const aBox = a.getBoundingClientRect();
    const bBox = b.getBoundingClientRect();

    if (!aBox.width || !aBox.height || !bBox.width || !bBox.height) return;

    const x1 = aBox.left + aBox.width / 2 - layerBox.left;
    const y1 = aBox.top + aBox.height / 2 - layerBox.top;
    const x2 = bBox.left + bBox.width / 2 - layerBox.left;
    const y2 = bBox.top + bBox.height / 2 - layerBox.top;

    const dx = x2 - x1;
    const dy = y2 - y1;
    const length = Math.sqrt(dx * dx + dy * dy);
    const angle = Math.atan2(dy, dx) * (180 / Math.PI);

    const line = document.createElement("div");
    line.className = `connection-line ${className || ""}`;
    line.style.width = `${length}px`;
    line.style.left = `${x1}px`;
    line.style.top = `${y1}px`;
    line.style.transform = `rotate(${angle}deg)`;

    layer.appendChild(line);
  }

  function renderDenseAttention() {
    const layer = document.getElementById("dense-connections");
    if (!layer) return;

    const slide = layer.closest(".slide");
    if (!slide || !slide.classList.contains("active")) return;

    const top = buildTokenRow("dense-top", words, 1);
    const bottom = buildTokenRow("dense-bottom", words, 1);

    layer.innerHTML = "";

    const catToken = top[1];
    bottom.forEach((b) => {
      lineBetween(layer, catToken, b, "dense");
    });
  }

  function renderSparseAttention() {
    const layer = document.getElementById("sparse-connections");
    if (!layer) return;

    const slide = layer.closest(".slide");
    if (!slide || !slide.classList.contains("active")) return;

    const top = buildTokenRow("sparse-top", words, 1);
    const bottom = buildTokenRow("sparse-bottom", words, 1);

    layer.innerHTML = "";

    const catToken = top[1];
    [0, 1, 2].forEach((j) => {
      lineBetween(layer, catToken, bottom[j], "sparse");
    });
    lineBetween(layer, catToken, bottom[5], "sparse");
  }

  function renderVisibleSlideVisuals() {
    renderDenseAttention();
    renderSparseAttention();
  }

  function showSlide(index) {
    current = (index + slides.length) % slides.length;

    slides.forEach((slide, i) => {
      slide.classList.toggle("active", i === current);
    });

    dots.forEach((dot, i) => {
      dot.classList.toggle("active", i === current);
    });

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        renderVisibleSlideVisuals();
      });
    });
  }

  prevBtn.addEventListener("click", function () {
    showSlide(current - 1);
  });

  nextBtn.addEventListener("click", function () {
    showSlide(current + 1);
  });

  dots.forEach((dot) => {
    dot.addEventListener("click", function () {
      showSlide(parseInt(dot.dataset.target, 10));
    });
  });

  window.addEventListener("resize", function () {
    renderVisibleSlideVisuals();
  });

  window.addEventListener("load", function () {
    renderVisibleSlideVisuals();
  });

  showSlide(0);
});