const header = document.querySelector(".site-header");
const menuToggle = document.querySelector(".menu-toggle");
const disabledLinks = document.querySelectorAll('[aria-disabled="true"]');
const modal = document.querySelector("[data-wechat-modal]");
const openWechat = document.querySelector("[data-open-wechat]");
const closeWechatControls = document.querySelectorAll("[data-close-wechat]");

menuToggle?.addEventListener("click", () => {
  const isOpen = header.classList.toggle("is-open");
  menuToggle.setAttribute("aria-expanded", String(isOpen));
});

document.querySelectorAll(".site-nav a").forEach((link) => {
  link.addEventListener("click", () => {
    header.classList.remove("is-open");
    menuToggle?.setAttribute("aria-expanded", "false");
  });
});

disabledLinks.forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
  });
});

function showWechatModal() {
  modal.hidden = false;
  document.body.style.overflow = "hidden";
}

function hideWechatModal() {
  modal.hidden = true;
  document.body.style.overflow = "";
}

openWechat?.addEventListener("click", showWechatModal);
closeWechatControls.forEach((control) => {
  control.addEventListener("click", hideWechatModal);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !modal.hidden) {
    hideWechatModal();
  }
});
