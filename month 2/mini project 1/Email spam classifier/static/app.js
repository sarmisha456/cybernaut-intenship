const form = document.querySelector("#checker-form");
const messageInput = document.querySelector("#message");
const characterCount = document.querySelector("#character-count");
const checkButton = document.querySelector("#check-button");
const buttonLabel = checkButton.querySelector(".button-label");
const result = document.querySelector("#result");

messageInput.addEventListener("input", () => {
  characterCount.textContent = `${messageInput.value.length.toLocaleString()} / 10,000`;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) {
    messageInput.focus();
    return;
  }

  checkButton.disabled = true;
  buttonLabel.textContent = "Checking…";
  result.hidden = true;
  result.className = "result";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "The message could not be checked.");
    }

    const isSpam = data.is_spam;
    result.classList.add(isSpam ? "result-spam" : "result-safe");
    result.replaceChildren();

    const icon = document.createElement("span");
    icon.className = "result-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = isSpam ? "!" : "✓";

    const copy = document.createElement("div");
    copy.className = "result-copy";
    const title = document.createElement("strong");
    title.textContent = isSpam ? "SPAM" : "HAM";
    const detail = document.createElement("span");
    detail.textContent = `${isSpam ? "Likely unwanted" : "Likely legitimate"} · Model confidence: ${data.confidence}%`;
    copy.append(title, detail);
    result.append(icon, copy);
    result.hidden = false;
  } catch (error) {
    result.classList.add("result-error");
    result.textContent = error.message || "Unable to reach the server. Please try again.";
    result.hidden = false;
  } finally {
    checkButton.disabled = false;
    buttonLabel.textContent = "Check message";
  }
});
