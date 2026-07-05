const HF_ENDPOINT = "https://sarge28-tiny-gpt-demo.hf.space/generate";

async function generateText(prompt) {
  if (HF_ENDPOINT === "TODO_REPLACE_WITH_HF_SPACE_URL") {
    throw new Error("DEMO_NOT_CONFIGURED");
  }

  const response = await fetch(HF_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt })
  });

  if (!response.ok) {
    throw new Error("REQUEST_FAILED");
  }

  const data = await response.json();
  return data.generated_text; // adjust field name to match actual HF Space response shape
}

const promptInput = document.getElementById("prompt-input");
const generateBtn = document.getElementById("generate-btn");
const statusEl = document.getElementById("demo-status");
const outputEl = document.getElementById("demo-output");

function showStatus(message, isError) {
  statusEl.textContent = message;
  statusEl.hidden = false;
  statusEl.classList.toggle("error", Boolean(isError));
}

function hideStatus() {
  statusEl.hidden = true;
  statusEl.classList.remove("error");
}

function showOutput(text) {
  outputEl.textContent = text;
  outputEl.hidden = false;
}

function hideOutput() {
  outputEl.hidden = true;
}

generateBtn.addEventListener("click", async () => {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    showStatus("Enter a prompt first.", true);
    return;
  }

  generateBtn.disabled = true;
  hideOutput();
  showStatus("Generating...");

  try {
    const generated = await generateText(prompt);
    hideStatus();
    showOutput(generated);
  } catch (err) {
    if (err.message === "DEMO_NOT_CONFIGURED") {
      showStatus("Demo isn't connected yet — check back soon.", true);
    } else {
      showStatus("Something went wrong generating a response. Try again in a moment.", true);
    }
  } finally {
    generateBtn.disabled = false;
  }
});
