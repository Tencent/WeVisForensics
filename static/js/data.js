/* Detection accuracy (%) from the main comparison table of the paper. */

(function () {
  const COLS = ["GB", "GB++", "Hailuo", "Jimeng", "Kling", "Seedance", "Sora2", "Wan2.5", "Visual", "Physical", "OCR", "Common sense", "Audio", "Avg."];

  // group: ours | spec (MLLM-based forgery detectors) | gen (generic MLLMs) | vlm (binary detectors)
  const ROWS = [
    ["NSG-VD", "vlm", [46.4, 53.7, 57.1, 48.5, 52.6, 54.7, 57.1, 51.0, 41.7, 43.2, 41.7, 34.2, 44.4, 48.7]],
    ["ReStraV", "vlm", [51.9, 66.9, 59.5, 56.8, 55.6, 64.0, 50.8, 54.9, 55.5, 54.6, 51.6, 47.0, 51.8, 57.0]],
    ["Qwen3.5-4B", "gen", [82.9, 73.2, 63.5, 56.5, 65.1, 73.8, 57.0, 73.2, 73.7, 71.8, 83.8, 94.4, 58.1, 74.3]],
    ["Qwen3.5-397B", "gen", [65.1, 66.7, 54.7, 52.0, 55.8, 63.0, 51.4, 63.4, 69.2, 66.7, 86.0, 94.5, 54.1, 65.7]],
    ["Qwen3-VL-8B", "gen", [44.8, 55.2, 52.0, 50.4, 50.5, 63.0, 52.6, 55.9, 61.7, 58.6, 72.2, 91.9, 47.8, 55.1]],
    ["Qwen3-Omni-30B-A3B", "gen", [50.7, 55.2, 51.7, 51.9, 52.9, 61.4, 54.6, 56.3, 57.5, 54.8, 70.6, 85.6, 43.1, 55.8]],
    ["Gemini-3.1-Pro", "gen", [69.4, 54.3, 55.1, 54.4, 54.6, 45.3, 48.6, 54.9, 61.8, 61.6, 63.4, 63.4, 63.1, 59.6]],
    ["BusterX++", "spec", [70.1, 79.6, 70.4, 71.7, 82.3, 71.8, 58.7, 87.3, 64.4, 65.1, 71.1, 87.1, 49.9, 72.7]],
    ["DeepTraceReward", "spec", [76.4, 58.8, 64.5, 64.1, 63.9, 56.5, 59.3, 65.5, 55.5, 54.2, 51.9, 55.4, 53.5, 62.9]],
    ["Skyra-RL", "spec", [68.2, 55.7, 51.7, 51.7, 53.5, 63.4, 50.6, 59.6, 83.4, 83.8, 74.0, 86.9, 68.5, 64.6]],
    ["VideoVeritas", "spec", [94.3, 91.4, 85.8, 80.8, 86.0, 86.6, 67.6, 86.3, 72.4, 75.3, 76.0, 91.5, 69.3, 86.2]],
    ["ARGUS", "spec", [86.4, 78.9, 61.8, 56.5, 65.1, 67.6, 56.1, 73.5, 73.3, 71.1, 80.4, 90.5, 64.2, 76.2]],
    ["WeVisForensics-DTPO", "ours", [97.7, 97.5, 88.9, 82.9, 89.6, 98.6, 82.4, 85.6, 97.3, 98.0, 98.0, 98.8, 97.0, 95.2]]
  ];

  const col = (name) => COLS.indexOf(name);
  const mean = (vals) => Math.round((vals.reduce((a, b) => a + b, 0) / vals.length) * 10) / 10;

  function item(name, pick, metric) {
    return {
      name,
      metric: metric || "Accuracy (%)",
      rows: ROWS.map(([model, group, v]) => ({ model, group, value: pick(v) }))
    };
  }
  const single = (name, label) => item(label || name, (v) => v[col(name)]);
  const avgOf = (label, names) => item(label, (v) => mean(names.map((n) => v[col(n)])), "Mean accuracy (%)");

  const MINT = ["Hailuo", "Jimeng", "Kling", "Seedance", "Sora2", "Wan2.5"];
  const AGENTIC = ["Visual", "Physical", "OCR", "Common sense", "Audio"];

  window.BENCH = [
    {
      label: "Overall",
      items: [item("Average", (v) => v[col("Avg.")], "Avg. of AgenticCue, GB, GB++, MintVid (%)")]
    },
    {
      label: "AgenticCue · agentic",
      items: [avgOf("Mean of 5", AGENTIC)].concat(AGENTIC.map((n) => single(n)))
    },
    {
      label: "GenBuster · ID",
      items: [single("GB", "GenBuster-Bench (ID)")]
    },
    {
      label: "GenBuster++ · OOD",
      items: [single("GB++", "GenBuster-Bench++")]
    },
    {
      label: "MintVid · new generators",
      items: [avgOf("Mean of 6", MINT)].concat(MINT.map((n) => single(n)))
    }
  ];
})();
