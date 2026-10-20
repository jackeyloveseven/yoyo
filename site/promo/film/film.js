/* YoYo 宣传片 — 逐帧 seek 渲染（同一 t 永远同一帧，不依赖 CSS 动画）。
 * 设计语言来自 wallstreet-explainer：镜头在有前后层的舞台里走，东西原地换状态。
 * 素材是真实的文件树、真实代码、真实 CLI 输出；数字来自 assets/data.json。 */
(() => {
  const { camPath, L, DEFS } = S25;
  const stage = document.getElementById("stage");
  const clamp = (x) => Math.max(0, Math.min(1, x));
  const ease = (x) => { x = clamp(x); return x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2; };
  const mix = (a, b, k) => a + (b - a) * clamp(k);
  const FPS = 30;

  let D, T, SHOTS;

  const fit = () => {
    const k = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
    stage.style.transform = `scale(${k})`;
    stage.style.transformOrigin = "0 0";
  };
  fit();
  window.addEventListener("resize", fit);

  const beat = (id) => T.beats.find((b) => b.id === id);
  const cue = (id, word) => {
    const b = beat(id);
    if (!b) return 0;
    if (!word) return b.start;
    const i = b.text.indexOf(word);
    return i < 0 ? b.start : (b.chars[i] ? b.chars[i][1] : b.start);
  };
  const span = (id) => { const b = beat(id); return [b.start, b.end]; };
  const prog = (t, id) => { const [a, b] = span(id); return clamp((t - a) / Math.max(0.1, b - a)); };

  // ---------- 组件 ----------
  const win = (x, y, w, h, title, body, opts = {}) => {
    const dim = opts.dim ? "filter:brightness(.62) saturate(.7);" : "";
    return `<div class="win" style="left:${x}px;top:${y}px;width:${w}px;height:${h}px;${dim}opacity:${opts.o ?? 1}">
      <div class="win-head"><span class="dots"><i style="background:#e06c75"></i><i style="background:#f0a93b"></i><i style="background:#4fd1b0"></i></span><span>${title}</span></div>
      <div class="code" style="font-size:${opts.fs || 21}px;line-height:${opts.lh || 36}px">${body}</div>
    </div>`;
  };

  const codeLine = (n, text, hl = false, indent = 0) =>
    `<div class="line ${hl ? "hl" : ""}" style="padding-left:${18 + indent * 26}px"><span class="ln">${n}</span><span>${text}</span></div>`;

  const snippet = (lines) => lines.map((ln, i) => codeLine(i + 1, ln[0], !!ln[1], ln[2] || 0)).join("");

  const tree = (items, t, a) =>
    items.map(([text, i, mark], k) => {
      const on = t >= a + i * 0.16;
      return `<div class="tree-line" style="opacity:${on ? 1 : 0};transform:translateX(${on ? 0 : -14}px);padding-left:${k ? 42 : 10}px;color:${mark === "hl" ? "#f0a93b" : mark === "dim" ? "#5d6678" : "#c9cfda"}">${text}</div>`;
    }).join("");

  const bg = (c, cp) => L(c, cp, 0.32, '<div class="worldgrid"></div>', { mb: false });
  const layer = (c, cp, z, html, o = {}) => L(c, cp, z, html, o);
  const cam = camPath;

  // ---------- 真实代码（来自 nn/flow.py） ----------
  const FLOW_CODE = [
    ['def flow_sample(clean, time, noise):'],
    ['    """x_t = (1 - t) * x_0 + t * noise"""'],
    ['    ...'],
    ['    return FlowSample(', false, 1],
    ['        noisy=(1.0 - expanded_time) * clean + expanded_time * noise,', true, 1],
    ['        target=noise - clean,', true, 1],
    ['    )', false, 1],
  ];

  const MODULE_LINES = [
    ["load_model(cfg)"],
    ["add_noise(x0, t)"],
    ["ode_step(x, v, dt)"],
    ["lr_schedule(step)"],
    ["fsdp_wrap(model)"],
  ];

  // 1. 每读一篇论文,仓库里多一套脚本
  function s01(t) {
    const [a] = span("s01");
    const c = cam(t, { x: 760, y: 540, s: 1.02, t: a }, [{ t: a, d: 3.6, to: { x: 860, s: 1.08 } }], 0.02);
    const cp = cam(t - 1 / FPS, { x: 760, y: 540, s: 1.02, t: a }, [{ t: a, d: 3.6, to: { x: 860, s: 1.08 } }], 0.02);
    const items = [
      ["repo/", 0, "dim"],
      ["train_sft.py", 1, "hl"],
      ["train_dpo.py", 2, "hl"],
      ["train_rl.py", 3, "hl"],
      ["train_distill.py", 4, "hl"],
      ["... 还有下一篇", 5, "dim"],
    ];
    const body = `<div class="tree">${tree(items, t, a)}</div>`;
    const html = `<div class="brow" style="left:300px;top:120px;opacity:${clamp(prog(t, "s01") * 2)}">你的仓库</div>` +
      `<div style="position:absolute;left:300px;top:182px;width:900px;min-height:520px;padding:26px 0;border-radius:14px;background:#0f141c;border:1px solid #2b3342">${body}</div>`;
    return bg(c, cp) + layer(c, cp, 1, html);
  }

  // 2. 四套脚本,五块同样的代码
  function s02(t) {
    const [a] = span("s02");
    const moves = [{ t: a, d: 1.3, to: { s: 0.82 } }, { t: a + 2.4, d: 1.2, to: { s: 0.88 } }];
    const c = cam(t, { x: 960, y: 540, s: 1.06, t: a }, moves, 0.02);
    const cp = cam(t - 1 / FPS, { x: 960, y: 540, s: 1.06, t: a }, moves, 0.02);
    const names = ["train_sft.py", "train_dpo.py", "train_rl.py", "train_distill.py"];
    const cols = names.map((name, i) => {
      const k = ease((t - a - 0.1 - i * 0.09) / 0.38);
      const x = 96 + i * 438;
      const body = snippet(MODULE_LINES.map((m) => [m[0]]));
      return `<div style="position:absolute;left:${mix(540, x, k)}px;top:170px;opacity:${k}">${win(0, 0, 400, 620, name, body, { fs: 19, lh: 62 })}</div>`;
    }).join("");
    return bg(c, cp) + layer(c, cp, 1, cols);
  }

  // 3. 同一个公式,四种写法
  function s03(t) {
    const [a] = span("s03");
    const moves = [
      { t: a + 0.2, d: 1.4, to: { x: 940, s: 1.04 } },
      { t: a + 2.4, d: 1.4, to: { x: 980, s: 1.04 } },
      { t: a + 4.6, d: 1.0, to: { x: 960, s: 1.0 } },
    ];
    const c = cam(t, { x: 960, y: 540, s: 1.0, t: a }, moves, 0.02);
    const cp = cam(t - 1 / FPS, { x: 960, y: 540, s: 1.0, t: a }, moves, 0.02);
    const variants = [
      ["x_t = (1 - t) * x0 + t * noise"],
      ["xt = (1 - t) * clean + t * eps"],
      ["noisy = (1 - t) * latent + t * noise"],
      ["x_t = lerp(x0, noise, t)"],
    ];
    const names = ["train_sft.py", "train_dpo.py", "train_rl.py", "train_distill.py"];
    const cols = names.map((name, i) => {
      const x = 96 + i * 438;
      const body = codeLine(1, "def add_noise(x0, noise, t):") +
        codeLine(2, variants[i][0], true) +
        codeLine(3, "    return");
      return `<div style="position:absolute;left:${x}px;top:210px">${win(0, 0, 400, 330, name, body, { fs: 18, lh: 54 })}</div>`;
    }).join("");
    const k = ease((t - a - 0.5) / 0.6);
    const note = `<div style="position:absolute;left:960px;top:640px;transform:translateX(-50%);font:600 34px var(--han);color:#e06c75;opacity:${k}">四个名字，一个公式</div>`;
    return bg(c, cp) + layer(c, cp, 1, cols + note);
  }

  // 4. 改一个公式,要改四遍
  function s04(t) {
    const [a] = span("s04");
    const moves = [{ t: a, d: 5.4, to: { s: 1.06, y: 520 } }];
    const c = cam(t, { x: 960, y: 540, s: 1.0, t: a }, moves, 0.02);
    const cp = cam(t - 1 / FPS, { x: 960, y: 540, s: 1.0, t: a }, moves, 0.02);
    const names = ["train_sft.py", "train_dpo.py", "train_rl.py", "train_distill.py"];
    const done = (i) => t >= a + 0.7 + i * 1.2;
    const cols = names.map((name, i) => {
      const x = 96 + i * 438;
      const body = codeLine(1, "def add_noise(x0, noise, t):") +
        codeLine(2, "    x_t = (1 - t) * x0 + t * noise", done(i)) +
        codeLine(3, "    return x_t");
      const mark = done(i) ? `<div style="position:absolute;right:16px;top:10px;font:700 20px var(--mono);color:#4fd1b0">✓</div>` : "";
      return `<div style="position:absolute;left:${x}px;top:240px">${win(0, 0, 400, 300, name, body, { fs: 18, lh: 54, dim: !done(i) && t > a + 0.4 })}${mark}</div>`;
    }).join("");
    const k = ease((t - a - 3.4) / 0.7);
    const count = `<div style="position:absolute;left:960px;top:660px;transform:translateX(-50%);font:700 96px var(--num);color:#e06c75;opacity:${k}">×4</div>`;
    return bg(c, cp) + layer(c, cp, 1, cols + count);
  }

  // 5. 拆开,只留数学
  function s05(t) {
    const [a] = span("s05");
    const c = cam(t, { x: 960, y: 560, s: 0.92, t: a }, [{ t: a, d: 3.2, to: { y: 500, s: 1.0 } }], 0.02);
    const cp = cam(t - 1 / FPS, { x: 960, y: 560, s: 0.92, t: a }, [{ t: a, d: 3.2, to: { y: 500, s: 1.0 } }], 0.02);
    const chunks = [
      ["add_noise", "nn/", 0],
      ["ode_step", "nn/", 1],
      ["loss_fn", "algorithms/", 2],
      ["adapter", "models/", 3],
      ["reward", "rewards/", 4],
    ];
    const blocks = chunks.map(([fn, dir], i) => {
      const k = ease((t - a - 0.3 - i * 0.12) / 0.55);
      const x = 210 + (i % 5) * 320;
      return `<div class="chunk" style="left:${x}px;top:${mix(760, 330 + (i % 2) * 120, k)}px;opacity:${k}">
        <b>${fn}</b><span>${dir}</span></div>`;
    }).join("");
    const label = `<div style="position:absolute;left:960px;top:180px;transform:translateX(-50%);font:600 30px var(--han);color:#a2aab8;opacity:${clamp(prog(t, "s05") * 2)}">只留下数学</div>`;
    return bg(c, cp) + layer(c, cp, 1, blocks + label);
  }

  // 6. 五块各归各位;真实 flow.py 片段
  function s06(t) {
    const [a] = span("s06");
    const c = cam(t, { x: 960, y: 480, s: 1.0, t: a }, [{ t: a + 0.4, d: 1.4, to: { x: 1000, s: 1.06 } }], 0.02);
    const cp = cam(t - 1 / FPS, { x: 960, y: 480, s: 1.0, t: a }, [{ t: a + 0.4, d: 1.4, to: { x: 1000, s: 1.06 } }], 0.02);
    const dirs = [
      ["nn/", "加噪 · 求解器 · 学习率"],
      ["algorithms/", "SFT / DPO / 蒸馏 / RL"],
      ["models/", "模型适配器"],
      ["datasets/", "JSONL 清单"],
      ["rewards/", "身份 · VLM · 美学"],
    ];
    const left = dirs.map(([name, desc], i) => {
      const k = ease((t - a - 0.1 - i * 0.1) / 0.4);
      return `<div style="position:absolute;left:110px;top:${170 + i * 116}px;opacity:${k}">
        <div style="font:700 30px var(--mono);color:#f0a93b">${name}</div>
        <div style="font:400 19px var(--han);color:#8b93a3;margin-top:4px">${desc}</div></div>`;
    }).join("");
    const body = snippet(FLOW_CODE);
    const codeWin = win(700, 210, 980, 420, "src/mmtrainer/nn/flow.py", body, { fs: 20, lh: 44 });
    const note = `<div style="position:absolute;left:960px;top:680px;transform:translateX(-50%);font:500 24px var(--han);color:#4fd1b0;opacity:${ease((t - a - 2.2) / 0.6)}">全仓库只有这一份</div>`;
    return bg(c, cp) + layer(c, cp, 1, left + codeWin + note);
  }

  // 7. 加方法:一个 loss 函数 + 一行注册
  function s07(t) {
    const [a] = span("s07");
    const c = cam(t, { x: 960, y: 540, s: 1.02, t: a }, [{ t: a, d: 3.6, to: { s: 1.1, y: 510 } }], 0.02);
    const cp = cam(t - 1 / FPS, { x: 960, y: 540, s: 1.02, t: a }, [{ t: a, d: 3.6, to: { s: 1.1, y: 510 } }], 0.02);
    const lines = [
      ["def my_loss(model, batch, ctx):"],
      ["    pred = model(batch.x_t, batch.t)"],
      ["    return ((pred - batch.target) ** 2).mean()"],
      [""],
      ['registry.register("my_loss", my_loss)', true],
    ];
    const body = lines.map((ln, i) => codeLine(i + 1, ln[0] || "", !!ln[1])).join("");
    const k = ease((t - a - 0.5) / 0.7);
    const badge = `<div style="position:absolute;left:1540px;top:250px;opacity:${k};padding:12px 18px;border-radius:10px;background:#221a10;border:1px solid #8a5d18;font:600 24px var(--mono);color:#f0a93b">+ 1 行</div>`;
    return bg(c, cp) + layer(c, cp, 1, win(360, 180, 1000, 420, "algorithms/my_loss.py", body, { fs: 21, lh: 62 }) + badge);
  }

  // 8. 加模型:一个适配器,十个目标
  function s08(t) {
    const [a] = span("s08");
    const c = cam(t, { x: 860, y: 540, s: 1.0, t: a }, [{ t: a + 0.4, d: 2.8, to: { x: 980, s: 1.08 } }], 0.02);
    const cp = cam(t - 1 / FPS, { x: 860, y: 540, s: 1.0, t: a }, [{ t: a + 0.4, d: 2.8, to: { x: 980, s: 1.08 } }], 0.02);
    const lines = [
      ["class MyModelAdapter(BaseModelAdapter):"],
      ['    plugin_id = "my_model"'],
      ['    capabilities = {"image_edit", "image_training"}'],
      ["    ..."],
    ];
    const body = lines.map((ln, i) => codeLine(i + 1, ln[0])).join("");
    const lights = [...Array(10)].map((_, i) => {
      const on = t >= cue("s08", "十个目标") + i * 0.06;
      return `<i style="position:absolute;left:${1380 + (i % 5) * 38}px;top:${420 + Math.floor(i / 5) * 38}px;width:18px;height:18px;border-radius:50%;background:${on ? "#4fd1b0" : "#26303e"};box-shadow:${on ? "0 0 16px #4fd1b066" : "none"}"></i>`;
    }).join("");
    return bg(c, cp) + layer(c, cp, 1, win(300, 220, 900, 360, "models/my_model.py", body, { fs: 21, lh: 62 }) + lights);
  }

  // 9. 真实终端:registry list
  function s09(t) {
    const [a] = span("s09");
    const c = cam(t, { x: 960, y: 540, s: 0.96, t: a }, [{ t: a + 0.4, d: 4.6, to: { y: 505, s: 1.02 } }], 0.02);
    const cp = cam(t - 1 / FPS, { x: 960, y: 540, s: 0.96, t: a }, [{ t: a + 0.4, d: 4.6, to: { y: 505, s: 1.02 } }], 0.02);
    const models = D.models.map((m) => `"${m.id}"`);
    const objs = D.matrix.rows.map((r) => `"${r.id}"`);
    const lines = [
      ['<span class="c">$</span> mmtrain registry list'],
      ["{"],
      [`  <span class="k">"models"</span>: [${models.slice(0, 3).join(", ")}, …],`],
      [`  <span class="k">"objectives"</span>: [${objs.slice(0, 3).join(", ")}, …],`],
      [`  <span class="k">"rewards"</span>: [${D.registry_list.slice(-7, -4).map((x) => x.trim().replace(/,$/, "")).join(", ")}, …]`],
      ["}"],
    ];
    const shown = lines.slice(0, Math.max(1, Math.floor((t - a) / 0.32) + 1));
    const nums = [
      [D.counts.models, "模型", 1500, 250],
      [D.counts.objectives, "目标", 1500, 480],
      [D.counts.rewards, "奖励", 1500, 710],
    ];
    const numHtml = nums.map(([n, label, x, y], i) => {
      const k = ease((t - a - 1.2 - i * 0.55) / 0.45);
      return k > 0
        ? `<div class="mega" style="left:${x}px;top:${y}px;font-size:${130 * mix(0.9, 1, k)}px;opacity:${k}">${n}</div>` +
          `<div class="sub" style="left:${x + 4}px;top:${y + 120}px;opacity:${k}">${label}</div>`
        : "";
    }).join("");
    const body = shown.map((x) => `<div>${x}</div>`).join("");
    return bg(c, cp) + layer(c, cp, 1,
      `<div class="term" style="left:250px;top:150px;width:1140px;height:740px">
        <div class="term-head"><span class="dots"><i style="background:#e06c75"></i><i style="background:#f0a93b"></i><i style="background:#4fd1b0"></i></span><span>mmtrain</span></div>
        <div class="term-body">${body}</div></div>` + numHtml);
  }

  // 10. compose 蓝图
  function s10(t) {
    const [a] = span("s10");
    const c = cam(t, { x: 960, y: 540, s: 0.98, t: a }, [{ t: a + 0.5, d: 1.4, to: { y: 470, s: 1.08 } }], 0.02);
    const cp = cam(t - 1 / FPS, { x: 960, y: 540, s: 0.98, t: a }, [{ t: a + 0.5, d: 1.4, to: { y: 470, s: 1.08 } }], 0.02);
    const cmd = "mmtrain compose --model qwen_image_edit_2x --objective image_sft --reward none";
    const typed = cmd.slice(0, Math.floor((t - a) / 0.042));
    const metrics = D.compose.metrics || [];
    const viz = D.compose.visualizations || [];
    const rows = [
      `<span class="c">$</span> ${typed}`,
      "",
      ...metrics.map((m) => `<span class="d">指标</span>    ${m}`),
      "",
      ...viz.map((v) => `<span class="d">可视化</span>  ${v}`),
    ];
    const shown = rows.slice(0, Math.max(1, Math.floor((t - a) / 0.34) + 1));
    const status = t > a + 5.2
      ? `<div style="position:absolute;left:340px;top:860px;font:600 24px var(--mono);color:#4fd1b0">✓ 已写入 output/blueprint.md</div>`
      : "";
    return bg(c, cp) + layer(c, cp, 1,
      `<div class="term" style="left:280px;top:120px;width:1360px;height:790px">
        <div class="term-head"><span class="dots"><i style="background:#e06c75"></i><i style="background:#f0a93b"></i><i style="background:#4fd1b0"></i></span><span>mmtrain</span></div>
        <div class="term-body" style="font-size:20px;line-height:42px">${shown.map((x) => `<div>${x}</div>`).join("")}</div></div>` + status);
  }

  // 11. 五道门 + 测试数
  function s11(t) {
    const [a] = span("s11");
    const c = cam(t, { x: 900, y: 540, s: 1.02, t: a }, [{ t: a + 0.2, d: 1.3, to: { y: 400 } }, { t: a + 1.7, d: 1.3, to: { y: 450 } }, { t: a + 3.2, d: 1.2, to: { y: 500 } }], 0.02);
    const cp = cam(t - 1 / FPS, { x: 900, y: 540, s: 1.02, t: a }, [{ t: a + 0.2, d: 1.3, to: { y: 400 } }, { t: a + 1.7, d: 1.3, to: { y: 450 } }, { t: a + 3.2, d: 1.2, to: { y: 500 } }], 0.02);
    const gates = D.gates.map(([name, desc], i) => {
      const on = t >= a + 0.4 + i * 0.42;
      return `<div class="gate ${on ? "on" : ""}" style="left:400px;top:${180 + i * 104}px">
        <div class="check">${on ? '<svg width="17" height="17" viewBox="0 0 16 16"><path d="M2 8.4l4 4L14 3" stroke="#4fd1b0" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>' : ""}</div>
        <div class="name">${name}</div><div class="desc">${desc}</div></div>`;
    }).join("");
    const k = ease((t - a - 1.6) / 0.5);
    const big = k > 0
      ? `<div class="mega" style="left:1440px;top:300px;font-size:${190 * mix(0.92, 1, k)}px;opacity:${k}">${D.counts.tests}</div>` +
        `<div class="sub" style="left:1444px;top:510px;opacity:${k}">个测试</div>`
      : "";
    return bg(c, cp) + layer(c, cp, 1, gates + big);
  }

  // 12. 五级晋级
  function s12(t) {
    const [a] = span("s12");
    const c = cam(t, { x: 960, y: 690, s: 0.99, t: a }, [{ t: a + 0.2, d: 5.0, to: { y: 570, s: 1.04 } }], 0.02);
    const cp = cam(t - 1 / FPS, { x: 960, y: 690, s: 0.99, t: a }, [{ t: a + 0.2, d: 5.0, to: { y: 570, s: 1.04 } }], 0.02);
    const rungs = [
      ["1", "CPU preflight", "组件可构造"],
      ["2", "GPU no-update", "数值有限"],
      ["3", "one-update", "梯度非 nan"],
      ["4", "checkpoint readback", "SHA-256 一致"],
      ["5", "冻结评估", "登记 receipt"],
    ];
    const steps = rungs.map(([n, name, desc], i) => {
      const on = t >= a + 0.4 + i * 0.42;
      return `<div class="rung ${on ? "on" : ""}" style="left:${120 + i * 350}px;top:${mix(760, 520, ease((t - a - 0.3 - i * 0.1) / 0.55))}px"><b>${n} · ${name}</b><span>${desc}</span></div>`;
    }).join("");
    const warnK = ease((t - cue("s12", "不等于")) / 0.4);
    const warn = `<div style="position:absolute;left:960px;top:240px;transform:translateX(-50%);font:600 32px var(--mono);color:#e06c75;opacity:${warnK}">launch_enabled = false</div>`;
    return bg(c, cp) + layer(c, cp, 1, warn + steps);
  }

  // 13. 片名
  function s13(t) {
    const [a] = span("s13");
    const c = cam(t, { x: 960, y: 540, s: 1.0, t: a }, [{ t: a, d: 2.6, to: { s: 1.05 } }], 0.02);
    const cp = cam(t - 1 / FPS, { x: 960, y: 540, s: 1.0, t: a }, [{ t: a, d: 2.6, to: { s: 1.05 } }], 0.02);
    const k = ease((t - a) / 0.55);
    const u = ease((t - a - 0.8) / 0.5);
    const html = `<div class="title-wrap" style="opacity:${k}">
      <div class="title-main">换<em>零件</em>，不换整套脚本</div>
      <div class="title-url" style="opacity:${u}">github.com/jackeyloveseven/yoyo</div>
    </div>`;
    return bg(c, cp) + layer(c, cp, 1, html);
  }

  const FNS = { s01, s02, s03, s04, s05, s06, s07, s08, s09, s10, s11, s12, s13 };

  function buildShots() {
    return T.beats.map((b, i) => ({
      a: i === 0 ? 0 : b.start - 0.16,
      z: i === T.beats.length - 1 ? T.duration : T.beats[i + 1].start - 0.02,
      id: b.id,
      fn: FNS[b.id] || s13,
    }));
  }

  function subtitle(t) {
    const b = T.beats.find((x) => t >= x.start - 0.08 && t <= x.end + 0.26);
    if (!b) return "";
    const k = Math.min(1, (t - b.start + 0.08) / 0.16, (b.end + 0.26 - t) / 0.16);
    return `<div style="position:absolute;left:0;right:0;bottom:44px;text-align:center;opacity:${clamp(k).toFixed(2)};z-index:60;pointer-events:none">
      <span style="display:inline-block;padding:7px 20px;border-radius:7px;background:#00000072;font:500 35px var(--han);color:#fff;text-shadow:0 2px 8px #000c">${b.text}</span></div>`;
  }

  function frame(t) {
    let shot = SHOTS[0];
    for (const s of SHOTS) if (t >= s.a) shot = s;
    return shot.fn(t) + subtitle(t);
  }

  window.ready = (async () => {
    D = await (await fetch("assets/data.json")).json();
    T = await (await fetch("timeline.json")).json();
    await document.fonts.load('100px "Bebas"');
    await document.fonts.ready;
    SHOTS = buildShots();
    stage.innerHTML = DEFS + frame(0.2);
  })();

  window.seek = (t) => {
    stage.innerHTML = DEFS + frame(t);
    return Promise.resolve();
  };
})();
