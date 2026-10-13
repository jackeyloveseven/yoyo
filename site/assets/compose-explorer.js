/* YoYo — compose 兼容性演示（纯手写，无框架）
 * 数据来自 assets/registry-data.js（离线快照，与仓库注册表同源）。
 * 交互只是把 "required_model_capabilities ⊆ model.capabilities" 这条判定可视化，
 * 不引入任何后端调用。 */
(function () {
  "use strict";

  var DATA = window.YOYO_REGISTRY;
  var app = document.getElementById("explorer-app");
  if (!DATA || !app) return;

  var els = {
    modes: app.querySelectorAll(".ex-mode"),
    chips: document.getElementById("ex-chips"),
    pickLabel: document.getElementById("ex-pick-label"),
    detail: document.getElementById("ex-detail"),
    outTitle: document.getElementById("ex-out-title"),
    count: document.getElementById("ex-count"),
    list: document.getElementById("ex-list"),
    hint: document.getElementById("ex-hint")
  };

  var mode = "model";
  var pickedModel = "qwen_image_edit_2x";
  var pickedObjective = "image_dpo";

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  function objectivesFor(modelId) {
    return DATA.objectives.filter(function (o) { return o.row[modelId]; });
  }

  function modelsFor(objectiveId) {
    var o = DATA.objectives.filter(function (x) { return x.id === objectiveId; })[0];
    return o ? DATA.models.filter(function (m) { return o.row[m.id]; }) : [];
  }

  function shortRewardHint(o) {
    return o.reward_required
      ? "需注册一个 RewardBackend（" + DATA.rewards.slice(0, 3).join(" / ") + " …）"
      : "无需 reward";
  }

  /* ---------- 左侧：选择器 + 详情 ---------- */
  function renderChips() {
    els.chips.textContent = "";
    var items = mode === "model" ? DATA.models : DATA.objectives;
    items.forEach(function (item) {
      var id = item.id;
      var label = mode === "model" ? item.id : item.id;
      var btn = el("button", "chip" + (isActive(id) ? " is-active" : ""), label);
      btn.type = "button";
      btn.setAttribute("role", "tab");
      btn.setAttribute("aria-selected", isActive(id) ? "true" : "false");
      btn.dataset.id = id;
      btn.addEventListener("click", function () {
        if (mode === "model") pickedModel = id; else pickedObjective = id;
        render();
      });
      els.chips.appendChild(btn);
    });
    els.pickLabel.textContent = mode === "model" ? "选模型" : "选目标";
  }

  function isActive(id) {
    return mode === "model" ? id === pickedModel : id === pickedObjective;
  }

  function renderDetail() {
    els.detail.textContent = "";
    if (mode === "model") {
      var m = DATA.models.filter(function (x) { return x.id === pickedModel; })[0];
      var h = el("h4", null, m.name);
      els.detail.appendChild(h);
      els.detail.appendChild(el("p", null, m.desc));
      var p = el("p", null, "已注册能力");
      els.detail.appendChild(p);
      var caps = el("div", "ex-caps");
      m.capabilities.forEach(function (c) { caps.appendChild(el("span", "ex-cap", c)); });
      els.detail.appendChild(caps);
    } else {
      var o = DATA.objectives.filter(function (x) { return x.id === pickedObjective; })[0];
      els.detail.appendChild(el("h4", null, o.id + " · " + o.family));
      els.detail.appendChild(el("p", null, o.desc));
      els.detail.appendChild(el("p", null, "需要模型具备：" + o.required.join(" + ")));
      els.detail.appendChild(el("p", null, shortRewardHint(o)));
    }
  }

  /* ---------- 右侧：结果列表 ---------- */
  function renderOut() {
    els.list.textContent = "";
    var rows;
    if (mode === "model") {
      var modelId = pickedModel;
      els.outTitle.textContent = DATA.models.filter(function (x) {
        return x.id === modelId;
      })[0].name + " 可跑的目标";
      rows = DATA.objectives.map(function (o) {
        return { id: o.id, yes: o.row[modelId], reward: o.reward_required };
      });
      els.count.textContent =
        objectivesFor(modelId).length + " / " + DATA.objectives.length + " 可跑";
      els.hint.textContent =
        "标 · 的目标缺所需能力：" + capabilityGap(modelId) + "。" +
        (objectivesFor(modelId).some(function (o) { return o.reward_required; })
          ? "其中 " + objectivesFor(modelId).filter(function (o) { return o.reward_required; }).length +
            " 个还需注册 reward 后端。"
          : "");
    } else {
      var objectiveId = pickedObjective;
      els.outTitle.textContent = "支持 " + objectiveId + " 的模型";
      rows = DATA.models.map(function (m) {
        var o = DATA.objectives.filter(function (x) { return x.id === objectiveId; })[0];
        return { id: m.name, yes: o.row[m.id], reward: false };
      });
      els.count.textContent =
        modelsFor(objectiveId).length + " / " + DATA.models.length + " 支持";
      els.hint.textContent = capabilityGapForObjective(objectiveId);
    }

    rows.forEach(function (row) {
      var li = el("li", "ex-item " + (row.yes ? "is-yes" : "is-no"));
      li.appendChild(el("span", "mk", row.yes ? "✓" : "·"));
      li.appendChild(el("span", "name", row.id));
      if (row.reward) li.appendChild(el("span", "req", "需 reward"));
      else li.appendChild(el("span"));
      els.list.appendChild(li);
    });
  }

  function capabilityGap(modelId) {
    var m = DATA.models.filter(function (x) { return x.id === modelId; })[0];
    var missing = {};
    DATA.objectives.forEach(function (o) {
      if (o.row[modelId]) return;
      o.required.forEach(function (c) {
        if (m.capabilities.indexOf(c) < 0) missing[c] = true;
      });
    });
    var keys = Object.keys(missing);
    return keys.length ? keys.join("、") : "无";
  }

  function capabilityGapForObjective(objectiveId) {
    var o = DATA.objectives.filter(function (x) { return x.id === objectiveId; })[0];
    return "该目标要求模型具备 " + o.required.join(" + ") +
      "；不具备此能力的模型即使注册也不出现在结果里。" +
      (o.reward_required ? " 它还需要一个 RewardBackend。" : "");
  }

  function render() {
    renderChips();
    renderDetail();
    renderOut();
  }

  els.modes.forEach(function (btn) {
    btn.addEventListener("click", function () {
      mode = btn.dataset.mode;
      els.modes.forEach(function (b) {
        var on = b === btn;
        b.classList.toggle("is-active", on);
        b.setAttribute("aria-selected", on ? "true" : "false");
      });
      render();
    });
  });

  render();
})();
