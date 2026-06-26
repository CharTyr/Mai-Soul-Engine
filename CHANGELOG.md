# Changelog

## [2.0.0] — 2026-06-26

### 用户可感知

- 适配新版 MaiBot（maibot-plugin-sdk 2.x），插件在独立 Runner 子进程中运行。
- 意识形态光谱：问卷初始化、群聊自动演化、回复前注入（`maisaka.planner.before_request`）。
- 思维阁（可选）：种子审核、内化、trait 管理命令。
- Soul @API（可选）：光谱 / 历史 / traits / 种子查询与健康检查。
- Notion 前端同步（可选，默认关闭）。
- 从旧版宿主 `data/MaiBot.db` 自动迁移 `soul_*` 数据到插件 `data/soul.db`（幂等）。

### 开发侧

- manifest v2，`id`: `char-tyr.mai-soul-engine`。
- 数据层：peewee → stdlib `sqlite3`，插件自有 `soul.db`。
- 旧版 `src.plugin_system` / POST_LLM 注入已移除；仅 SDK 2.x。
- 旧版代码归档分支：`archive/legacy-sdk1-v1`，标签 `archive-sdk1-last`。