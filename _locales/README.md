# _locales

当前状态：**仅 zh-CN，i18n 未实施**。

`zh-CN.json` 仅含 plugin.name 和 plugin.description（SDK 要求的元数据）。
所有用户可见文本（命令回复、错误消息、进度提示）目前硬编码中文。

未来如需多语言支持，需要：
1. 把用户可见字符串抽到 `_locales/<locale>.json`
2. 命令 handler 通过 `plugin.i18n.t("key")` 读取（待 SDK 支持）
3. manifest 的 `supported_locales` 声明目标语言

当前不做 i18n 的理由：插件用户群全中文，无实际多语言需求。
