# 实验数据备份 · 2026-11-25

本目录保存服务器回收前的必要实验证据。覆盖 0、5、6 号机的自有实验目录，以及本地已有的冻结数据、报告与代码。完整文件数、体积和分卷 SHA256 见 [public_manifest.json](public_manifest.json)。

保留训练/评测输入、数据拆分、manifest/hash、任务与 gold、完整消息轨迹、逐项评测、汇总结果、配置、运行脚本、实验日志及论文实验材料。模型权重、LoRA 权重、优化器/分布式 checkpoint 张量、Docker 镜像、框架源码副本、Ray 运行日志、临时沙箱副本和登录材料不在备份范围。

只有在原始事件的 **SHA256 被含完整 messages 的汇总轨迹明确引用** 时，才用汇总轨迹替代重复事件流水；对应关系保存在加密包的 `*_event_compaction_audit.json` 中。大于 4 MiB 的设备监控流水保留首尾各 2 MiB，正式实验日志保持完整。备份以每个文件读取时的快照为准，服务器原文件未被清理。

## 文件位置

- 本地 `local/`：按 SHA256 去重的压缩内容库；`*_manifest.jsonl` 记录原路径及校验值。
- 本地加密分卷：同一备份的 AES-256-GCM 加密副本。为减少 D 盘重复占用，可暂存在本机另一磁盘；实际位置见 `encrypted_parts_location.txt`，恢复工具会自动读取。
- GitHub [Release 附件](https://github.com/linmumu009/qwen3.6-27b-trajory-dpo/releases/tag/experiment-backup-20261125)：加密分卷，避免大文件进入 Git 历史。
- 本地 `decryption_key.txt`：**解密密钥，仅在本机保存，不提交 Git，也不包含在备份包内。请另存到独立的安全位置；丢失后无法解密 GitHub 上的备份。**

仓库公开，因此原始数据、路径清单和解密密钥均由本目录 `.gitignore` 排除，采集脚本保存在加密包内；公开部分仅包含说明、恢复工具和不含服务器访问路径的汇总校验清单。

## 在本机查看或恢复

以下命令在项目根目录执行。恢复工具使用 Python 3.10+；解密需要 `cryptography`，本机已安装。

直接从本地内容库恢复指定实验，无需先解密：

```powershell
python experiment_backup_20261125/restore_backup.py --snapshot experiment_backup_20261125 --materialize experiment_backup_20261125/restored --match p001_frozen
```

`--match` 按原始路径中的文字筛选；不传该参数会展开全部文件，所需空间显著大于压缩备份。恢复后的文件按来源机器分目录。

## 从 GitHub 恢复

先克隆本仓库，再下载 Release 附件，并从独立保存的位置取回密钥：

```powershell
gh release download experiment-backup-20261125 --repo linmumu009/qwen3.6-27b-trajory-dpo --pattern "evidence-*.enc" --dir experiment_backup_20261125/encrypted
python experiment_backup_20261125/restore_backup.py --key-file experiment_backup_20261125/decryption_key.txt --snapshot experiment_backup_20261125/restored_snapshot --verify
python experiment_backup_20261125/restore_backup.py --snapshot experiment_backup_20261125/restored_snapshot --materialize experiment_backup_20261125/restored --match p001_frozen
```

解密会先校验全部分卷 SHA256 与认证标签，`--verify` 再检查所有原始内容的 SHA256 和清单引用。`restored_snapshot` 保持压缩去重结构，只有 `--materialize` 才展开为原始文件。
