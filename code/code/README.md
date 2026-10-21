# 路桥巡检统一版本：全图视觉与有条件复核

线上与线下使用同一run.py、配置、底座、LoRA、监督头和案例资产，只更换合法输入及新输出目录。代码和离线依赖在code目录；模型权重单独预置，详见UPLOAD_README.md及EXTERNAL_MODEL_MANIFEST.json。方案在design目录，result目录只含同次实跑的result.json、infer_time.json。

## 恢复和执行

在赛方Python3.12与PPU匹配镜像上，先 source /etc/profile.d/dockerenv.sh，再直接用 /usr/local/bin/python3 运行 run.py。精简包不含91个离线wheel；需要重建独立环境时从外部 wheel 仓库恢复。模型权重不在上传包中：OSS 冷备份位于 ~/track4/assets/p2_runtime_assets，重启后先执行 ~/track4/reports/restore_models_to_root.sh 复制回 ~/track4_current 再跑，以保证本地 NVMe 速度。torch、torchvision、triton 使用赛方PPU 版本，禁止整体升级替换。

统一命令：

```bash
/usr/local/bin/python3 ~/result/code/run.py --input-dir INPUT --metadata META --output-dir NEW_OUTPUT
```

现场执行 ~/result/code/run.py；旧 ~/track4_current 与 ~/track4_current 入口已退役。输入元数据仅提供questionCategory、合法bridgeName、filename、imageRelativePath。桥名必须随当前输入提供。输出目录必须为新目录。所有模型加载、外层入口、shell、视觉计算和复核均计入infer_time。

## 已实现算法

冻结Qwen3-VL-Embedding-2B将每张图实时编码为2048维向量，长边1024，batch4、预处理8线程。轨道22类线性头；桥梁presence LR阈值0.8，698条病害案例、48类型、k=10投票决定初判类型与标度。视角头只选择健康描述及复核抽检分层，不能直接覆盖病害判断。线性头合计55,323参数，不含2B/27B底座。

最多6条、一般不超过批次5%（非空小批次至少1条）的训练后Qwen3.8-27B盲复核；120秒含加载，首图25秒、后续12秒、最多128输出token。最多2个健康抽检、2个病害结果抽检，其余按风险与任务配额选择。预算不足时缩减抽检保留风险名额。教师不读取学生答案、邻居标签、文件名、桥名或金标；匹配官方底座与clean_full LoRA。

桥梁非完好初判只有p_disease在[0.8,0.9]、教师返回合法“完好”、标度为空、uncertain=false且提供可见依据时，才允许改回完好。这是保守接受条件，未校准为正确率；航拍、近景使用同一条件。轨道病害改完好仍阻断，因为诊断发现教师可能清除真实锈蚀。其他非法、不确定、超时或未选中行保留初判；未知桥梁标度可保留案例投票标度。类型更新后描述仍按固定训练代表句填报。

TongAgents2.7.19实际执行embed_head_inspect、teacher_review，使用track4-inspect Skill和运行内SimpleMemory。工具流程固定，不是开放ReAct多智能体协商。七字段合同检查身份、条数、标签闭集及标度。没有Wiki/国标自动定级，没有蒸馏或在线训练。

## 来源与边界

底座来自Qwen官方开源权重，Qwen3.8-27B为项目目录别名，底座配置与LoRA匹配身份见code/model_assets/base_identity.json。自研部分为监督判别头、案例投票、限额复核与接受规则、字段合同及离线交付。教师仍可能判断错误，已有筛查不能证明新集得分必升；本版官方分待提交。运行覆盖、SHA一致、内存验收均不等于准确率。

本次上传排除全部模型权重，根目录只保留code、design、result。91个离线wheel保留；完整模型仍在服务器独立存储，按清单SHA恢复。测试原图、实验脚本、缓存、日志、虚拟环境和旧结果不装入提交目录，平台接收以实际回执为准。
